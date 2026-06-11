#  Copyright [2025] [Huawei Technologies Co., Ltd.] 
#  Licensed under the Code Sharing Policy of the UHD World Association (the "Policy"); 
#  http://www.theuwa.com/UWA_Code_Sharing_Policy.pdf.
#  you may not use this file except in compliance with the Policy. 
#  Unless agreed to in writing, software distributed under the Policy is distributed on an "AS IS" BASIS, 
#  WITHOUT WARRANTIES OF ANY KIND, either express or implied. 
#  See the Policy for the specific language governing permissions and 
#  limitations under the Policy.

import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from .utils import inverse_sigmoid
from .utils import strip_symmetric, build_scaling_rotation


class BaseGaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        self.semantics = 0
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._features_dc = torch.empty(0)
        # self._features_rest = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.setup_functions()

    def get_attribute(self, name, activate=True) -> Tensor:
        if name == 'means':
            return self.get_xyz
        elif name == 'quats':
            return self.get_rotation if activate else self._rotation
        elif name == 'scales':
            return self.get_scaling if activate else self._scaling
        elif name == 'opacitys':
            return self.get_opacity if activate else self._opacity
        elif name == 'features_dc':
            return self.get_features_dc
        elif name == 'features_sh':
            return self.get_features_rest
        else:
            raise RuntimeError(f'error parameter name:{name}')

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        if hasattr(self,"_features_rest"):
            return self._features_rest
        else:
            return torch.zeros_like(self._features_dc)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def load_ply(self, path):
        names = []
        isgs_content = True
        # try:
        num_vertices = 0
        with open(path,'rb') as op:
            for line in op:
                line = line.decode('utf-8').rstrip()
                if line.startswith('element vertex') or line.startswith('element skybox'):
                    num_vertices = int(line.split(" ")[2])
                elif line.startswith('element'):
                    isgs_content = False
                if line.startswith("property") and isgs_content:
                    name = line.split(" ")[-1]
                    names.append(name)
                if line == "end_header":
                    break
            points = np.frombuffer(op.read(), dtype=np.float32)
            points = points[:(len(points)//num_vertices)*num_vertices].reshape(num_vertices,-1)
            vertex_data = {}
            for idx,name in enumerate(names):
                vertex_data[name] = points[:,idx]
            
        # except Exception as e:
        #     print("not support current ply file")

        xyz = np.stack((np.asarray(vertex_data["x"]),
                        np.asarray(vertex_data["y"]),
                        np.asarray(vertex_data["z"])), axis=1)
        opacities = np.asarray(vertex_data["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(vertex_data["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(vertex_data["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(vertex_data["f_dc_2"])

        extra_f_names = [name for name in names if name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        self.max_sh_degree = int(((len(extra_f_names) + 3) / 3) ** 0.5) - 1
        if self.max_sh_degree > 0:
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(vertex_data[attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [name for name in names if name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(vertex_data[attr_name])

        rot_names = [name for name in names if name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(vertex_data[attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=self.device).requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(False))
        if self.max_sh_degree > 0:
            self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=self.device).requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=self.device).requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=self.device).requires_grad_(False))
        self.active_sh_degree = int(((len(extra_f_names) + 3) / 3) ** 0.5) - 1
        self.device = self._xyz.device

    def save_ply(self, path):
        path_dir = os.path.dirname(path)
        if len(path_dir) > 0:
            os.makedirs(path_dir, exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        ) if self.active_sh_degree > 0 and self._features_rest is not None else None
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        names = [name for name in self.construct_list_of_attributes()]
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        ) if self.active_sh_degree > 0 and f_rest is not None else np.concatenate(
            (xyz, normals, f_dc, opacities, scale, rotation), axis=1
        ).astype(np.float32)
        with open(path, "wb") as op:
            lines = []
            lines.append("ply\n".encode())
            lines.append("format binary_little_endian 1.0\n".encode())
            lines.append(f"element vertex {xyz.shape[0]}\n".encode())
            for name in names:
                lines.append(f"property float {name}\n".encode())
            lines.append("end_header\n".encode())
            op.writelines(lines)
            float_dtype = np.dtype(np.float32).newbyteorder("<")
            op.write(attributes.astype(float_dtype).tobytes())

    def save_quantized_ply(self, path, degree):
        self.active_sh_degree = degree
        self._opacity = self._opacity.clamp(-10,10)
        path_dir = os.path.dirname(path)
        if len(path_dir) > 0:
            os.makedirs(path_dir, exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        ) if self.active_sh_degree > 0 and self._features_rest is not None else None
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        names = [name for name in self.construct_list_of_attributes()]
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        ) if self.active_sh_degree > 0 and f_rest is not None else np.concatenate(
            (xyz, normals, f_dc, opacities, scale, rotation), axis=1
        ).astype(np.float32)
        with open(path, "wb") as op:
            lines = []
            lines.append("ply\n".encode())
            lines.append("format binary_little_endian 1.0\n".encode())
            lines.append(f"element vertex {xyz.shape[0]}\n".encode())
            for name in names:
                lines.append(f"property float {name}\n".encode())
            lines.append("end_header\n".encode())
            op.writelines(lines)
            float_dtype = np.dtype(np.float32).newbyteorder("<")
            op.write(attributes.astype(float_dtype).tobytes())

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
            
        if hasattr(self, "_features_rest") and self._features_rest is not None:
            component = (self._features_rest.shape[1] * self._features_rest.shape[2]) // 3
            if self.active_sh_degree > 0:
                for i in range(3):
                    for j in range(component):
                        l.append("f_rest_{}".format(i*component+j))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def convert_from_splat(self, gaussian_data):
        self._xyz = nn.Parameter(gaussian_data["means"].clone().requires_grad_(False))
        self._features_dc = nn.Parameter(gaussian_data["features_dc"].clone().requires_grad_(False))
        self._features_rest = nn.Parameter(gaussian_data["features_rest"].clone().requires_grad_(False))
        self._opacity = nn.Parameter(gaussian_data["opacity"].clone().requires_grad_(False))
        self._scaling = nn.Parameter(gaussian_data["scaling"].clone().requires_grad_(False))
        self._rotation = nn.Parameter(gaussian_data["rotation"].clone().requires_grad_(False))
        self.active_sh_degree = int(((gaussian_data["features_rest"].flatten().shape[-1] + 3) / 3) ** 0.5) - 1
        self.device = self._xyz.device