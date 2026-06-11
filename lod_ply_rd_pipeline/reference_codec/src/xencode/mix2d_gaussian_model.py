#  Copyright [2025] [Huawei Technologies Co., Ltd.] 
#  Licensed under the Code Sharing Policy of the UHD World Association (the "Policy"); 
#  http://www.theuwa.com/UWA_Code_Sharing_Policy.pdf.
#  you may not use this file except in compliance with the Policy. 
#  Unless agreed to in writing, software distributed under the Policy is distributed on an "AS IS" BASIS, 
#  WITHOUT WARRANTIES OF ANY KIND, either express or implied. 
#  See the Policy for the specific language governing permissions and 
#  limitations under the Policy.

import os
import time
import zlib
import numpy as np
import torch
import struct
import platform
import tempfile
import subprocess

from io import BytesIO
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from math import sqrt
from torch import nn, Tensor
from .decode import decode_stream
from .base_gaussian_model import BaseGaussianModel
from .utils import *
from .utils import mortonEncode

_module_path = Path(__file__).resolve()
_module_dir = _module_path.parent
astc_cmd = None
if platform.system() == "Linux":
    astc_cmd = os.path.join(str(_module_dir), "tools", "astcenc-sse2")
elif platform.system() == "Windows":
    astc_cmd = os.path.join(str(_module_dir), "tools", "astcenc-sse2.exe")
else:
    raise RuntimeError("Only execute in Linux or Windows!")

class Mix2DGaussianModel(BaseGaussianModel):

    def __init__(self, sh_degree: int):

        self._importance = None
        self.encoded_data_stream = None
        self.active_sh_degree = None
        self.max_sh_degree = None
        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._opacity = None
        self._rotation = None
        self._scaling = None
        self._xyz_min = None
        self._xyz_max = None
        self.chunk_size = None
        self._index = None
        self._chunk_index = None
        self._xyz_quant = None
        self.min_scale = None
        self.max_scale = None
        self._scaling_quant = None
        self._rotation_quant = None
        self._dc_min = None
        self._dc_max = None
        self._features_dc_quant = None
        self._features_rest_min = None
        self._features_rest_max = None
        self._features_rest_quant = None
        self._importance = None
        self._importance_quant = None
        self._opacity_quant = None
        self._2d_size = None
        self.max_tex_num_in_a_merged_tex = None
        self.merged_sh_texture_num = None

        super().__init__(sh_degree)

    def restore_fromgaussian(self, gaussian: BaseGaussianModel):
        """
        restore from a gaussian model
        """
        self.active_sh_degree = gaussian.active_sh_degree
        self.max_sh_degree = gaussian.max_sh_degree
        device = gaussian.device
        self._xyz = nn.Parameter(gaussian.get_attribute('means', False).detach().clone().requires_grad_(False))
        self._features_dc = nn.Parameter(gaussian.get_attribute('features_dc', False).detach().clone().requires_grad_(False))
        self._features_rest = nn.Parameter(gaussian.get_attribute('features_sh', False).detach().clone().requires_grad_(True))
        self._opacity = nn.Parameter(gaussian.get_attribute('opacitys', False).detach().clone().requires_grad_(False))
        self._rotation = nn.Parameter(gaussian.get_attribute('quats', False).detach().clone().requires_grad_(False))
        self._scaling = nn.Parameter(gaussian.get_attribute('scales', False).detach().clone().requires_grad_(False))
        self._sort_morton()
    
    def _sort_morton(self):
        """
        sort with morton order
        """
        with torch.no_grad():
            xyz_q = (
                    (2 ** 21 - 1)
                    * (self._xyz - self._xyz.min(0).values)
                    / (self._xyz.max(0).values - self._xyz.min(0).values)
            ).long()
            order = mortonEncode(xyz_q).sort().indices
            self._xyz = nn.Parameter(self._xyz[order], requires_grad=False)
            self._opacity = nn.Parameter(self._opacity[order], requires_grad=False)

            self._features_rest = nn.Parameter(
                self._features_rest[order], requires_grad=True
            )
            self._features_dc = nn.Parameter(
                self._features_dc[order], requires_grad=False
            )

            self._scaling = nn.Parameter(self._scaling[order], requires_grad=False)
            self._rotation = nn.Parameter(self._rotation[order], requires_grad=False)
    
    def apply_minmaxquant(self, config):
        """
        quantize the 3DGS attributes
        """
        # quant xyz
        self.chunk_size = 256
        if "pos" in config and config["pos"]:
            # split chunk
            self._xyz_min = []
            self._xyz_max = []
            for idx in range(len(self._xyz) // self.chunk_size + 1):
                if idx * self.chunk_size >= len(self._xyz):
                    break
                final_idx = (idx + 1) * self.chunk_size if (idx + 1) * self.chunk_size <= len(self._xyz) else len(self._xyz)
                self._xyz_min.append(torch.min(self._xyz[idx*self.chunk_size:final_idx], axis = 0, keepdim = True)[0])
                self._xyz_max.append(torch.max(self._xyz[idx*self.chunk_size:final_idx], axis = 0, keepdim = True)[0])
            self._xyz_min = torch.cat(self._xyz_min, axis = 0)
            self._xyz_max = torch.cat(self._xyz_max, axis = 0)
            self._index = torch.arange(len(self._xyz))
            self._chunk_index = torch.arange(len(self._xyz)) // self.chunk_size
            _x, _x_quant = self.quant(self._xyz[:,0], self._xyz_min[self._chunk_index,0], self._xyz_max[self._chunk_index,0], 16)
            _y, _y_quant = self.quant(self._xyz[:,1], self._xyz_min[self._chunk_index,1], self._xyz_max[self._chunk_index,1], 16)
            _z, _z_quant = self.quant(self._xyz[:,2], self._xyz_min[self._chunk_index,2], self._xyz_max[self._chunk_index,2], 16)
            _xyz  = torch.stack([_x, _y, _z], axis = -1).clone().detach().requires_grad_(False)

            self._xyz = _xyz
            self._xyz_quant = torch.stack([_x_quant, _y_quant, _z_quant], axis = -1).to(torch.int32)
        else:
            self._xyz = self._xyz
            self._xyz_quant = self._xyz
        
        # quant scale
        self.min_scale = torch.min(self._scaling)
        self.max_scale = torch.max(self._scaling)
        _scaling, _scaling_quant = self.quant(self._scaling, self.min_scale, self.max_scale, 8)
        self._scaling_quant = _scaling_quant.to(torch.uint8)
        if "scale" in config and config["scale"]:
            self._scaling = _scaling

        # quant rotation, rotation is special
        _rotation = torch.nn.functional.normalize(self._rotation).detach().requires_grad_(False)
        _rotation, _rotation_quant = self.quant(_rotation, -1, 1, 8)
        self._rotation_quant = _rotation_quant.to(torch.uint8).to(self.device)
        if "rot" in config and config["rot"]:
            self._rotation = _rotation.to(torch.float32).to(self.device)

        # quant dc
        SH_C0 = 0.28209479177387814
        dcs = torch.sort(self._features_dc.reshape(-1)).values
        self._dc_min = dcs[int(len(dcs) * 0.0005)]
        self._dc_max = dcs[int(len(dcs) * 0.9995)]
        _features_dc, _features_dc_quant = self.quant(
            self._features_dc, self._dc_min, self._dc_max, 8)
        self._features_dc_quant = _features_dc_quant.to(torch.uint8).squeeze()
        if "dc" in config and config["dc"]:
            self._features_dc = _features_dc
        
        # quant sh_rest
        self._features_rest_min = torch.min(self._features_rest)
        self._features_rest_max = torch.max(self._features_rest)
        _features_rest, _features_rest_quant = self.quant(
            self._features_rest, torch.min(self._features_rest), torch.max(self._features_rest), 8)
        self._features_rest_quant = _features_rest_quant.to(torch.uint8).squeeze()
        if "rest" in config and config["rest"]:
            self._features_rest = _features_rest

        # quant importance 
        if self._importance is not None:
            imp = self._importance.reshape(-1)
            imp, _ = torch.sort(imp, descending=True)
            max_imp = imp[int(0.004 * len(imp))]
            _importance , _importance_quant = self.quant(
            self._importance + max_imp / 250, 0, max_imp, 8)
            self._importance = _importance
            self._importance_quant = _importance_quant.to(torch.uint8).squeeze()
            self._importance_quant[self._importance_quant < 1] = 1

        # reset attributes( <-quant_recon value)
        device = self._opacity.device
        _opacity, _opacity_quant = self.quant(
            self.opacity_activation(self._opacity), 
            torch.tensor([0.0]).expand(self._opacity.shape).to(device), torch.tensor([1.0]).expand(self._opacity.shape).to(device), 8)
        _opacity[_opacity<0.001] = 0.001
        _opacity[_opacity>0.999] = 0.999   # avoid lose in inverse opacity activation
        if "opacity" in config and config["opacity"]:
            self._opacity = self.inverse_opacity_activation(_opacity).clone().detach().requires_grad_(False)
        self._opacity_quant = _opacity_quant.to(torch.uint8).squeeze()

        self._xyz = nn.Parameter(self._xyz.clone().detach().requires_grad_(False))
        self._features_dc = nn.Parameter(self._features_dc.clone().detach().requires_grad_(False))
        self._features_rest = nn.Parameter(self._features_rest.clone().detach().requires_grad_(False))
        self._opacity = nn.Parameter(self._opacity.clone().detach().requires_grad_(False))
        self._rotation = nn.Parameter(self._rotation.clone().detach().requires_grad_(False))
        self._scaling = nn.Parameter(self._scaling.clone().detach().requires_grad_(False))

    # quant the value
    def quant(self, value, chunk_min, chunk_max, bit_num):

        delta = chunk_max - chunk_min
        norm = value - chunk_min
        if not isinstance(delta, int):
            norm[delta < 1.e-6] = 0.0
            delta[delta < 1.e-6] = 1.e-6
        norm = norm / delta
        t = 2 ** bit_num - 1
        quant = torch.floor(norm * t + 0.5).to(torch.int32)
        quant[quant < 0] = 0
        quant[quant > t] = t

        new_value = (1.0 * quant / t) * delta + chunk_min
        return new_value, quant

    def _morton_sort(self, **kwargs):
        """
        sort with morton order and reshape
        """
        if kwargs["importance"].dim() <= 1:
            kwargs["importance"] = kwargs["importance"].unsqueeze(-1).repeat(1, self._features_rest.shape[1] * self._features_rest.shape[2])
        full_importance = kwargs["importance"]
        importance = torch.sum(full_importance, axis = 1)

        xyz_q = ((2**21 - 1)*(self._xyz - self._xyz.min(0).values)/(self._xyz.max(0).values - self._xyz.min(0).values)).long()
        order = mortonEncode(xyz_q).sort().indices
        self._xyz = self._xyz[order]
        self._opacity = self._opacity[order]

        self._features_rest = self._features_rest[order]
        self._features_dc = self._features_dc[order]

        self._scaling = self._scaling[order]
        self._rotation = self._rotation[order]
        importance = importance[order]
        full_importance = full_importance[order]

        block = kwargs["block"]
        args=kwargs["args"]
        if args.codec_sh_degree > 0 and self.active_sh_degree > 0:
            if args.merge_sh_texture:
                # Merge several sh parameters in several Textures
                width, height, max_tex_num_in_a_merged_tex, merged_number = self.calculate_merge(
                    len(self._xyz), (min(self.active_sh_degree, args.codec_sh_degree) + 1) ** 2 -1, block, args.max_merged_tex_size)
                self._2d_size =[height, width]
                self.max_tex_num_in_a_merged_tex = max_tex_num_in_a_merged_tex
                self.merged_sh_texture_num = merged_number
            else:
                # Save each sh parameter in a Texture!
                width = int(np.round(np.sqrt(len(self._xyz)) / block)) * block
                height = int((len(self._xyz) / width)/block +1)*block
                self._2d_size =[height, width]
                self.max_tex_num_in_a_merged_tex = 1
                self.merged_sh_texture_num = (min(self.active_sh_degree, args.codec_sh_degree) + 1) ** 2 - 1

            cat_len = width * height - len(self._xyz)
            _importance = torch.cat([importance, torch.zeros([cat_len]).to(self.device)], axis = 0)
            _xyz = torch.cat([self._xyz, torch.zeros([cat_len, 3]).to(self.device) + torch.mean(self._xyz, axis = 0).reshape(1,-1)], axis = 0)
            _scale = torch.cat([self._scaling, torch.zeros([cat_len, 3]).to(self.device) + torch.min(self._scaling)], axis = 0)
            _rotation = torch.cat([self._rotation, torch.ones([cat_len, 4]).to(self.device)], axis = 0)
            _opacity = torch.cat([self._opacity, torch.zeros([cat_len, 1]).to(self.device) - 10], axis = 0)
            _features_dc = torch.cat([self._features_dc, torch.zeros([cat_len, 1, 3]).to(self.device)], axis = 0)
            _features_rest = torch.cat([self._features_rest, torch.zeros([cat_len, self._features_rest.shape[1], self._features_rest.shape[2]]).to(self.device)], axis = 0)
            _full_importance = torch.cat([full_importance, torch.zeros([cat_len, self._features_rest.shape[1] * self._features_rest.shape[2]]).to(self.device)], axis = 0)


            xyz, scale, rotation, opacity, features_dc, features_rest, full_importances = \
            _xyz, _scale, _rotation, _opacity, _features_dc, _features_rest, _full_importance
            
            h,w = self._2d_size[0] // kwargs["block"], self._2d_size[1] // kwargs["block"]
            self._xyz = xyz.reshape(-1, 3)
            self._scaling = scale.reshape(-1, 3)
            self._rotation = rotation.reshape(-1, 4)
            self._opacity = opacity.reshape(-1, 1)
            self._features_dc = features_dc.reshape(-1, 1, 3)
            self._features_rest = features_rest.reshape((h,w,kwargs["block"],kwargs["block"],-1)).permute((0,2,1,3,4)).reshape(len(self._xyz), -1, 3)
            self._importance = full_importances.reshape(len(self._xyz), -1, 3)
        else:
            self.merged_sh_texture_num = 0

    def _morton_and_blocksort(self, **kwargs):
        """
        sort with morton order and re-sort in blocks
        """
        if kwargs["importance"].dim() <= 1:
            kwargs["importance"] = kwargs["importance"].unsqueeze(-1).repeat(1, self._features_rest.shape[1] * self._features_rest.shape[2])
        full_importance = kwargs["importance"]
        importance = torch.sum(full_importance, axis = 1)

        xyz_q = ((2**21 - 1)*(self._xyz - self._xyz.min(0).values)/(self._xyz.max(0).values - self._xyz.min(0).values)).long()
        order = mortonEncode(xyz_q).sort().indices
        self._xyz = self._xyz[order]
        self._opacity = self._opacity[order]

        self._features_rest = self._features_rest[order]
        self._features_dc = self._features_dc[order]

        self._scaling = self._scaling[order]
        self._rotation = self._rotation[order]
        importance = importance[order]
        full_importance = full_importance[order]

        block = kwargs["block"]
        args=kwargs["args"]
        if args.codec_sh_degree > 0 and self.active_sh_degree > 0:
            if args.merge_sh_texture:
                # Merge several sh parameters in several Textures
                width, height, max_tex_num_in_a_merged_tex, merged_number = self.calculate_merge(
                    len(self._xyz), (min(self.active_sh_degree, args.codec_sh_degree) + 1) ** 2 -1, block, args.max_merged_tex_size)
                self._2d_size =[height, width]
                self.max_tex_num_in_a_merged_tex = max_tex_num_in_a_merged_tex
                self.merged_sh_texture_num = merged_number
            else:
                # Save each sh parameter in a Texture!
                width = int(np.round(np.sqrt(len(self._xyz)) / block)) * block
                height = int((len(self._xyz) / width)/block +1)*block
                self._2d_size =[height, width]
                self.max_tex_num_in_a_merged_tex = 1
                self.merged_sh_texture_num = (min(self.active_sh_degree, args.codec_sh_degree) + 1) ** 2 - 1

            cat_len = width * height - len(self._xyz)
            _importance = torch.cat([importance, torch.zeros([cat_len]).to(self.device)], axis = 0)
            _xyz = torch.cat([self._xyz, torch.zeros([cat_len, 3]).to(self.device) + torch.mean(self._xyz, axis = 0).reshape(1,-1)], axis = 0)
            _scale = torch.cat([self._scaling, torch.zeros([cat_len, 3]).to(self.device) + torch.min(self._scaling)], axis = 0)
            _rotation = torch.cat([self._rotation, torch.ones([cat_len, 4]).to(self.device)], axis = 0)
            _opacity = torch.cat([self._opacity, torch.zeros([cat_len, 1]).to(self.device) - 10], axis = 0)
            _features_dc = torch.cat([self._features_dc, torch.zeros([cat_len, 1, 3]).to(self.device)], axis = 0)
            _features_rest = torch.cat([self._features_rest, torch.zeros([cat_len, self._features_rest.shape[1], self._features_rest.shape[2]]).to(self.device)], axis = 0)
            _full_importance = torch.cat([full_importance, torch.zeros([cat_len, self._features_rest.shape[1] * self._features_rest.shape[2]]).to(self.device)], axis = 0)

            try:
                from block_sort._C import block_sort
                xyz, scale, rotation, opacity, features_dc, features_rest, full_importances = \
                block_sort(_importance, _xyz, _scale, _rotation, _opacity, _features_dc, _features_rest, _full_importance, \
                    block, width, height, self._features_rest.shape[1] * self._features_rest.shape[2])
            except Exception as e:
                print("Not Support Cuda, using cpu sorting! ")
                xyz, scale, rotation, opacity, features_dc, features_rest, full_importances = \
                self._block_sort(_importance, _xyz, _scale, _rotation, _opacity, _features_dc, _features_rest, _full_importance, \
                    block, width, height)
            h,w = xyz.shape[0] // kwargs["block"], xyz.shape[1] // kwargs["block"]
            self._xyz = xyz.reshape((h,kwargs["block"],w,kwargs["block"],-1)).permute((0,2,1,3,4)).reshape(-1, 3)
            self._scaling = scale.reshape((h,kwargs["block"],w,kwargs["block"],-1)).permute((0,2,1,3,4)).reshape(-1, 3)
            self._rotation = rotation.reshape((h,kwargs["block"],w,kwargs["block"],-1)).permute((0,2,1,3,4)).reshape(-1, 4)
            self._opacity = opacity.reshape((h,kwargs["block"],w,kwargs["block"],-1)).permute((0,2,1,3,4)).reshape(-1, 1)
            self._features_dc = features_dc.reshape((h,kwargs["block"],w,kwargs["block"],-1)).permute((0,2,1,3,4)).reshape(-1, 1, 3)
            self._features_rest = features_rest.reshape(len(self._xyz), -1, 3)
            self._importance = full_importances.reshape((h,kwargs["block"],w,kwargs["block"],-1)).permute((0,2,1,3,4)).reshape(len(self._xyz), -1, 3)
        else:
            self.merged_sh_texture_num = 0

    def zigzag_traverse(self, width, height, args=None):
        """
        generate zigzag order
        """
        indices = []
        
        # 遍历所有对角线
        for d in range(width + height - 1):
            # 确定对角线的方向
            if d % 2 == 0:
                # 向上遍历
                x = min(d, width - 1)
                y = max(0, d - width + 1)
                while x >= 0 and y < height:
                    indices.append([y, x])
                    x -= 1
                    y += 1
            else:
                # 向下遍历
                y = min(d, height - 1)
                x = max(0, d - height + 1)
                while y >= 0 and x < width:
                    indices.append([y, x])
                    y -= 1
                    x += 1
        
        indices = torch.tensor(indices).to(torch.int)
        #print(f"zig_indices={indices}")
        order = torch.sum(indices, axis = 1)
        dif = torch.abs(indices[:,0] - indices[:,1])
        sort_by = order * (width + height) + dif
        #print(f"before sort, sort_by={sort_by}")
        val, idx = torch.sort(sort_by)
        #print(f"after sort, sort_by={val}, idx={idx}")
        return indices[idx]

    def sort_with_distance(self, img, order = None, args=None):
        K = img.reshape(img.shape[0], -1)
        height = img.shape[1]
        width = img.shape[2]

        dis = torch.sum(torch.abs(K.unsqueeze(-1) - K.unsqueeze(1)), axis = 0)
        full_index = torch.arange(img.shape[1] * img.shape[2])
        exists = full_index > -1
        
        if order is None:
            order = self.zigzag_traverse(width, height, args)
        #print(f"order={order}")
        val = torch.zeros_like(img)
        idx = torch.zeros([height, width]).to(torch.int)
        
        val[:, 0, 0] = img[:, order[0][0], order[0][1]]
        idx[0, 0] = order[0][0] * width + order[0][1]
        exists[idx[0,0]] = False

        for i in range(1, len(order)):
            item = order[i]
            item_index = item[0] * width + item[1]
            
            d = None
            if (item[0] - 1) >= 0:
                adj_idx = idx[item[0] - 1, item[1]]
                if d is None:
                    d = dis[adj_idx][exists]
                else:
                    d += dis[adj_idx][exists]
                
            if (item[1] - 1) >= 0:
                adj_idx = idx[item[0], item[1] - 1]
                if d is None:
                    d = dis[adj_idx][exists]
                else:
                    d += dis[adj_idx][exists]

            exists_index = full_index[exists]
            v, sel_i = torch.sort(d)
            sel_idx = exists_index[sel_i[0]]
            exists[sel_idx] = False
            idx[item[0], item[1]] = sel_idx
            val[:,item[0], item[1]] = K[:, sel_idx]
        return val, idx

    def _block_sort(self, _importance, _xyz, _scale, _rotation, _opacity, _features_dc, _features_rest, _full_importance, block, width, height):
        """
        resort within each block
        """
        xyz = torch.zeros(height, width, 3).to(self.device)
        scale = torch.zeros(height, width, 3).to(self.device)
        rotation = torch.zeros(height, width, 4).to(self.device)
        opacity = torch.zeros(height, width, 1).to(self.device) - 10
        features_dc = torch.zeros(height, width, 1, 3).to(self.device)
        features_rest = torch.zeros(height, width, self._features_rest.shape[1], self._features_rest.shape[2]).to(self.device)
        full_importances = torch.zeros(height, width, self._features_rest.shape[1], self._features_rest.shape[2]).to(self.device)


        for _idy in tqdm(range((height // block))):
            for _idx in range(width // block):
                _id = _idy*(width // block)+_idx
                count = _id*block*block
                h_point = _idy*block
                w_point = _idx*block
                value, indices = torch.sort(_importance[count: count+block*block], descending=True)
                # 然后计算相似性            
                distance = torch.sum(torch.pow(_features_rest[count: count+block*block][indices[0]:indices[0] + 1] - _features_rest[count: count+block*block], 2), axis = [1,2])
                v, dis_indices = torch.sort(distance)
                sh_block = _features_rest[count: count+block*block][dis_indices].reshape([block, block, -1]).permute(2,0,1)
                v, sort_indices = self.sort_with_distance(sh_block)
                sort_indices = dis_indices[sort_indices.reshape(-1).long()]
                xyz[h_point: h_point + block, w_point: w_point + block, :] = _xyz[count: count+block*block][sort_indices].reshape([block, block, -1])
                scale[h_point:h_point + block, w_point:w_point + block, :] = _scale[count: count+block*block][sort_indices].reshape([block, block, -1])
                rotation[h_point:h_point + block, w_point:w_point + block, :] = _rotation[count: count+block*block][sort_indices].reshape([block, block, -1])
                opacity[h_point:h_point + block, w_point:w_point + block, :] = _opacity[count: count+block*block][sort_indices].reshape([block, block, -1])
                features_dc[h_point:h_point + block, w_point:w_point + block, :] = _features_dc[count: count+block*block][sort_indices].reshape([block, block, 1, 3])
                features_rest[h_point:h_point + block, w_point:w_point + block, :] = _features_rest[count: count+block*block][sort_indices].reshape([block, block, -1, 3])
                full_importances[h_point:h_point + block, w_point:w_point + block, :] = _full_importance[count: count+block*block][sort_indices].reshape([block, block, -1, 3])


        return xyz, scale, rotation, opacity, features_dc, features_rest, full_importances
                    

    def save_encoded_model(self, filepath, args):
        """
        save the encoded 3dgs into bitstreams
        """
        path = os.path.dirname(filepath)
        os.makedirs(path, exist_ok=True)

        if self.encoded_data_stream is None:
            stream = self.save_encoded_model_to_stream(args)
        else:
            stream = self.encoded_data_stream

        with open(filepath, "wb") as f:
            f.write(stream)

    def save_encoded_model_to_stream(self, args):
        """
        transform the encoded 3dgs into bitstreams
        """
        # save the image
        if hasattr(self, "_2d_size"):
            img_size = self._2d_size

        astc_block = args.astc_block
        # TODO: passing cmd parameter instead of using constant value
        astc_file_sizes = []
        astc_stream = None

        def generate_shell_script(image_paths, output_paths, block_size='4x4', preset='-medium', log_path = "log"):
            """
            生成一个 Shell 脚本，用于批量压缩图片。
            """
            script_content = ""
            for img_path, out_path in zip(image_paths, output_paths):
                cmd = [
                    astc_cmd,
                    '-cl', img_path,
                    out_path,
                    block_size,
                    preset, f">{log_path} 2>&1"
                ]
                script_content += " ".join(cmd) + "\n"
            return script_content

        start = time.time()
        if self.merged_sh_texture_num > 0:
            temp_dir_path = '/dev/shm' if platform.system() == 'Linux' else None
            with tempfile.TemporaryDirectory(dir=temp_dir_path) as temp_dir:
                image_paths = []
                output_paths = []
                # log_path = os.path.join(temp_dir, f"out.log")
                log_path = os.path.join(".", f"out.log")

                for idx in range(self.merged_sh_texture_num):
                    start_idx = self.max_tex_num_in_a_merged_tex * idx
                    end_idx = min(self.max_tex_num_in_a_merged_tex * (idx + 1), self._features_rest_quant.shape[1])

                    img_data = [self._features_rest_quant[:, ti].reshape(img_size + [3]).detach() for ti in range(start_idx, end_idx)]
                    img_data = torch.cat(img_data, axis = 1).cpu().numpy()
                    img = Image.fromarray(img_data)
                    temp_image_path = os.path.join(temp_dir, f"image{idx}.bmp")
                    img.save(temp_image_path, format="BMP")
                    image_paths.append(temp_image_path)

                    if self._importance is not None:
                        img_data = [self._importance_quant[:,idx].reshape(img_size + [3]).detach() for ti in range(start_idx, end_idx)]
                        img_data = torch.cat(img_data, axis = 1).cpu().numpy()
                        img = Image.fromarray(img_data)
                        temp_image_path = os.path.join(temp_dir, f"image{idx}_weight.bmp")
                        img.save(temp_image_path)

                    output_paths.append(os.path.join(temp_dir, f"image{idx}.astc"))
                
                # 创建并写入 Shell 脚本
                shell_script = generate_shell_script(image_paths, output_paths, f"{astc_block}x{astc_block}", f"-{args.astc_mode}", log_path)
                shell_script_path = None
                if platform.system() == 'Linux':
                    shell_script_path = os.path.join(temp_dir, "compress.sh")
                    with open(shell_script_path, 'w') as f:
                        f.write(shell_script)                # 赋予脚本执行权限
                    # 执行 Shell 脚本
                    subprocess.call(["bash", shell_script_path], shell=False, stdout=open(os.path.join(temp_dir, "command.log"), "w"))
                else:
                    shell_script_path = os.path.join(temp_dir, "compress.bat")
                    with open(shell_script_path, 'w') as f:
                        f.write(shell_script)                # 赋予脚本执行权限
                    subprocess.call(shell_script_path, shell=False, stdout=open(os.path.join(temp_dir, "command.log"), "w"))
                

                for idx in range(self.merged_sh_texture_num):
                    tmp_file = open(output_paths[idx], "rb")
                    stream = tmp_file.read()
                    astc_file_sizes.append(len(stream))
                    tmp_file.close()
                    if astc_stream is not None:
                        astc_stream += stream
                    else:
                        astc_stream = stream

        end = time.time()

        # pack stream first!
        data_buffer = BytesIO()
        xyz = self._xyz_quant.cpu().numpy().astype(np.uint16).tobytes()
        scale = self._scaling_quant.cpu().numpy().astype(np.uint8).tobytes()
        rot = self._rotation_quant.cpu().numpy().astype(np.uint8).tobytes()
        dc_op = torch.cat([self._features_dc_quant, self._opacity_quant.unsqueeze(-1)], axis = -1).squeeze().cpu().numpy().astype(np.uint8).tobytes()
        data_buffer.write(xyz)
        data_buffer.write(scale)
        data_buffer.write(rot)
        data_buffer.write(dc_op)
        if astc_stream is not None:
            data_buffer.write(astc_stream)
        data_bitstream = data_buffer.getvalue()
        data_bitstream_unzip_len = len(data_bitstream)
        if int(args.xencode_compress) == 1:
            data_bitstream = zlib.compress(data_bitstream)
        data_buffer.close()
        
        # write the buffer!
        buffer = BytesIO()
        buffer.write("gsct".encode("ascii")) # magic number
        buffer.write(struct.pack("<BBB", 1, 0, 0))
        buffer.write(struct.pack("<I", data_bitstream_unzip_len)) # length of bitstream 
        buffer.write(struct.pack("<B", int(args.xencode_compress))) 
        buffer.write(struct.pack("<I", 0)) # 保留字段
        buffer.write(struct.pack("<I", len(self._xyz)))
        numAttribute = 5 if (self._features_rest is not None and self.merged_sh_texture_num > 0) else 4
        buffer.write(struct.pack("<B", numAttribute))

        # 打包属性数据
        # 打包position数据
        attributeType = 0
        componentsCount = 3
        uncompressedDataType = 4
        attributeQuantizationFlag = 1
        attributeEncoderScheme = 0
        buffer.write(struct.pack("<IBBB", attributeType, componentsCount, uncompressedDataType, (attributeQuantizationFlag << 4) | attributeEncoderScheme))
        buffer.write(struct.pack("<B", 16))
        buffer.write(torch.min(self._xyz, axis = 0).values.detach().cpu().numpy().astype(np.float32).tobytes())
        buffer.write(torch.max(self._xyz, axis = 0).values.detach().cpu().numpy().astype(np.float32).tobytes())
        patchNum = (len(self._xyz) + self.chunk_size - 1) // self.chunk_size
        buffer.write(struct.pack("<IIII", 0, len(xyz), 4 * componentsCount * len(self._xyz), patchNum))
        byteOffset = len(xyz)
        # write the patch number!
        patchFlag = (0 << 15) | (0 << 14) | (0 << 13) | (1 << 12) | (0 << 11)
        lastPatchSize = len(self._xyz) % self.chunk_size
        if lastPatchSize == 0:
            lastPatchSize = self.chunk_size
        buffer.write(struct.pack("<HI", patchFlag, lastPatchSize))
        for i in range(patchNum):
            buffer.write(self._xyz_min[i].detach().cpu().numpy().astype(np.float32).tobytes())
            buffer.write(self._xyz_max[i].detach().cpu().numpy().astype(np.float32).tobytes())
        
        # 打包scale数据
        attributeType = 2
        componentsCount = 3
        uncompressedDataType = 4
        attributeQuantizationFlag = 1
        attributeEncoderScheme = 0
        buffer.write(struct.pack("<IBBB", attributeType, componentsCount, uncompressedDataType, (attributeQuantizationFlag << 4) | attributeEncoderScheme))
        buffer.write(
            struct.pack("<Bffffff", 8, 
                self.min_scale.item(), self.min_scale.item(), self.min_scale.item(),
                self.max_scale.item(), self.max_scale.item(), self.max_scale.item()
                )
            )
        buffer.write(struct.pack("<IIII", byteOffset, len(scale), 4 * componentsCount * len(self._xyz), 1))
        byteOffset += len(scale)

        # 打包rotation 数据
        attributeType = 1
        componentsCount = 4
        uncompressedDataType = 4
        attributeQuantizationFlag = 1
        attributeEncoderScheme = 0
        buffer.write(struct.pack("<IBBB", attributeType, componentsCount, uncompressedDataType, (attributeQuantizationFlag << 4) | attributeEncoderScheme))
        buffer.write(
            struct.pack("<Bffffffff", 8, 
                -1.0, -1.0, -1.0, -1.0,
                1.0, 1.0, 1.0, 1.0
                )
            )
        buffer.write(struct.pack("<IIII", byteOffset, len(rot), 4 * 4 * len(self._xyz), 1))
        byteOffset += len(rot)

        # 打包dc_op数据
        attributeType = 3
        componentsCount = 4
        uncompressedDataType = 4
        attributeQuantizationFlag = 1
        attributeEncoderScheme = 0
        buffer.write(struct.pack("<IBBB", attributeType, componentsCount, uncompressedDataType, (attributeQuantizationFlag << 4) | attributeEncoderScheme))
        buffer.write(
            struct.pack("<Bffffffff", 8, 
                self._dc_min.item(), self._dc_min.item(), self._dc_min.item(), 0.0,
                self._dc_max.item(), self._dc_max.item(), self._dc_max.item(), 1.0
                )
            )
        buffer.write(struct.pack("<IIII", byteOffset, len(dc_op), 4 * componentsCount * len(self._xyz), 1))
        byteOffset += len(dc_op)

        # 打包 sh 数据
        if astc_stream is not None:
            attributeType = 4
            componentsCount = 3 * ((min(self.active_sh_degree, args.codec_sh_degree) + 1)**2 - 1)
            uncompressedDataType = 4
            attributeQuantizationFlag = 1
            attributeEncoderScheme = 2
            buffer.write(struct.pack("<IBBB", attributeType, componentsCount, uncompressedDataType, (attributeQuantizationFlag << 4) | attributeEncoderScheme))
            buffer.write(struct.pack("<B", 8))
            buffer.write(np.array([self._features_rest_min.item() for i in range(componentsCount)]).astype(np.float32).tobytes())
            buffer.write(np.array([self._features_rest_max.item() for i in range(componentsCount)]).astype(np.float32).tobytes())
            # 2D 化的特征写入
            buffer.write(
                struct.pack("<BHHBBBBB", 
                    0, img_size[0], img_size[1], 0, 
                    1, self.max_tex_num_in_a_merged_tex, 1, len(astc_file_sizes)))
            buffer.write(np.array(astc_file_sizes).astype(np.uint32).tobytes())
            buffer.write(struct.pack("<IIII", byteOffset, len(astc_stream), 4 * componentsCount * len(self._xyz), 1))

        buffer.write(data_bitstream)
        total_data_bitstream = buffer.getvalue()
        buffer.close()

        return total_data_bitstream

    def load_encoded_model(self, file_path):
        """
        load encoded model from path
        """
        with open(file_path, 'rb') as ply_file:
            data_stream = bytearray(ply_file.read())
            self.load_encoded_model_from_stream(data_stream)


    def load_encoded_model_from_stream(self, data_stream):
        """
        load encoded model from stream file
        """
        
        info = decode_stream(data_stream)

        # Step5. reset parameters
        self._xyz = nn.Parameter(torch.from_numpy(info["positions"]).to(torch.float).to(device=self.device).contiguous()).requires_grad_(False)
        self._features_dc = nn.Parameter(torch.from_numpy(info["dc"]).to(torch.float).to(device=self.device).contiguous()).requires_grad_(False)
        self._features_dc = self._features_dc.unsqueeze(1)
        self._features_dc.reshape(self._features_dc.shape[0], 1, 3)
        if info["sh_degree"] > 0:
            self._features_rest = nn.Parameter(torch.from_numpy(info["shs"].reshape(len(self._xyz), -1 ,3)).to(torch.float).to(device=self.device).contiguous()).requires_grad_(False)
        else:
            self._features_rest = None
        self._opacity = nn.Parameter(torch.from_numpy(info["opacity"]).to(device=self.device).to(torch.float).contiguous()).requires_grad_(False)
        self._opacity = self._opacity.reshape(-1, 1)
        self._scaling = nn.Parameter(torch.from_numpy(info["scale"]).to(device=self.device).to(torch.float).contiguous()).requires_grad_(False)
        self._rotation = nn.Parameter(torch.from_numpy(info["rotation"]).to(device=self.device).to(torch.float).contiguous()).requires_grad_(False)
        self.active_sh_degree = int(info["sh_degree"])

        mask = torch.sigmoid(self._opacity.reshape(-1)) > 0.0011
        self._xyz = self._xyz[mask]
        self._features_dc = self._features_dc[mask]
        if self.active_sh_degree > 0:
            self._features_rest = self._features_rest[mask]
        self._opacity = self._opacity[mask]
        self._scaling = self._scaling[mask]
        self._rotation = self._rotation[mask]
    
    def calculate_merge(self, gs_num, tex_num, block_size, merged_max_tex_size = 4096):
        """
        calculate appropriate size for shN textures
        """

        single_tex_num = np.power(int(round(np.sqrt(gs_num) / block_size + 0.4999999999999) * block_size), 2)
        tuned_merged_max_tex_size = (merged_max_tex_size // block_size) * block_size

        max_tex_num_in_a_merged_tex = (tuned_merged_max_tex_size * tuned_merged_max_tex_size) // single_tex_num
        if max_tex_num_in_a_merged_tex < 1:
            raise("GS Model is too big, single texture can't constain one SH parameter!")
        elif max_tex_num_in_a_merged_tex > tex_num:
            max_tex_num_in_a_merged_tex = tex_num
        merged_number = int(round(tex_num / max_tex_num_in_a_merged_tex + 0.499999999999999))
        
        tuned_width = int(round(np.sqrt(single_tex_num * max_tex_num_in_a_merged_tex) / (block_size * max_tex_num_in_a_merged_tex) + 0.4999999999999) * block_size)
        if tuned_width > int(round(np.sqrt(gs_num) / block_size + 0.5) * block_size):
            tuned_width = int(round(np.sqrt(gs_num) / block_size + 0.5) * block_size)
        tuned_height = int((gs_num / tuned_width) / block_size + 0.9999999999999) * block_size
        
        return tuned_width, tuned_height, max_tex_num_in_a_merged_tex, merged_number