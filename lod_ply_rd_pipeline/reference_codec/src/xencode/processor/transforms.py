#  Copyright [2025] [Huawei Technologies Co., Ltd.] 
#  Licensed under the Code Sharing Policy of the UHD World Association (the "Policy"); 
#  http://www.theuwa.com/UWA_Code_Sharing_Policy.pdf.
#  you may not use this file except in compliance with the Policy. 
#  Unless agreed to in writing, software distributed under the Policy is distributed on an "AS IS" BASIS, 
#  WITHOUT WARRANTIES OF ANY KIND, either express or implied. 
#  See the Policy for the specific language governing permissions and 
#  limitations under the Policy.

import torch
from .sorters import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from ..utils import *

def approx_quantile(x, q, bins=1000):
    """
    Fast approximate quantile using histogram.
    ~10-20x faster than torch.quantile for large tensors.
    """
    if x.numel() < 1000:
        return torch.quantile(x, q)
    
    x_flat = x.flatten()
    hist = torch.histc(x_flat, bins=bins, min=x_flat.min().item(), max=x_flat.max().item())
    cumsum = torch.cumsum(hist, dim=0)
    target = q * x_flat.numel()
    idx = torch.searchsorted(cumsum, target)
    idx = torch.clamp(idx, 0, bins - 1)
    
    bin_edges = torch.linspace(x_flat.min(), x_flat.max(), bins + 1)
    bin_width = (bin_edges[1] - bin_edges[0])
    q_approx = bin_edges[0] + idx * bin_width
    
    return q_approx 

class AttributeTransform:
    transform_map_dict = {
        "default": {
            "global": "None",
            "post_global": "None",
            "means": "None",
            "scaling": "None",
            "rotation": "norm",
            "opacity": "None",
            "features_dc": "None",
            "features_rest": "None",
            "importance": "None",
        },
        "reduction_rsnorm": {
            "global": "rsnorm",
            "post_global": "None",
            "means": "None",
            "scaling": "None",
            "rotation": "reduction",
            "opacity": "None",
            "features_dc": "None",
            "features_rest": "None",
            "importance": "None",
        },
        "reduction_rsnorm_imp": {
            "global": "rsnormimp",
            "post_global": "None",
            "means": "None",
            "scaling": "None",
            "rotation": "reduction",
            "opacity": "None",
            "features_dc": "None",
            "features_rest": "imp",
            "importance": "None",
        },
    }

    def get_transform_processor(self, key, item):
        if key == "global":
            if item == "rsnorm": return RSNormTransform()
            elif item == "rsnormimp": return RSNormTransformImp()
        elif key == "means":
            if item == "imp": return ImportanceTransform()
        elif key == "scaling":
            if item == "imp": return ImportanceTransform()
        elif key == "rotation":
            if item == "reduction": return QuatReductionTransform()
            elif item == "imp": return ImportanceTransform()
        elif key == "opacity":
            if item == "imp": return ImportanceTransform()
        elif key == "features_dc":
            if item == "imp": return ImportanceTransform()
        elif key == "features_rest":
            if item == "imp": return ImportanceTransform()

    def __init__(self, transform_map = None, version="default") -> None:
        if transform_map is None:
            self.transform_map = self.transform_map_dict[version]
        else:
            self.transform_map = transform_map
        self.function_map = {}
        for key, item in self.transform_map.items():
            self.function_map[key] = self.get_transform_processor(key, item)

    def process(self, gaussian_data, cameras=None):
        metas = {}
        for key, value in self.transform_map.items():
            if key in self.function_map and self.function_map[key] is not None:
                gaussian_data, metas[key] = self.function_map[key].process(gaussian_data, cameras, key)
        return gaussian_data, metas

    def deprocess(self, gaussian_data, metas):
        for key, value in self.transform_map.items():
            if key in self.function_map and self.function_map[key] is not None:
                gaussian_data = self.function_map[key].deprocess(gaussian_data, metas.get(key, {}), key)
        return gaussian_data

class ImportanceTransform:
    def process(self, gaussian_data, camera=[], key=None):
        # factor = self.opacity_scaling_to_imp(gaussian_data).view(-1,1)
        if key in gaussian_data and gaussian_data[key] is not None:
            if (gaussian_data[key].dim()) == 2:
                factor = gaussian_data["importance"].view(-1,1)
            elif (gaussian_data[key].dim()) == 3:
                factor = gaussian_data["importance"].view(-1,1,1)
            gaussian_data[key] *= factor
        return gaussian_data, {}
    
    def deprocess(self, gaussian_data, meta, key):
        if key in gaussian_data and gaussian_data[key] is not None:
            factor = (gaussian_data["importance"]).clamp(1e-6,1)
            if (gaussian_data[key].dim()) == 2:
                factor = factor.view(-1,1)
            elif (gaussian_data[key].dim()) == 3:
                factor = factor.view(-1,1,1)
            gaussian_data[key] /= factor
        return gaussian_data

class RSNormTransform:
    
    def process(self, gaussian_data, camera=[], key=None):
        gaussian_data["rotation"], gaussian_data["scaling"] = canonicalize_rot_scaling(gaussian_data["rotation"], gaussian_data["scaling"])
        meta = None
        return gaussian_data, meta

    def deprocess(self, gaussian_data, meta, key):
        return gaussian_data

class RSNormTransformImp:
    
    def process(self, gaussian_data, camera=[], key=None):
        gaussian_data["rotation"], gaussian_data["scaling"] = canonicalize_rot_scaling(gaussian_data["rotation"], gaussian_data["scaling"])
        meta = None

        def get_norm(x):
            min = x.min()
            qt = approx_quantile(x - x.min(), 0.9) + 1e-6
            return (x - min).clamp(0, qt) / qt

        def func3(gaussian_data, a=10, b=1):
            scaling = get_norm(torch.log(torch.exp(gaussian_data["scaling"])).sum(dim=-1,keepdim=True))
            opacity = get_norm(torch.log(torch.sigmoid(gaussian_data["opacity"])+1e-6))
            factor =  ((opacity + 5 * scaling) / (1+5))
            factor = factor.view(-1,1) ** (b)
            factor = 0.05 + 0.95 * torch.sigmoid(a * (factor - 0.2))
            _min = torch.min(factor)
            _max = torch.max(factor)
            factor = torch.round(((factor - _min) / (_max - _min + 1e-6)) * (3)) / (3) * (_max - _min) + _min
            return factor

        factor = func3(gaussian_data)
        gaussian_data["importance"] = factor.view(-1,1)

        return gaussian_data, meta

    def deprocess(self, gaussian_data, meta, key):
        return gaussian_data

class QuatReductionTransform:
    def __init__(self, method="rotvec") -> None:
        self.method = method
        pass

    def quaternion_to_euler(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        quaternions: (N, 4) tensor, where each row is (w, x, y, z)
        return: (N, 3) tensor, where each row is (roll, pitch, yaw)
        """
        w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

        # Roll (x-axis)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis)
        sinp = 2 * (w * y - z * x)
        pitch = torch.where(
            torch.abs(sinp) >= 1,
            torch.sign(sinp) * (torch.pi / 2),
            torch.asin(sinp)
        )

        # Yaw (z-axis)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return torch.stack([roll, pitch, yaw], dim=1)
    
    def euler_to_quaternion(self, euler_angles: torch.Tensor) -> torch.Tensor:
        """
        euler_angles: (N, 3) tensor, where each row is (roll, pitch, yaw)
        return: (N, 4) tensor, where each row is (w, x, y, z)
        """
        roll, pitch, yaw = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
        
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return torch.stack([w, x, y, z], dim=1)

    def process(self, gaussian_data, cameras=[], key=None):
        if key in gaussian_data and gaussian_data[key] is not None:
            tmp = torch.zeros_like(gaussian_data[key])
            tmp[:,:3] = self.quaternion_to_euler(gaussian_data[key])
            gaussian_data[key] = tmp
        return gaussian_data, {}
    
    def deprocess(self, gaussian_data, meta, key):
        if key in gaussian_data and gaussian_data[key] is not None:
            gaussian_data[key] = gaussian_data[key][:,:3]
            gaussian_data[key] =  self.euler_to_quaternion(gaussian_data[key])
        return gaussian_data
