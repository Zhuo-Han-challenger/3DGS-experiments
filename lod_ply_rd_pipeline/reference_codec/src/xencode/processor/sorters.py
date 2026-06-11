#  Copyright [2025] [Huawei Technologies Co., Ltd.] 
#  Licensed under the Code Sharing Policy of the UHD World Association (the "Policy"); 
#  http://www.theuwa.com/UWA_Code_Sharing_Policy.pdf.
#  you may not use this file except in compliance with the Policy. 
#  Unless agreed to in writing, software distributed under the Policy is distributed on an "AS IS" BASIS, 
#  WITHOUT WARRANTIES OF ANY KIND, either express or implied. 
#  See the Policy for the specific language governing permissions and 
#  limitations under the Policy.

import torch
import math
import block_sort_cpp
from ..utils import find_densest_voxel

class MortonSpatialSorter:
    def __init__(self, padding = True, block_size=4, pca=False, distance=False) -> None:
        self.padding = padding
        self.block_size = block_size
        self.pca = pca
        self.distance = distance

    """Morton码空间排序"""
    def process(self, gaussian_data, **kwargs):
        if self.padding:
            gs_num = gaussian_data["means"].shape[0]
            sidelen = math.ceil(gs_num ** 0.5 / self.block_size) * self.block_size
            padding = sidelen * sidelen - gs_num
            for key, value in gaussian_data.items():
                if key == "opacity":
                    padded_tensor = torch.zeros([padding] + list(value.shape[1:]), device=value.device, dtype=value.dtype) - 10
                elif key == "importance":
                    padded_tensor = torch.ones([padding] + list(value.shape[1:]), device=value.device, dtype=value.dtype)
                else:
                    padded_tensor = torch.zeros([padding] + list(value.shape[1:]), device=value.device, dtype=value.dtype)
                gaussian_data[key] = torch.cat([
                    value, padded_tensor
                ],dim=0)

        xyz = gaussian_data["means"]
        center, _ = find_densest_voxel(xyz, (xyz.max() - xyz.min()) / 1024)

        if self.pca:
            rest = gaussian_data["features_rest"].flatten(1)
            xyz = torch.cat([xyz, rest],dim=-1)
            mean_vals = xyz.mean(dim=0,keepdim=True)
            # std_vals = xyz.std(dim=0,keepdim=True)
            xyz = (xyz - mean_vals)
            _, _, Vt = torch.linalg.svd(xyz, full_matrices=False)
            v = Vt[:3, :]
            xyz = xyz @ v.T

        if "importance" in gaussian_data:
            imp = gaussian_data["importance"]
            if self.distance:
                xyz_q = (
                            (2 ** 19 - 1)
                            * (xyz - xyz.min(0).values)
                            / (xyz.max(0).values - xyz.min(0).values)
                    ).long()
                distance = ((xyz-center) ** 2).mean(dim=-1,keepdim=True)
                distance = ((distance - distance.min()) / (distance.max() - distance.min()) * 7).long()
                xyz_q[:,:1] += (distance // 4) * (2 ** 19)
                xyz_q[:,1:2] += ((distance % 4) // 2) * (2 ** 19)
                xyz_q[:,2:3] += (distance % 2) * (2 ** 19)
                imp_q = torch.round(
                        (2 ** 3 - 1)
                        * (imp - imp.min(0).values)
                        / (imp.max(0).values - imp.min(0).values)
                ).to(torch.long)
                xyz_q[:,:1] += (imp_q // 4) * (2 ** 20)
                xyz_q[:,1:2] += ((imp_q % 4) // 2) * (2 ** 20)
                xyz_q[:,2:3] += (imp_q % 2) * (2 ** 20)
            else:
                xyz_q = (
                            (2 ** 20 - 1)
                            * (xyz - xyz.min(0).values)
                            / (xyz.max(0).values - xyz.min(0).values)
                    ).long()
                imp_q = (
                        (2 ** 3 - 1)
                        * (imp - imp.min(0).values)
                        / (imp.max(0).values - imp.min(0).values)
                ).long()
                xyz_q[:,:1] += (imp_q // 4) * (2 ** 20)
                xyz_q[:,1:2] += ((imp_q % 4) // 2) * (2 ** 20)
                xyz_q[:,2:3] += (imp_q % 2) * (2 ** 20)
            order = self.mortonEncode(xyz_q).sort().indices
        else:
            if self.distance:
                xyz_q = (
                            (2 ** 20 - 1)
                            * (xyz - xyz.min(0).values)
                            / (xyz.max(0).values - xyz.min(0).values)
                    ).long()
                distance = ((xyz-center) ** 2).mean(dim=-1,keepdim=True)
                distance = ((distance - distance.min()) / (distance.max() - distance.min()) * 7).long()
                xyz_q[:,:1] += (distance // 4) * (2 ** 20)
                xyz_q[:,1:2] += ((distance % 4) // 2) * (2 ** 20)
                xyz_q[:,2:3] += (distance % 2) * (2 ** 20)

            else:
                xyz_q = (
                            (2 ** 21 - 1)
                            * (xyz - xyz.min(0).values)
                            / (xyz.max(0).values - xyz.min(0).values)
                    ).long()
            order = self.mortonEncode(xyz_q).sort().indices
        
        # 重排所有属性
        sorted_data = {}
        for attr, tensor in gaussian_data.items():
            sorted_data[attr] = tensor[order]
        
        return sorted_data

    def splitBy3(self, a):
        x = a & 0x1FFFFF  # we only look at the first 21 bits
        x = (x | x << 32) & 0x1F00000000FFFF
        x = (x | x << 16) & 0x1F0000FF0000FF
        x = (x | x << 8) & 0x100F00F00F00F00F
        x = (x | x << 4) & 0x10C30C30C30C30C3
        x = (x | x << 2) & 0x1249249249249249
        return x

    def mortonEncode(self, pos: torch.Tensor) -> torch.Tensor:
        x, y, z = pos.unbind(-1)
        answer = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
        answer |= self.splitBy3(x) | self.splitBy3(y) << 1 | self.splitBy3(z) << 2
        return answer
    
    def mortonImpEncode(self, pos: torch.Tensor, imp: torch.Tensor) -> torch.Tensor:
        x, y, z = pos.unbind(-1)
        answer = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
        answer |= self.splitBy3(x) | self.splitBy3(y) << 1 | self.splitBy3(z) << 2
        answer = (answer & 0xFFFFFFFFFFFFFFF) | ((imp.reshape(x.shape) << 60) & 0x7000000000000000)
        return answer

class PlasSorter:
    def __init__(self, sort_keys = ["means", "features_rest", "scaling", "rotation", "features_dc", "opacity"], block_size=4) -> None:
        self.block_size = block_size
        self.sort_keys = sort_keys

    def norm(self, param):
        param = param.reshape(param.shape[0], -1)
        mins = torch.amin(param,dim=(0))
        maxs = torch.amax(param,dim=(0))
        param = (param - mins) / (maxs - mins + 1e-6)
        return param

    """plas空间排序"""
    def process(self, gaussian_data, **kwargs):
        try:
            from plas import sort_with_plas
        except:
            raise ImportError(
                "Please install PLAS with 'pip install git+https://github.com/fraunhoferhhi/PLAS.git' to use sorting"
            )
        
        gs_num = gaussian_data["means"].shape[0]
        sidelen = math.ceil(gs_num ** 0.5 / self.block_size) * self.block_size
        padding = sidelen * sidelen - gs_num
        for key, value in gaussian_data.items():
            if key == "opacity":
                padded_tensor = torch.zeros([padding] + list(value.shape[1:]), device=value.device, dtype=value.dtype) - 10
            elif key == "importance":
                padded_tensor = torch.ones([padding] + list(value.shape[1:]), device=value.device, dtype=value.dtype)
            else:
                padded_tensor = torch.zeros([padding] + list(value.shape[1:]), device=value.device, dtype=value.dtype)
            gaussian_data[key] = torch.cat([
                value, padded_tensor
            ],dim=0)
        if "importance" in gaussian_data:
            self.sort_keys.append("importance")
        weight_dict = {
            "means": 15, "rotation": 1, "scaling": 1, "opacity": 1, "features_dc":1, "features_rest": 1, "importance": 1
        }
        params_to_sort = torch.cat([weight_dict[k] * self.norm(gaussian_data[k].reshape(sidelen**2, -1)) for k in self.sort_keys], dim=-1)
        params_to_sort = params_to_sort.flatten(1)
        shuffled_indices = torch.randperm(
            params_to_sort.shape[0], device=params_to_sort.device
        )
        params_to_sort = params_to_sort[shuffled_indices]
        grid = params_to_sort.reshape((sidelen, sidelen, -1))
        _, sorted_indices = sort_with_plas(
            grid.permute(2, 0, 1), min_block_size=16, min_blur_radius=1, improvement_break=1e-4, verbose=True
        )
        sorted_indices = sorted_indices.squeeze().flatten()
        sorted_indices = shuffled_indices[sorted_indices]
        for k, v in gaussian_data.items():
            gaussian_data[k] = v[sorted_indices].reshape((sidelen//self.block_size,self.block_size,sidelen//self.block_size,self.block_size,-1)).permute((0,2,1,3,4)).reshape(v.shape)
        return gaussian_data

class CMortonBlockSortorN:
    def __init__(self, block_size=4, padding=True, pca=False, distance=False) -> None:
        self.block_size = block_size
        self.padding = padding
        self.pca = pca
        self.distance = distance

    def process(self, gaussian_data, **kwargs):
        sorter = MortonSpatialSorter(padding=False, pca=self.pca,distance=self.distance, block_size=self.block_size)
        gaussian_data = sorter.process(gaussian_data)
        gs_num = gaussian_data["means"].shape[0]
        if int(gs_num ** 0.5) ** 2 != gs_num:
            sidelen = math.ceil(gs_num ** 0.5 / self.block_size) * self.block_size
            padding = sidelen * sidelen - gs_num
        else:
            sidelen = int(gs_num ** 0.5)
            padding = 0
        for key, value in gaussian_data.items():
            if key == "opacity":
                padded_tensor = torch.zeros([padding] + list(value.shape[1:]), device=value.device, dtype=value.dtype) - 10
            elif key == "importance":
                padded_tensor = torch.ones([padding] + list(value.shape[1:]), device=value.device, dtype=value.dtype)
            else:
                padded_tensor = torch.zeros([padding] + list(value.shape[1:]), device=value.device, dtype=value.dtype)
            gaussian_data[key] = torch.cat([
                value,
                padded_tensor
            ],dim=0)

        if "features_rest" in gaussian_data and gaussian_data["features_rest"] is not None:
            features = gaussian_data["features_rest"].flatten(1)
        else:
            features = torch.cat([
                gaussian_data["means"].flatten(1)/(gaussian_data["means"].max() - gaussian_data["means"].min() + 1e-8)*65536//256,
                gaussian_data["opacity"].flatten(1)/(gaussian_data["opacity"].max() - gaussian_data["opacity"].min() + 1e-8)*256,
                gaussian_data["scaling"].flatten(1)/(gaussian_data["scaling"].max() - gaussian_data["scaling"].min() + 1e-8)*256,
                gaussian_data["rotation"].flatten(1)/(gaussian_data["rotation"].max() - gaussian_data["rotation"].min() + 1e-8)*256,
            ], dim=-1)
        indices = block_sort_cpp.block_sort(
            features.detach().flatten().cpu().numpy(), sidelen, sidelen, features.detach().flatten(1).shape[1], self.block_size
        )
        for key, value in gaussian_data.items():
            gaussian_data[key] = value[indices]

        for key, value in gaussian_data.items():
            gaussian_data[key] = value.reshape((sidelen//self.block_size,self.block_size,sidelen//self.block_size,self.block_size,-1)).permute((0,2,1,3,4)).reshape(value.shape)
        
        return gaussian_data

class HybridAttributeBlockSorter:
    def __init__(
        self,
        block_size=4,
        padding=True,
        pca=False,
        distance=True,
        rest_channels=18,
    ) -> None:
        self.block_size = block_size
        self.padding = padding
        self.pca = pca
        self.distance = distance
        self.rest_channels = rest_channels

    def _normalize_feature(self, value: torch.Tensor, scale: float) -> torch.Tensor:
        value = value.reshape(value.shape[0], -1).to(torch.float32)
        min_vals = value.min(dim=0).values
        max_vals = value.max(dim=0).values
        value = (value - min_vals) / (max_vals - min_vals + 1e-8)
        return value * scale

    def _gather_features(self, gaussian_data):
        features = [self._normalize_feature(gaussian_data["means"], 4)]

        if "opacity" in gaussian_data and gaussian_data["opacity"] is not None:
            features.append(self._normalize_feature(gaussian_data["opacity"], 0.5))
        if "scaling" in gaussian_data and gaussian_data["scaling"] is not None:
            features.append(self._normalize_feature(gaussian_data["scaling"], 0.5))
        if "rotation" in gaussian_data and gaussian_data["rotation"] is not None:
            features.append(self._normalize_feature(gaussian_data["rotation"], 0.5))
        if "features_dc" in gaussian_data and gaussian_data["features_dc"] is not None:
            features.append(self._normalize_feature(gaussian_data["features_dc"], 2))
        if "importance" in gaussian_data and gaussian_data["importance"] is not None:
            features.append(self._normalize_feature(gaussian_data["importance"], 0.5))
        if "features_rest" in gaussian_data and gaussian_data["features_rest"] is not None:
            rest = gaussian_data["features_rest"].reshape(gaussian_data["features_rest"].shape[0], -1)
            # Sample lower-order channels first; these are most likely to affect ASTC correlation.
            rest = rest[:, : min(rest.shape[1], self.rest_channels)]
            features.append(self._normalize_feature(rest, 1))

        return torch.cat(features, dim=-1)

    def process(self, gaussian_data, **kwargs):
        base_sorter = MortonSpatialSorter(
            padding=False,
            block_size=self.block_size,
            pca=self.pca,
            distance=self.distance,
        )
        gaussian_data = base_sorter.process(gaussian_data)

        gs_num = gaussian_data["means"].shape[0]
        if int(gs_num ** 0.5) ** 2 != gs_num:
            sidelen = math.ceil(gs_num ** 0.5 / self.block_size) * self.block_size
            padding = sidelen * sidelen - gs_num
        else:
            sidelen = int(gs_num ** 0.5)
            padding = 0

        for key, value in gaussian_data.items():
            if key == "opacity":
                padded_tensor = torch.zeros([padding] + list(value.shape[1:]), device=value.device, dtype=value.dtype) - 10
            elif key == "importance":
                padded_tensor = torch.ones([padding] + list(value.shape[1:]), device=value.device, dtype=value.dtype)
            else:
                padded_tensor = torch.zeros([padding] + list(value.shape[1:]), device=value.device, dtype=value.dtype)
            gaussian_data[key] = torch.cat([value, padded_tensor], dim=0)

        features = self._gather_features(gaussian_data)
        indices = block_sort_cpp.block_sort(
            features.detach().cpu().numpy().astype("float32").reshape(-1),
            sidelen,
            sidelen,
            features.shape[1],
            self.block_size,
        )

        for key, value in gaussian_data.items():
            gaussian_data[key] = value[indices]

        for key, value in gaussian_data.items():
            gaussian_data[key] = value.reshape((sidelen // self.block_size, self.block_size, sidelen // self.block_size, self.block_size, -1)).permute((0, 2, 1, 3, 4)).reshape(value.shape)

        return gaussian_data
