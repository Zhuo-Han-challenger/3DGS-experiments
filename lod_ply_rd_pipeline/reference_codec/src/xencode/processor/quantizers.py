#  Copyright [2025] [Huawei Technologies Co., Ltd.] 
#  Licensed under the Code Sharing Policy of the UHD World Association (the "Policy"); 
#  http://www.theuwa.com/UWA_Code_Sharing_Policy.pdf.
#  you may not use this file except in compliance with the Policy. 
#  Unless agreed to in writing, software distributed under the Policy is distributed on an "AS IS" BASIS, 
#  WITHOUT WARRANTIES OF ANY KIND, either express or implied. 
#  See the Policy for the specific language governing permissions and 
#  limitations under the Policy.

import torch
import numpy as np
from typing import Dict, Tuple, List

class QuantizationError(Exception):
    pass

class MinmaxQuantizer:
    """Minmax量化器， 逐通道最大最小量化"""
    def __init__(self, bits: int = 16):
        if not 1 <= bits <= 32:
            raise QuantizationError(f"量化位数必须在1-32之间，当前值: {bits}")
        self.bits = bits
    
    def process(self, data) -> Tuple[torch.Tensor, Dict]:
        '''
        name: [region tensor] -> name: [quantized region tensor]
        '''
        levels = 2 ** self.bits - 1
        d = data.flatten(1)
        if not isinstance(d, torch.Tensor):
            raise QuantizationError("输入数据必须是torch.Tensor")
        if d.numel() == 0:
            raise QuantizationError("输入数据不能为空")
        try:
            min_vals = d.min(dim=0)[0]
            max_vals = d.max(dim=0)[0]
            normalized = (d - min_vals) / (max_vals - min_vals + 1e-6)
            quantized = (normalized * levels).round().detach()
            
            if self.bits > 8:
                quantized = quantized.to(torch.uint16)
            else:
                quantized = quantized.to(torch.uint8)
            metadata = {'min_vals': min_vals.flatten(), 'max_vals': max_vals.flatten(), 'bit_depth': self.bits}

        except Exception as e:
            raise QuantizationError(f"Minmax 量化失败: {str(e)}")

        return quantized, metadata

    def deprocess(self, quantized: List[torch.Tensor], metadata: List[Dict]) -> List[torch.Tensor]:
        '''
        name: [quantized region tensor] -> name: [region tensor]
        '''

        # if metadata['bit_depth'] > 8:
        #     quantized = quantized.to(torch.uint16)
        # else:
        #     quantized = quantized.to(torch.uint8)
        min_vals = metadata['min_vals'].view([-1]+list(quantized.shape[1:]))
        max_vals = metadata['max_vals'].view([-1]+list(quantized.shape[1:]))
        return quantized.float() / (2 ** self.bits - 1) * (max_vals - min_vals + 1e-6) + min_vals

class GlobalMinmaxQuantizer:
    """Minmax量化器, 全局最大最小量化"""
    def __init__(self, bits: int = 16):
        if not 1 <= bits <= 32:
            raise QuantizationError(f"量化位数必须在1-32之间，当前值: {bits}")
        self.bits = bits
    
    def process(self, data) -> Tuple[torch.Tensor, Dict]:
        '''
        name: [region tensor] -> name: [quantized region tensor]
        '''
        levels = 2 ** self.bits - 1
        d = data.flatten(1)
        if not isinstance(d, torch.Tensor):
            raise QuantizationError("输入数据必须是torch.Tensor")
        if d.numel() == 0:
            raise QuantizationError("输入数据不能为空")
        try:
            min_vals = d.min()
            max_vals = d.max()
            normalized = (d - min_vals) / (max_vals - min_vals + 1e-6)
            quantized = (normalized * levels).round().detach()
            
            if self.bits > 16:
                quantized = quantized.to(torch.uint32)
            elif self.bits > 8:
                quantized = quantized.to(torch.uint16)
            else:
                quantized = quantized.to(torch.uint8)

            metadata = {'min_vals': min_vals.flatten(), 'max_vals': max_vals.flatten(), 'bit_depth': self.bits}

        except Exception as e:
            raise QuantizationError(f"Minmax 量化失败: {str(e)}")

        return quantized, metadata

    def deprocess(self, quantized: List[torch.Tensor], metadata: List[Dict]) -> List[torch.Tensor]:
        '''
        name: [quantized region tensor] -> name: [region tensor]
        '''
        quantized, metadata
        if metadata['bit_depth'] > 8:
            quantized = quantized.to(torch.uint16)
        else:
            quantized = quantized.to(torch.uint8)

        min_vals = metadata['min_vals']
        max_vals = metadata['max_vals'] 

        return quantized.float() / (2 ** self.bits - 1) * (max_vals - min_vals + 1e-6) + min_vals

class GroupMinmaxQuantizer:
    """分组 Minmax 量化器"""
    def __init__(self, bits: int = 16, group_size=256):
        if not 1 <= bits <= 32:
            raise QuantizationError(f"量化位数必须在1-32之间，当前值: {bits}")
        self.bits = bits
        self.group_size = group_size
    
    def process(self, data) -> Tuple[torch.Tensor, Dict]:
        '''
        name: [region tensor] -> name: [quantized region tensor]
        '''
        levels = 2 ** self.bits - 1
        d = data.flatten(1)
        if not isinstance(d, torch.Tensor):
            raise QuantizationError("输入数据必须是torch.Tensor")
        
        if d.numel() == 0:
            raise QuantizationError("输入数据不能为空")
        
        try:
            gs_num = len(d)
            if gs_num % self.group_size != 0:
                group_num = gs_num // self.group_size
                d_main = d[:-(gs_num % self.group_size)]
                d_rest = d[-(gs_num % self.group_size):]
            else:
                group_num = gs_num // self.group_size
                d_main = d
                d_rest = None
            min_vals = torch.amin(d_main.reshape([group_num, self.group_size, -1]), dim=(1))
            max_vals = torch.amax(d_main.reshape([group_num, self.group_size, -1]), dim=(1))
            normalized = (d_main.reshape([group_num, self.group_size, -1]) - min_vals.unsqueeze(1)) / (max_vals.unsqueeze(1) - min_vals.unsqueeze(1) + 1e-6)
            quantized = (normalized * levels).round().detach().flatten(0,1)
            if d_rest is not None:
                min_rest = torch.amin(d_rest.reshape([1, gs_num % self.group_size, -1]), dim=(1))
                max_rest = torch.amax(d_rest.reshape([1, gs_num % self.group_size, -1]), dim=(1))
                min_vals = torch.cat([min_vals, min_rest],dim=0)
                max_vals = torch.cat([max_vals, max_rest],dim=0)
                normalized_rest = (d_rest.reshape([1, gs_num % self.group_size, -1]) - min_rest.unsqueeze(1)) / (max_rest.unsqueeze(1) - min_rest.unsqueeze(1) + 1e-6)
                quantized_rest = (normalized_rest * levels).round().detach().flatten(0,1)
                quantized = torch.cat([quantized, quantized_rest], dim=0)

            if self.bits > 16:
                quantized = quantized.to(torch.uint32)
            elif self.bits > 8:
                quantized = quantized.to(torch.uint16)
            else:
                quantized = quantized.to(torch.uint8)
            metadata = {'min_vals': min_vals.flatten(), 'max_vals': max_vals.flatten(), 'group_size': [self.group_size]*len(min_vals), 'bit_depth': self.bits}
        except Exception as e:
            raise QuantizationError(f"Minmax 量化失败: {str(e)}")
        return quantized, metadata
        

    def deprocess(self, quantized: List[torch.Tensor], metadata: List[Dict]) -> List[torch.Tensor]:
        '''
        name: [quantized region tensor] -> name: [region tensor]
        '''

        min_vals = metadata['min_vals']
        max_vals = metadata['max_vals']
        group_size = int(metadata['group_size'][0])
        gs_num = len(quantized)
        levels = 2 ** metadata['bit_depth'] - 1

        if metadata['bit_depth'] > 8:
            quantized = quantized.to(torch.uint16)
        else:
            quantized = quantized.to(torch.uint8)

        if gs_num % group_size != 0:
            group_num = gs_num // group_size
            q_main = quantized[:-(gs_num % group_size)]
            q_rest = quantized[-(gs_num % group_size):]
            rest_ = 1
        else:
            group_num = gs_num // group_size
            q_main = quantized
            q_rest = None
            rest_ = 0

        max_vals = max_vals.reshape(group_num+rest_, 1, -1)
        min_vals = min_vals.reshape(group_num+rest_, 1, -1)
        quantized = (q_main.reshape(group_num, group_size, -1).float() / levels * (max_vals[:group_num] - min_vals[:group_num] + 1e-6) + min_vals[:group_num]).flatten(0,1)
        if q_rest is not None:
            q_rest = (q_rest.reshape(1, gs_num % group_size, -1).float() / levels * (max_vals[-1:] - min_vals[-1:] + 1e-6) + min_vals[-1:]).flatten(0,1)
            quantized = torch.cat([quantized, q_rest], dim=0)

        return quantized

class AdaptiveGroupMinmaxQuantizer:
    def __init__(self, bits: int = 16, group_size=256):
        if not 1 <= bits <= 32:
            raise QuantizationError(f"量化位数必须在1-32之间，当前值: {bits}")
        self.bits = bits
        self.group_size = group_size
    
    def process(self, data) -> Tuple[torch.Tensor, Dict]:
        '''
        name: [region tensor] -> name: [quantized region tensor]
        '''
        def find_densest_voxel(points: torch.Tensor, voxel_size: float = 1.0) -> tuple:
            """
            Find the voxel containing the most points.

            Args:
                points: Tensor of shape (N, 3) containing 3D points
                voxel_size: Size of each voxel (default: 1.0)

            Returns:
                Tuple of (voxel_center, count) where:
                - voxel_center: Tensor (x, y, z) world coordinates of the voxel's center
                - count: number of points in that voxel
            """
            # Compute voxel index for each point
            voxel_indices = torch.floor(points / voxel_size).to(torch.int64)

            # Encode (i,j,k) -> flat integer for torch.unique
            mins = voxel_indices.min(dim=0).values
            shifted = voxel_indices - mins
            dims = shifted.max(dim=0).values + 1
            flat = shifted[:, 0] * (dims[1] * dims[2]) + shifted[:, 1] * dims[2] + shifted[:, 2]

            # Count and find densest voxel
            unique, counts = torch.unique(flat, return_counts=True)
            best_flat = unique[counts.argmax()]
            count = counts.max()

            # Decode flat index back to (i, j, k)
            k = best_flat % dims[2]
            j = (best_flat // dims[2]) % dims[1]
            i = best_flat // (dims[1] * dims[2])
            densest_voxel = torch.stack([i, j, k]) + mins

            # Compute center: (index + 0.5) * voxel_size
            voxel_center = (densest_voxel.float() + 0.5) * voxel_size

            return voxel_center, count.item()

        levels = 2 ** self.bits - 1
        d = data.flatten(1)
        if not isinstance(d, torch.Tensor):
            raise QuantizationError("输入数据必须是torch.Tensor")
        
        if d.numel() == 0:
            raise QuantizationError("输入数据不能为空")
        
        # try:
        global_min = d.min()
        global_max = d.max()
        global_range = global_max - global_min
        gs_num = len(d)

        group_num = gs_num // self.group_size
        center,_ = find_densest_voxel(data, global_range/(1024))
        q_main = d[:group_num*self.group_size]
        sigma = (q_main  / global_range).std()
        q_main_range = 2 * torch.quantile(torch.sqrt(((q_main - center)**2).mean(dim=-1)), 0.9)
        
        count = 0
        def get_weight(x, sigma):
            return 0.5 + torch.exp( 10 * (x / (0.2 * sigma) - 1))

        metadata = {}
        metadata["min_vals"] = []
        metadata["max_vals"] = []
        metadata["group_size"] = []
        while(count<gs_num):
            group_size = self.group_size
            while(count+group_size<=gs_num):
                d_group = d[count:count + group_size]
                d_group_range = (d_group.max(dim=-1,keepdim=True)[0] - d_group.min(dim=-1,keepdim=True)[0]).max()
                weight =  get_weight((torch.sqrt(((d_group - center) ** 2).sum(dim=-1)) / global_range).min(), sigma)
                if d_group_range / weight < q_main_range:
                    if count+group_size == gs_num:
                        break
                    group_size = min(int(group_size * 1.5), gs_num - count)
                else:
                    group_size = max(self.group_size, int(group_size/1.5))
                    break
            d_group = d[count:count + group_size]
            min_vals = d_group.min(dim=0, keepdim=True)[0]
            max_vals = d_group.max(dim=0, keepdim=True)[0]
            metadata["min_vals"].append(min_vals)
            metadata["max_vals"].append(max_vals)
            metadata["group_size"].append(group_size)
            d[count:count + group_size] = (d_group - min_vals) / (max_vals - min_vals + 1e-8)
            count += group_size
        quantized = (d * levels).round().detach()
        metadata["min_vals"] = torch.cat(metadata["min_vals"],dim=0).flatten()
        metadata["max_vals"] = torch.cat(metadata["max_vals"],dim=0).flatten()
        
        if self.bits > 16:
            quantized = quantized.to(torch.uint32)
        elif self.bits > 8:
            quantized = quantized.to(torch.uint16)
        else:
            quantized = quantized.to(torch.uint8)

        metadata['bit_depth'] = self.bits
        return quantized, metadata
        

    def deprocess(self, quantized: List[torch.Tensor], metadata: List[Dict]) -> List[torch.Tensor]:
        '''
        name: [quantized region tensor] -> name: [region tensor]
        '''
        levels = 2 ** self.bits - 1
        group_size = metadata['group_size']
        min_vals = metadata['min_vals'].reshape([len(group_size),-1])
        max_vals = metadata['max_vals'].reshape([len(group_size),-1])
        # if metadata['bit_depth'] > 8:
        #     quantized = quantized.to(torch.uint16)
        # else:
        #     quantized = quantized.to(torch.uint8)
        count = 0
        q = quantized.to(torch.float) / levels
        for ind in range(len(group_size)):
            gp = group_size[ind].to(torch.long)
            miv = min_vals[ind].to(q.device).unsqueeze(0)
            mav = max_vals[ind].to(q.device).unsqueeze(0)
            q[count:count+gp] = q[count:count+gp] * (mav - miv + 1e-6) + miv
            count += gp
        return q