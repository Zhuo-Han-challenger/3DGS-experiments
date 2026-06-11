#  Copyright [2025] [Huawei Technologies Co., Ltd.] 
#  Licensed under the Code Sharing Policy of the UHD World Association (the "Policy"); 
#  http://www.theuwa.com/UWA_Code_Sharing_Policy.pdf.
#  you may not use this file except in compliance with the Policy. 
#  Unless agreed to in writing, software distributed under the Policy is distributed on an "AS IS" BASIS, 
#  WITHOUT WARRANTIES OF ANY KIND, either express or implied. 
#  See the Policy for the specific language governing permissions and 
#  limitations under the Policy.

import torch
from typing import Dict
import math

class OneSizePacker:
    map_list = {
        "three_map_stream_15bit": {
            "position_lsb":{
                "attribute_type": "means",
                "byteshift":-7,
            },
            "position_msb":{
                "attribute_type": "means",
                "byteshift":7,
            },
            "color":{
                "attribute_type": "features_dc",
                "byteshift":0,
            },
            "high_map":{
                "grid_shape": [2,4,1,1],
                "regions":[
                    # [attribute_type, frame_index, channel_offset, channel_num, byteshift/(-maxBitdepth), grid_u, grid_v]
                    ["rotation",0,0,1,0,0,0],
                    ["rotation",0,1,1,0,1,0],
                    ["rotation",0,2,1,0,2,0],
                    ["scaling",0,0,1,0,3,0],
                    ["scaling",0,1,1,0,0,1],
                    ["scaling",0,2,1,0,1,1],
                    ["opacity",0,0,1,0,2,1],
                    ["importance",0,0,1,0,3,1],
                ],
            },
            "low_map":{
                "attribute_type": "features_rest",
                "grid_shape": [4,4,1,3],
                "regions":[
                    ["features_rest",0,0,3,0,0,0],
                    ["features_rest",0,3,3,0,1,0],
                    ["features_rest",0,6,3,0,2,0],
                    ["features_rest",0,9,3,0,3,0],
                    ["features_rest",0,12,3,0,0,1],
                    ["features_rest",0,15,3,0,1,1],
                    ["features_rest",0,18,3,0,2,1],
                    ["features_rest",0,21,3,0,3,1],
                    ["features_rest",0,24,3,0,0,2],
                    ["features_rest",0,27,3,0,1,2],
                    ["features_rest",0,30,3,0,2,2],
                    ["features_rest",0,33,3,0,3,2],
                    ["features_rest",0,36,3,0,0,3],
                    ["features_rest",0,39,3,0,1,3],
                    ["features_rest",0,42,3,0,2,3],
                    
                ],
            },
        },
        "three_map_stream_14bit": {
            "position_lsb":{
                "attribute_type": "means",
                "byteshift":-6,
            },
            "position_msb":{
                "attribute_type": "means",
                "byteshift":6,
            },
            "color":{
                "attribute_type": "features_dc",
                "byteshift":0,
            },
            "high_map":{
                "grid_shape": [2,4,1,1],
                "regions":[
                    # [attribute_type, frame_index, channel_offset, channel_num, byteshift/(-maxBitdepth), grid_u, grid_v]
                    ["rotation",0,0,1,0,0,0],
                    ["rotation",0,1,1,0,1,0],
                    ["rotation",0,2,1,0,2,0],
                    ["scaling",0,0,1,0,3,0],
                    ["scaling",0,1,1,0,0,1],
                    ["scaling",0,2,1,0,1,1],
                    ["opacity",0,0,1,0,2,1],
                    ["importance",0,0,1,0,3,1],
                ],
            },
            "low_map":{
                "attribute_type": "features_rest",
                "grid_shape": [4,4,1,3],
                "regions":[
                    ["features_rest",0,0,3,0,0,0],
                    ["features_rest",0,3,3,0,1,0],
                    ["features_rest",0,6,3,0,2,0],
                    ["features_rest",0,9,3,0,3,0],
                    ["features_rest",0,12,3,0,0,1],
                    ["features_rest",0,15,3,0,1,1],
                    ["features_rest",0,18,3,0,2,1],
                    ["features_rest",0,21,3,0,3,1],
                    ["features_rest",0,24,3,0,0,2],
                    ["features_rest",0,27,3,0,1,2],
                    ["features_rest",0,30,3,0,2,2],
                    ["features_rest",0,33,3,0,3,2],
                    ["features_rest",0,36,3,0,0,3],
                    ["features_rest",0,39,3,0,1,3],
                    ["features_rest",0,42,3,0,2,3],
                    
                ],
            },
        },
    }

    def __init__(self, packing_map: Dict = None, version="default", block_size=4, scaning="block", bitdepth = 8):
        if packing_map is None:
            self.packing_map = self.map_list[version]
        else:
            self.packing_map = packing_map
        self.block_size = block_size
        self.scaning = scaning
        self.version = version
        self.bitdepth = bitdepth

    """3DGS属性拼接器"""
    def packing(self, gaussian_data, **kwargs) -> Dict[str, torch.Tensor]:
        '''
        attribute name: tensor -> map name: tensor
        '''
        result = {}
        # debug position
        # print(gaussian_data["means"][:3])

        sidelen = (math.ceil(len(gaussian_data["means"])**0.5 / self.block_size) * self.block_size)
        padding = sidelen ** 2 - len(gaussian_data["means"])
        if self.scaning == "dualmorton":
            dualmorton = Morton2DMapping(sidelen=sidelen, device=gaussian_data["means"].device)
        for key, value in self.packing_map.items():
            if "grid_shape" not in value:
                byteshift = value.get("byteshift", 0)
                dtype = torch.uint8 if self.bitdepth <= 8 else torch.uint16
                result[key] = (gaussian_data[value["attribute_type"]].to(torch.int32) // (2 ** byteshift)).to(dtype) if byteshift >= 0 else (gaussian_data[value["attribute_type"]].to(torch.int32) % (2 ** (-byteshift))).to(dtype)
            else:
                # slicing
                rows, cols, t, c = value["grid_shape"]
                result[key] = torch.zeros([rows*sidelen, cols*sidelen, t, c], device=gaussian_data["means"].device,dtype=torch.uint8 if self.bitdepth <= 8 else torch.uint16)
                for idx, region in enumerate(value["regions"]):
                    att_type, f_index, c_offset, c_num, byteshift, col, row = region
                    if att_type in gaussian_data:
                        sliced = gaussian_data[att_type].flatten(1)[:,c_offset:c_offset+c_num]
                        if key == "opacity":
                            padded_tensor = torch.zeros([padding] + list(sliced.shape[1:]), 
                                dtype=sliced.dtype, device=sliced.device) - 10
                        elif key == "importance":
                            padded_tensor = torch.ones([padding] + list(sliced.shape[1:]), 
                                dtype=sliced.dtype, device=sliced.device)
                        else:
                            padded_tensor = torch.zeros([padding] + list(sliced.shape[1:]), 
                                dtype=sliced.dtype, device=sliced.device)
                        if self.scaning == "block":
                            sliced = torch.cat([
                                    sliced,padded_tensor],dim=0).reshape([sidelen//self.block_size, sidelen//self.block_size, self.block_size, self.block_size, -1]).permute((0,2,1,3,4)).reshape([sidelen, sidelen, 1, -1])
                        elif self.scaning == "rowsfirst":
                            sliced = torch.cat([
                                    sliced, 
                                    padded_tensor
                                ],dim=0).reshape([sidelen, sidelen, 1, -1])
                        elif self.scaning == "dualmorton":
                            sliced = dualmorton.pack_morton(torch.cat([
                                    sliced, 
                                    padded_tensor
                                ],dim=0)).reshape([sidelen, sidelen, 1, -1])
                        sliced = sliced.to(torch.int32) // (2 ** byteshift) if byteshift >= 0 else sliced.to(torch.int32) % (2 ** -byteshift)
                    else:
                        sliced = torch.zeros([sidelen, sidelen, 1, c_num], device=gaussian_data["means"].device,dtype=gaussian_data["means"].dtype)
                    sliced = sliced.to(torch.int32)
                    sliced = sliced % (2 ** self.bitdepth)
                    sliced = sliced.to(torch.uint8) if self.bitdepth <= 8 else sliced.to(torch.uint16)
                    result[key][row*sidelen:row*sidelen+sidelen,col*sidelen:col*sidelen+sidelen, f_index:f_index+1, :c_num] = sliced
        
        return result

    def depacking(self, data: Dict[str, torch.Tensor], gs_num=1000, sh_degree=3) -> Dict[str, torch.Tensor]:
        '''
        name: quantized map tensor -> name: attribute tensor
        '''
        result = createSplat(gs_num, data[list(data.keys())[0]].device, sh_degree)
        
        for key, value in data.items():
            if "grid_shape" in self.packing_map[key].keys():
                # result[key] = []
                rows, cols, t, c = self.packing_map[key]["grid_shape"]
                sidelen = value.shape[0] // rows
                if self.scaning == "dualmorton":
                    dualmorton = Morton2DMapping(sidelen, device=value.device)
                for idx, region in enumerate(self.packing_map[key]["regions"]):
                    att_type, f_index, c_offset, c_num, byteshift, col, row = region
                    # byteshift = max(0, byteshift)
                    result[att_type] = result[att_type].to(torch.int32)
                    shape = result[att_type].shape
                    result[att_type] = result[att_type].flatten(1)
                    if self.scaning == "block":
                        tmp = value[row*sidelen:(row+1)*sidelen, col*sidelen:(col+1)*sidelen, f_index, :c_num].reshape([sidelen//self.block_size,self.block_size, sidelen//self.block_size , self.block_size, c_num]).permute((0,2,1,3,4)).reshape(-1, c_num).to(torch.int32)
                    elif self.scaning == "rowsfirst":
                        tmp = value[row*sidelen:(row+1)*sidelen, col*sidelen:(col+1)*sidelen, f_index, :c_num].reshape(-1, c_num).to(torch.int32)
                    elif self.scaning == "dualmorton":
                        tmp =  dualmorton.unpack_morton(value[row*sidelen:(row+1)*sidelen, col*sidelen:(col+1)*sidelen, f_index:f_index+1, :c_num]).reshape(-1, c_num).to(torch.int32)
                    tmp = tmp * (2 ** byteshift) if byteshift >= 0 else tmp.clamp(0, 2 ** (-byteshift))
                    result[att_type][:,c_offset:c_offset+c_num] += tmp
                    result[att_type] = result[att_type].reshape(shape)
            else:
                byteshift = self.packing_map[key].get("byteshift", 0)
                result[self.packing_map[key]["attribute_type"]] = result[self.packing_map[key]["attribute_type"]].to(torch.int32)
                result[self.packing_map[key]["attribute_type"]] += (value.to(torch.int32) * (2 ** byteshift)).view(result[self.packing_map[key]["attribute_type"]].shape) if byteshift >= 0 else value.to(torch.int32).clamp(0, 2 ** (-byteshift))
        return result

class PackingError(Exception):
    pass

def createSplat(num, device, sh_degree=3):
    splat = {}
    splat["means"] = torch.zeros([num,3], device=device)
    splat["features_dc"] = torch.zeros([num,1,3], device=device)
    if sh_degree > 0:
        splat["features_rest"] = torch.zeros([num,(sh_degree + 1)**2-1,3], device=device)
    splat["opacity"] = torch.zeros([num,1], device=device)
    splat["rotation"] = torch.zeros([num,4], device=device)
    splat["scaling"] = torch.zeros([num,3], device=device)
    splat["importance"] = torch.zeros([num,1], device=device)
    return splat

class Morton2DMapping:
    def __init__(self, sidelen=None, device=None) -> None:
        codes, self.rows, self.cols = self.morton_codes_for_grid(sidelen, device=device)
        self.order = torch.argsort(codes)
        self.rows = self.rows[self.order]
        self.cols = self.cols[self.order]
        self.device = device

    def morton_codes_for_grid(self, sidelen: int, device=None, dtype=torch.long):
        """为 SxS 网格计算每个格子的 Morton code（Z-order），返回 codes, rows, cols 三个一维 tensor，长度 sidelen*sidelen。
        codes 是用于排序的 Morton code；rows, cols 是对应的行列索引。
        支持非 2 的幂的 S：使用 bits = ceil(log2(sidelen))，并且生成所有 0..sidelen-1 的行列。
        """
        device = device
        rows = torch.arange(sidelen, device=device, dtype=torch.long).repeat_interleave(sidelen)
        cols = torch.arange(sidelen, device=device, dtype=torch.long).repeat(sidelen)
        codes = torch.zeros(sidelen * sidelen, dtype=torch.long, device=device)
        codes |= self.splitBy2(cols) | self.splitBy2(rows) << 1 

        return codes, rows, cols
    
    def splitBy2(self, a):
        x = a & 0x1FFFFF  # we only look at the first 21 bits
        x = (x | x << 16) & 0x1F0000FFFF
        x = (x | x << 8) & 0x1F00FF00FF
        x = (x | x << 4) & 0x10F0F0F0F0F
        x = (x | x << 2) & 0x13333333333
        x = (x | x << 1) & 0x15555555555
        return x

    def pack_morton(self, x: torch.Tensor):
        result = torch.zeros_like(x).to(torch.int32)
        result[self.order.flatten()] = x.to(torch.int32)
        return result.to(x.dtype)

    def unpack_morton(self, img: torch.Tensor):
        values = img.to(torch.int32)[self.rows, self.cols].to(img.dtype)
        return values
    
    def unpack_morton_float(self, img: torch.Tensor):
        values = img[self.rows, self.cols]
        return values