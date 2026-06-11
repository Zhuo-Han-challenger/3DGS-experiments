#  Copyright [2025] [Huawei Technologies Co., Ltd.] 
#  Licensed under the Code Sharing Policy of the UHD World Association (the "Policy"); 
#  http://www.theuwa.com/UWA_Code_Sharing_Policy.pdf.
#  you may not use this file except in compliance with the Policy. 
#  Unless agreed to in writing, software distributed under the Policy is distributed on an "AS IS" BASIS, 
#  WITHOUT WARRANTIES OF ANY KIND, either express or implied. 
#  See the Policy for the specific language governing permissions and 
#  limitations under the Policy.

import torch
from ..utils import * 

class AttributePridiction:
    prediction_map_dict = {
        "default": {
            "means": "None",
            "scaling": "None",
            "rotation": "None",
            "opacity": "None",
            "features_dc": "None",
            "features_rest": "None",
            "importance": "None",
        },
        "pos_minor": {
            "means": "minor",
            "scaling": "None",
            "rotation": "None",
            "opacity": "None",
            "features_dc": "minor",
            "features_rest": "None",
            "importance": "None",
        },
    }

    def get_prediction_processor(self, key, item):
        if key == "means":
            if item == "minor": return MinorBlockPrediction()
        elif key == "features_dc":
            if item == "minor": return MinorBlockPrediction()

    def __init__(self, prediction_map = None, version="default") -> None:
        if prediction_map is None:
            self.prediction_map = self.prediction_map_dict[version]
        else:
            self.prediction_map = prediction_map
        self.function_map = {}
        for key, item in self.prediction_map.items():
            self.function_map[key] = self.get_prediction_processor(key, item)

    def process(self, gaussian_data, cameras=None):
        metas = {}
        for key, value in self.prediction_map.items():
            if value != "None":
                gaussian_data[key], metas[key] = self.function_map[key].process(gaussian_data[key])
        return gaussian_data, metas

    def deprocess(self, gaussian_data, metas):
        for key, value in self.prediction_map.items():
            if value != "None":
                gaussian_data[key] = self.function_map[key].deprocess(gaussian_data[key], metas.get(key, {}))
        return gaussian_data

class MinorPrediction:
    def __init__(self, bitdepth=14) -> None:
        self.bitdepth = bitdepth
        pass
    def process(self, x):
        levels = 2 ** max(self.bitdepth - 8,0)
        x_h = x.to(torch.int32) // levels
        x_l = x.to(torch.int32) % levels
        x_h[1:] = (x_h[1:].to(torch.uint8) - x_h[:-1].to(torch.uint8)).to(torch.int32)
        x = (x_h * levels + x_l).to(x.dtype)
        return x, {"byteshift": max(self.bitdepth - 8,0)}
    
    def deprocess(self, x, meta):
        
        levels = 2 ** meta["byteshift"]
        x_h = x.to(torch.int32) // levels
        x_l = x.to(torch.int32) % levels
        x_h = ((torch.cumsum(x_h,dim=0)) % 256).to(torch.int32)
        x = (x_h * levels + x_l).to(x.dtype)
        return x


class MinorBlockPrediction:
    def __init__(self, bitdepth=14, block_size=16) -> None:
        self.bitdepth = bitdepth
        self.block_size = block_size
        pass
    def process(self, x):
        levels = 2 ** max(self.bitdepth - 8,0)
        x_h = x.to(torch.int32) // levels
        x_l = x.to(torch.int32) % levels
        x_h = x_h.reshape(-1,self.block_size**2,3)
        x_h[:,1:] = (x_h[:,1:].to(torch.uint8) - x_h[:,:-1].to(torch.uint8)).to(torch.int32)
        x = (x_h.reshape(-1,3) * levels + x_l).to(x.dtype)
        return x, {"byteshift": max(self.bitdepth - 8,0), "block_size": self.block_size}
    
    def deprocess(self, x, meta):
        levels = 2 ** meta["byteshift"]
        block_size = meta["block_size"]
        x_h = x.to(torch.int32) // levels
        x_l = x.to(torch.int32) % levels
        x_h = x_h.reshape(-1,block_size**2,3)
        x_h = ((torch.cumsum(x_h,dim=(1))) % 256).to(torch.int32).reshape(-1,3)
        x_l = x_l.reshape(-1,3)
        x = (x_h * levels + x_l).to(x.dtype)
        return x
