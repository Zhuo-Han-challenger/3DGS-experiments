#  Copyright [2025] [Huawei Technologies Co., Ltd.] 
#  Licensed under the Code Sharing Policy of the UHD World Association (the "Policy"); 
#  http://www.theuwa.com/UWA_Code_Sharing_Policy.pdf.
#  you may not use this file except in compliance with the Policy. 
#  Unless agreed to in writing, software distributed under the Policy is distributed on an "AS IS" BASIS, 
#  WITHOUT WARRANTIES OF ANY KIND, either express or implied. 
#  See the Policy for the specific language governing permissions and 
#  limitations under the Policy.

import os
import tyro

from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, Optional
from src.xencode.base_gaussian_model import BaseGaussianModel
from src.xencode.codec_gaussian_model import CodecGaussianModels
from src.xencode.arguments import UniCodingParams
from jsonlogger import JSONLogger

@dataclass
class Config:
    # 待编码 bin 文件路径
    bin_path: Path
    # 结果保存文件夹路径
    save_dir: Path
    log_path: Path
    scene_name: str = "default"

if __name__ == "__main__":
    args = UniCodingParams()
    cfg = tyro.cli(Config)


    cameras = []
    os.makedirs(cfg.save_dir, exist_ok=True)
    if not os.path.exists(cfg.bin_path):
        print("Error, Bin not exist")
    else:
        jl = JSONLogger(cfg.log_path, cfg.scene_name)
        jl.set_item_key(os.path.basename(cfg.bin_path).split(".")[0])
        encode_gaussians = CodecGaussianModels(3, args=args)
        encode_gaussians.load_encoded_model(cfg.bin_path, jl, True)
        encode_gaussians.save_ply(os.path.join(cfg.save_dir, "decode_point_cloud.ply"))
        print("解码文件保存至：", os.path.join(cfg.save_dir, "decode_point_cloud.ply"))