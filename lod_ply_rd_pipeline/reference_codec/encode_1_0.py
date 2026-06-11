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
import torch

from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, Optional
from src.xencode.base_gaussian_model import BaseGaussianModel
from src.xencode.codec_gaussian_model import CodecGaussianModels
from src.xencode.arguments import UniCodingParams
from jsonlogger import JSONLogger

@dataclass
class Config:
    # 待编码 ply 文件路径
    ply_path: Path
    # 结果保存文件夹路径
    save_dir: Path
    # 场景名
    scene_name: str
    # log 地址
    log_path: Path
    # 基础配置
    version: str = "mix_d3"
    # 排序 block 大小
    block_size: int = 16
    # 排序方法
    sort_method: Literal["morton", "plas", "dblockn"] = "dblockn"
    # 视频编码器
    video_codec: Literal["x265","x264","hm","vtm"] = "x265"
    # 质量
    quality: list[float] = field(default_factory=lambda: [100,66,33,0])
    # 输出 EGSC 码流
    uwa: bool = True
    # 位深
    pos_bitdepth: int = 15
    rot_bitdepth: int = 8
    sca_bitdepth: int = 8
    op_bitdepth: int = 8
    sh0_bitdepth: int = 8
    shn_bitdepth: int = 8
    # sh degree
    sh_degree: int = 3
    # importance 感知
    importance: bool = False
    # 配置文件
    config: Optional[Path] = None

def config_informing(cfg):
    print(
        "Configs:###########################\n",
        f"ply_path: {cfg.ply_path}\n", 
        f"codec: {cfg.video_codec}\n", 
        f"version: {cfg.version}\n", 
        f"sort_method: {cfg.sort_method}\n", 
        f"quality: ", cfg.quality,
        f"\n pos_bitdepth: {cfg.pos_bitdepth}\n", 
        f"rot_bitdepth: {cfg.rot_bitdepth}\n", 
        f"sca_bitdepth: {cfg.sca_bitdepth}\n", 
        f"op_bitdepth : {cfg.op_bitdepth }\n", 
        f"sh0_bitdepth: {cfg.sh0_bitdepth}\n", 
        f"shn_bitdepth: {cfg.shn_bitdepth}\n", 
        f"shn_degree: {cfg.sh_degree}\n", 
        f"##################################", 
    )

if __name__ == "__main__":
    args = UniCodingParams()
    cfg = tyro.cli(Config)
    cfg = args.cfg_to_params(cfg)
    config_informing(cfg)
    
    cameras = []
    os.makedirs(cfg.save_dir, exist_ok=True)
    if not os.path.exists(cfg.ply_path):
        print("Error, Ply not exist")
    else:
        baseModel = BaseGaussianModel(sh_degree=args.sh_degree)
        baseModel.load_ply(cfg.ply_path)
        jl = JSONLogger(cfg.log_path, cfg.scene_name)
        for rate in range(len(cfg.quality)):
            args.set_compression_level(cfg.quality[rate])
            jl.set_item_key(f"GSCompressed_r{rate:02d}")
            encode_gaussians = CodecGaussianModels(args.sh_degree, args=args)
            encode_gaussians.restore_fromgaussian(baseModel)
            encode_gaussians.encode(jl, cameras=cameras, uwabitstream=cfg.uwa)
            encode_gaussians.save_encoded_model(os.path.join(cfg.save_dir, f"./GSCompressed_r{rate:02d}.egsc"), args)
            encode_gaussians.load_encoded_model(os.path.join(cfg.save_dir, f"./GSCompressed_r{rate:02d}.egsc"), jl,uwabitstream=cfg.uwa)
            encode_gaussians.save_ply(os.path.join(cfg.save_dir, f"./GSCompressed_r{rate:02d}.ply"))