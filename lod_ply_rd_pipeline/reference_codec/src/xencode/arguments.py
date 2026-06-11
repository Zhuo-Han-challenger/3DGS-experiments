#  Copyright [2025] [Huawei Technologies Co., Ltd.] 
#  Licensed under the Code Sharing Policy of the UHD World Association (the "Policy"); 
#  http://www.theuwa.com/UWA_Code_Sharing_Policy.pdf.
#  you may not use this file except in compliance with the Policy. 
#  Unless agreed to in writing, software distributed under the Policy is distributed on an "AS IS" BASIS, 
#  WITHOUT WARRANTIES OF ANY KIND, either express or implied. 
#  See the Policy for the specific language governing permissions and 
#  limitations under the Policy.

from argparse import ArgumentParser
import yaml
import os
from dataclasses import replace

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        if parser is not None:
            group = parser.add_argument_group(name)
            for key, value in vars(self).items():
                shorthand = False
                t = type(value)
                value = value if not fill_none else None
                if shorthand:
                    if t == bool:
                        if value:
                            group.add_argument("--no-" + key, ("-n" + key[0:1]), dest=key, action="store_false")
                        else:
                            group.add_argument("--" + key, ("-" + key[0:1]), action="store_true")
                    elif t == int or t == float or t == str:
                        group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
                else:
                    if t == bool:
                        if value:
                            group.add_argument("--no-" + key, dest=key, action="store_false")
                        else:
                            group.add_argument("--" + key, action="store_true")
                    elif t == int or t == float or t == str:
                        group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(self, arg[0], arg[1])
        return self

class CodingParams(ParamGroup):
    def __init__(self, parser: ArgumentParser = None, sentinel=False):
        self.debug = False
        # ---------------------------------------Xencode Parameter---------------------------------------
        # 1.0 ：高阶球协采用ASTC压缩，更快编码
        self.xencode_version = 1.0
        # version choices: 
        # 编码后码流中包含的 sh 级数
        self.codec_sh_degree = 0 
        # 是否对属性码流进行熵编码，0 ： 不熵编码， 1 ： zlib熵编码
        self.xencode_compress = 1

        # astc block 的大小：4, 5, 6, 8, 10, 12, block 越大压缩率越高
        self.astc_block = 4
        self.astc_mode = "medium" # exhaustive, medium, fast
        # 是否将高阶 SH 系数拼接成一张纹理
        self.merge_sh_texture = True
        # 纹理大小限制
        self.max_merged_tex_size = 4096
        # 选择排序模式： block/morton, block 可被 gpu 加速，压缩性能较高，morton 效率高压缩性能较低
        self.sort_method = "morton"
        self.bitdepth_choice = 0
        self.bitdepth_config = [
            {
                "pos":12,
                "other":8,
                "shn":8
            },
            {
                "pos":12,
                "other":7,
                "shn":7
            },
            {
                "pos":12,
                "other":7,
                "shn":5
            },
            {
                "pos":10,
                "other":6,
                "shn":4
            },
        ]
        super().__init__(parser, "Coding Parameters", sentinel)

class UniCodingParams(ParamGroup):
    def __init__(self, parser: ArgumentParser = None, sentinel=False):
        # sorting 
        self.sort_method = "block"
        self.block_size = 4
        # codec
        self.astc_mode = "medium" # exhaustive, medium, fast
        self.qps = {}
        self.astc_blocks = {}
        self.video_codec = "hm"
        # 量化相关参数
        self.pos_bitdepth = 15
        self.rot_bitdepth = 8
        self.sca_bitdepth = 8
        self.op_bitdepth = 8
        self.sh0_bitdepth = 8
        self.shn_bitdepth = 8
        # 变换
        self.importance = True
        self.xencode_version = ""
        self.sh_degree = 3

        super().__init__(parser, "Coding Parameters", sentinel)
    
    def cfg_to_params(self, cfg):
        if cfg.config is not None and os.path.exists(cfg.config):
            with open(cfg.config) as f:
                overrides = yaml.safe_load(f)
            cfg = replace(cfg, **overrides)
        self.xencode_version = cfg.version
        self.block_size = cfg.block_size
        self.video_codec = cfg.video_codec
        self.sort_method = cfg.sort_method
        self.pos_bitdepth = cfg.pos_bitdepth
        self.rot_bitdepth = cfg.rot_bitdepth
        self.sca_bitdepth = cfg.sca_bitdepth
        self.op_bitdepth = cfg.op_bitdepth
        self.sh0_bitdepth = cfg.sh0_bitdepth
        self.shn_bitdepth = cfg.shn_bitdepth
        self.importance = cfg.importance
        self.sh_degree = cfg.sh_degree
        
        # check
        assert self.pos_bitdepth > 8 and self.pos_bitdepth <= 16, "position 位宽应大于 8 小于等于 16"
        assert self.rot_bitdepth > 0 and self.rot_bitdepth <= 8, "rotation 位宽应大于 0 小于等于 8"
        assert self.sca_bitdepth > 0 and self.sca_bitdepth <= 8, "scaling 位宽应大于 0 小于等于 8"
        assert self.op_bitdepth  > 0 and self.op_bitdepth <= 8, "opacity 位宽应大于 0 小于等于 8"
        assert self.sh0_bitdepth > 0 and self.sh0_bitdepth <= 8, "sh0 位宽应大于 0 小于等于 8"
        assert self.shn_bitdepth > 0 and self.shn_bitdepth <= 8, "shn 位宽应大于 0 小于等于 8"
        assert self.sh_degree >= 0 and self.sh_degree <= 3, "目前实现仅支持 0~3 阶 sh"
        return cfg
    
    def set_compression_level(self, quality):
        def get_compression_level(quality):
            if quality > 100 or quality < 0:
                print("Error, quality is out of range")
            else:
                compression_level = 1 - quality / 100
                return compression_level
        compression_level = min(1, max(0, get_compression_level(quality)))
        self.astc_blocks = {'features_rest': [4,5,5,6,8,10,12,12][round(7 * compression_level)],'low_map': [4,5,5,6,6,8,10,12][round(7 * compression_level)],'high_map': [4,5,5,6,6,8,8,10][round(7 * compression_level)]}
        if quality > 30:
            self.qps = {'high_map': 4 + 21 * compression_level, 'low_map': 12 + 39 * compression_level}
        else:
            self.qps = {'high_map': 18 + 13 * compression_level, 'low_map': 36 + 2 * compression_level}
            self.pos_bitdepth = 12
