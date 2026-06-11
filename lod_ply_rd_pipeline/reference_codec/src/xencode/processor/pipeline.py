#  Copyright [2025] [Huawei Technologies Co., Ltd.] 
#  Licensed under the Code Sharing Policy of the UHD World Association (the "Policy"); 
#  http://www.theuwa.com/UWA_Code_Sharing_Policy.pdf.
#  you may not use this file except in compliance with the Policy. 
#  Unless agreed to in writing, software distributed under the Policy is distributed on an "AS IS" BASIS, 
#  WITHOUT WARRANTIES OF ANY KIND, either express or implied. 
#  See the Policy for the specific language governing permissions and 
#  limitations under the Policy.

from .sorters import *
from .packers import *
from .codecs import *
from .quantizers import *
from .transforms import *
from .prediction import *
from .segment import *

# pipeline_registry.py
class PipelineRegistry:
    """全局注册器"""
    _configs = {}
    
    @classmethod
    def register_config(cls, name):
        """装饰器：注册 pipeline 配置"""
        def decorator(func):
            cls._configs[name] = func
            return func
        return decorator
    
    @classmethod
    def build(cls, name):
        """构建新的 pipeline 实例"""
        if name not in cls._configs:
            raise ValueError(f"Config '{name}' not found. Available: {list(cls._configs.keys())}")
        
        # 调用配置函数，返回新实例的字典
        return cls._configs[name]()
    
    @classmethod
    def list_configs(cls):
        """列出所有可用配置"""
        return list(cls._configs.keys())

@PipelineRegistry.register_config("ff_d0")
def recommend_pipeline():
    return {
        'transform': AttributeTransform(version="reduction_rsnorm"),
        'spatial_sorter': CMortonBlockSortorN(block_size=4),
        'quantizers': {
            "means":AdaptiveGroupMinmaxQuantizer(bits=14, group_size=256),
            "opacity":MinmaxQuantizer(bits=6),
            "scaling":MinmaxQuantizer(bits=6),
            "rotation":MinmaxQuantizer(bits=6),
            "features_dc":MinmaxQuantizer(bits=6),
        },
        'prediction': AttributePridiction(version="pos_minor"),
        'attribute_packer': OneSizePacker(version="three_map_stream_14bit", bitdepth=8),
        'codecs':{
            'position_lsb': EntropyCodec(),
            'position_msb': EntropyCodec(),
            'color': EntropyCodec(),
            'high_map': VideoCodec(pix_fmt="yuv420p", qp=12),
        },
    }

@PipelineRegistry.register_config("mix_d3")
def recommend_pipeline():
    return {
        'transform': AttributeTransform(version="reduction_rsnorm_imp"),
        'spatial_sorter': CMortonBlockSortorN(block_size=4),
        'quantizers': {
            "means":AdaptiveGroupMinmaxQuantizer(bits=15, group_size=256),
            "opacity":MinmaxQuantizer(bits=8),
            "scaling":MinmaxQuantizer(bits=8),
            "rotation":MinmaxQuantizer(bits=8),
            "features_dc":MinmaxQuantizer(bits=8),
            "features_rest":MinmaxQuantizer(bits=8),
            "importance":MinmaxQuantizer(bits=8),
        },
        'prediction': AttributePridiction(version="pos_minor"),
        'attribute_packer': OneSizePacker(version="three_map_stream_15bit", bitdepth=8),
        'codecs':{
            'position_lsb': EntropyCodec(),
            'position_msb': EntropyCodec(),
            'color': EntropyCodec(),
            'high_map': VideoCodec(pix_fmt="yuv420p"),
            'low_map': AstcCodecPy(),
        },
    }

@PipelineRegistry.register_config("mix_d0")
def recommend_pipeline():
    return {
        'transform': AttributeTransform(version="reduction_rsnorm"),
        'spatial_sorter': CMortonBlockSortorN(block_size=4),
        'quantizers': {
            "means":AdaptiveGroupMinmaxQuantizer(bits=15, group_size=256),
            "opacity":MinmaxQuantizer(bits=8),
            "scaling":MinmaxQuantizer(bits=8),
            "rotation":MinmaxQuantizer(bits=8),
            "features_dc":MinmaxQuantizer(bits=8),
        },
        'prediction': AttributePridiction(version="pos_minor"),
        'attribute_packer': OneSizePacker(version="three_map_stream_15bit", bitdepth=8),
        'codecs':{
            'position_lsb': EntropyCodec(),
            'position_msb': EntropyCodec(),
            'color': EntropyCodec(),
            'high_map': VideoCodec(pix_fmt="yuv420p"),
        },
    }