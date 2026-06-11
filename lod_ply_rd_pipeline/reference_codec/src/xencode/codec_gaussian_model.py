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
import zstd
import numpy as np
import torch
import struct
import platform
import tempfile
import subprocess
import inspect
import copy

from io import BytesIO
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from math import sqrt
from torch import nn, Tensor
from loguru import logger
import concurrent.futures
import msgpack

from .decode import decode_stream
from .base_gaussian_model import BaseGaussianModel
from .utils import *
from .processor.sorters import *
from .processor.packers import *
from .processor.codecs import *
from .processor.quantizers import *
from .processor.transforms import *
from .processor.segment import *
from .processor.pipeline import *
from .processor.stream import *
from .processor.prediction import *


_module_path = Path(__file__).resolve()
_module_dir = _module_path.parent
astc_cmd = None
if platform.system() == "Linux":
    astc_cmd = os.path.join(str(_module_dir), "tools", "astcenc-sse2")
elif platform.system() == "Windows":
    astc_cmd = os.path.join(str(_module_dir), "tools", "astcenc-sse2.exe")
else:
    raise RuntimeError("Only execute in Linux or Windows!")

def synchronize_device(device=None):
    """
    根据设备类型自动选择同步方式。
    如果 device 是 'cuda' 或当前环境支持 CUDA，则执行 GPU 同步。
    """
    if torch.cuda.is_available():
        # 如果指定了特定 device (如 'cuda:0')，则同步该设备
        torch.cuda.synchronize(device)
    else:
        pass

class CodecGaussianModel:
    def __init__(self, sh_degree: int, custom_pipeline: Dict = None, args = None):
        super().__init__()
        self.splat = {}
        self.max_workers = 10
        if custom_pipeline is None:
            if args is not None:
                self.pipeline = PipelineRegistry.build(args.xencode_version)
        else:
            self.pipeline = custom_pipeline
        self.max_sh_degree = sh_degree
        if args is not None:
            self.version = args.xencode_version
        self.refine_pipeline(args)

    def refine_pipeline(self,args):
        if args is not None:
            if args.xencode_version == "mix_d0" or args.xencode_version == "ff_d0":
                self.active_sh_degree = 0
            else:
                self.active_sh_degree = 3.
            # transform settings
            if args.importance:
                self.pipeline["transform"].function_map["global"] = RSNormTransformImp()
            else:
                self.pipeline["transform"].function_map["global"] = RSNormTransform()
            for key, value in self.pipeline["transform"].function_map.items():
                if key != "global":
                    if not args.importance and self.pipeline["transform"].transform_map[key] == "imp":
                        self.pipeline["transform"].function_map[key] = None
                        self.pipeline["transform"].transform_map[key] = "None"


            # sorting settings
            if args.sort_method == "dblockn":
                self.pipeline["spatial_sorter"] = CMortonBlockSortorN(distance=True)
            elif args.sort_method == "hybrid":
                self.pipeline["spatial_sorter"] = HybridAttributeBlockSorter(
                    distance=True,
                    rest_channels=45,
                )
            elif args.sort_method == "plas":
                self.pipeline["spatial_sorter"] = PlasSorter()
            elif args.sort_method == "dual":
                self.pipeline["spatial_sorter"] = MortonSpatialSorter(distance=True)
                self.pipeline["attribute_packer"].scaning = "dualmorton"
            else:
                self.pipeline["spatial_sorter"] = MortonSpatialSorter(distance=True)
            self.pipeline["spatial_sorter"].block_size = args.block_size

            # packing setting
            self.pipeline["attribute_packer"].block_size = args.block_size
            for key, value in self.pipeline["attribute_packer"].packing_map.items():
                if "lsb" in key:
                    value["byteshift"] = -(args.pos_bitdepth - 8)
                if "msb" in key:
                    value["byteshift"] = (args.pos_bitdepth - 8)

            # quantizer setting
            for key, value in self.pipeline["quantizers"].items():
                if hasattr(value, "bits"):
                    if key == "means":
                        bits = args.pos_bitdepth
                    elif key == "rotation":
                        bits = args.rot_bitdepth
                    elif key == "opacity":
                        bits = args.op_bitdepth
                    elif key == "scaling":
                        bits = args.sca_bitdepth
                    elif key == "features_dc":
                        bits = args.sh0_bitdepth
                    elif key == "features_rest":
                        bits = args.shn_bitdepth
                    else:
                        bits = None
                    value.bits = bits if bits is not None else value.bits
            if args.sort_method == "morton" or args.sort_method == "dual" or args.sort_method == "plas":
                self.pipeline["quantizers"]["means"] = GroupMinmaxQuantizer(bits=args.pos_bitdepth, group_size=256)
            if not args.importance:
                self.pipeline["quantizers"].pop('importance', None)

            # prediction
            for key, value in self.pipeline["prediction"].function_map.items():
                if isinstance(value, MinorBlockPrediction) or isinstance(value, MinorPrediction):
                    self.pipeline["prediction"].function_map[key].bitdepth = self.pipeline["quantizers"][key].bits
                    if hasattr(self.pipeline["prediction"].function_map[key], "block_size"):
                        self.pipeline["prediction"].function_map[key].block_size = args.block_size

            # codec setting
            for key, value in self.pipeline["codecs"].items():
                if isinstance(value, AstcCodecPy) and key in args.astc_blocks:
                    value.block_size = args.astc_blocks[key]
                    value.astc_mode = args.astc_mode
                    
                if isinstance(value, VideoCodec):
                    value.codec = args.video_codec
                    self.max_workers = 10
                    if key in args.qps.keys():
                        value.qp = args.qps[key]

    def pipeline_to_dict(self):
        def convert_value(value):
            if isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            elif isinstance(value, (int, float, str, bool)):
                return value
            elif isinstance(value, object) and hasattr(value, '__dict__'):
                class_ = value.__class__
                init_params = inspect.signature(class_.__init__).parameters
                params = {k: v for k, v in value.__dict__.items() if k in init_params}
                return {
                    "method": class_.__name__,
                    "params": {k: convert_value(v) for k, v in params.items()}
                }
            else:
                raise ValueError(f"Unsupported type: {type(value)}")
        return {k: convert_value(v) for k, v in self.pipeline.items()}
    
    def dict_to_pipeline(self, char_dict):
        def convert_value(value):
            if isinstance(value, dict):
                if "method" in value:
                    method_name = value["method"]
                    
                    class_ = globals().get(method_name)
                    if class_ is None:
                        raise ValueError(f"Class {method_name} not found")
                    if "params" in value:
                        return class_(**value["params"])
                    else:
                        return class_()
                else:
                    return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            elif isinstance(value, (int, float, str, bool)):
                return value
            else:
                raise ValueError(f"Unsupported type: {type(value)}")
        return {k: convert_value(v) for k, v in char_dict.items()}

    def restore_fromgaussian(self, gaussian: BaseGaussianModel):
        """
        restore from a gaussian model
        """
        self.active_sh_degree = gaussian.active_sh_degree
        self.max_sh_degree = gaussian.max_sh_degree
        self.device = gaussian.device
        self.splat["means"] = gaussian.get_attribute('means', False).detach().clone()
        self.splat["features_dc"] = gaussian.get_attribute('features_dc', False).detach().clone()
        self.splat["features_rest"] = gaussian.get_attribute('features_sh', False).detach().clone()
        self.splat["opacity"] = gaussian.get_attribute('opacitys', False).clamp(-10,10).detach().clone()
        self.splat["rotation"] = gaussian.get_attribute('quats', False).detach().clone()
        self.splat["scaling"] = gaussian.get_attribute('scales', False).detach().clone()

    def encode(self, jl, sub="main", cameras=None, uwabitstream=False):
        synchronize_device()
        zero_time = time.time()
        encode_stream = {}
        encode_stream["version"] = str(self.version)
        encode_stream["pipeline"] = self.pipeline_to_dict()
        encode_stream["att_meta"] = {
            "sh_degree": self.active_sh_degree, 
            "means_max": self.splat["means"].max(dim=0,keepdim=True)[0].tolist(),
            "means_min": self.splat["means"].min(dim=0,keepdim=True)[0].tolist(),
            }
        encode_stream["stream_meta"] = {}
        encode_stream["streams"] = {}

        # 1~2.排序与变换
        logger.info("开始变换")
        synchronize_device()
        start_time = time.time()
        splat, encode_stream["att_meta"]["transform_meta"] = self.pipeline['transform'].process(self.splat, cameras=cameras)
        synchronize_device()
        jl.log({f"{sub}_transform_time": time.time() - start_time})

        logger.info("开始排序")
        synchronize_device()
        start_time = time.time()
        splat = self.pipeline['spatial_sorter'].process(splat)
        synchronize_device()
        jl.log({f"{sub}_sorting_time": time.time() - start_time})
        encode_stream["att_meta"]["num_points"] = splat["means"].shape[0]

        # 3.量化 name: tensor -> name: quantized tensor 
        logger.info("quantize...")
        synchronize_device()
        start_time = time.time()
        for key, value in splat.items():
            if key in self.pipeline["quantizers"]:
                encode_stream["att_meta"][key] = {}
                splat[key], quant_metadata = self.pipeline["quantizers"][key].process(value)
                bit_depth = self.pipeline["quantizers"][key].bits
                encode_stream["att_meta"][key]["quantmeta"] = self._tensor_to_json_serializable(quant_metadata)
                encode_stream["att_meta"][key]["bit_depth"] = bit_depth
        synchronize_device()
        jl.log({f"{sub}_quantize_time": time.time() - start_time})

        # 3.5 prediction
        synchronize_device()
        start_time = time.time()
        splat, encode_stream["att_meta"]["prediction_meta"] = self.pipeline['prediction'].process(splat, cameras=cameras)
        synchronize_device()
        jl.log({f"{sub}_prediction_time": time.time() - start_time})

        # 4.packing - name: [quantized region tensor] -> name: map_tensor
        logger.info("开始 packing...")
        synchronize_device()
        start_time = time.time()
        splat = self.pipeline["attribute_packer"].packing(splat)
        synchronize_device()
        jl.log({f"{sub}_packing_time": time.time() - start_time})

        # 5.codec - name: map_tensor -> name: map_stream
        logger.info("调用标准编解码器...")
        def encode_and_store(key, value, pipeline, encode_stream):
            start_time = time.time()
            streams = pipeline['codecs'][key].encode(value, name=key)
            encode_stream["streams"][key] = streams
            encode_stream["stream_meta"][key] = {}
            encode_stream["stream_meta"][key]["shapes"] = list(value.shape)
            jl.log({f"{sub}_{key}_size": len(streams[0]) / 1024 / 1024})
            
            if value.dtype is torch.uint32:
                encode_stream["stream_meta"][key]["bit_depth"] = 32
            elif value.dtype is torch.float32:
                encode_stream["stream_meta"][key]["bit_depth"] = 32
            elif value.dtype is torch.uint16:
                encode_stream["stream_meta"][key]["bit_depth"] = 16
            elif value.dtype is torch.uint8:
                encode_stream["stream_meta"][key]["bit_depth"] = 8
            else:
                logger.warning(f"注意不常见数据类型: {value.dtype}")
            jl.log({f"{key}_codec_time": time.time() - start_time})

        synchronize_device()
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for key, value in splat.items():
                if key in self.pipeline["codecs"]:
                    future = executor.submit(encode_and_store, key, value, self.pipeline, encode_stream)
                    futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"处理过程中发生错误: {e}")
                
        # 6.write bitstreams
        logger.info("整合码流...")
        if uwabitstream:
            self.encode_stream = encode_stream
        else:
            data_buffer = BytesIO()
            data_buffer.write(msgpack.packb(encode_stream, use_bin_type=True))
            self.encode_stream = data_buffer.getvalue()
            data_buffer.close()
        logger.info("编码完毕！")
        synchronize_device()
        jl.log({f"{sub}_codec_time": time.time() - start_time})
        jl.log({f"{sub}_encoding_time": time.time() - zero_time})
        jl.log({f"{sub}_size": len(self.encode_stream) / 1024 / 1024})

    def save_encoded_model(self, filepath, args):
        """
        save the encoded 3dgs into bitstreams
        """
        path = os.path.dirname(filepath)
        os.makedirs(path, exist_ok=True)

        stream = self.encode_stream

        with open(filepath, "wb") as f:
            f.write(stream)

    def load_encoded_model(self, file_path, jl, sub=0):
        """
        load encoded model from path
        """
        
        logger.info(f"开始解码码流文件: {file_path}")

        # 1. 读取码流，解析元数据
        logger.info("读取码流...")
        with open(file_path, 'rb') as f:
            encoded_stream = bytearray(f.read())
        self.load_encoded_stream(encoded_stream, jl, sub=sub)
    
    def load_encoded_stream(self, encoded_stream, jl, sub="main"):
        synchronize_device()
        zero_time = time.time()
        start_time = zero_time
        synchronize_device()
        jl.log({f"{sub}_entropy_decoding_time": time.time() - start_time})
        
        if isinstance(encoded_stream, bytes):
            encoded_stream = msgpack.unpackb(encoded_stream, raw=False)
        self.active_sh_degree = encoded_stream['att_meta'].get('sh_degree', 3)
        stream_version = encoded_stream.get('version', 'v1.0')
        self.pipeline = self.dict_to_pipeline(encoded_stream.get('pipeline'))
        logger.warning(f"注意配置版本: {stream_version}")

        # 2. 调用解码器解码 name: [streams] -> name: tensor
        logger.info("调用解码器解码...")
        synchronize_device()
        start_time = time.time()
        decoded_attrs = {}
        for substream_name, streams in encoded_stream['streams'].items():
            decoded_attrs[substream_name] = self.pipeline['codecs'][substream_name].decode(streams, shape=encoded_stream["stream_meta"][substream_name]["shapes"], bit_depth=encoded_stream["stream_meta"][substream_name]["bit_depth"])
        synchronize_device()
        jl.log({f"{sub}_codec_decoding_time": time.time() - start_time})

        # 3. unpacking name: tensor -> name: [quantized tensors]
        logger.info("unpacking...")
        synchronize_device()
        start_time = time.time()
        decoded_attrs = self.pipeline['attribute_packer'].depacking(decoded_attrs, gs_num=encoded_stream["att_meta"]["num_points"])
        synchronize_device()
        jl.log({f"{sub}_depacking_time": time.time() - start_time})
        # 3.5 unpreciction
        decoded_attrs = self.pipeline["prediction"].deprocess(decoded_attrs, encoded_stream["att_meta"].get("prediction_meta", {}))
        
        
        # 4. 反量化 name: quantized tensors -> name: tensors
        logger.info("按属性反量化...")
        start_time = time.time()
        for attr_name, quantized_data in decoded_attrs.items():
            if attr_name in self.pipeline['quantizers']:
                attr_data = self.pipeline['quantizers'][attr_name].deprocess(quantized_data, self._json_serializable_to_tensor(encoded_stream["att_meta"][attr_name]["quantmeta"]))
                decoded_attrs[attr_name] = attr_data
            else:
                decoded_attrs[attr_name] = None
        synchronize_device()
        jl.log({f"{sub}_dequantization_time": time.time() - start_time})

        # 5. 反变换 name: tensors -> name: tensors
        logger.info("反变换 3DGS...")
        start_time = time.time()
        self.splat = self.pipeline["transform"].deprocess(decoded_attrs, encoded_stream["att_meta"].get("transform_meta", {}))
        jl.log({f"{sub}_de_transform_time": time.time() - start_time})
        mask = (self.splat["opacity"] > -9).flatten()
        for key, value in self.splat.items():
            if value is not None:
                # debug
                # if key != "features_rest":
                #     print(key,":: ",value[:1])
                self.splat[key] = value[mask]
        logger.info("解码完毕！")
        synchronize_device()
        jl.log({f"{sub}_decoding_time": time.time() - zero_time})

    def _tensor_to_json_serializable(self, obj):
        """将包含tensor的对象转换为JSON可序列化格式"""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._tensor_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._tensor_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _json_serializable_to_tensor(self, obj):
        """将JSON可序列化格式转换回tensor对象"""
        if isinstance(obj, dict):
            return {k: self._json_serializable_to_tensor(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            # 检查是否是tensor数据（数字列表）
            if obj and isinstance(obj[0], (int, float)):
                return torch.tensor(obj, dtype=torch.float32)
            else:
                return [self._json_serializable_to_tensor(item) for item in obj]
        else:
            return obj

    def save_ply(self, ply_path):
        gaussian = BaseGaussianModel(self.max_sh_degree)
        gaussian.active_sh_degree = self.active_sh_degree
        gaussian._xyz = self.splat["means"]
        gaussian._features_dc = self.splat["features_dc"].view([-1,1,3])
        gaussian._features_rest = self.splat["features_rest"].view([self.splat["features_rest"].shape[0],-1,3])
        gaussian._opacity = self.splat["opacity"].view([-1,1])
        gaussian._scaling = self.splat["scaling"].view([-1,3])
        gaussian._rotation = self.splat["rotation"].view([-1,4])
        gaussian.save_ply(ply_path)

def get_subset_pipelines(key):
    if key == "":
        return None
    elif key == "default":
        return {
            "segment": CompleteSeg(),
            "pipelines": {"main": "default"},
        }
    else:
        return None

class CodecGaussianModels:
    def __init__(self, sh_degree: int, custom_pipeline: Dict = None, args = None):
        import copy
        super().__init__()
        self.splat = {}
        if custom_pipeline is None:
            self.pipeline = get_subset_pipelines(args.xencode_version)
            if self.pipeline is None:
                self.pipeline = {
                "segment": CompleteSeg(),
                "pipelines": {"main":args.xencode_version},
            }
        else:
            self.pipeline = custom_pipeline
        self.max_sh_degree = sh_degree
        self.version = args.xencode_version
        self.args = copy.deepcopy(args)

    def restore_fromgaussian(self, gaussian: BaseGaussianModel):
        """
        restore from a gaussian model
        """
        self.active_sh_degree = gaussian.active_sh_degree
        self.device = gaussian.device
        self.splat["means"] = gaussian.get_attribute('means', False).detach().clone()
        self.splat["features_dc"] = gaussian.get_attribute('features_dc', False).detach().clone()
        if self.max_sh_degree > 0:
            self.splat["features_rest"] = gaussian.get_attribute('features_sh', False).detach().clone()
        self.splat["opacity"] = torch.nan_to_num(gaussian.get_attribute('opacitys', False).clamp(-10,10).detach().clone(),nan=-10.0)
        self.splat["rotation"] = gaussian.get_attribute('quats', False).detach().clone()
        self.splat["scaling"] = gaussian.get_attribute('scales', False).detach().clone()
    
    def encode(self, jl, cameras=None, uwabitstream=False):
        # 0. 分割模型
        splats = self.pipeline["segment"].process(self.splat)
        streams = {}
        
        # 1. subset 分别编码
        synchronize_device()
        start = time.time()
        for ind, key in enumerate(splats.keys()):
            args = copy.deepcopy(self.args)
            args.xencode_version = self.pipeline["pipelines"][key]
            if key in self.args.astc_blocks:
                args.astc_blocks = self.args.astc_blocks[key]
            if key in self.args.qps:
                args.qps  = self.args.qps[key]
            sub_splat = CodecGaussianModel(self.max_sh_degree, args=args)
            sub_splat.splat = splats[key]
            sub_splat.encode(jl, sub=key, cameras=cameras, uwabitstream=uwabitstream)
            streams[key] = sub_splat.encode_stream
        
        # 2. 整合码流
        if uwabitstream:
            self.encode_stream = UWABitstreamPacker().encode_streamer(streams)
            data_buffer = BytesIO()
            data_buffer.write(self.encode_stream)
            self.encode_stream = data_buffer.getvalue()
            data_buffer.close()
        else:
            data_buffer = BytesIO()
            data_buffer.write(msgpack.packb(streams, use_bin_type=True))
            self.encode_stream = zstd.compress(data_buffer.getvalue())
            data_buffer.close()
        synchronize_device()
        jl.log({"total_encoding_time": time.time() - start})
        jl.log({"total_size": len(self.encode_stream) / 1024 / 1024})
    
    def save_encoded_model(self, filepath, args):
        """
        save the encoded 3dgs into bitstreams
        """
        path = os.path.dirname(filepath)
        os.makedirs(path, exist_ok=True)

        stream = self.encode_stream

        with open(filepath, "wb") as f:
            f.write(stream)
        
    def load_encoded_model(self, file_path, jl, uwabitstream=False):
        """
        load encoded model from path
        """
        synchronize_device()
        start = time.time()
        # 1. 读取码流，解析元数据
        logger.info("读取码流...")
        with open(file_path, 'rb') as f:
            encoded_stream = bytearray(f.read())

        # 2. subset 解码
        decoded_splats = []
        if uwabitstream:
            encoded_stream = bytes(encoded_stream)
            encoded_stream = UWABitstreamPacker().decode_streamer(encoded_stream)
        else:
            encoded_stream = zstd.decompress(bytes(encoded_stream))
            encoded_stream = msgpack.unpackb(encoded_stream, raw=False)
        for ind, key in enumerate(encoded_stream.keys()):
            sub_splat = CodecGaussianModel(self.max_sh_degree)
            sub_splat.load_encoded_stream(encoded_stream[key], jl, sub=key)
            decoded_splats.append(sub_splat.splat)
            self.active_sh_degree = sub_splat.active_sh_degree
            
        
        # 3. subset 合并
        if len(decoded_splats) > 1:
            for splat in decoded_splats[1:]:
                for key, value in splat.items():
                    if key in decoded_splats[0].keys():
                        decoded_splats[0][key] = torch.cat([decoded_splats[0][key], value.view([-1]+list(decoded_splats[0][key].shape[1:]))], dim=0)
        self.splat = decoded_splats[0]
        
        synchronize_device()
        jl.log({"total_decoding_time": time.time() - start})

    def save_ply(self, ply_path):
        gaussian = BaseGaussianModel(self.max_sh_degree)
        gaussian.active_sh_degree = self.active_sh_degree
        gaussian._xyz = self.splat["means"]
        gaussian._features_dc = self.splat["features_dc"].view([-1,1,3])
        if "features_rest" in self.splat and self.splat["features_rest"] is not None and self.active_sh_degree > 0:
            gaussian._features_rest = self.splat["features_rest"].view([self.splat["features_rest"].shape[0],-1,3])
        else:
            gaussian._features_rest = None
        gaussian._opacity = self.splat["opacity"].view([-1,1])
        gaussian._scaling = self.splat["scaling"].view([-1,3])
        gaussian._rotation = self.splat["rotation"].view([-1,4])
        gaussian.save_ply(ply_path)