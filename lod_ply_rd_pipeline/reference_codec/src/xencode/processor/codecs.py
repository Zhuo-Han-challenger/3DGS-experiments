#  Copyright [2025] [Huawei Technologies Co., Ltd.] 
#  Licensed under the Code Sharing Policy of the UHD World Association (the "Policy"); 
#  http://www.theuwa.com/UWA_Code_Sharing_Policy.pdf.
#  you may not use this file except in compliance with the Policy. 
#  Unless agreed to in writing, software distributed under the Policy is distributed on an "AS IS" BASIS, 
#  WITHOUT WARRANTIES OF ANY KIND, either express or implied. 
#  See the Policy for the specific language governing permissions and 
#  limitations under the Policy.

import math
import cv2
import torch
import numpy as np
import tempfile
import subprocess
import os
import zstd
import ffmpeg
import struct
import astcenc_py

from pathlib import Path
from typing import List, Dict, Literal
from io import BytesIO

class EntropyCodec():
    """Entropy编解码器（适用于整型数据）
    """
    def __init__(self, compression_level: int = 19, bitdepth=8, entropy=True):
        """
        Args:
            compression_level: zstd 压缩级别 (1-22)，默认 19。
                - 1-3: 快速压缩，较低压缩率
                - 10-15: 平衡模式
                - 19: 高压缩率，合理速度（推荐）
                - 22: 最高压缩率，最慢（原默认值）
        """
        self.compression_level = compression_level
        self.bitdepth = bitdepth
        self.entropy = entropy
    
    def encode(self, value: torch.Tensor, **kwargs) -> bytes:
        # 直接转换为 numpy 并获取 bytes，避免 BytesIO 开销
        if self.bitdepth <= 8:
            data_bytes = value.cpu().numpy().astype('>u1').tobytes()
        elif self.bitdepth <= 16:
            data_bytes = value.cpu().numpy().astype('>u2').tobytes()
        else:
            data_bytes = value.cpu().numpy().astype('>u4').tobytes()
        # 使用可配置的压缩级别
        if self.entropy:
            encode_stream = zstd.compress(data_bytes, self.compression_level)
        else:
            encode_stream = data_bytes
        return [encode_stream]
    
    def decode(self, data: List, **kwargs) -> torch.Tensor:
        if self.entropy:
            encoded_stream = zstd.decompress(data[0])
        else:
            encoded_stream = data[0]
        if kwargs["bit_depth"] <= 8:
            dtype = '>u1'
            dtype2 = np.uint8
        elif kwargs["bit_depth"] <= 16:
            dtype = '>u2'
            dtype2 = np.uint16
        else:
            dtype = '>u4'
            dtype2 = np.uint32
        result = torch.tensor(np.frombuffer(encoded_stream, dtype=dtype).astype(dtype2)).reshape(kwargs["shape"])
        return result

class NoneCodec():
    """无压缩编解码器（适用于不需要压缩的场景）
    """
    def encode(self, value: torch.Tensor, **kwargs) -> bytes:
        # 直接转换为 bytes，避免 BytesIO 开销
        if value.dtype == torch.uint8:
            encode_stream = value.cpu().numpy().astype(np.uint8).tobytes()
        else:
            encode_stream = value.cpu().numpy().astype(np.uint16).tobytes()
        return [encode_stream]
    
    def decode(self, data: List, **kwargs) -> torch.Tensor:
        dtype = np.uint8 if kwargs["bit_depth"][0] <= 8 else np.uint16
        result = torch.tensor(np.frombuffer(data, dtype=dtype)).reshape(kwargs["shape"])
        return result


class AstcCodecPy():
    """ASTC 编解码器"""
    def __init__(self, block_size = 4, astc_mode = "medium", vis=False, save_dir = "results", entropy=True) -> None:
        self.vis = vis
        self.save_dir = save_dir
        self.block_size = block_size
        self.astc_mode = astc_mode
        self.log_path = os.path.join(".", f"out.log")
        self.entropy = entropy

    def encode(self, value: torch.Tensor, name = None, **kwargs):
        '''
        tensor -> list(astc stream)
        '''
        h,w,t,c = value.shape
        img_datas = []
        results = []
        if value.dtype == torch.uint8:
            img_data = value.flatten(2).detach().cpu().numpy().astype(np.uint8)
        else:
            img_data = value.flatten(2).detach().cpu().numpy().astype(np.uint16)

        # 处理不同通道数
        if t*c == 1:
            img_datas.append(np.concatenate([img_data[:,:,:],np.zeros_like(img_data[:,:,:]),np.zeros_like(img_data[:,:,:])],axis=-1))
        elif t*c == 3:
            img_datas.append(img_data)
        elif t*c > 3:
            for ind in range(math.ceil(img_data.shape[2]/3)):
                img_datas.append(img_data[:,:,ind*3:ind*3+3])

        encoder = astcenc_py.ASTCEncoder(
            block_x=self.block_size,
            block_y=self.block_size,
            quality=astcenc_py.QUALITY_MEDIUM,
            profile=astcenc_py.Profile.LDR,
            thread_count=64
        )
        for ind, img_data in enumerate(img_datas):
            stream = encoder.compress(img_data)
            if self.entropy:
                data_buffer = BytesIO()
                data_buffer.write(stream)
                stream = zstd.compress(data_buffer.getvalue())
                data_buffer.close()
            results.append(stream)
            encoder.reset_compress()
        
            if self.vis:
                cv2.imwrite(f"tmp/tmp/{name}_{ind}.png", img_data)

        return results
    
    def decode(self, data: List, **kwargs) -> torch.Tensor:
        '''
        list(astc stream) -> tensor
        '''
        img_arrays = []

        for idx, astc_data in enumerate(data):
            if self.entropy:
                astc_data = zstd.decompress(bytes(astc_data))
            block_size = struct.unpack("<BBBBB", astc_data[:5])[-1]
            encoder = astcenc_py.ASTCEncoder(
                block_x=block_size,
                block_y=block_size,
                quality=astcenc_py.QUALITY_MEDIUM,
                profile=astcenc_py.Profile.LDR,
                thread_count=1
            )
            img_arrays.append(encoder.decompress(astc_data)[:,:,:3])
            encoder.reset_decompress()
        
        if any(img_array is None for img_array in img_arrays):
            raise RuntimeError("ASTC解码失败")
        
        # 处理BGR到RGB的转换
        tmp = []
        for img_array in img_arrays:
            # 根据原始数据类型选择合适的torch类型
            if img_array.dtype == np.uint16:
                result = torch.tensor(img_array, dtype=torch.uint16)
            else:
                result = torch.tensor(img_array, dtype=torch.uint8)
            if len(result.shape) == 2:
                result = result.unsqueeze(-1)
            tmp.append(result)
        h,w,t,c = kwargs["shape"]
        results = torch.cat(tmp,dim=-1).reshape([h,w,t,-1])#[:,:,:,:c]
        return results

class VideoCodec():
    configs = {
        "hm": {
            "encoder": "dependency/HM-18.0/bin/TAppEncoderStatic",
            "decoder": "dependency/HM-18.0/bin/TAppDecoderStatic",
            "intra_lossy_config_path": "src/xencode/codec_cfg/hm_cfg/intra_yuv444p.cfg",
            "intra_lossless_config_path": "src/xencode/codec_cfg/hm_cfg/intra_lossless_yuv444p.cfg",
            "inter_lossy_config_path": "src/xencode/codec_cfg/hm_cfg/inter_yuv444p.cfg",
            "inter_lossless_config_path": "src/xencode/codec_cfg/hm_cfg/inter_lossless_yuv444p.cfg",
        },
        "vtm": {
            "encoder": "dependency/VTM-23.11/bin/EncoderAppStatic",
            "decoder": "dependency/VTM-23.11/bin/DecoderAppStatic",
            "intra_lossy_config_path": "src/xencode/codec_cfg/vtm_cfg/cfg/intra_yuv444p.cfg",
            "intra_lossless_config_path": "src/xencode/codec_cfg/vtm_cfg/cfg/intra_lossless_yuv444p.cfg",
            "inter_lossy_config_path": "src/xencode/codec_cfg/vtm_cfg/cfg/inter_yuv444p.cfg",
            "inter_lossy_420_config_path": "src/xencode/codec_cfg/vtm_cfg/cfg/inter_yuv444p.cfg",
            "inter_lossless_config_path": "src/xencode/codec_cfg/vtm_cfg/cfg/inter_lossless_yuv444p.cfg",
        },
        "x265" : {
            
        },
        "x264" : {
            
        },
    }
    def __init__(self, codec="x265", pix_fmt="yuv444p", qp=24, intra=False, lossy=True, vis=False, save_dir = "results", preset = "medium") -> None:
        self.vis = vis
        self.save_dir = save_dir
        self.codec = codec
        self.pix_fmt = pix_fmt
        self.qp = qp if lossy else -1
        self.lossy = lossy
        self.intra = intra
        self.preset = preset

    """video编解码器（适用于空间相关的2D数据）"""
    def encode(self, value: torch.Tensor, name = None, **kwargs) -> bytes:
        h,w,t,c = value.shape
        retults = []
        self.qp = self.qp if self.lossy else -1
        # 处理不同通道数
        img_datas = []
        if c == 1:
            if self.pix_fmt != "yuv400p":
                other_channel = torch.zeros_like(value[:,:,:,0])
                img_datas.append(torch.stack([value[:,:,:,0], other_channel, other_channel], dim=-1))
                if self.vis:
                    cv2.imwrite(f"tmp/tmp/{name}.png", value.flatten(2).detach().cpu().numpy().astype(np.uint8))
            else:
                img_datas.append(value[:,:,:,:])  # 单通道
        elif c == 3:
            img_datas.append(value)
        elif c > 3:
            for jnd in range(math.ceil(value.shape[-1]/3)):
                img_datas.append(value[:,:,:,jnd*3:jnd*3+3])
        
        for jnd, img_data in enumerate(img_datas):
            height, width = img_data.shape[:2]
            # x265, x264 管道模式：直接在内存中处理
            if self.codec in ["x265", "x264"]:
                # 将 tensor 转换为 numpy 数组 (T, H, W, C)
                video_np = img_data.permute(2, 0, 1, 3).cpu().numpy().astype(np.uint8)
                
                if self.codec in ["x265", "x264"]:
                    stream = self._encode_ffmpeg_pipe(
                        video_data=video_np,
                        width=width,
                        height=height,
                        pix_fmt=self.pix_fmt,
                        qp=self.qp,
                        use_all_intra=self.intra,
                        use_chroma_qp_offset=False,
                        preset=self.preset,
                        codec=self.codec
                    )
                retults.append(stream)
            else:
                # 文件模式：使用临时文件
                temp_dir_path = 'tmp'
                os.makedirs(temp_dir_path, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=temp_dir_path) as temp_dir:
                    yuv_file = os.path.join(temp_dir, f"video_{name}_{jnd:02d}.yuv")
                    chroma_format = self.pix_fmt.replace("yuv", "").replace("p", "")
                    self._save_tensor_to_yuv(img_data.permute(2,0,1,3), yuv_file, chroma_format)
                    file_extension = ".vtm" if self.codec == "vtm" else ".hevc"
                    file_extension = ".jpg" if self.codec == "jpeg" else file_extension
                    video_file = os.path.join(temp_dir, f"video_{name}_{jnd:02d}{file_extension}")
                    
                    try:
                        self._encode_yuv_to_video_stream_multi_encoder(
                            yuv_file_path=yuv_file,
                            output_video_path=video_file,
                            width=width,
                            height=height,
                            pix_fmt=self.pix_fmt,
                            qp=self.qp,
                            use_all_intra=self.intra,
                            use_chroma_qp_offset=False,
                            encoder_type=self.codec,
                            encoder_config=self.configs[self.codec],
                            frame_num=t,
                            preset=self.preset
                        )
                    except Exception as e:
                        raise RuntimeError(f"Video 编码失败:{e}")

                    tmp_file = open(video_file, "rb")
                    stream = tmp_file.read()
                    retults.append(stream)
                    tmp_file.close()
                
        return retults
    
    def decode(self, data: List, **kwargs) -> torch.Tensor:
        tmp = []
        h,w,t,c = kwargs["shape"]
        # 管道模式：直接在内存中处理，避免文件 I/O
        if self.codec in ["x265", "x264"]:
            for ind, stream in enumerate(data):
                if self.codec in ["x265", "x264"]:
                    video_np = self._decode_ffmpeg_pipe(
                        video_data=stream,
                        width=w,
                        height=h,
                        pix_fmt=self.pix_fmt,
                        codec="hevc"
                    )

                # 转换为 tensor: (T, H, W, C) -> (H, W, T, C)
                video_tensor = torch.from_numpy(video_np).view([t, h, w, -1]).permute([1, 2, 0, 3])
                tmp.append(video_tensor)
        else:
            # 文件模式：使用临时文件
            temp_dir_path = 'tmp'
            os.makedirs(temp_dir_path, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=temp_dir_path) as temp_dir:
                video_files = []
                yuv_files = []
                for ind, stream in enumerate(data):
                    file_extension = ".vtm" if self.codec == "vtm" else ".hevc"
                    video_file = os.path.join(temp_dir, f"video_{ind:02d}"+file_extension)
                    yuv_file = os.path.join(temp_dir, f"video_{ind:02d}_decoded.yuv")
                    tmp_file = open(video_file, "wb")
                    tmp_file.write(stream)
                    tmp_file.close()
                    video_files.append(video_file)
                    yuv_files.append(yuv_file)
                
                    self._decode_video_stream_to_yuv_multi_encoder(
                        video_file=video_file,
                        yuv_file=yuv_file,
                        width=w,
                        height=h,
                        pix_fmt=self.pix_fmt,
                        encoder_type=self.codec,
                        encoder_config=self.configs[self.codec])

                    tmp.append(self._load_yuv_to_tensor(yuv_file, h, w, self.pix_fmt).view([t,h,w,-1]).permute([1,2,0,3]))
        
        results = torch.cat(tmp,dim=-1).reshape([h,w,t,-1])
        return results
    
    def _save_tensor_to_yuv(self, video: torch.Tensor,
                        yuv_file_path: str,
                        chroma_subsampling: Literal["420", "444", "400"] = "420"):
        '''
        Save a tensor to a YUV file in planar format.

        Args:
            video (torch.Tensor): video in the shape of [T, H, W, 3], assumed to be uint8 and contain YUV data.
                                For 420, the U and V channels might conceptually represent subsampling
                                (e.g., via `chroma_downsampling` function), but still have HxW resolution.
            yuv_file_path (str): path to save the YUV file
            chroma_subsampling (str): chroma subsampling format ("420", "444", or "400")
        '''
        n_frames = int(video.size(0))
        H = int(video.size(1))
        W = int(video.size(2))

        # Ensure tensor is uint8 and on CPU
        if video.dtype != torch.uint8:
            if video.dtype.is_floating_point:
                video = (video.clamp(0, 1) * 255).round().to(torch.uint8)
            else:
                video = video.to(torch.uint8)

        video_np = video.cpu().numpy()  # Shape: (T, H, W, 3)
        
        # 预计算输出大小并一次性分配
        if chroma_subsampling == "400":
            bytes_per_frame = H * W
        elif chroma_subsampling == "444":
            bytes_per_frame = H * W * 3
        elif chroma_subsampling == "420":
            bytes_per_frame = H * W + 2 * (H // 2) * (W // 2)
        else:
            raise ValueError(f"Unsupported chroma subsampling format: {chroma_subsampling}")
        
        # 预分配输出缓冲区
        output_buffer = bytearray(n_frames * bytes_per_frame)
        offset = 0
        
        if chroma_subsampling == "400":
            # YUV400: 只写入 Y 平面
            for t in range(n_frames):
                y_plane = video_np[t, ..., 0].tobytes()
                output_buffer[offset:offset + len(y_plane)] = y_plane
                offset += len(y_plane)
        elif chroma_subsampling == "444":
            # YUV444: 逐帧写入 Y, U, V 平面
            for t in range(n_frames):
                frame = video_np[t]
                y_plane = frame[..., 0].tobytes()
                u_plane = frame[..., 1].tobytes()
                v_plane = frame[..., 2].tobytes()
                output_buffer[offset:offset + len(y_plane)] = y_plane
                offset += len(y_plane)
                output_buffer[offset:offset + len(u_plane)] = u_plane
                offset += len(u_plane)
                output_buffer[offset:offset + len(v_plane)] = v_plane
                offset += len(v_plane)
        elif chroma_subsampling == "420":
            # YUV420: Y 平面 + 下采样后的 U, V 平面
            for t in range(n_frames):
                frame = video_np[t]
                y_plane = frame[..., 0].tobytes()
                u_subsampled = frame[::2, ::2, 1].tobytes()
                v_subsampled = frame[::2, ::2, 2].tobytes()
                output_buffer[offset:offset + len(y_plane)] = y_plane
                offset += len(y_plane)
                output_buffer[offset:offset + len(u_subsampled)] = u_subsampled
                offset += len(u_subsampled)
                output_buffer[offset:offset + len(v_subsampled)] = v_subsampled
                offset += len(v_subsampled)
        
        # 一次性写入文件
        with open(yuv_file_path, 'wb') as f:
            f.write(output_buffer)

    def _load_yuv_to_tensor(
        self,
        yuv_file: str,
        height: int,
        width: int,
        pix_fmt: Literal["yuv420p", "yuv444p", "yuv400p"],
        bit_depth: int = 8
    ):
        """
        Load a raw YUV file into a PyTorch tensor.

        Args:
            yuv_file (str): Path to the input raw YUV file.
            height (int): Height of the video frames.
            width (int): Width of the video frames.
            pix_fmt (Literal["yuv420p", "yuv444p", "yuv400p"]): Pixel format of the YUV file.
            bit_depth (int): Bit depth (8 or 10).

        Returns:
            Tensor: A tensor containing the video data in shape [T, H, W, 3] and dtype uint8.
        """
        if bit_depth == 8:
            pix_bytes = 1
            _dtype = torch.uint8
            np_dtype = np.uint8
        elif bit_depth == 10:
            pix_bytes = 2
            _dtype = torch.int16
            np_dtype = np.uint16
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}. Only 8 and 10 bits are supported.")
        
        if not os.path.exists(yuv_file):
            raise FileNotFoundError(f"YUV file not found: {yuv_file}")

        file_size = os.path.getsize(yuv_file)
        
        # 自动检测像素格式
        if file_size % (pix_bytes * height * width * 3) != 0:
            if file_size % (pix_bytes * int(height * width * 1.5)) != 0:
                pix_fmt = "yuv400p"
            else:
                pix_fmt = "yuv420p"
        
        if pix_fmt == "yuv444p":
            bytes_per_frame = pix_bytes * height * width * 3
            chroma_height, chroma_width = height, width
        elif pix_fmt == "yuv420p":
            bytes_per_frame = pix_bytes * int(height * width * 1.5)
            chroma_height, chroma_width = height // 2, width // 2
        elif pix_fmt == "yuv400p":
            bytes_per_frame = pix_bytes * height * width
            chroma_height, chroma_width = 0, 0
        else:
            raise ValueError(f"Unsupported pix_fmt: {pix_fmt}")

        if bytes_per_frame == 0:
            raise ValueError("Calculated bytes_per_frame is zero. Check height/width.")

        if file_size == 0:
            return torch.empty((0, height, width, 3), dtype=_dtype)

        if file_size % bytes_per_frame != 0:
            raise ValueError(
                f"File size {file_size} is not a multiple of calculated frame size {bytes_per_frame} "
                f"for H={height}, W={width}, pix_fmt={pix_fmt}."
            )

        num_frames = file_size // bytes_per_frame
        
        # 一次性读取整个文件
        with open(yuv_file, 'rb') as f:
            raw_data = f.read()
        
        # 使用 numpy 批量解析
        data = np.frombuffer(raw_data, dtype=np_dtype)
        
        if pix_fmt == "yuv400p":
            # YUV400: 只有 Y 平面
            y_data = data.reshape(num_frames, height, width)
            # 扩展为 (T, H, W, 1)
            video_tensor = torch.from_numpy(y_data.copy()).unsqueeze(-1)
            # 扩展为 (T, H, W, 3) 用 Y 填充所有通道
            video_tensor = video_tensor.expand(-1, -1, -1, 3).clone()
            
        elif pix_fmt == "yuv444p":
            # YUV444: 直接 reshape
            frames = data.reshape(num_frames, height, width, 3)
            video_tensor = torch.from_numpy(frames.copy())
            
        elif pix_fmt == "yuv420p":
            # YUV420: 需要分离 Y, U, V 并上采样
            y_plane_size = height * width
            uv_plane_size = chroma_height * chroma_width
            
            # 预分配输出张量
            video_tensor = torch.zeros((num_frames, height, width, 3), dtype=_dtype)
            
            for t in range(num_frames):
                frame_start = t * bytes_per_frame // pix_bytes
                
                # Y 平面
                y_plane = data[frame_start:frame_start + y_plane_size].reshape(height, width)
                
                # U 平面
                u_start = frame_start + y_plane_size
                u_plane = data[u_start:u_start + uv_plane_size].reshape(chroma_height, chroma_width)
                
                # V 平面
                v_start = u_start + uv_plane_size
                v_plane = data[v_start:v_start + uv_plane_size].reshape(chroma_height, chroma_width)
                
                # 上采样 U 和 V（使用 repeat 比 interpolate 更快）
                u_upsampled = np.repeat(np.repeat(u_plane, 2, axis=0), 2, axis=1)
                v_upsampled = np.repeat(np.repeat(v_plane, 2, axis=0), 2, axis=1)
                
                # 组合帧
                frame = np.stack([y_plane, u_upsampled, v_upsampled], axis=-1)
                video_tensor[t] = torch.from_numpy(frame)
        
        if bit_depth == 10:
            video_tensor = (video_tensor >> 2).to(torch.uint8)
            
        return video_tensor

    def _encode_yuv_to_video_stream_multi_encoder(
        self,
        yuv_file_path: str,
        output_video_path: str,
        width: int,
        height: int,
        pix_fmt: str,
        qp: int,
        use_all_intra: bool = False,
        use_chroma_qp_offset: bool = False,
        encoder_type: str = "x265",
        encoder_config: Dict = None,
        frame_num: int = None,
        preset: str = "medium",
    ):

        if encoder_config is None:
            encoder_config = self.configs.get(encoder_type, self.configs["hm"])
        
        if encoder_type in ["x265", "x264"]:
            self._encode_ffmpeg(yuv_file_path, output_video_path, width, height, pix_fmt, 
                        qp, use_all_intra, use_chroma_qp_offset, preset)

        elif encoder_type == "hm":
            self._encode_hm(yuv_file_path, output_video_path, width, height, pix_fmt, 
                    qp, use_all_intra, encoder_config, frame_num)

        elif encoder_type == "vtm":
            self._encode_vtm(yuv_file_path, output_video_path, width, height, pix_fmt, 
                    qp, use_all_intra, encoder_config, frame_num)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

    def _decode_video_stream_to_yuv_multi_encoder(
        self,
        video_file: str,
        yuv_file: str,
        width: int,
        height: int,
        pix_fmt: str,
        encoder_type: str = "x265",
        encoder_config: Dict = None
    ):

        if encoder_config is None:
            encoder_config = self.configs.get(encoder_type, self.configs["hm"])
        if encoder_type in ["x265", "x264"]:
            self._decode_ffmpeg(video_file, yuv_file, width, height, pix_fmt)
        elif encoder_type in ["hm", "vtm"]:
            self._decode_hm_vtm(video_file, yuv_file, encoder_config)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

    def _encode_hm(self, yuv_file_path: str, output_video_path: str, width: int, height: int, 
               pix_fmt: str, qp: int, use_all_intra: bool, encoder_config: Dict, frame_num: int):
        pix_fmt_map = {'yuv420p': '420', 'yuv422p': '422', 'yuv444p': '444', 'yuv400p': '400'}
        chroma_format = pix_fmt_map.get(pix_fmt)
        if not chroma_format:
            raise ValueError(f"Unsupported pix_fmt for HM: {pix_fmt}")

        if qp < 0:
            config_path = (encoder_config["intra_lossless_config_path"]
                        if use_all_intra else encoder_config["inter_lossless_config_path"])
            qp_value = 4
        else:
            config_path = (encoder_config["intra_lossy_config_path"]
                        if use_all_intra else encoder_config["inter_lossy_config_path"])
            qp_value = qp

        cmd = [
            encoder_config["encoder"],
            '-c', config_path,
            '-i', yuv_file_path,
            '-b', output_video_path,
            '-wdt', str(width),
            '-hgt', str(height),
            '-q', str(qp_value),
            '-cf', chroma_format,
            '-fr', '30',
            '-f', str(frame_num),
            f'--InputChromaFormat={chroma_format}',
        ]
        # print(cmd)
        if use_all_intra:
            cmd.extend(['--IntraPeriod', '1'])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"HM encoding failed: {result.stderr}")

    def _encode_vtm(self, yuv_file_path: str, output_video_path: str, width: int, height: int, 
                pix_fmt: str, qp: int, use_all_intra: bool, encoder_config: Dict, frame_num: int):
        pix_fmt_map = {'yuv420p': '420', 'yuv422p': '422', 'yuv444p': '444', 'yuv400p': '400'}
        chroma_format = pix_fmt_map.get(pix_fmt)
        if not chroma_format:
            raise ValueError(f"Unsupported pix_fmt for VTM: {pix_fmt}")

        if qp < 0:
            config_path = (encoder_config["intra_lossless_config_path"]
                        if use_all_intra else encoder_config["inter_lossless_config_path"])
            qp_value = 4
        else:
            config_path = (encoder_config["intra_lossy_config_path"]
                        if use_all_intra else encoder_config["inter_lossy_config_path"])
            qp_value = qp

        cmd = [
            encoder_config["encoder"],
            '-c', config_path,
            '-i', yuv_file_path,
            '-b', output_video_path,
            f'--SourceWidth={width}',
            f'--SourceHeight={height}',
            f"--InputBitDepth={8}",
            '-q', str(qp_value),
            '-cf', chroma_format,
            '-fr', '30',
            '-f', str(frame_num),
            f'--InputChromaFormat={chroma_format}',
            "--ConformanceWindowMode=1",
            "--TemporalSubsampleRatio=1",
            "--OutputBitDepth=8",
            "--InputBitDepth=8",
            "--InternalBitDepth=8",
        ]

        
        # print(cmd)
        
        if use_all_intra:
            cmd.extend(['-ip', '1'])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"VTM encoding failed: {result.stderr}")
        
    def _decode_ffmpeg(self, video_file: str, yuv_file: str, width: int, height: int, pix_fmt: str):
        if pix_fmt == "yuv400p":
            pix_fmt = "gray"

        cmd = [
            "ffmpeg", "-y",
            "-i", video_file,
            "-pix_fmt", pix_fmt,
            "-c:v", "rawvideo",
            yuv_file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg decoding failed: {result.stderr}")
    
    def _decode_hm_vtm(self, video_file: str, yuv_file: str, encoder_config: Dict):
        cmd = [
            encoder_config["decoder"],
            '-b', video_file,
            '-o', yuv_file,
            '--OutputBitDepth=8'
        ]
        # print(cmd)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"HM/VTM decoding failed: {result.stderr}")
    
    # def _decode_jpeg(self, video_file: str, yuv_file: str, width: int, height: int, pix_fmt: str):
    #     # if pix_fmt == "yuv400p":
    #     pix_fmt = "gray"

    #     cmd = [
    #         "ffmpeg", "-y",
    #         "-i", video_file,
    #         "-pix_fmt", pix_fmt,
    #         "-c:v", "rawvideo",
    #         yuv_file
    #     ]

    #     result = subprocess.run(cmd, capture_output=True, text=True)
    #     if result.returncode != 0:
    #         raise RuntimeError(f"FFmpeg decoding failed: {result.stderr}")
    
    def _encode_ffmpeg_pipe(
        self,
        video_data: np.ndarray,
        width: int,
        height: int,
        pix_fmt: str,
        qp: int,
        use_all_intra: bool = False,
        use_chroma_qp_offset: bool = False,
        preset: str = "medium",
        codec: str = "x265",
    ) -> bytes:
        """
        使用 ffmpeg-python 通过管道编码视频数据。
        
        Args:
            video_data: numpy 数组，形状为 (T, H, W, C)，dtype 为 uint8
            width: 视频宽度
            height: 视频高度
            pix_fmt: 像素格式 (yuv420p, yuv444p, yuv400p)
            qp: 量化参数
            use_all_intra: 是否使用全帧内编码
            use_chroma_qp_offset: 是否使用色度 QP 偏移
            preset: 编码预设
        
        Returns:
            编码后的视频字节流
        """
        n_frames = video_data.shape[0]
        
        # 处理 yuv400p 的像素格式映射
        input_pix_fmt = "gray" if pix_fmt == "yuv400p" else pix_fmt
        
        # 构建 x265 编码参数
        if qp <= 4:
            qp = 0
        qp_params = f"qp={qp}" if qp >= 0 else 'lossless=1'
        intra_params = ":keyint=1:min-keyint=1:scenecut=0" if use_all_intra else ""
        chroma_qp_params = ":chroma-qp-offset=2" if use_chroma_qp_offset else ""
        
        # 基础 x265 参数
        base_x265_params = (
            f"{qp_params}{intra_params}{chroma_qp_params}"
            ":scenecut=0:rect=1:ctu=32:amp=1:cpp=1"
            ":strong-intra-smoothing=1:screen-content-mode=1"
            ":screen-content-prediction=1:tskip=1:tskip-fast=0"
            ":psy-rd=0:psy-rdoq=0:ipratio=1.1:pbratio=1.1:rd=1"
            ":intra=dc,planar,angular:tu-inter-depth=2:tu-intra-depth=2"
            ":max-tu-size=32:aq-mode=0"
            # "-vf"
            # f"addroi=0:ih/4:iw:ih/4:-1,addroi=0:ih/4*2:iw/4:ih/4:-1,addroi=iw/4*3:0:iw/4:ih/4:-1"
        )

        base_x264_params = (
            f"{qp_params}{intra_params}{chroma_qp_params}"
            ":scenecut=0"                    
            ":partitions=all"                
            ":no-8x8dct"                     
            ":no-fast-pskip"                 
            ":psy-rd=0:psy-trellis=0"        
            ":ipratio=1.1:pbratio=1.1"       
            ":subme=7"                       
            ":trellis=0"                     
            ":aq-mode=0"                     
            ":tune=fastdecode"               
        )
        
        if qp > 4:
            x265_params = base_x265_params
            x264_params = base_x264_params
        else:
            # lossless 模式添加 cu-lossless
            x265_params = base_x265_params + ":cu-lossless=1"
            x264_params = base_x264_params + ":lossless=1"
            
        
        # 构建 ffmpeg 管道
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt=input_pix_fmt, 
                   s=f'{width}x{height}', r=30)
            .output('pipe:', format='hevc', vcodec='libx265' if codec == "x265" else 'libx264',
                    preset=preset, **{f'{codec}-params': x265_params if codec == "x265" else x264_params} )
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )
        
        # 将视频数据写入 stdin
        # 需要将 (T, H, W, C) 转换为 ffmpeg 期望的 rawvideo 格式
        if pix_fmt == "yuv400p":
            # 单通道：直接写入 Y 平面
            raw_data = video_data[..., 0].tobytes()
        elif pix_fmt == "yuv444p":
            # YUV444: 逐帧写入 Y, U, V 平面
            raw_buffer = []
            for t in range(n_frames):
                frame = video_data[t]
                raw_buffer.append(frame[..., 0].tobytes())  # Y
                raw_buffer.append(frame[..., 1].tobytes())  # U
                raw_buffer.append(frame[..., 2].tobytes())  # V
            raw_data = b''.join(raw_buffer)
        elif pix_fmt == "yuv420p":
            # YUV420: Y 平面 + 下采样后的 U, V 平面
            raw_buffer = []
            for t in range(n_frames):
                frame = video_data[t]
                raw_buffer.append(frame[..., 0].tobytes())  # Y
                # U 和 V 下采样
                u_subsampled = frame[::2, ::2, 1].tobytes()
                v_subsampled = frame[::2, ::2, 2].tobytes()
                raw_buffer.append(u_subsampled)
                raw_buffer.append(v_subsampled)
            raw_data = b''.join(raw_buffer)
        else:
            raise ValueError(f"Unsupported pix_fmt: {pix_fmt}")
        
        # 写入数据并获取输出
        stdout_data, stderr_data = process.communicate(input=raw_data)
        
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg pipe encoding failed: {stderr_data.decode('utf-8', errors='ignore')}")
        
        return stdout_data

    def _decode_ffmpeg_pipe(
        self,
        video_data: bytes,
        width: int,
        height: int,
        pix_fmt: str,
        codec: str = "hevc"
    ) -> np.ndarray:
        """
        使用 ffmpeg-python 通过管道解码视频数据。
        
        Args:
            video_data: 编码后的视频字节流
            width: 视频宽度
            height: 视频高度
            pix_fmt: 像素格式 (yuv420p, yuv444p, yuv400p)
            codec: 输入视频编码格式 (hevc, h264, etc.)
        
        Returns:
            解码后的 numpy 数组，形状为 (T, H, W, C)，dtype 为 uint8
        """
        # 处理 yuv400p 的像素格式映射
        output_pix_fmt = "gray" if pix_fmt == "yuv400p" else pix_fmt
        
        # 构建 ffmpeg 解码管道
        process = (
            ffmpeg
            .input('pipe:', format=codec)
            .output('pipe:', format='rawvideo')
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )
        
        # 写入编码数据并获取原始视频输出
        stdout_data, stderr_data = process.communicate(input=video_data)
        
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg pipe decoding failed: {stderr_data.decode('utf-8', errors='ignore')}")
        
        # 解析原始视频数据
        raw_data = np.frombuffer(stdout_data, dtype=np.uint8)
        
        if len(raw_data) == height * width * 1:
            # 单通道
            bytes_per_frame = height * width
            n_frames = len(raw_data) // bytes_per_frame
            video = raw_data.reshape(n_frames, height, width)
            # 扩展为 (T, H, W, 3)
            video = np.stack([video, video, video], axis=-1)
            
        if len(raw_data) == height * width * 3:
            bytes_per_frame = height * width * 3
            n_frames = len(raw_data) // bytes_per_frame
            # YUV444 平面格式：逐帧 Y, U, V
            video = np.zeros((n_frames, height, width, 3), dtype=np.uint8)
            for t in range(n_frames):
                frame_start = t * bytes_per_frame
                y_plane = raw_data[frame_start:frame_start + height * width].reshape(height, width)
                u_plane = raw_data[frame_start + height * width:frame_start + 2 * height * width].reshape(height, width)
                v_plane = raw_data[frame_start + 2 * height * width:frame_start + 3 * height * width].reshape(height, width)
                video[t, ..., 0] = y_plane
                video[t, ..., 1] = u_plane
                video[t, ..., 2] = v_plane
                

        if len(raw_data) == height * width + 2 * (height // 2) * (width // 2):
            y_plane_size = height * width
            uv_plane_size = (height // 2) * (width // 2)
            bytes_per_frame = y_plane_size + 2 * uv_plane_size
            n_frames = len(raw_data) // bytes_per_frame
            video = np.zeros((n_frames, height, width, 3), dtype=np.uint8)
            
            for t in range(n_frames):
                frame_start = t * bytes_per_frame
                # Y 平面
                y_plane = raw_data[frame_start:frame_start + y_plane_size].reshape(height, width)
                # U 平面
                u_start = frame_start + y_plane_size
                u_plane = raw_data[u_start:u_start + uv_plane_size].reshape(height // 2, width // 2)
                # V 平面
                v_start = u_start + uv_plane_size
                v_plane = raw_data[v_start:v_start + uv_plane_size].reshape(height // 2, width // 2)
                
                # 上采样 U 和 V
                u_upsampled = np.repeat(np.repeat(u_plane, 2, axis=0), 2, axis=1)
                v_upsampled = np.repeat(np.repeat(v_plane, 2, axis=0), 2, axis=1)
                
                video[t, ..., 0] = y_plane
                video[t, ..., 1] = u_upsampled
                video[t, ..., 2] = v_upsampled
        
        
        return video