#  Copyright [2025] [Huawei Technologies Co., Ltd.] 
#  Licensed under the Code Sharing Policy of the UHD World Association (the "Policy"); 
#  http://www.theuwa.com/UWA_Code_Sharing_Policy.pdf.
#  you may not use this file except in compliance with the Policy. 
#  Unless agreed to in writing, software distributed under the Policy is distributed on an "AS IS" BASIS, 
#  WITHOUT WARRANTIES OF ANY KIND, either express or implied. 
#  See the Policy for the specific language governing permissions and 
#  limitations under the Policy.

from dataclasses import dataclass, field
import numpy as np
import torch
import struct
import io

class BitStreamWriter:
    """支持批量写入的比特流写入器"""
    
    def __init__(self, stream: io.BytesIO = None):
        self.stream = stream or io.BytesIO()
        self._current_byte = 0
        self._bit_position = 0  # 当前字节中的比特位置 (0-7)
        self._bit_count = 0
        self._reserved_positions = {} 
    def write_bits(self, value: int, num_bits: int):
        """写入指定比特数的值"""
        if num_bits <= 0:
            return
            
        # 确保值在有效范围内
        max_value = (1 << num_bits) - 1
        if value > max_value:
            raise ValueError(f"值 {value} 超出 {num_bits} 比特的表示范围 (0-{max_value})")
        
        value = np.int32(value)
        # 处理剩余比特
        remaining_bits = num_bits
        while remaining_bits:
            # 当前字节剩余可用比特
            available_bits = 8 - self._bit_position
            
            # 计算本次可写入的比特数
            bits_to_write = min(remaining_bits, available_bits)
            
            # 提取要写入的比特段 (从最高位开始)
            shift = remaining_bits - bits_to_write
            mask = (1 << bits_to_write) - 1
            bits = (value >> shift) & mask
            
            # 将比特移动到正确位置并存入当前字节
            self._current_byte |= bits << (available_bits - bits_to_write)
            
            # 更新位置
            self._bit_position += bits_to_write
            self._bit_count += bits_to_write
            remaining_bits -= bits_to_write
            
            # 如果当前字节已满，则刷新到流
            if self._bit_position == 8:
                self._flush_current_byte()
    
    def _flush_current_byte(self):
        """将当前字节写入流并重置状态"""
        if self._bit_position > 0:
            self.stream.write(struct.pack('B', self._current_byte))
            self._current_byte = 0
            self._bit_position = 0
    
    def byte_align(self):
        """执行字节对齐，填充0直到达到字节边界"""
        if self._bit_position > 0:
            self._flush_current_byte()
    
    def write_uint8(self, value: int):
        """写入单个8位无符号整数"""
        self.write_bits(value, 8)
    

    def write_bytes(self, data: bytes):
        """优化版字节写入方法，特别针对长字节串"""
        if not data:
            return
        
        # 字节对齐情况：直接写入
        if self._bit_position == 0:
            self.stream.write(data)
            self._bit_count += len(data) * 8
            return
        
        # 非对齐情况处理
        saved_position = self._bit_position  # 保存当前位位置
        available_bits = 8 - saved_position
        mask = (1 << saved_position) - 1  # 低位掩码
        
        # 处理第一个字节
        first_byte = data[0]
        
        # 合并到当前字节并刷新
        self._current_byte |= first_byte >> saved_position
        self._flush_current_byte()  # 会重置_bit_position为0
        
        # 提取第一个字节剩余的低位
        remaining_bits = first_byte & mask
        
        # 单字节情况处理
        if len(data) == 1:
            self._current_byte = remaining_bits << available_bits
            self._bit_position = saved_position
            self._bit_count += 8
            return
        
        # ===== 多字节优化处理 =====
        # 1. 准备批量转换缓冲区
        buffer = bytearray(len(data) - 1)
        prev_low = remaining_bits
        
        # 2. 批量处理中间字节
        for i in range(1, len(data)):
            current_byte = data[i]
            # 组合前字节低位 + 当前字节高位
            buffer[i-1] = (prev_low << available_bits) | (current_byte >> saved_position)
            # 更新前字节低位为当前字节低位
            prev_low = current_byte & mask
        
        # 3. 写入转换后的字节
        self.stream.write(buffer)
        
        # 4. 设置新的当前字节
        self._current_byte = prev_low << available_bits
        self._bit_position = saved_position
        
        # 5. 更新比特计数 (总字节数*8 - 缓存的比特数)
        self._bit_count += len(data) * 8
    
    def write_uint16(self, value: int):
        """写入16位无符号整数（大端序）"""
        # 如果当前字节对齐，直接写入
        if self._bit_position == 0:
            self.stream.write(struct.pack('>H', value))
            self._bit_count += 16
        else:
            # 否则使用比特写入
            self.write_bits(value >> 8, 8)
            self.write_bits(value & 0xFF, 8)
    
    def write_uint32(self, value: int):
        """写入32位无符号整数（大端序）"""
        # 如果当前字节对齐，直接写入
        if self._bit_position == 0:
            self.stream.write(struct.pack('>I', value))
            self._bit_count += 32
        else:
            # 否则使用比特写入
            self.write_bits(value >> 24, 8)
            self.write_bits((value >> 16) & 0xFF, 8)
            self.write_bits((value >> 8) & 0xFF, 8)
            self.write_bits(value & 0xFF, 8)
    

    def write_float32(self, value: float):
        """写入32位浮点数（IEEE 754标准）"""
        # 将浮点数转换为字节表示
        packed = struct.pack('>f', value)
        # 写入字节
        self.write_bytes(packed)
    
    
    def write_bool(self, value: bool):
        """写入布尔值（1位）"""
        self.write_bits(1 if value else 0, 1)
    
    def write_string(self, s: str, encoding: str = 'utf-8'):
        """写入字符串"""
        encoded = s.encode(encoding)
        # self.write_uint16(len(encoded))
        self.write_bytes(encoded)  # 使用批量写入方法
    
    def get_bytes(self) -> bytes:
        """获取所有写入数据的字节表示"""
        self.byte_align()
        return self.stream.getvalue()
    
    def write_to_file(self, filename: str):
        """将数据写入文件"""
        self.byte_align()
        with open(filename, 'wb') as f:
            f.write(self.get_bytes())
    
    def __enter__(self):
        """支持上下文管理器"""
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """退出上下文时自动对齐字节"""
        self.byte_align()
    
    def get_values(self):
        values = self.stream.getvalue()
        self.stream.close()
        return values
        



class BitStreamReader:
    def __init__(self):
        self.data = None
        self.current_byte_index = 0
        self.current_bit_pos = 0  # 当前字节中的位位置（0-7）
        # self.total_bits = len(data) * 8
        
    def set_data(self, data, current_byte_index = 0):
        self.data = data
        self.current_byte_index = 0
        self.current_bit_pos = 0  # 当前字节中的位位置（0-7）
        self.total_bits = len(data) * 8
    
    
    def _ensure_data_available(self, num_bits):
        """确保有足够的数据可读"""
        available_bits = (len(self.data) - self.current_byte_index) * 8 - self.current_bit_pos
        if num_bits > available_bits:
            raise EOFError(f"Not enough data: requested {num_bits} bits, only {available_bits} available")
    
    def read_bits(self, num_bits):
        """读取指定位数的数据"""
        if num_bits <= 0:
            return 0
            
        self._ensure_data_available(num_bits)
        
        result = 0
        remaining_bits = num_bits
        
        while remaining_bits > 0:
            # 当前字节剩余可读位数
            bits_available_in_byte = 8 - self.current_bit_pos
            
            # 计算本次可读取的位数
            bits_to_read = min(remaining_bits, bits_available_in_byte)
            
            # 从当前字节读取位
            current_byte = self.data[self.current_byte_index]
            
            # 创建掩码提取位
            mask = ((1 << bits_available_in_byte) - 1)
            bits = (current_byte & mask) >> (bits_available_in_byte - bits_to_read)
            
            # 将读取的位添加到结果中
            result = (result << bits_to_read) | bits
            
            # 更新位置
            self.current_bit_pos += bits_to_read
            remaining_bits -= bits_to_read
            
            # 如果当前字节已读完，移动到下一个字节
            if self.current_bit_pos == 8:
                self.current_byte_index += 1
                self.current_bit_pos = 0
        return result
    def byte_align(self):
        """字节对齐（跳过当前字节的剩余位）"""
        if self.current_bit_pos > 0:
            self.current_byte_index += 1
            self.current_bit_pos = 0
    
    def read_uint8(self):
        """读取8位无符号整数"""
        return self.read_bits(8)
    
    def read_float32(self):
        """读取32位浮点数（IEEE 754格式）"""
        # 读取32位整数
        int_val = self.read_bits(32)
        # 转换为字节
        float_bytes = int_val.to_bytes(4, 'big')
        # 使用struct解析为浮点数
        return struct.unpack('>f', float_bytes)[0]
    
    def read_uint32(self):
        """读取32位浮点数（IEEE 754格式）"""
        # 读取32位整数
        int_val = self.read_bits(32)
        return int_val
    
    def read_bytes(self, length):
        """读取指定长度的字节数据"""
        # 如果已经是字节对齐状态，直接读取
        if self.current_bit_pos == 0:
            result = self.data[self.current_byte_index:self.current_byte_index + length]
            self.current_byte_index += length
            return bytes(result)
        
        # 非字节对齐状态：逐字节读取
        result = bytearray()
        for _ in range(length):
            result.append(self.read_bits(8))
        return bytes(result)
    
    def read_string(self, length, encoding='utf-8'):
        """读取指定长度的字符串（使用指定编码）"""
        byte_data = self.read_bytes(length)
        return byte_data.decode(encoding)
    
    def get_remaining_bits(self):
        """获取剩余位数"""
        return (len(self.data) - self.current_byte_index) * 8 - self.current_bit_pos
    
    def get_current_bit_position(self):
        """获取当前位位置（用于调试）"""
        return self.current_byte_index * 8 + self.current_bit_pos
    
    def skip_bits(self, num_bits):
        """跳过指定数量的位"""
        if num_bits <= 0:
            return
            
        self._ensure_data_available(num_bits)
        
        # 计算新位置
        total_bits = self.current_byte_index * 8 + self.current_bit_pos + num_bits
        self.current_byte_index = total_bits // 8
        self.current_bit_pos = total_bits % 8
    
    def peek_bits(self, num_bits):
        """预览指定位数的数据（不移动读取位置）"""
        if num_bits <= 0:
            return 0
            
        # 保存当前状态
        saved_byte_index = self.current_byte_index
        saved_bit_pos = self.current_bit_pos
        
        # 读取数据
        result = self.read_bits(num_bits)
        
        # 恢复状态
        self.current_byte_index = saved_byte_index
        self.current_bit_pos = saved_bit_pos
        
        return result

class reconstruction_information():
    def __init__(self):
        self.attribute_type = 0
        self.component = 0
        self.quantization_type = 0
        self.quantization_bitdepth = 0
        self.prediction_type = 0
        self.byteshift = 0
        self.blocksize = 0
        self.transformation_type = 0
        self.quantization_min_value = []
        self.quantization_max_value = []
        self.patch_num = 0
        self.patch_size = [0]
        self.patch_quantization_min_value = [0]
        self.patch_quantization_max_value = [0]

    def initialize(self):
        if self.quantization_type == 2:
            self.quantization_min_value = np.zeros((self.component), dtype=np.float32)
            self.quantization_max_value = np.zeros((self.component), dtype=np.float32)
        elif self.quantization_type == 3:
            self.patch_size = np.zeros(self.patch_num,dtype = np.int32)
            self.patch_quantization_min_value = np.zeros((self.patch_num, self.component),dtype = np.float32)
            self.patch_quantization_max_value = np.zeros((self.patch_num, self.component),dtype = np.float32)
    
    def write(self, writer):
        writer.write_bits(self.attribute_type,8)
        writer.write_bits(self.component,8)
        writer.write_bits(self.quantization_type,4)
        writer.write_bits(self.quantization_bitdepth,8)
        writer.write_bits(self.prediction_type,4)
        if self.prediction_type == 1:
            writer.write_bits(self.byteshift, 4)
            writer.write_bits(self.blocksize, 16)
        elif self.prediction_type > 1:
            print("The prediction is not supported currently!")
        writer.write_bits(self.transformation_type,4)
        if self.transformation_type > 2:
            print("The prediction is not supported currently!")
        if self.quantization_type == 2:
            for i in range(self.component):
                writer.write_float32(self.quantization_min_value[i])
                writer.write_float32(self.quantization_max_value[i])
        elif self.quantization_type == 3:
            writer.write_uint32(self.patch_num)
            self.patch_quantization_min_value = self.patch_quantization_min_value.reshape(self.patch_num,-1)
            self.patch_quantization_max_value = self.patch_quantization_max_value.reshape(self.patch_num,-1)
            for ind in range(self.patch_num):
                writer.write_uint32(self.patch_size[ind])
                writer.write_bytes(self.patch_quantization_min_value[ind].astype('>f4').tobytes())
                writer.write_bytes(self.patch_quantization_max_value[ind].astype('>f4').tobytes())
        writer.byte_align()

    def read(self, reader):
        self.attribute_type = reader.read_bits(8)
        self.component = reader.read_bits(8)
        self.quantization_type = reader.read_bits(4)
        self.quantization_bitdepth = reader.read_bits(8)
        self.prediction_type = reader.read_bits(4)
        if self.prediction_type == 1:
            self.byteshift = reader.read_bits(4)
            self.blocksize = reader.read_bits(16)
        self.transformation_type = reader.read_bits(4)
        
        if self.quantization_type == 2:
            self.quantization_min_value = np.zeros(self.component)
            self.quantization_max_value = np.zeros(self.component)
            for i in range(self.component):
                self.quantization_min_value[i] = reader.read_float32()
                self.quantization_max_value[i] = reader.read_float32()
        elif self.quantization_type == 3:
            self.patch_num = reader.read_uint32()
            self.initialize()
            for ind in range(self.patch_num):
                self.patch_size[ind] = reader.read_uint32()
                self.patch_quantization_min_value[ind] = np.frombuffer(reader.read_bytes( 4 * self.component) , dtype='>f4').astype(np.float32)
                self.patch_quantization_max_value[ind] = np.frombuffer(reader.read_bytes( 4 * self.component) , dtype='>f4').astype(np.float32)
        reader.byte_align()

class texture_decode_information():
    def __init__(self):
        self.entropy_decode_type = 0
        self.packing_map_texture_codec_id = 0
        
    def write(self, writer):  
        writer.write_uint8(self.entropy_decode_type) 
        writer.write_uint8(self.packing_map_texture_codec_id) 

    def read(self, reader):  
        self.entropy_decode_type = reader.read_uint8() 
        self.packing_map_texture_codec_id = reader.read_uint8() 

class texture_packing_information():
    def __init__(self):
        self.packing_map_width = 0
        self.packing_map_height = 0
        self.region_width = 0
        self.region_height = 0
        self.packing_scaning_type = 0
        self.packing_scaning_block_size = 0
        self.packing_region_count_minus1 = 0
        self.region_top_left_x = []
        self.region_top_left_y = []
        self.texture_channel_num = 3
        self.byteshift = 0
        
    def initialize(self):
        self.region_top_left_x = np.zeros(self.packing_region_count_minus1 + 1,dtype= np.int16)
        self.region_top_left_y = np.zeros(self.packing_region_count_minus1 + 1,dtype= np.int16)
        rows = self.packing_map_height // self.region_height
        cols = self.packing_map_width // self.region_width
        for ind in range(self.packing_region_count_minus1 + 1):
            row = ind // rows
            col = ind % rows
            self.region_top_left_x[ind] = col * self.region_width
            self.region_top_left_y[ind] = row * self.region_height
        
    def write(self, writer):  
        writer.write_uint16(self.packing_map_width) 
        writer.write_uint16(self.packing_map_height) 
        writer.write_uint16(self.region_width) 
        writer.write_uint16(self.region_height) 
        writer.write_bits(self.packing_scaning_type,4) 
        if self.packing_scaning_type == 1:
            writer.write_bits(self.packing_scaning_block_size,8)   
        writer.write_bits(self.packing_region_count_minus1,8) 
        for i in range(self.packing_region_count_minus1+ 1):
            writer.write_uint16(self.region_top_left_x[i])
            writer.write_uint16(self.region_top_left_y[i])
        writer.write_uint8(self.texture_channel_num)
        writer.write_uint8(self.byteshift)
        writer.byte_align()

    def read(self, reader):  
        self.packing_map_width = reader.read_bits(16)
        self.packing_map_height = reader.read_bits(16)
        self.region_width = reader.read_bits(16) 
        self.region_height = reader.read_bits(16)
        self.packing_scaning_type = reader.read_bits(4) 
        if self.packing_scaning_type == 1:
            self.packing_scaning_block_size = reader.read_bits(8)
        self.packing_region_count_minus1 = reader.read_bits(8) 
        self.region_top_left_x = np.zeros(self.packing_region_count_minus1 + 1,dtype= np.int16)
        self.region_top_left_y = np.zeros(self.packing_region_count_minus1 + 1,dtype= np.int16)
        for i in range(self.packing_region_count_minus1 + 1):
            self.region_top_left_x[i] = reader.read_bits(16) 
            self.region_top_left_y[i] = reader.read_bits(16) 
        self.texture_channel_num = reader.read_bits(8)
        self.byteshift = reader.read_bits(8)
        reader.byte_align()    
    
class video_packing_information():
    def __init__(self):
        self.packing_map_width = 0
        self.packing_map_height = 0
        self.region_width = 0
        self.region_height = 0
        self.packing_map_frame_num_minus1 = 0
        self.packing_scaning_type = 0
        self.packing_scaning_block_size = 0
        self.packing_region_count_minus1 = 0
        self.region_frame_index = [0]
        self.region_top_left_x = [0]
        self.region_top_left_y = [0]
        self.attribute_type = [0]
        self.attribute_channel_offset = [0]
        self.attribute_channel_num = [0]
        self.byteshift = [0]
    
    def initialize(self):
        self.region_frame_index = np.zeros((self.packing_region_count_minus1 + 1),dtype= np.int8)
        self.region_top_left_x = np.zeros((self.packing_region_count_minus1 + 1),dtype= np.int16)
        self.region_top_left_y = np.zeros((self.packing_region_count_minus1 + 1),dtype= np.int16)
        self.attribute_type = np.zeros((self.packing_region_count_minus1 + 1),dtype= np.int8)
        self.attribute_channel_offset = np.zeros((self.packing_region_count_minus1 + 1),dtype= np.int8)
        self.attribute_channel_num = np.zeros((self.packing_region_count_minus1 + 1),dtype= np.int8)
        self.byteshift = np.zeros((self.packing_region_count_minus1 + 1),dtype= np.int8)
        # default the frame num is one
        rows = self.packing_map_height // self.region_height
        cols = self.packing_map_width // self.region_width
        for ind in range(self.packing_region_count_minus1 + 1):
            row = ind // rows
            col = ind % rows
            self.region_top_left_x[ind] = col * self.region_width
            self.region_top_left_y[ind] = row * self.region_height

    
    def write(self, writer):  
        writer.write_uint16(self.packing_map_width) 
        writer.write_uint16(self.packing_map_height) 
        writer.write_uint16(self.region_width) 
        writer.write_uint16(self.region_height) 
        writer.write_uint16(self.packing_map_frame_num_minus1) 
        writer.write_bits(self.packing_scaning_type,4) 
        if self.packing_scaning_type == 1:
            writer.write_bits(self.packing_scaning_block_size, 8)
        writer.write_bits(self.packing_region_count_minus1,8) 
        for i in range(self.packing_region_count_minus1 + 1):
            writer.write_uint8(self.region_frame_index[i])
            writer.write_uint16(self.region_top_left_x[i])
            writer.write_uint16(self.region_top_left_y[i]) 
            writer.write_uint8(self.attribute_type[i])
            writer.write_uint8(self.attribute_channel_offset[i])
            writer.write_uint8(self.attribute_channel_num[i])
            writer.write_uint8(self.byteshift[i])
        writer.byte_align()    
    def read(self, reader):  
        self.packing_map_width = reader.read_bits(16) 
        self.packing_map_height = reader.read_bits(16)
        self.region_width = reader.read_bits(16)
        self.region_height = reader.read_bits(16)
        self.packing_map_frame_num_minus1 = reader.read_bits(16)
        self.packing_scaning_type = reader.read_bits(4) 
        if self.packing_scaning_type == 1:
            self.packing_scaning_block_size = reader.read_bits(8)
        self.packing_region_count_minus1 = reader.read_bits(8)
        
        self.region_frame_index = np.zeros((self.packing_region_count_minus1 + 1),dtype= np.int8)
        self.region_top_left_x = np.zeros((self.packing_region_count_minus1 + 1),dtype= np.int16)
        self.region_top_left_y = np.zeros((self.packing_region_count_minus1 + 1),dtype= np.int16)
        self.attribute_type = np.zeros((self.packing_region_count_minus1 + 1),dtype= np.int8)
        self.attribute_channel_offset = np.zeros((self.packing_region_count_minus1 + 1),dtype= np.int8)
        self.attribute_channel_num = np.zeros((self.packing_region_count_minus1 + 1),dtype= np.int8)
        self.byteshift = np.zeros((self.packing_region_count_minus1 + 1),dtype= np.int8)
        for i in range(self.packing_region_count_minus1 + 1):
            self.region_frame_index[i] = reader.read_bits(8) 
            self.region_top_left_x[i] = reader.read_bits(16) 
            self.region_top_left_y[i] = reader.read_bits(16) 
            self.attribute_type[i] = reader.read_uint8()
            self.attribute_channel_offset[i] = reader.read_uint8()
            self.attribute_channel_num[i] = reader.read_uint8()
            self.byteshift[i] = reader.read_uint8()
        reader.byte_align()



class video_decode_information():
    def __init__(self):
        self.packing_map_video_codec_id = 0
    def write(self, writer):  
        writer.write_uint8(self.packing_map_video_codec_id)
    def read(self, reader):
        self.packing_map_video_codec_id = reader.read_uint8()

class gsbs_meta():
    @dataclass
    class entropy_meta:
        entropy_decode_type: int = 0
        attribute_type: int = 0
        bitdepth: int = 0
        byteshift: int = 0

        def write(self, writer):
            writer.write_uint8(self.entropy_decode_type)
            writer.write_uint8(self.attribute_type)
            writer.write_bits(self.bitdepth,8)
            writer.write_bits(self.byteshift,8)
        
        def read(self, reader):
            self.entropy_decode_type = reader.read_uint8()
            self.attribute_type = reader.read_uint8()
            self.bitdepth = reader.read_bits(8)
            self.byteshift = reader.read_bits(8)
    
    @dataclass
    class texture_meta:
        texture_decode_information: texture_decode_information = field(default_factory=lambda: texture_decode_information())
        texture_packing_information: texture_packing_information = field(default_factory=lambda: texture_packing_information())
        attribute_type: int = 0

        def write(self, writer):
            self.texture_decode_information.write(writer)
            self.texture_packing_information.write(writer)
            writer.write_uint8(self.attribute_type)
        
        def read(self, reader):
            self.texture_decode_information.read(reader)
            self.texture_packing_information.read(reader)
            self.attribute_type = reader.read_uint8()
    
    @dataclass
    class video_meta:
        video_decode_information: video_decode_information = field(default_factory=lambda: video_decode_information())
        video_packing_information: video_packing_information = field(default_factory=lambda: video_packing_information())

        def write(self, writer):
            self.video_decode_information.write(writer)
            self.video_packing_information.write(writer)
        
        def read(self, reader):
            self.video_decode_information.read(reader)
            self.video_packing_information.read(reader)

    def __init__(self):
        self.profile_idc = 1
        self.gs_points_num = 0
        self.sub_bitstream_num = 0
        self.SH_degree = 0
        self.gs_subset_num = 0
        self.sub_gs_points_num = 0
        
        self.sub_bitstream_size = [0]
        self.gs_subset_id = [0]
        self.sub_bitstream_decode_type = [0]
        self.sub_bitstream_meta = []

        self.position_min_value=[0,0,0]
        self.position_max_value=[1,1,1]
        self.reconstruction_count = [0]
        self.reconstruction_information = [[reconstruction_information()]]
       
        
    def initialize2(self, gs_points_num, sub_bitstream_num, SH_degree, gs_subset_num, maxs, mins):
        self.gs_points_num = gs_points_num
        self.sub_bitstream_num = sub_bitstream_num
        self.SH_degree = SH_degree
        self.gs_subset_num = gs_subset_num

        self.sub_gs_points_num = [0 for i in range(self.gs_subset_num)]
        self.sub_bitstream_size = [0 for i in range(self.sub_bitstream_num)]
        self.gs_subset_id = [0 for i in range(self.sub_bitstream_num)]
        self.sub_bitstream_decode_type = [0 for i in range(self.sub_bitstream_num)]
        self.position_min_value = mins.tolist()
        self.position_max_value = maxs.tolist()

        self.reconstruction_count = [0 for i in range(self.gs_subset_num)]
    
    def initialize(self):
        self.sub_gs_points_num = [0 for i in range(self.gs_subset_num)]
        self.sub_bitstream_size = [0 for i in range(self.sub_bitstream_num)]
        self.gs_subset_id = [0 for i in range(self.sub_bitstream_num)]
        self.sub_bitstream_decode_type = [0 for i in range(self.sub_bitstream_num)]
        self.reconstruction_count = [0 for i in range(self.gs_subset_num)]
        
    def write(self, writer):    
        writer.write_uint8(self.profile_idc)
        writer.write_uint32(self.gs_points_num)
        writer.write_bits(self.sub_bitstream_num,5)
        writer.write_bits(self.SH_degree,3)
        for i in range(3):
            writer.write_float32(self.position_min_value[i])
            writer.write_float32(self.position_max_value[i])
        writer.write_uint8(self.gs_subset_num)
        for i in range(self.gs_subset_num):
            writer.write_uint32(self.sub_gs_points_num[i])
        for i in range(self.sub_bitstream_num):
            writer.write_uint32(self.sub_bitstream_size[i])
            writer.write_uint8(self.gs_subset_id[i])
            writer.write_uint8(self.sub_bitstream_decode_type[i])
            if(self.sub_bitstream_decode_type[i] == 0):
                self.sub_bitstream_meta[i].write(writer)
            elif(self.sub_bitstream_decode_type[i] == 1):
                self.sub_bitstream_meta[i].write(writer)
            elif(self.sub_bitstream_decode_type[i] == 2):
                self.sub_bitstream_meta[i].write(writer)
        for i in range(self.gs_subset_num):
            writer.write_bits(self.reconstruction_count[i],8)
            for j in range(self.reconstruction_count[i]):
                self.reconstruction_information[i][j].write(writer)
        writer.byte_align() # byte_alignment()        

    def read(self, reader):    
        self.profile_idc = reader.read_uint8()
        self.gs_points_num = reader.read_uint32()
        self.sub_bitstream_num = reader.read_bits(5)
        self.SH_degree = reader.read_bits(3)
        self.position_min_value = []
        self.position_max_value = []
        for i in range(3):
            self.position_min_value.append(reader.read_float32())
            self.position_max_value.append(reader.read_float32())
        self.gs_subset_num = reader.read_bits(8)
        self.initialize()
        for i in range(self.gs_subset_num):
            self.sub_gs_points_num[i] = reader.read_uint32()
        for i in range(self.sub_bitstream_num):
            self.sub_bitstream_size[i] = reader.read_uint32()
            self.gs_subset_id[i] = reader.read_bits(8)
            self.sub_bitstream_decode_type[i] = reader.read_uint8()
            if(self.sub_bitstream_decode_type[i] == 0):
                self.sub_bitstream_meta.append(self.entropy_meta())
                self.sub_bitstream_meta[i].read(reader)
            elif(self.sub_bitstream_decode_type[i] == 1):
                self.sub_bitstream_meta.append(self.texture_meta())
                self.sub_bitstream_meta[i].read(reader)
            elif(self.sub_bitstream_decode_type[i] == 2):
                self.sub_bitstream_meta.append(self.video_meta())
                self.sub_bitstream_meta[i].read(reader)
        for i in range(self.gs_subset_num):
            self.reconstruction_count[i] = reader.read_uint8()
            self.reconstruction_information[i] = []
            for j in range(self.reconstruction_count[i]):
                self.reconstruction_information[i].append(reconstruction_information())
                self.reconstruction_information[i][j].read(reader)
        reader.byte_align() # byte_alignment() 
            
                     
class gsbs_sub_bitstreams():
    def __init__(self):
        self.gstc_sub_bitstream_data = []
        self.sub_bitstream_num = 0
        self.sub_bitstream_size = []
        
    def initialize(self,sub_bitstream_num):
        self.sub_bitstream_num = sub_bitstream_num
        self.gstc_sub_bitstream_data = [[] for i in range(self.sub_bitstream_num)]
        
        
    def write(self, writer):
       for i in range(self.sub_bitstream_num):
          writer.write_bytes(self.gstc_sub_bitstream_data[i])

    def read(self, reader):
       self.gstc_sub_bitstream_data = [[] for i in range(self.sub_bitstream_num)]
       for i in range(self.sub_bitstream_num):
          self.gstc_sub_bitstream_data[i] = reader.read_bytes(self.sub_bitstream_size[i])

class unit_header():
    def __init__(self, unit_type):
        self.unit_type = unit_type
        self.reserved = 0
        
    def write(self, writer):
        writer.write_bits(self.unit_type,4)
        writer.write_bits(self.reserved,28)
        
    def read(self, reader):
        self.unit_type = reader.read_bits(4)
        self.reserved = reader.read_bits(28)
        
class unit_payload():
    def __init__(self, unit_type):
        self.unit_type = unit_type
        self.gsbs_metadata = gsbs_meta() if unit_type == 0 else None
        self.gsbs_sub_bitstreams = gsbs_sub_bitstreams() if unit_type == 1 else None
        
    def write(self, writer):
        if(self.unit_type == 0):
            self.gsbs_metadata.write(writer)
        elif(self.unit_type == 1):
            self.gsbs_sub_bitstreams.write(writer)
            
    def read(self, reader):
        if(self.unit_type == 0):
            self.gsbs_metadata.read(reader)
        elif(self.unit_type == 1):
            self.gsbs_sub_bitstreams.read(reader)
    
class unit():
    def __init__(self, unit_type):
        self.unit_header = unit_header(unit_type)
        self.unit_payload = unit_payload(unit_type)
        self.writer = BitStreamWriter()
        self.reader = BitStreamReader()
        
    def write(self):
        self.unit_header.write(self.writer)
        self.unit_payload.write(self.writer)

    def read(self):
        self.unit_header.read(self.reader)
        self.unit_payload.read(self.reader)

codecs_dict_str_to_num = {
    'EntropyCodec':0,
    'AstcCodec': 1,
    'AstcCodecPy': 1,
    'VideoCodec':2
    }

codecs_dict_num_to_str = { v: k for k, v in codecs_dict_str_to_num.items()}

attribute_dict_str_to_num = {
    'POSITION':0,
    'OPACITY': 1,
    'SCALE':2,
    'ROTATION':3,
    'SPHERICAL_HARMONICS_DEGREE_0_COEFFICIENT_0': 4,
    'SPHERICAL_HARMONICS_DEGREE_1_COEFFICIENT_0': 5,
    'SPHERICAL_HARMONICS_DEGREE_1_COEFFICIENT_1': 6,
    'SPHERICAL_HARMONICS_DEGREE_1_COEFFICIENT_2': 7,
    'SPHERICAL_HARMONICS_DEGREE_2_COEFFICIENT_0': 8,
    'SPHERICAL_HARMONICS_DEGREE_2_COEFFICIENT_1': 9,
    'SPHERICAL_HARMONICS_DEGREE_2_COEFFICIENT_2': 10,
    'SPHERICAL_HARMONICS_DEGREE_2_COEFFICIENT_3': 11,
    'SPHERICAL_HARMONICS_DEGREE_2_COEFFICIENT_4': 12,
    'SPHERICAL_HARMONICS_DEGREE_3_COEFFICIENT_0': 13,
    'SPHERICAL_HARMONICS_DEGREE_3_COEFFICIENT_1': 14,
    'SPHERICAL_HARMONICS_DEGREE_3_COEFFICIENT_2': 15,
    'SPHERICAL_HARMONICS_DEGREE_3_COEFFICIENT_3': 16,
    'SPHERICAL_HARMONICS_DEGREE_3_COEFFICIENT_4': 17,
    'SPHERICAL_HARMONICS_DEGREE_3_COEFFICIENT_5': 18,
    'SPHERICAL_HARMONICS_DEGREE_3_COEFFICIENT_6': 19,
    'SPHERICAL_HARMONICS_DEGREE_1_AND_HIGHER': 20,
    'importance':21
    }
splat_attribute_to_attribute_type = {
    "means": "POSITION",
    "scaling": "SCALE",
    "rotation": "ROTATION",
    "opacity": "OPACITY",
    "features_dc": "SPHERICAL_HARMONICS_DEGREE_0_COEFFICIENT_0",
    "features_rest": "SPHERICAL_HARMONICS_DEGREE_1_AND_HIGHER",
    "importance": "importance",
}
attribute_type_to_splat_attribute = { v: k for k, v in splat_attribute_to_attribute_type.items()}
    
attribute_dict_num_to_str = { v: k for k, v in attribute_dict_str_to_num.items()}


attribute_component_dict_str_to_num = {
    'means':3,
    'opacity': 1,
    'scaling':3,
    'rotation':4,
    'features_dc':3,
    'importance':1,
    }
attribute_component_dict_num_to_str = { v: k for k, v in attribute_component_dict_str_to_num.items()}

transform_type_dict_str_to_num = {
    'None':0,
    'reduction':1,
    'imp':2,
}
transform_type_dict_num_to_str = { v: k for k, v in transform_type_dict_str_to_num.items()}

prediction_type_dict_str_to_num = {
    'None':0,
    'minor':1,
    'minor2':2,
}
prediction_type_dict_num_to_str = { v: k for k, v in prediction_type_dict_str_to_num.items()}


quantization_dict_str_to_num = {
    'NoQuantizer': 0,
    'Quantizer': 1,
    'MinmaxQuantizer':2,
    'GroupMinmaxQuantizer':3,
    'AdaptiveGroupMinmaxQuantizer':3,
    }
quantization_dict_num_to_str = {
    0: 'NoQuantizer',
    1: 'Quantizer',
    2: 'MinmaxQuantizer',
    3: 'AdaptiveGroupMinmaxQuantizer',
    }

video_codec_str_to_num = {
    'x264':1,
    'x265':2,
    'hm':2,
    'vtm':3,
}
video_codec_num_to_str = { 
    1:"x264",
    2:"x265",
    3:"vtm",
}

entropy_dict_str_to_num = {
    'NoEntropy':0,
    'zstd': 1,
    }
entropy_dict_num_to_str = { v: k for k, v in entropy_dict_str_to_num.items()}

scaning_dict_str_to_num = {
    'rowsfirst': 0,
    'block': 1,
    'dualmorton': 2,
}
scaning_dict_num_to_str = { v: k for k, v in scaning_dict_str_to_num.items()}

class UWABitstreamPacker:
    def encode_streamer(self, encode_stream):

        gs_points_num = sum(encode_stream[key]["att_meta"]["num_points"] for key in encode_stream.keys())
        means_maxs = torch.cat([torch.tensor(encode_stream[key]["att_meta"]["means_max"]) for key in encode_stream.keys()], dim=0).max(dim=0)[0]
        means_mins = torch.cat([torch.tensor(encode_stream[key]["att_meta"]["means_min"]) for key in encode_stream.keys()], dim=0).min(dim=0)[0]
        unit_metadata = unit(unit_type=0)
        unit_substream = unit(unit_type=1)

        #元数据赋值
        unit_metadata.unit_payload.gsbs_metadata.initialize2(
            gs_points_num=gs_points_num,
            sub_bitstream_num=sum([len(sub_set_stream["streams"]) for (key, sub_set_stream) in encode_stream.items()]),
            SH_degree=[sub_set_stream["att_meta"]["sh_degree"] for (key, sub_set_stream) in encode_stream.items()][0],
            gs_subset_num=len(encode_stream),
            maxs=means_maxs,
            mins=means_mins
        )

        # 遍历 subset
        sub_bitstream_offset = 0
        gsbs_metadata = unit_metadata.unit_payload.gsbs_metadata
        for ind, (key, sub_set_stream) in enumerate(encode_stream.items()):
            # parse the header!
            # subset info
            gsbs_metadata.sub_gs_points_num[ind] = sub_set_stream["att_meta"]["num_points"]
            gsbs_metadata.reconstruction_count[ind] = len(sub_set_stream["pipeline"]["quantizers"])
            gsbs_metadata.reconstruction_information[ind] = [reconstruction_information() for i in range(gsbs_metadata.reconstruction_count[ind])]

            # 遍历该 subset 下的 subbitstream
            for jnd, sub_bs_key in enumerate(sub_set_stream['stream_meta'].keys()):
                # sub_bit_stream information
                sbgs_id = sub_bitstream_offset + jnd
                gsbs_metadata.sub_bitstream_size[sbgs_id] = len(sub_set_stream["streams"][sub_bs_key][0])
                gsbs_metadata.sub_bitstream_decode_type[sbgs_id] = codecs_dict_str_to_num[sub_set_stream["pipeline"]["codecs"][sub_bs_key]["method"]]
                gsbs_metadata.gs_subset_id[sbgs_id] = ind
                decode_type = gsbs_metadata.sub_bitstream_decode_type[sbgs_id]

                if decode_type == 0:
                    gsbs_metadata.sub_bitstream_meta.append(
                        gsbs_meta.entropy_meta(
                            1 if sub_set_stream["pipeline"]["codecs"][sub_bs_key]["params"]["entropy"] else 0,
                            attribute_dict_str_to_num[splat_attribute_to_attribute_type[sub_set_stream["pipeline"]["attribute_packer"]["params"]["packing_map"][sub_bs_key]["attribute_type"]]],
                            sub_set_stream["stream_meta"][sub_bs_key]["bit_depth"],
                            max(0,sub_set_stream["pipeline"]["attribute_packer"]["params"]["packing_map"][sub_bs_key]["byteshift"])
                        )
                    )
                elif decode_type == 1:
                    texture_meta = gsbs_meta.texture_meta()
                    texture_meta.attribute_type = attribute_dict_str_to_num[splat_attribute_to_attribute_type[sub_set_stream["pipeline"]["attribute_packer"]["params"]["packing_map"][sub_bs_key]["attribute_type"]]]
                    texture_meta.texture_decode_information.entropy_decode_type = 1 if sub_set_stream["pipeline"]["codecs"][sub_bs_key]["params"]["entropy"] else 0 # zstd
                    texture_meta.texture_decode_information.packing_map_texture_codec_id = 1 # astc
                    texture_meta.texture_packing_information.packing_map_height = sub_set_stream["stream_meta"][sub_bs_key]["shapes"][0]
                    texture_meta.texture_packing_information.packing_map_width = sub_set_stream["stream_meta"][sub_bs_key]["shapes"][1]
                    texture_meta.texture_packing_information.region_height = sub_set_stream["stream_meta"][sub_bs_key]["shapes"][0] // sub_set_stream["pipeline"]["attribute_packer"]["params"]["packing_map"][sub_bs_key]["grid_shape"][0]
                    texture_meta.texture_packing_information.region_width = sub_set_stream["stream_meta"][sub_bs_key]["shapes"][1] // sub_set_stream["pipeline"]["attribute_packer"]["params"]["packing_map"][sub_bs_key]["grid_shape"][1]
                    texture_meta.texture_packing_information.packing_scaning_type = scaning_dict_str_to_num[sub_set_stream["pipeline"]["attribute_packer"]["params"]["scaning"]]
                    if texture_meta.texture_packing_information.packing_scaning_type == 1:
                        texture_meta.texture_packing_information.packing_scaning_block_size = sub_set_stream["pipeline"]["attribute_packer"]["params"]["block_size"]
                    texture_meta.texture_packing_information.packing_region_count_minus1 = len(sub_set_stream["pipeline"]["attribute_packer"]["params"]["packing_map"][sub_bs_key]["regions"]) - 1
                    texture_meta.texture_packing_information.initialize()
                    for knd,region in enumerate(sub_set_stream["pipeline"]["attribute_packer"]["params"]["packing_map"][sub_bs_key]["regions"]):
                        att_type, f_index, c_offset, c_num, byteshift, col, row = region
                        texture_meta.texture_packing_information.region_top_left_x[knd] = col * texture_meta.texture_packing_information.region_width
                        texture_meta.texture_packing_information.region_top_left_y[knd] = row * texture_meta.texture_packing_information.region_height
                    texture_meta.texture_packing_information.texture_channel_num = sub_set_stream["pipeline"]["attribute_packer"]["params"]["packing_map"][sub_bs_key]["grid_shape"][-1]
                    texture_meta.texture_packing_information.byteshift = max(0,byteshift)
                    gsbs_metadata.sub_bitstream_meta.append(
                        texture_meta
                    )
                elif decode_type == 2:
                    video_meta = gsbs_meta.video_meta()
                    video_meta.video_decode_information.packing_map_video_codec_id = video_codec_str_to_num[sub_set_stream["pipeline"]["codecs"][sub_bs_key]["params"]["codec"]]
                    video_meta.video_packing_information.packing_map_frame_num_minus1 = sub_set_stream["pipeline"]["attribute_packer"]["params"]["packing_map"][sub_bs_key]["grid_shape"][2] - 1
                    video_meta.video_packing_information.packing_map_height = sub_set_stream["stream_meta"][sub_bs_key]["shapes"][0]
                    video_meta.video_packing_information.packing_map_width = sub_set_stream["stream_meta"][sub_bs_key]["shapes"][1]
                    video_meta.video_packing_information.region_height = sub_set_stream["stream_meta"][sub_bs_key]["shapes"][0] // sub_set_stream["pipeline"]["attribute_packer"]["params"]["packing_map"][sub_bs_key]["grid_shape"][0]
                    video_meta.video_packing_information.region_width = sub_set_stream["stream_meta"][sub_bs_key]["shapes"][1] // sub_set_stream["pipeline"]["attribute_packer"]["params"]["packing_map"][sub_bs_key]["grid_shape"][1]
                    video_meta.video_packing_information.packing_scaning_type = scaning_dict_str_to_num[sub_set_stream["pipeline"]["attribute_packer"]["params"]["scaning"]]
                    if video_meta.video_packing_information.packing_scaning_type == 1:
                        video_meta.video_packing_information.packing_scaning_block_size = sub_set_stream["pipeline"]["attribute_packer"]["params"]["block_size"]
                    video_meta.video_packing_information.packing_region_count_minus1 = len(sub_set_stream["pipeline"]["attribute_packer"]["params"]["packing_map"][sub_bs_key]["regions"]) - 1
                    video_meta.video_packing_information.initialize()
                    for knd,region in enumerate(sub_set_stream["pipeline"]["attribute_packer"]["params"]["packing_map"][sub_bs_key]["regions"]):
                        att_type, f_index, c_offset, c_num, byteshift, col, row = region
                        video_meta.video_packing_information.region_frame_index[knd] = f_index
                        video_meta.video_packing_information.region_top_left_x[knd] = col * video_meta.video_packing_information.region_width
                        video_meta.video_packing_information.region_top_left_y[knd] = row * video_meta.video_packing_information.region_height
                        video_meta.video_packing_information.attribute_type[knd] = attribute_dict_str_to_num[splat_attribute_to_attribute_type[att_type]]
                        video_meta.video_packing_information.attribute_channel_offset[knd] = c_offset
                        video_meta.video_packing_information.attribute_channel_num[knd] = c_num
                        video_meta.video_packing_information.byteshift[knd] = byteshift
                    gsbs_metadata.sub_bitstream_meta.append(
                        video_meta
                    )
                else:
                    print("The decode type is not supported currently!")

            # 遍历该 subset 下的 reconstruction info
            for jnd, attribute in enumerate(sub_set_stream["pipeline"]["quantizers"].keys()):
                re_inf = gsbs_metadata.reconstruction_information[ind][jnd]
                re_inf.attribute_type = attribute_dict_str_to_num[splat_attribute_to_attribute_type[attribute]]
                re_inf.component = int(attribute_component_dict_str_to_num[attribute] if attribute != "features_rest" else 3 * ((gsbs_metadata.SH_degree+1)** 2 - 1))
                re_inf.quantization_type = quantization_dict_str_to_num[sub_set_stream["pipeline"]["quantizers"][attribute]["method"]]
                re_inf.quantization_bitdepth = sub_set_stream["pipeline"]["quantizers"][attribute]["params"]["bits"]
                re_inf.prediction_type = prediction_type_dict_str_to_num[sub_set_stream["pipeline"]["prediction"]["params"]["prediction_map"][attribute]]
                if re_inf.prediction_type == 1:
                    re_inf.byteshift = sub_set_stream["att_meta"]["prediction_meta"][attribute]["byteshift"]
                    re_inf.blocksize = sub_set_stream["att_meta"]["prediction_meta"][attribute]["block_size"]
                re_inf.transformation_type = transform_type_dict_str_to_num[sub_set_stream["pipeline"]["transform"]["params"]["transform_map"][attribute]]
                if re_inf.quantization_type == 2:
                    re_inf.initialize()
                    for knd in range(re_inf.component):
                        re_inf.quantization_min_value[knd] = sub_set_stream["att_meta"][attribute]["quantmeta"]["min_vals"][knd]
                        re_inf.quantization_max_value[knd] = sub_set_stream["att_meta"][attribute]["quantmeta"]["max_vals"][knd]
                elif re_inf.quantization_type == 3:
                    re_inf.initialize()
                    re_inf.patch_num = len(sub_set_stream["att_meta"][attribute]["quantmeta"]["group_size"])
                    re_inf.patch_size = np.array(sub_set_stream["att_meta"][attribute]["quantmeta"]["group_size"],dtype = np.int32)
                    re_inf.patch_quantization_min_value = np.array(sub_set_stream["att_meta"][attribute]["quantmeta"]["min_vals"],dtype = np.float32)
                    re_inf.patch_quantization_max_value = np.array(sub_set_stream["att_meta"][attribute]["quantmeta"]["max_vals"],dtype = np.float32)
                
            unit_metadata.write()
            unit_substream.unit_payload.gsbs_sub_bitstreams.initialize(unit_metadata.unit_payload.gsbs_metadata.sub_bitstream_num)
            for jnd,sub_bs_key in enumerate(sub_set_stream['stream_meta'].keys()):
                sbgs_id = sub_bitstream_offset + jnd
                unit_substream.unit_payload.gsbs_sub_bitstreams.gstc_sub_bitstream_data[sbgs_id] = sub_set_stream["streams"][sub_bs_key][0]
            
            sub_bitstream_offset += len(sub_set_stream['stream_meta'].keys())

        unit_substream.write()
        buffer = io.BytesIO()
        unit_metadata_stream = unit_metadata.writer.get_values()
        unit_substream_stream = unit_substream.writer.get_values()
        buffer.write(struct.pack(">I",len(unit_metadata_stream)))
        buffer.write(unit_metadata_stream)
        buffer.write(struct.pack(">I",len(unit_substream_stream)))
        buffer.write(unit_substream_stream)
        stream = buffer.getvalue()
        buffer.close()
        return stream
        
    def decode_streamer(self, stream):
        sub_set_id = 0
        encode_stream = {}
        
        # parsing streams
        meta_byte_size = struct.unpack(">I", stream[:4])[0]
        unit_metadata = unit(0)
        unit_stream_data = unit(1)
        
        unit_metadata.reader.set_data(stream[4:])
        unit_metadata.read()
        assert unit_metadata.reader.current_byte_index == meta_byte_size

        sub_bitstream_byte_size = struct.unpack(">I", stream[4+meta_byte_size:8+meta_byte_size])[0]
        unit_stream_data.reader.set_data(stream[8+meta_byte_size:])
        unit_stream_data.unit_payload.gsbs_sub_bitstreams.sub_bitstream_num = unit_metadata.unit_payload.gsbs_metadata.sub_bitstream_num
        for i in range(unit_metadata.unit_payload.gsbs_metadata.sub_bitstream_num):
            unit_stream_data.unit_payload.gsbs_sub_bitstreams.sub_bitstream_size.append(unit_metadata.unit_payload.gsbs_metadata.sub_bitstream_size[i])
        unit_stream_data.read()
        assert unit_stream_data.reader.current_byte_index == sub_bitstream_byte_size

        for ind in range(unit_metadata.unit_payload.gsbs_metadata.gs_subset_num):
            sub_set_id = ind
            encode_stream[f"sub_set_{sub_set_id}"] = {}
            sub_set_stream = encode_stream[f"sub_set_{sub_set_id}"]
            # convert metadata into pipelines
            sub_set_stream["pipeline"] = {}
            sub_set_stream["att_meta"] = {
                "num_points": unit_metadata.unit_payload.gsbs_metadata.sub_gs_points_num[ind],
                "sh_degree": unit_metadata.unit_payload.gsbs_metadata.SH_degree
            }
            sub_set_stream["stream_meta"] = {}
            sub_set_stream["streams"] = {}

            # set pipeline
            sub_set_stream["pipeline"]["transform"] = {
                "method": "AttributeTransform",
                "params": {
                    "transform_map": {
                        "global": "rsnorm",
                        "post_global": "None",
                    }
                }
            }
            sub_set_stream["pipeline"]["prediction"] = {
                "method": "AttributePridiction",
                "params": {
                    "prediction_map": {
                    }
                }
            }
            sub_set_stream["pipeline"]["quantizers"] = {}
            for re_inf in unit_metadata.unit_payload.gsbs_metadata.reconstruction_information[ind]:
                attribute_type = attribute_type_to_splat_attribute[attribute_dict_num_to_str[re_inf.attribute_type]]
                sub_set_stream["pipeline"]["transform"]["params"]["transform_map"][attribute_type] = transform_type_dict_num_to_str[re_inf.transformation_type]
                sub_set_stream["pipeline"]["quantizers"][attribute_type] = {}
                sub_set_stream["pipeline"]["quantizers"][attribute_type]["methods"] = quantization_dict_num_to_str[re_inf.quantization_type]
                sub_set_stream["pipeline"]["prediction"]["params"]["prediction_map"][attribute_type] = prediction_type_dict_num_to_str[re_inf.prediction_type]
                
        
            sub_set_stream["pipeline"]["attribute_packer"] = {
                "method": "OneSizePacker",
                "params": {
                    "packing_map": {}
                },
            }
        sub_set_stream["pipeline"]["codecs"] = {}
        
        # set sub_bitstreams
        for ind in range(unit_metadata.unit_payload.gsbs_metadata.sub_bitstream_num):
            sub_set_id = unit_metadata.unit_payload.gsbs_metadata.gs_subset_id[ind]
            sub_meta = unit_metadata.unit_payload.gsbs_metadata.sub_bitstream_meta[ind]
            sub_set_stream = encode_stream[f"sub_set_{sub_set_id}"]
            if isinstance(sub_meta, gsbs_meta.entropy_meta):
                key = f"stream_{ind}"
                attribute_type = attribute_type_to_splat_attribute[attribute_dict_num_to_str[sub_meta.attribute_type]]
                sub_set_stream["pipeline"]["attribute_packer"]["params"]["packing_map"][key] = {
                    "attribute_type": attribute_type,
                    "byteshift": sub_meta.byteshift
                }
                sub_set_stream["pipeline"]["codecs"][key] = {}
                sub_set_stream["pipeline"]["codecs"][key]["params"] = {}
                sub_set_stream["pipeline"]["codecs"][key]["method"] = "EntropyCodec"
                sub_set_stream["pipeline"]["codecs"][key]["params"]["entropy"] = True if sub_meta.entropy_decode_type == 1 else False
                sub_set_stream["stream_meta"][key] = {}
                sub_set_stream["stream_meta"][key]["bit_depth"] = sub_meta.bitdepth
                sub_set_stream["stream_meta"][key]["shapes"] = [unit_metadata.unit_payload.gsbs_metadata.sub_gs_points_num[sub_set_id],-1]
                sub_set_stream["streams"][key] = [unit_stream_data.unit_payload.gsbs_sub_bitstreams.gstc_sub_bitstream_data[ind]]
            elif isinstance(sub_meta, gsbs_meta.texture_meta):
                key = f"stream_{ind}"
                attribute_type = attribute_type_to_splat_attribute[attribute_dict_num_to_str[sub_meta.attribute_type]]
                sub_set_stream["pipeline"]["attribute_packer"]["params"]["packing_map"][key] = {
                    "attribute_type": attribute_type,
                    "regions":
                        [[attribute_type, 0, sub_meta.texture_packing_information.texture_channel_num * x, sub_meta.texture_packing_information.texture_channel_num, sub_meta.texture_packing_information.byteshift, u//sub_meta.texture_packing_information.region_width, v//sub_meta.texture_packing_information.region_height] for x,u,v in zip(range(sub_meta.texture_packing_information.packing_region_count_minus1+1), sub_meta.texture_packing_information.region_top_left_x, sub_meta.texture_packing_information.region_top_left_y)],
                    "grid_shape":[sub_meta.texture_packing_information.packing_map_height//sub_meta.texture_packing_information.region_height,0,0,0]
                }
                sub_set_stream["pipeline"]["attribute_packer"]["params"]["scaning"] = scaning_dict_num_to_str[sub_meta.texture_packing_information.packing_scaning_type]
                if sub_meta.texture_packing_information.packing_scaning_type == 1:
                    sub_set_stream["pipeline"]["attribute_packer"]["params"]["block_size"] = sub_meta.texture_packing_information.packing_scaning_block_size

                sub_set_stream["pipeline"]["codecs"][key] = {}
                sub_set_stream["pipeline"]["codecs"][key]["params"] = {}
                sub_set_stream["pipeline"]["codecs"][key]["method"] = "AstcCodecPy"
                sub_set_stream["pipeline"]["codecs"][key]["params"]["entropy"] = True if sub_meta.texture_decode_information.entropy_decode_type == 1 else False
                sub_set_stream["stream_meta"][key] = {}
                sub_set_stream["stream_meta"][key]["bit_depth"] = 8
                sub_set_stream["stream_meta"][key]["shapes"] = [sub_meta.texture_packing_information.packing_map_height, sub_meta.texture_packing_information.packing_map_width,1,-1]
                sub_set_stream["streams"][key] = [unit_stream_data.unit_payload.gsbs_sub_bitstreams.gstc_sub_bitstream_data[ind]]

            elif isinstance(sub_meta, gsbs_meta.video_meta):
                key = f"stream_{ind}"
                sub_set_stream["pipeline"]["attribute_packer"]["params"]["packing_map"][key] = {
                    "regions":
                        [[ attribute_type_to_splat_attribute[attribute_dict_num_to_str[type]],f, x, n, b, u//sub_meta.video_packing_information.region_width, v//sub_meta.video_packing_information.region_height] for type,f,x,n,b,u,v in zip(sub_meta.video_packing_information.attribute_type,sub_meta.video_packing_information.region_frame_index, sub_meta.video_packing_information.attribute_channel_offset, sub_meta.video_packing_information.attribute_channel_num,sub_meta.video_packing_information.byteshift,sub_meta.video_packing_information.region_top_left_x, sub_meta.video_packing_information.region_top_left_y)],
                    "grid_shape":[sub_meta.video_packing_information.packing_map_height//sub_meta.video_packing_information.region_height,0,0,0]
                }
                sub_set_stream["pipeline"]["attribute_packer"]["params"]["scaning"] = scaning_dict_num_to_str[sub_meta.video_packing_information.packing_scaning_type]
                if sub_meta.video_packing_information.packing_scaning_type == 1:
                    sub_set_stream["pipeline"]["attribute_packer"]["params"]["block_size"] = sub_meta.video_packing_information.packing_scaning_block_size
                
                # key = f"video_{ind}"
                sub_set_stream["pipeline"]["codecs"][key] = {}
                sub_set_stream["pipeline"]["codecs"][key]["params"] = {}
                sub_set_stream["pipeline"]["codecs"][key]["method"] = "VideoCodec"
                sub_set_stream["pipeline"]["codecs"][key]["params"]["codec"] = video_codec_num_to_str[sub_meta.video_decode_information.packing_map_video_codec_id]
                sub_set_stream["stream_meta"][key] = {}
                sub_set_stream["stream_meta"][key]["bit_depth"] = 8
                sub_set_stream["stream_meta"][key]["shapes"] = [sub_meta.video_packing_information.packing_map_height, sub_meta.video_packing_information.packing_map_width, sub_meta.video_packing_information.packing_map_frame_num_minus1+1,-1]
                sub_set_stream["streams"][key] = [unit_stream_data.unit_payload.gsbs_sub_bitstreams.gstc_sub_bitstream_data[ind]]
    
        # unit_metadata.unit_payload.gsbs_metadata.reconstruction_count = 6
        for ind in range(unit_metadata.unit_payload.gsbs_metadata.gs_subset_num):
            sub_set_id = unit_metadata.unit_payload.gsbs_metadata.gs_subset_id[ind]
            sub_set_stream = encode_stream[f"sub_set_{sub_set_id}"]
            count = unit_metadata.unit_payload.gsbs_metadata.reconstruction_count[ind]
            sub_set_stream["pipeline"]["quantizers"] = {}
            for jnd in range(count):
                recon_inf = unit_metadata.unit_payload.gsbs_metadata.reconstruction_information[ind][jnd]
                index = recon_inf.attribute_type
                attribute_type = attribute_type_to_splat_attribute[attribute_dict_num_to_str[index]]

                if recon_inf.prediction_type == 1:
                    if "prediction_meta" not in sub_set_stream["att_meta"]:
                        sub_set_stream["att_meta"]["prediction_meta"] = {}
                    sub_set_stream["att_meta"]["prediction_meta"][attribute_type] = {
                            "byteshift": recon_inf.byteshift,
                            "block_size": recon_inf.blocksize
                    }

                
                sub_set_stream["pipeline"]['quantizers'][attribute_type] = {}
                sub_set_stream['pipeline']['quantizers'][attribute_type]['method'] = quantization_dict_num_to_str[recon_inf.quantization_type]
                sub_set_stream['pipeline']['quantizers'][attribute_type]['params'] = {}
                sub_set_stream['pipeline']['quantizers'][attribute_type]['params']["bits"] = recon_inf.quantization_bitdepth

                # quantization meta
                if recon_inf.quantization_type == 2:
                    sub_set_stream["att_meta"][attribute_type] = {
                        "quantmeta":{
                            "min_vals": torch.from_numpy(np.array([recon_inf.quantization_min_value])),
                            "max_vals": torch.from_numpy(np.array([recon_inf.quantization_max_value])),
                        }
                    }
                elif recon_inf.quantization_type == 3:
                    sub_set_stream["att_meta"][attribute_type] = {
                        "quantmeta":{
                            "min_vals": torch.from_numpy(np.array(recon_inf.patch_quantization_min_value, dtype=np.float64)),
                            "max_vals": torch.from_numpy(np.array(recon_inf.patch_quantization_max_value, dtype=np.float64)),
                            "group_size": torch.tensor(recon_inf.patch_size)
                        }
                    }
                elif recon_inf.quantization_type == 1:
                    # min-max quantization
                    sub_set_stream["att_meta"][attribute_type] = {
                        "quantmeta":{
                            "min_vals": torch.tensor([0.] * recon_inf.component),
                            "max_vals": torch.tensor([1.] * recon_inf.component),
                        }
                    }
                else:
                    print("currently not support this quantization type!")
        return encode_stream
