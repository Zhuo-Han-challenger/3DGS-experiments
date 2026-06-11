#  Copyright [2025] [Huawei Technologies Co., Ltd.] 
#  Licensed under the Code Sharing Policy of the UHD World Association (the "Policy"); 
#  http://www.theuwa.com/UWA_Code_Sharing_Policy.pdf.
#  you may not use this file except in compliance with the Policy. 
#  Unless agreed to in writing, software distributed under the Policy is distributed on an "AS IS" BASIS, 
#  WITHOUT WARRANTIES OF ANY KIND, either express or implied. 
#  See the Policy for the specific language governing permissions and 
#  limitations under the Policy.

import os
import zlib
import struct
import platform
import subprocess
import tempfile

import numpy as np

from PIL import Image
from pathlib import Path

_module_path = Path(__file__).resolve()
_module_dir = _module_path.parent
astc_cmd = None
if platform.system() == "Linux":
    astc_cmd = os.path.join(str(_module_dir), "tools", "astcenc-sse2")
elif platform.system() == "Windows":
    astc_cmd = os.path.join(str(_module_dir), "tools", "astcenc-sse2.exe")
else:
    raise RuntimeError("Only execute in Linux or Windows!")
        
def parsePatch(data_stream, pointer, attribute_info):
    """
    parse Patch meta-data
    """
    info = {}
    if attribute_info["patchGlobalEnableIndexFlag"] == 1:
        info["patchIndex"] = struct.unpack("<I", data_stream[pointer:pointer + 4])[0]
        pointer += 4
    if attribute_info["patchGlobalEnableSizeFlag"] == 1:
        info["patchSize"] = struct.unpack("<I", data_stream[pointer:pointer + 4])[0]
        pointer += 4
    if attribute_info["attributeQuantizationFlag"] == 1:
        if attribute_info["patchGlobalEnableQuantBitFlag"] == 1:
            info["patchQuantBits"] = struct.unpack("<B", data_stream[pointer:pointer + 1])[0]
            pointer += 1
        if attribute_info["patchGlobalEnableQuantMinMaxFlag"] == 1:
            info["patchQuantMinValue"] = np.array([struct.unpack("<f", data_stream[pointer + i*4:pointer + i*4 + 4])[0] for i in range(attribute_info["componentsCount"])])
            pointer += attribute_info["componentsCount"] * 4
            info["patchQuantMaxValue"] = np.array([struct.unpack("<f", data_stream[pointer + i*4:pointer + i*4 + 4])[0] for i in range(attribute_info["componentsCount"])])
            pointer += attribute_info["componentsCount"] * 4
    return info, pointer


def parseAttributeMeta(data_stream, pointer):
    """
    parse attribute meta-data
    """
    info = {}
    info["attributeType"], info["componentsCount"], info["uncompressedDataType"], Tmp = struct.unpack("<IBBB", data_stream[pointer:pointer+7])
    info["attributeQuantizationFlag"] = Tmp >> 4
    info["attributeEncoderScheme"] = Tmp & 15
    pointer += 7
    if info["attributeQuantizationFlag"] == 1:
        info["quantizationBits"] = struct.unpack("<B", data_stream[pointer:pointer+1])[0]
        pointer += 1
        info["quantMinValue"] = np.array([struct.unpack("<f", data_stream[pointer + i*4:pointer + i*4 + 4])[0] for i in range(info["componentsCount"])])
        pointer += info["componentsCount"] * 4
        info["quantMaxValue"] = np.array([struct.unpack("<f", data_stream[pointer + i*4:pointer + i*4 + 4])[0] for i in range(info["componentsCount"])])
        pointer += info["componentsCount"] * 4
    if info["attributeEncoderScheme"] == 1:
        info["attributeIndexType"], info["attributeCodebookLength"] =\
                struct.unpack("<BI", data_stream[pointer:pointer + 5])
        pointer += 5
    elif info["attributeEncoderScheme"] == 2:
        info["attribute2DmimeType"], info["attribute2DSingleHeight"], info["attribute2DSingleWidth"], info["attribute2DSingleAlign"], \
            info["attribute2DConcat"], info["attribute2DConcatMaxInWidth"], info["attribute2DConcatMaxInHeight"], info["attribute2DTexNum"] =\
                struct.unpack("<BHHBBBBB", data_stream[pointer:pointer + 10])
        pointer += 10
        info["attribute2DTexSizes"] = [struct.unpack("<I", data_stream[pointer + i*4:pointer + i*4 + 4])[0] for i in range(info["attribute2DTexNum"])]
        pointer += info["attribute2DTexNum"] * 4
    
    info["byteOffset"], info["byteLength"], info["uncompressedByteLength"], info["patchNum"] =\
        struct.unpack("<IIII", data_stream[pointer:pointer + 4*4])
    pointer += 16

    if info["patchNum"] > 1:
        info["patchMetas"] = []
        Tmp = struct.unpack("<H", data_stream[pointer:pointer + 2])[0]
        pointer += 2
        info["patchGlobalEnableIndexFlag"] = 1 & (Tmp >> 15)
        info["patchGlobalEnableSizeFlag"] = 1 & (Tmp >> 14)
        info["patchGlobalEnableQuantBitFlag"] = 1 & (Tmp >> 13)
        info["patchGlobalEnableQuantMinMaxFlag"] = 1& (Tmp >> 12)
        info["patchGlobalEnable2DmapingFlag"] = 1 & (Tmp >> 11)
        info["reversedFlag"] = 0x7FF & Tmp
        if info["patchGlobalEnableSizeFlag"] == 0:
            info["lastPatchSize"] = struct.unpack("<I", data_stream[pointer:pointer + 4])[0]
            pointer += 4

        for i in range(info["patchNum"]):
            patch_info, pointer = parsePatch(data_stream, pointer, info)
            info["patchMetas"].append(patch_info)

    return info, pointer

uncompressed_dataType = {
    1 : np.int8,
    2 : np.uint8,
    3 : np.int16,
    4 : np.uint16,
    5 : np.float16,
    6 : np.int32,
    7 : np.uint32,
    8 : np.float32,
    9 : np.int64,
    10 : np.uint64,
    11 : np.float64
}

def decode_attribute(attribute_stream, attribute_info, numGS):
    """
    decode attributes according to metadata
    """
    # 解码后数据类型
    if attribute_info["attributeQuantizationFlag"] not in uncompressed_dataType:
        raise RuntimeError(f"不支持的数据类型 {attribute_info['attributeQuantizationFlag']}") 
    # 没有量化的情况下，直接读取数据返回
    if attribute_info["attributeQuantizationFlag"] == 0:
        if attribute_info["attributeEncoderScheme"] != 0:
            raise RuntimeError(f"不量化的情况下，不支持2D化")
        decoded_data = np.frombuffer(attribute_stream, dtype = uncompressed_dataType[attribute_info["attributeQuantizationFlag"]]).reshape(numGS, attribute_info["componentsCount"])
        return decoded_data

    # 获得数据
    if attribute_info["attributeQuantizationFlag"] == 1:
        decoded_data = None
        if attribute_info["attributeEncoderScheme"] == 1:
            # 支持VQ量化化
            dtype = uncompressed_dataType[attribute_info["attributeIndexType"]]
            if attribute_info["attributeIndexType"] != 7:
                raise RuntimeError("不支持对该属性进行 VQ 量化")
            indices = np.frombuffer(attribute_stream[:numGS * 4], dtype = dtype).reshape(numGS)
            dtype = np.uint16 if attribute_info["quantizationBits"] > 8 else np.uint8
            decoded_data = np.frombuffer(attribute_stream[numGS * 4:], dtype = dtype).reshape(-1, attribute_info["componentsCount"]) / 255.0
            decoded_data = decoded_data[indices]
        elif attribute_info["attributeEncoderScheme"] == 0:
            dtype = np.uint16 if attribute_info["quantizationBits"] > 8 else np.uint8
            decoded_data = np.frombuffer(attribute_stream, dtype = dtype).reshape(numGS, attribute_info["componentsCount"]) / (2 ** attribute_info["quantizationBits"] - 1)
        elif attribute_info["attributeEncoderScheme"] == 2:
            astc_file_sizes = attribute_info["attribute2DTexSizes"]
            astc_data = attribute_stream
            astc_files = []
            ptr = 0
            for astc_file_size in astc_file_sizes:
                astc_files.append(astc_data[ptr:ptr + astc_file_size])
                ptr += astc_file_size

            #   Decode
            data_recon = []
            def generate_shell_script(image_paths, output_paths, log_path):
                """
                生成一个 Shell 脚本，用于批量压缩图片。
                """
                script_content = ""
                for img_path, out_path in zip(image_paths, output_paths):
                    cmd = [
                        astc_cmd,
                        "-dl", img_path,
                        out_path, f">{log_path} 2>&1"
                    ]
                    script_content += " ".join(cmd) + "\n"
                return script_content

            temp_dir_path = '/dev/shm' if platform.system() == 'Linux' else None
            with tempfile.TemporaryDirectory(dir=temp_dir_path) as temp_dir:
                image_paths = []
                output_paths = []
                log_path = os.path.join(temp_dir, f"out.log")
                for idx, astc_data in enumerate(astc_files):
                    astc_block_size = struct.unpack("<BBBBB", astc_data[:5])[-1]
                    temp_image_path = os.path.join(temp_dir, f"image{idx}.astc")
                    image_paths.append(temp_image_path)
                    tmp_file = open(temp_image_path, "wb")
                    tmp_file.write(astc_data)
                    tmp_file.close()
                    output_paths.append(os.path.join(temp_dir, f"image{idx}.bmp"))
                
                # 创建并写入 Shell 脚本
                shell_script = generate_shell_script(image_paths, output_paths, log_path)
                shell_script_path = None
                if platform.system() == 'Linux':
                    shell_script_path = os.path.join(temp_dir, "compress.sh")
                    with open(shell_script_path, 'w') as f:
                        f.write(shell_script)                # 赋予脚本执行权限
                    # 执行 Shell 脚本
                    subprocess.call(["bash", shell_script_path], shell=False, stdout=open(os.path.join(temp_dir, "command.log"), "w"))
                else:
                    shell_script_path = os.path.join(temp_dir, "compress.bat")
                    with open(shell_script_path, 'w') as f:
                        f.write(shell_script)                # 赋予脚本执行权限
                    subprocess.call(shell_script_path, shell=False, stdout=open(os.path.join(temp_dir, "command.log"), "w"))

                for temp_output_file in output_paths:
                    img = Image.open(temp_output_file).convert("RGB") # 目前只支持三通道！
                    total_img_data = np.array(img)

                    if attribute_info["attribute2DConcat"] == 0:
                        img_data = total_img_data.reshape(-1, 1, 3)
                        data_recon.append(img_data)

                    elif attribute_info["attribute2DConcat"] == 1:
                        for idx in range(int(attribute_info["attribute2DConcatMaxInWidth"])):
                            img_data = total_img_data[:, idx * int(attribute_info["attribute2DSingleWidth"]) : (idx + 1) * int(attribute_info["attribute2DSingleWidth"])]
                            img_data = img_data.reshape((attribute_info["attribute2DSingleHeight"]//astc_block_size,astc_block_size,attribute_info["attribute2DSingleWidth"]//astc_block_size,astc_block_size,-1)).transpose((0,2,1,3,4)).reshape(-1, 1, 3)
                            data_recon.append(img_data)

                            if len(data_recon) >= int(attribute_info["componentsCount"] // 3):
                                break
                    else:
                        # 目前只支持按行合并！
                        raise RuntimeError("当前只支持不拼接或者按行拼接")
            decoded_data = np.concatenate(data_recon, axis = 1).reshape(-1, attribute_info["componentsCount"]) / 255.0
        else:
            raise RuntimeError()

        # 反最大最小量化！
        if attribute_info["patchNum"] == 1:
            decoded_data = decoded_data * (attribute_info["quantMaxValue"] - attribute_info["quantMinValue"]) + attribute_info["quantMinValue"]
        else:
            tmp_decoded_data = []
            if attribute_info["patchGlobalEnableSizeFlag"] == 0:
                chunk_size = int((numGS - attribute_info["lastPatchSize"]) * 1.0 / (attribute_info["patchNum"] - 1))
                for i in range(numGS):
                    patch = attribute_info["patchMetas"][i // chunk_size]
                    tmp_decoded_data.append(decoded_data[i] * (patch["patchQuantMaxValue"] - patch["patchQuantMinValue"]) + patch["patchQuantMinValue"])
            else:
                pointer = 0
                for i in range(attribute_info["patchNum"]):
                    patch = attribute_info["patchMetas"][i]
                    for j in range(patch["patchSize"]):
                        tmp_decoded_data.append(decoded_data[pointer] * (patch["patchQuantMaxValue"] - patch["patchQuantMinValue"]) + patch["patchQuantMinValue"])
                        pointer += 1
                        if pointer >= numGS:
                            break
                    if pointer >= numGS:
                            break
            decoded_data = np.array(tmp_decoded_data)
        return decoded_data    

    raise RuntimeError("不支持的格式！")
    return None

def decode_stream(data_stream):
    # parse the header!
    pointer = 0
    if data_stream[:4] != "gsct".encode("ascii"):
        raise RuntimeError("不支持的格式！")
    pointer += 4
    version = list(data_stream[pointer:pointer+3])
    pointer += 3
    totalByteLength, superCompressionScheme, reversed = struct.unpack("<IBI", data_stream[pointer:pointer+9])
    pointer += 9
    numGS, numAttribute = struct.unpack("<IB", data_stream[pointer:pointer+5])
    pointer += 5
    print(f"Load compressed 3DGS data: Number = {numGS}; Attributes = {numAttribute}")
    attr_metas = []
    for i in range(numAttribute):
        attr_meta, pointer = parseAttributeMeta(data_stream, pointer)
        attr_metas.append(attr_meta)
    attribute_datas = data_stream[pointer:]
    if superCompressionScheme == 1:
        attribute_datas = zlib.decompress(attribute_datas)
    # decode the attributes
    datas = {"sh_degree" : 0}
    for attr_meta in attr_metas:
        attribute_data_stream = attribute_datas[attr_meta["byteOffset"]:attr_meta["byteOffset"] + attr_meta["byteLength"]]
        attributeType = attr_meta["attributeType"]
        decoded_attribute = decode_attribute(attribute_data_stream, attr_meta, numGS)
        if attributeType == 0:
            datas["positions"] = decoded_attribute
        elif attributeType == 1:
            datas["rotation"] = decoded_attribute
        elif attributeType == 2:
            datas["scale"] = decoded_attribute
        elif attributeType == 3:
            dc = decoded_attribute[:,:3]
            op = decoded_attribute[:,3]
            op[op<0.001] = 0.001
            op[op>0.999] = 0.999   # avoid lose in inverse opacity activation
            op = np.log(op / (1 - op))
            datas["dc"] = dc
            datas["opacity"] = op
        elif attributeType == 4:
            datas["shs"] = decoded_attribute
            datas["sh_degree"] = int(np.sqrt(attr_meta["componentsCount"] // 3 + 1) - 1)
        else:
            raise RuntimeError("Wrong Attribute Type")
    return datas