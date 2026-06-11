#  Copyright [2025] [Huawei Technologies Co., Ltd.] 
#  Licensed under the Code Sharing Policy of the UHD World Association (the "Policy"); 
#  http://www.theuwa.com/UWA_Code_Sharing_Policy.pdf.
#  you may not use this file except in compliance with the Policy. 
#  Unless agreed to in writing, software distributed under the Policy is distributed on an "AS IS" BASIS, 
#  WITHOUT WARRANTIES OF ANY KIND, either express or implied. 
#  See the Policy for the specific language governing permissions and 
#  limitations under the Policy.

import time
import torch
from loguru import logger
from .mix2d_gaussian_model import Mix2DGaussianModel
from .base_gaussian_model import BaseGaussianModel
from .arguments import CodingParams

def apply_xencode(gaussians: BaseGaussianModel, args: CodingParams):
    '''
    将 3DGS 原始表达编码为码流，返回编码后的 3DGS 对象。
    
    Args:
        gaussians (BaseGaussianModel): 待编码 3DGS 对象
        args (CodingParams): 编码相关参数
    Returns:
        Mix2DGaussianModel: 编码后的 3DGS 对象
    '''
    start_time = time.time()
    encode_gaussians = Mix2DGaussianModel(gaussians.max_sh_degree)
    encode_gaussians.restore_fromgaussian(gaussians)
    init_end_time = time.time()
    logger.info(f"[XEncode] Init time : {init_end_time - start_time}")

    color_importance = torch.abs(encode_gaussians.get_features_rest)
    color_importance = torch.sum(color_importance, axis = [1, 2])
    color_importance *= encode_gaussians.get_opacity.squeeze()
    imp_value = torch.sort(color_importance).values[int(0.2 * len(color_importance))]
    color_importance += imp_value * 0.05

    st_time = time.time()
    if args.sort_method == "block":
        encode_gaussians._morton_and_blocksort(importance = color_importance, block = args.astc_block, args = args)
    else:
        encode_gaussians._morton_sort(importance = color_importance, block = args.astc_block, args = args)
    config = {"pos" : True, "scale" : True, "rot" : True, "dc" : True,  "rest" : True, "opacity" : True}
    encode_gaussians.apply_minmaxquant(config)
    logger.info(f"[XEncode] Sort + Quant time : {time.time() - st_time:.2f} s")

    astc_time = time.time()
    data_stream = encode_gaussians.save_encoded_model_to_stream(args)
    encode_gaussians.encoded_data_stream = data_stream
    logger.info(f"[XEncode] ASTC Codec time : {time.time() - astc_time:.2f} s")

    return encode_gaussians
