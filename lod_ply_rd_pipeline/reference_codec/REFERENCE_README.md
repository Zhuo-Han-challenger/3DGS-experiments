# 说明
此目录为符合《支持六自由度交互的三维图像格式标准》 定义的 3DGS 编码的 python 版本参考实现，适合快速测试验证。

# 环境配置流程

## 第三方依赖库

通过以下命令将第三方依赖拉取至本地：

```bash
git submodule update --init --recursive
cd dependency/astcenc/astcenc
git checkout 701503966b1ac2ebd2616cba94adee5ae8ba6363
```

## python 环境配置

新建环境一键安装，程序将在纯 cpu 环境运行

```bash
    conda env create -f env.yml
    conda activate xcodec
```

## 标准编码软件配置

Video Codec (默认使用 ffmpeg x265)，若使用 hm 或 VTM，则需将其编译至 denpendency 文件夹下：

如 HM 编码器路径为： dependency/HM-18.0/bin/TAppEncoderStatic， VTM 编码器路径为：dependency/VTM-23.11/bin/EncoderAppStatic。

或者自行修改 `src/xencode/processor/codecs.py` 中的 VideoCodec 路径至实际编译好的编码器路径。

# 编解码示例

## 单文件编码示例
以下为单文件编码示例，ply-path 为 ply 文件地址，结果保存在 save-dir 指定的文件夹下，日志保存在 log-path 指定的 json 文件中，quality 指定压缩后文件质量，config 使用指定配置文件：

```bash
python encode_1_0.py \
    --ply-path bag.ply \
    --save-dir results \
    --scene-name bag \
    --log-path results/bag/log.json \
    --quality 66 \
    --config config/3dgs_d3.yaml
```

可选参数：

| **参数**              | **取值** | **说明**|
|---------------------|------------------------|------------------------------------|
| sort_method | morton/dblock | 选择排序方法，morton速度最快，dblock 需要安装 dependency/sorting 依赖 |
| video_codec | x264/x265/hm/vtm | 选择编码方法 |
| quality | 0~100 | 选择编码后文件质量，100最高, 100/66/33/0 分别对应测试报告中的 r0,r1,r2,r3 |
| pos_bitdepth | 9~16 | 选择位置位深 |
| rot_bitdepth | 1~8 | 选择旋转位深 |
| sca_bitdepth | 1~8 | 选择尺度位深 |
| op_bitdepth | 1~8 | 选择不透明度位深 |
| sh0_bitdepth | 1~8 | 选择 0 阶球谐系数位深 |
| shn_bitdepth | 1~8 | 选择高阶球谐系数位深 |
| config | path_to_config.yaml | 配置文件地址，提供配置组合，**配置文件的参数将会完全覆盖命令行参数**。提供三个推荐配置：3dgs_d3 编码 3 阶 3DGS，3dgs_d0 编码 0 阶 3DGS，3dgs_ff 编码单视图前馈式 3DGS。|

## 批量编码示例
数据集目录为 dataset 且文件结构为:
```
|-dataset
|-|-scene01
|-|-|-model.ply
|-|-scene02
...
```

则批量编码脚本如下：
```bash
bash scripts/compress_stream.sh
```
