# Vendored Reference Codec

This folder contains the Python reference 3DGS codec files copied from:

```text
D:\Documents\《支持六自由度交互的三维图像格式标准》参考工具集v1.0.3\基于glTF存储三维图像格式的参考实现\glTF三维图像格式存储和解析\3DGS编码实现\python
```

Copied contents:

- `encode_1_0.py`
- `decode_1_0.py`
- `jsonlogger.py`
- `env.yml`
- `config/`
- `src/`
- `REFERENCE_README.md`

Not copied:

- `dependency/`
- compiled external codec binaries
- submodule working trees

The wrapper scripts in the parent directory use this folder as their default `--codec-root`. External tools such as x265/x264/HM/VTM/ASTC dependencies still need to be installed or configured according to the reference README.
