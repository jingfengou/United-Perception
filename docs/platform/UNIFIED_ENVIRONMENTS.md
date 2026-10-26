# 通用镜像与环境隔离

完整构建和运行命令见 [快速开始](QUICKSTART_UNIFIED.md)。本页说明镜像内部布局、启动选择和无 Docker 时的目录模式。

## 镜像布局

```text
/opt/up/
  envs/nvidia/       NVIDIA PyTorch、TorchVision、UP 和厂商依赖
  envs/metax/        沐曦 PyTorch、TorchVision、UP 和厂商依赖
  sdk/cuda/          NVIDIA CUDA 工具链
  sdk/maca/          沐曦 MACA 工具链
  sdk/mxdriver/      沐曦用户态库，内核驱动由宿主机提供
  wheels/           两家分别编译的 UP 安装包
  source/           UP 配方、工具和构建源码
  profiles.json     各厂商 Python、SDK 和动态库路径
  launch.py         环境选择器
  probe.py          包和设备检查
```

沐曦环境继承基础镜像的厂商包；NVIDIA 环境不继承这些包。普通依赖分别安装，Torch/TorchVision 受各自版本约束。专用扩展使用独立构建目录和二进制。

## 两级启动

```text
宿主机 ./up_docker nvidia|metax train|eval|inference ...
    ↓ Docker 设备、工作目录和数据挂载
容器内 up-run nvidia|metax train|inference ...
    ↓ Python、SDK、动态库与 UP 后端
UP 模型及对应平台实现
```

`eval` 在宿主机转成 `train --evaluate`。镜像入口会补充 `runtime.platform.backend`，发现冲突时拒绝启动。

`up-run` 清理继承的其他 SDK、Conda 和 Python 路径，按 profile 设置当前厂商环境，在新进程中运行。切换不能发生在已经导入另一家 Torch 的同一个 Python 进程中。

NVIDIA 运行时使用所选 Torch 配套的数值库；CUDA 编译目录的 `lib64` 不加入运行时搜索路径。缓存按厂商隔离，默认位于 `/tmp/up-cache-<uid>/<vendor>/`；可用 `UP_CACHE_ROOT` 修改根目录。

## 构建参数

| 参数 | 用途 |
| --- | --- |
| `CUDA_IMAGE` | NVIDIA CUDA 基础镜像，默认 12.6.3-devel-ubuntu24.04 |
| `METAX_IMAGE` | 沐曦基础镜像，默认 MACA/PyTorch 3.8.1.2-torch2.10-py312 |
| `NVIDIA_ARCH_LIST` | 默认 `8.0;8.9`，替换为所需 NVIDIA 架构 |
| `UP_PIP_INDEX_URL` | 普通 Python 依赖源；厂商 Torch 使用自己的来源 |

更换基础镜像后，应核对 SDK 目录、Torch 版本和 ABI，再重新构建。两套环境会增大镜像体积；数据、权重和凭据放在镜像之外。

## 目录模式

没有 Docker 时，可使用同一环境选择器，但需要自行准备两套完整且可重定位的 Python/SDK。布局与上方一致，`profiles.json` 指向实际路径：

```bash
python docker/unified/launch.py --root /path/to/bundle nvidia check
python docker/unified/launch.py --root /path/to/bundle metax python -c "import torch; print(torch.__version__)"
```

目录模式不调用 `up_docker`，也不会自动创建虚拟环境或下载厂商 SDK。每家使用自己的 Python 安装对应 UP wheel：

```bash
/path/to/bundle/envs/nvidia/bin/python -m pip install --no-deps /path/to/nvidia/up-*.whl
/path/to/bundle/envs/metax/bin/python -m pip install --no-deps /path/to/metax/up-*.whl
```

wheel 分别在正确厂商环境下用 `UP_BUILD_BACKEND=nvidia` 或 `metax` 构建。不要直接搬移带绝对路径的 Conda/venv 后假定可用；先检查解释器、动态库和扩展。

## 环境检查

`check --no-device` 只检查包；`check` 执行 GPU 矩阵前后向、加载当前厂商原生扩展，并输出实际加载的动态库。模型所需操作和第三方工具链还应按 [接入验收](ADDING_VENDOR.md#4-配置并验证) 检查。
