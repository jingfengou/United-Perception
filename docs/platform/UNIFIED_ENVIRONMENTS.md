# 通用镜像与厂商环境切换

一个 Linux/amd64 镜像保存两套独立 Python 包、SDK 路径、UP wheel 和缓存。`up-run` 选择厂商后，在新进程中执行 UP；模型算法和配置格式保持原样。

## 布局

```text
/opt/up/
  envs/nvidia/       NVIDIA Torch 2.10.0+cu126、TorchVision、spconv、UP wheel
  envs/metax/        沐曦 Torch 2.10.0+metax3.8.1.0、厂商依赖、UP wheel
  sdk/cuda/          NVIDIA CUDA 开发工具链
  sdk/maca/          沐曦 MACA 开发工具链
  sdk/mxdriver/      沐曦用户态驱动库；内核驱动由宿主机提供
  wheels/           两家分别编译的 UP 安装包
  source/           UP 配方、工具与构建源码
  profiles.json     各厂商 Python、PATH、动态库和 SDK 路径
  launch.py         标准库启动器，不导入 Torch 或 UP
  probe.py          环境及设备检查
```

沐曦虚拟环境继承同一个镜像中的厂商基础包，普通依赖安装在自己的环境内；NVIDIA 环境不继承这些包。
安装普通依赖时固定各自 Torch/TorchVision 版本；两个扩展分别编译成 `_nvidia_ops`、`_metax_ops`，不共用构建目录。

## 构建与分发

```bash
docker build -f docker/unified/Dockerfile -t up-unified:local .
# 指定其他 NVIDIA 编译架构，例如仅 SM89
docker build -f docker/unified/Dockerfile --build-arg NVIDIA_ARCH_LIST=8.9 -t up-unified:sm89 .
```

默认 CUDA 镜像来自 NVIDIA NGC，MACA 镜像来自沐曦仓库。可用 `CUDA_IMAGE`、`METAX_IMAGE` 构建参数指定已核对的镜像标签或 digest；升级版本时须同步核对 SDK 目录、Torch 和依赖，再重新验证。
构建要求网络可访问厂商镜像仓库、PyTorch/PyPI 索引和系统包源。构建只编译和检查包，不需要 GPU，不修改默认数值策略。
普通依赖默认从官方 PyPI 下载，可用 `--build-arg UP_PIP_INDEX_URL=https://your-mirror/simple` 改为自己的镜像源；Torch 仍使用 NVIDIA 官方 PyTorch 索引或沐曦基础镜像。
镜像包含两套依赖，体积会大于单厂商镜像。数据、权重和 SSH 密钥不打入镜像；`.dockerignore` 排除本地实验及凭据。

构建一次后可通过私有镜像仓库或 `docker save` / `docker load` 在两台服务器间分发。宿主机须分别安装 NVIDIA/沐曦驱动，并按厂商要求映射设备。
无需同时在两台服务器上安装两家的内核驱动。厂商组件许可及分发条件仍由使用者按其来源确认。

## 使用

容器入口为 `up-run`，后面依次指定厂商、命令和原有参数：

```bash
export DEFAULT_TASKS=cls:det  # 本页示例为分类 / 二维检测
up-run nvidia train --config model.yaml --launch pytorch --ng 1
up-run metax train --evaluate --config model.yaml --launch pytorch --ng 1
up-run metax inference --config model.yaml --img_path images/ --ckpt model.pth --launch pytorch --ng 1
up-run nvidia check
up-run metax check --no-device
up-run nvidia python -c "import torch; print(torch.__version__)"
```

这些是容器内命令；外部用 README 中的 `docker run` 传入设备和工作目录。`check --no-device` 仅检查包，不算 GPU 验证。
UP 未设置 `DEFAULT_TASKS` 时会导入所有任务，可能在启动时就要求点云、量化等可选依赖。
例如点云选择 `DEFAULT_TASKS=det_3d`，同时安装该配方需要的 scikit-image 等依赖；本镜像的普通依赖清单覆盖本页分类/二维检测回归，其他任务按 [环境说明](ENVIRONMENT.md) 准备。
`check` 默认执行 GPU 矩阵前向/反向、加载专用扩展，并列出实际加载的厂商动态库。它不代表所有模型或接口都已验证。

启动器保留显卡可见性、显式随机种子和当前厂商的数值开关，不自动开启 AMP、确定性、量化或 CPU 代算。
它清理继承的 `PYTHONPATH`、另一家 SDK 的路径、Conda 激活状态和预加载库，按 profile 重建路径，并将缓存分到 `/tmp/up-cache-<uid>/<vendor>/`。
NVIDIA 运行时使用所选 PyTorch 配套的 CUDA 数值库；编译工具链的 `lib64` 不加入运行时搜索路径，避免抢先加载另一版本的 cuBLAS。
可用 `UP_CACHE_ROOT` 指定缓存根目录。切换时用另一条命令启动新进程；禁止在已经导入 Torch 的进程内替换厂商。
自定义插件要安装到所选厂商环境中；启动器不继承宿主机的 `PYTHONPATH`。

## 目录模式

启动器也可用于没有容器运行时的临时验证。按上述布局准备可用 Python 环境和 SDK，在对应目录保存 `profiles.json`、`launch.py`、`probe.py`，再运行：

```bash
python docker/unified/launch.py --root /path/to/bundle nvidia check
python docker/unified/launch.py --root /path/to/bundle metax python -c "import torch; print(torch.__version__)"
```

目录模式需要自行准备和核对依赖，不提供自动复制/重定位 Conda 环境的承诺。虚拟环境中的绝对路径和本机动态库仍须核对。
本轮用户指定：C500 验证镜像，4090 使用临时目录环境；二者的结果分别记录，不把目录执行写成 NVIDIA Docker 已验证。

目录模式同样用厂商自己的 Python 安装 UP：

```bash
# 每个环境分别执行；不能在同一个 site-packages 中覆盖两家的 Torch。
/path/to/bundle/envs/nvidia/bin/python -m pip install --no-deps /path/to/nvidia/up-*.whl
/path/to/bundle/envs/metax/bin/python -m pip install --no-deps /path/to/metax/up-*.whl
```

wheel 要在对应厂商环境下，用 `UP_BUILD_BACKEND=nvidia` 或 `metax` 编译；SDK 和库路径由 `profiles.json` 指向实际安装位置。
仅创建两个空 venv，或只更改 `runtime.platform.backend`，都不能替代这一步。

## 验证

软件测试覆盖环境污染清理、后端参数冲突、缓存隔离和新进程路由。真机检查使用 UP ResNet18 的短程训练、训练内 eval、检查点恢复及独立推理，并测试原生 NMS/RoIAlign。
同平台对照使用相同权重初始化、输入、种子和配置，沿用 `rtol=1e-4, atol=1e-5`。这是环境与调用链回归，不是完整数据集精度训练。
2026-10-26 已在两家真机完成回归。比较的是同一份 UP 代码在原独立环境和新通用环境中的结果：

| 检查 | NVIDIA RTX 4090：目录模式 | 沐曦 C500：通用 Docker 镜像 |
| --- | --- | --- |
| 切换两家 PyTorch、UP 命令入口 | 通过 | 通过 |
| GPU 矩阵前后向、NMS、RoIAlign 前后向 | 通过 | 通过 |
| 同权重独立推理 | 最大差 0 | 最大差 0 |
| 确定性设置下，2 步训练及训练内 eval | 最大差 0 | 最大差 0，需下述原生 DNN 控制 |
| 权重恢复后复算训练内 eval | 最大差 0 | 最大差 0 |

NVIDIA 还使用了通用镜像实际编译的 CUDA 12.6 wheel，在 4090 目录环境中通过同一对照。
4090 按用户要求使用目录模式；NVIDIA Docker 的 GPU 运行尚未验证。C500 验证了实际镜像。
两家训练对照各比较 257 个 Tensor 字段，同权重推理各比较 255 个字段；样例为 ResNet18、固定 2 张 224×224 输入、2 次 SGD 更新。

**默认 native 的训练数值有波动，原环境自身重复也会出现。** 以下为 Tensor 最大绝对差，单位不是准确率百分点；超出原容差的结果如实保留：

| 运行方式 | 原环境自身重复 | 原环境 / 新环境 | 数值对照 |
| --- | ---: | ---: | --- |
| NVIDIA 目录 native 诊断 | 2.062012 | 6.991455 | 未通过；最终镜像 wheel 另做确定性对照 |
| C500 镜像 native | 0.375061 | 0.040588 | 未通过 |
| C500 仅 PyTorch 原生确定性 | 0.014648 | 0.588135 | 未通过 |
| NVIDIA 原生确定性 + BLAS workspace | 0 | 0 | 通过 |
| C500 原生确定性 + BLAS workspace + 两项 DNN 控制 | 0 | 0 | 通过 |

NVIDIA 设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 并在回归工具中加 `--deterministic`。
C500 在此基础上，**两边均在进程启动前**设置 `MCDNN_USE_DETERMINISTIC_ALGO=1`、`MCDNN_STABLE_RESULT_MOD=1`。
这些是本次 MACA 环境的显式测试设置；启动器保持用户原有数值策略，默认不启用它们。

两套 Torch/SDK 在同一镜像中完成包检查；加载日志显示 NVIDIA 使用 cuBLAS/cuDNN，沐曦使用 mcBLAS/mcDNN。
C500 新旧环境的三份 BLAS/DNN 库 SHA256 相同。镜像 ID 为 `f1bb287ddc3a`，
NVIDIA 实测扩展 SHA256 为 `6a236cdf4e915071afdbbcd6ad01beabdc090a71f9360e26a8e8ad88429f8b8e`。
实测镜像在 C500 上保留为 `up-unified:local`，可直接使用 README 中的启动命令。

4090 临时环境、SDK 副本和编译目录已删除；原有 Python、Torch、扩展和配方删除前后校验一致，原始结果保留。
[结果摘要](validation/20261026/unified-environments/summary.json)、
[NVIDIA 原始证据索引](validation/20261026/unified-environments/nvidia/evidence-index.json)、
[C500 原始证据索引](validation/20261026/unified-environments/metax/evidence-index.json)、
[清理回执](validation/20261026/unified-environments/nvidia-cleanup.json) 可逐项核查。
C500 的本次失败构建容器和临时 SDK 导出副本也已清理，实测镜像和原有参考容器保留，见 [构建清理记录](validation/20261026/unified-environments/metax/cleanup.json)。

本次修正了 wheel 漏装 11 个旧算子兼容入口的打包问题，未修改 `up/`、`up_platform/`、`configs/`。
构建验证使用已保存的厂商镜像和离线依赖缓存继续安装，版本及文件哈希保存在镜像记录中；
缓存准备中的失败、native 超差和完整模型状态都保留在证据存储中。旧数据集精度和多种子任务保持暂停。

可复用回归工具 `tools/validate_unified_environment.py`。以下命令在已安装镜像内执行，数据由固定 CPU 种子生成后搬到 GPU：

```bash
up-run metax python /opt/up/source/tools/validate_unified_environment.py \
  --vendor metax --phase training --output /results/training
up-run metax python /opt/up/source/tools/validate_unified_environment.py \
  --vendor metax --phase inference --checkpoint /results/training/checkpoint.pt \
  --output /results/inference
```

同厂商旧环境也执行同一工具，新环境加 `--reference /results/old-training`（推理用 `old-inference`）比较。
每次使用新的输出目录；失败结果不覆盖。`--source-root` 可指定配方所在的源码目录。
NVIDIA 排除原生训练波动的复测可在两边同时设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 并加 `--deterministic`。
本次 C500 复现命令，在容器内、已映射设备和可写 `/results` 目录时执行：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MCDNN_USE_DETERMINISTIC_ALGO=1 MCDNN_STABLE_RESULT_MOD=1
up-run metax python /opt/up/source/tools/validate_unified_environment.py \
  --vendor metax --phase training --deterministic --output /results/sdk-training
up-run metax python /opt/up/source/tools/validate_unified_environment.py \
  --vendor metax --phase inference --deterministic \
  --checkpoint /results/sdk-training/checkpoint.pt --output /results/sdk-inference
```
