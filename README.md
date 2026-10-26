# UP Platform

<img src="up-logo.png" alt="United Perception" width="160" />

基于 [United-Perception](https://github.com/ModelTC/United-Perception) 的多厂商平台适配层。模型、数据和训练配方保留在 `up/`，设备、专用算子及厂商差异集中在 `up_platform/`。

- 设备、15 类算子、训练与推理使用统一接口，保留 UP 配置、checkpoint 和旧算子入口。
- NVIDIA 和沐曦分别绑定厂商 PyTorch 与独立扩展，兼容实现共享源码。
- 一个通用镜像隔离两套环境，通过仓库脚本选择厂商运行。

## 快速开始

在 Linux 服务器上安装 Docker 和对应显卡驱动。NVIDIA 还需配置 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

```bash
git clone --branch codex/up-platform-clean https://github.com/jingfengou/United-Perception.git
cd United-Perception
docker build -f docker/unified/Dockerfile -t up-unified:local .

# 在对应厂商服务器上检查环境
./up_docker nvidia check
./up_docker metax check
```

构建无需 GPU，默认 NVIDIA 扩展覆盖 A100/A800、RTX 4090。数据、权重和输出保存在宿主机，不打入镜像。

**[完整示例：构建镜像 → 准备 COCO → 训练 → 验证 → 图片推理](docs/platform/QUICKSTART_UNIFIED.md)**

## 选择与运行

按完整示例准备好 `model.yaml` 和工作目录后，选择本机厂商：

```bash
export UP_WORK="$HOME/up-demo"
export UP_DATA="/datasets/coco2017"
export VENDOR=metax  # NVIDIA 使用 nvidia

./up_docker "$VENDOR" train --config model.yaml --launch pytorch --ng 1 \
  --opts saver.save_dir "outputs/$VENDOR/checkpoints" saver.results_dir "outputs/$VENDOR/eval"

./up_docker "$VENDOR" eval --config model.yaml --launch pytorch --ng 1 \
  --opts saver.auto_resume False saver.resume_model "outputs/$VENDOR/checkpoints/ckpt_latest.pth" \
  saver.save_dir "outputs/$VENDOR/checkpoints" saver.results_dir "outputs/$VENDOR/eval"

./up_docker "$VENDOR" inference --config model.yaml --img_path images/ \
  --ckpt "outputs/$VENDOR/checkpoints/ckpt_latest.pth" --vis_dir "outputs/$VENDOR/predictions" \
  --launch pytorch --ng 1
```

`eval` 是 UP `train --evaluate` 的简写，使用训练适配路径；独立 `inference` 使用推理适配路径。`UP_GPU` 选择显卡，`UP_IMAGE` 选择镜像；全部选项见 `./up_docker --help`。

多卡训练默认 DDP。使用 **FSDP2** 时，在 YAML 中设置 `runtime.distributed.strategy: fsdp2`，并以 `--backend dist --ng 2` 或更多卡启动。NVIDIA、沐曦均经平台训练接口调用厂商 PyTorch；配置、保存恢复和限制见 [FSDP2 使用指南](docs/platform/FSDP.md)。

## 如何调用

<p align="center">
  <img src="docs/platform/assets/platform-overview.svg" width="100%" alt="UP 平台：训练和推理共用设备及分类算子接口，分别绑定 NVIDIA、沐曦或新厂商实现" />
</p>

15 类算子：**矩阵、卷积、注意力、归一化、激活、池化/插值、损失、检测、可变形卷积、Focal Loss、点云、稀疏、通信、量化、剪枝**。

设备和算子是公共能力；训练/推理适配提供 AMP、DDP/FSDP2、优化器和导出等执行能力。用途由入口显式选择，训练中的 eval 继续使用训练后端。

**基础张量操作仍由厂商 PyTorch 提供。** 加减乘除、拼接、索引、reshape、归约、随机数和 autograd 没有逐一包装；新增厂商必须同时具备这些 PyTorch 设备能力。平台边界详见 [架构与开发指南](docs/platform/README.md)。

## 新增厂商

<p align="center">
  <img src="docs/platform/assets/vendor-onboarding.svg" width="100%" alt="厂商接入：准备厂商环境、生成模板、实现或绑定接口、注册并测试" />
</p>

准备厂商 PyTorch/SDK，生成模板，按父类签名绑定兼容实现或补齐缺失内核，再注册并测试。

```bash
python tools/new_platform_vendor.py --vendor acme --device privateuseone --output acme_platform
```

完整步骤和两种实现示例见 [厂商接入指南](docs/platform/ADDING_VENDOR.md)。

## 支持范围

| 平台 | 当前实现 | GPU 接口验收 |
| --- | --- | --- |
| NVIDIA | 厂商 PyTorch/CUDA、独立 `_nvidia_ops` 扩展 | 151 项通过、1 项部分通过、3 项依赖缺失 |
| 沐曦 C500 | 厂商 PyTorch/MACA、独立 `_metax_ops` 扩展 | 151 项通过、1 项部分通过、3 项依赖缺失 |
| 昇腾、寒武纪 | 接入模板，尚需补齐实现和真机验证 | 模板阶段 |

验收比较各厂商环境内的平台调用与直接 API 调用，检查前向、所需梯度及状态。Conv2d 保留真实批次的梯度超差记录；3 项缺少 Amba 剪枝所需依赖。[查看逐项结果和评测代码](docs/platform/GPU_COVERAGE.md)。

普通安装不自动编译 GPU 扩展；原生环境须显式选择构建厂商。可选工具链按所用配方安装，见 [环境与依赖](docs/platform/ENVIRONMENT.md)。

## 文档

| 指南 | 内容 |
| --- | --- |
| [在线交互教程](https://jingfengou.github.io/United-Perception/) | 改了什么 → 功能与实现 → 使用示例 → 新厂商扩展 |
| [完整运行示例](docs/platform/QUICKSTART_UNIFIED.md) | 从镜像构建到训练、验证、推理 |
| [环境与依赖](docs/platform/ENVIRONMENT.md) | 原生安装、厂商版本和可选 SDK |
| [通用镜像参考](docs/platform/UNIFIED_ENVIRONMENTS.md) | 隔离布局、启动器和目录模式 |
| [FSDP2 训练](docs/platform/FSDP.md) | 多卡分片、AMP、验证和保存恢复 |
| [架构与开发](docs/platform/README.md) | 模块边界、调用关系和维护检查 |
| [接口参考](docs/platform/INTERFACES.md) | 全部接口签名和行为契约 |
| [GPU 接口代码对照表](docs/platform/GPU_COVERAGE.md) | UP 引用、两家适配、评测代码和逐项真机状态 |
| [厂商接入](docs/platform/ADDING_VENDOR.md) | 模板、实现和注册步骤 |

教程可[直接在线打开](https://jingfengou.github.io/United-Perception/)，也可[下载 HTML](https://github.com/jingfengou/United-Perception/raw/refs/heads/codex/up-platform-clean/docs/platform/TUTORIAL.html) 后离线阅读。[教程源码](docs/platform/TUTORIAL.html) 随代码维护。

原版 UP 模型与数据配方见 [官方文档](https://modeltc-up.readthedocs.io/en/latest/index.html)。

## 来源与许可

基于 UP `52aaeb9e`，平台结构参考 LightX2V `19c19614`。遵循 [Apache-2.0](LICENSE)。[查看相对原版 UP 的全部改动](https://github.com/jingfengou/United-Perception/compare/main...codex/up-platform-clean)。
