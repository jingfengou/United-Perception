# UP · 多厂商版本

从 [原版 UP 52aaeb9](https://github.com/ModelTC/United-Perception/tree/52aaeb9ed27a490820696d7754d19f98cd6ecd11) 重建。用 config 选择显卡，保留原训练、验证、独立推理和 ONNX 导出入口。

| 模型 | 配置 |
|---|---|
| ResNet50 | [res50.yaml](configs/cls/resnet/res50.yaml) |
| YOLO11s | [yolo11s.yaml](configs/det/yolo11/yolo11s.yaml) |
| YOLOX-S（原版 UP） | [yolox_small.yaml](configs/det/yolox/yolox_small.yaml) |
| ViT-B/16 | [vit_b16_224.yaml](configs/cls/vision_transformer/vit_b16_224.yaml) |
| DETR-R50 | [detr_r50.yaml](configs/det/detr/detr_r50.yaml) |

## 架构

```text
config → RuntimeBackend → 设备、AMP、通信、计算库策略
UP 模型 → 标准 Torch 操作 → 厂商 Torch / 计算库
        → UP 专用操作 → NativeOps → 厂商实现
```

ResNet/ViT 的模型计算保持原版。YOLO11s 使用官方 Ultralytics 结构和损失，DETR 使用随仓库保留的官方源码。普通 ReLU、卷积和矩阵乘法直接用 Torch，不逐层套包装。原版 43 个 C++/CUDA 文件保留，NVIDIA 显式编译使用。

## 安装与选择厂商

Python 3.10 及以上。每家使用独立环境，先安装该厂商提供的 Torch、匹配的 TorchVision、设备插件与驱动。然后在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -r requirements/yolo11.txt
python -m pip install -r requirements/export.txt
python -m pip install --no-deps -e .
```

这些命令不会主动安装 NVIDIA Torch。普通安装不编译 GPU 扩展。**YOLOX 的 NVIDIA GPU 训练需要原 IoU 扩展**，请先用匹配的 CUDA Toolkit 编译：

```bash
UP_BUILD_PLATFORM=nvidia python setup.py build_ext --inplace
# 无卡构建机还需显式设置 TORCH_CUDA_ARCH_LIST 为目标卡架构。
```

config 中只修改厂商选择，数据和配方保持相同：

```yaml
runtime:
  platform: nvidia   # metax_cuda / ascend_npu / cambricon_mlu
  fp16: false
```

切换时先进入该厂商环境，再启动新进程。`platform` 选择当前环境里的实现，不会把 NVIDIA Torch 换成沐曦 Torch。NPU/MLU 分别需要 `torch_npu` / `torch_mlu`。

## 训练、验证、推理、导出

先在对应 YAML 中设置数据路径。下面以 ResNet50 为例；其他模型替换配置路径即可。启动器默认单卡，多卡使用 `--ng 2`。

```bash
# 训练；可在命令末尾覆盖厂商
python -m up train --config configs/cls/resnet/res50.yaml \
  --opts runtime.platform metax_cuda

# 验证：沿用训练后端，加载指定权重
python -m up train --evaluate --config configs/cls/resnet/res50.yaml \
  --opts runtime.platform metax_cuda saver.auto_resume false saver.resume_model /data/resnet50.pth

# 独立图片/目录推理：使用 inference 用途
python -m up inference --config configs/cls/resnet/res50.yaml \
  --ckpt /data/resnet50.pth --img_path /data/example.jpg --vis_dir /data/predictions \
  --opts runtime.platform metax_cuda

# ONNX：不读取训练集；固定 batch=1 和输入尺寸
python -m up to_onnx --config configs/cls/resnet/res50.yaml \
  --ckpt /data/resnet50.pth --save_prefix /data/resnet50.onnx \
  --input_size 3x224x224 --verify
```

也可使用 `scripts/train.sh`、`evaluate.sh`、`inference.sh`、`to_onnx.sh`。检测导出保留原始预测，NMS 在图外；输出含义见 [架构说明](docs/platform/ARCHITECTURE.md#onnx-输出)。

## 训练配置

ImageNet / COCO 配置位于 [`configs/benchmarks/three_vendor/`](configs/benchmarks/three_vendor/)，包含 ResNet50、YOLOX-S 和 YOLO11s。配置固定为四卡，训练采用 AMP FP16，验证采用 FP32。数据准备、公共初始化、恢复和运行命令见 [训练指南](docs/platform/TRAINING.md)。

## 厂商扩展

设备、AMP、通信和计算库策略通过 `RuntimeBackend` 扩展；专用操作在 `up/platform/<vendor>/native_ops.py` 中绑定。标准 Torch 操作直接使用当前厂商环境。厂商不支持的接口明确报错，所需实现及注册方式见 [厂商接入指南](docs/platform/ADDING_VENDOR.md)。

[架构说明](docs/platform/ARCHITECTURE.md) · [训练指南](docs/platform/TRAINING.md) · [厂商接入](docs/platform/ADDING_VENDOR.md) · [接口浏览](docs/platform/HANDOFF.html)

原版 UP 保留其 [Apache-2.0 许可](LICENSE)。外部模型源码和依赖的来源、版本与各自许可见设计文档。
