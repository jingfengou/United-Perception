# 完整 ImageNet / COCO 三路径训练对照

日期：2026-10-25。**状态：数据准备和最终双卡预检查进行中，完整预算训练尚未启动，最终精度待测。**

本轮与先前 CIFAR-10 / PennFudan 实验分别记录；旧多随机种子任务保持暂停。
不修改 `up/`、`up_platform/` 或原模型配置文件，不迁入 portable/v2，不放宽算子比较容差。

## 比较什么

| 模型 | 数据 | 完整预算 | 每卡 batch / 两卡全局 batch | 精度模式 |
| --- | --- | ---: | ---: | --- |
| UP ResNet50 bag-of-tricks | ImageNet-1K | 200 epochs | 64 / 128 | 原生 AMP FP16 |
| UP ViT-B/16 | ImageNet-1K | 300 epochs | 64 / 128 | 原生 AMP FP16 |
| UP YOLOv5-S | COCO2017 | 300 epochs | 32 / 64 | 原生 FP32 |
| 同一官方 DETR R50 适配器 | COCO2017 | 300 epochs，资源安排待确认 | 2 / 4 | 原生 FP32 |

每个模型比较原版 UP/NVIDIA、新版 UP/NVIDIA、新版 UP/C500，种子均为 **131**。
原版和新版 NVIDIA 顺序使用同一对 RTX 4090；C500 使用两张卡，保持相同全局 batch。
每个模型只生成一次公共初始权重，再分发给三条路径。最终 epoch 为主指标，保留原配置中启用的 EMA，不挑最好一次。

这是用户确认的**对应 UP 模型**，不是逐项复现 model-verify：原版 UP 没有 YOLO26 或原生 DETR，
ViT 和 ResNet 的训练配方也不同。不能用本轮结果直接核对该仓库表中的精度。

### DETR 的资源问题

64 步真实图片诊断中，双 4090 的 FP32 DETR 预热后中位耗时约 **3.21 秒/步**。
当前全局 batch 4 对完整 COCO 的 300 epochs 约为 8,871,600 步，按短测粗估单条 NVIDIA 路径约 330 天，
尚未计入评估和保存。动态尺寸的算法选择也会影响短测，这不是可靠的完整数据 ETA，需要补充稳定态吞吐或调整资源。
已向用户确认是否等待更合适的资源或另测统一 AMP 配置；未得到新选择前，不自动启动这个长期 DETR 作业。
其余三模型的准备和既定预算不依赖该选择。

## 数据准备与校验

- ImageNet：读取服务器上有权限访问的完整 Parquet 副本，294 个训练分片、14 个验证分片；
  共 **1,281,167 / 50,000** 张、1,000 类。逐行保留原编码图片字节和标签，不重编码。
  没有绕过受保护 ILSVRC 目录权限，也尚未与该目录逐文件比较。
- COCO：完整原版 ZIP，**118,287 / 5,000** 张、原始 80 类。
  因官方源下载过慢，使用固定版本的公开镜像在本地下载，再上传两台服务器。
  三个 ZIP 均符合原始对象的 MD5，另记录 SHA256。来源见
  [COCO 下载页](https://cocodataset.org/#download)及
  [固定镜像版本](https://huggingface.co/datasets/pcuenq/coco-2017-mirror/tree/a4cd8b69bd45a35e9a0a9c8692f3a7f4f23321fe)。
- `READY.json` 只表示准备完成。正式训练另要求逐图 SHA/CRC/大小检查，以及每张图片实际 RGB 解码成功。
- 两台服务器的机器路径可以不同；必须有相同的图片、标签、划分、初始权重和路径无关的数据身份。
  正式计划再固定各机器绑定文件的 SHA256，训练时拒绝变更后的绑定。

| COCO ZIP | MD5 | SHA256 |
| --- | --- | --- |
| annotations_trainval2017.zip | `f4bbac642086de4f52a3fdda2de5fa2c` | `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` |
| val2017.zip | `442b8da7639aecaf257c1dceb8ba8c80` | `4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05` |
| train2017.zip | `cced6f7f71b7629ddf16f17bbcfab6b2` | `69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929` |

UP 默认 COCO 数据集会排除空标注图片。实验外部数据视图显式包含原始全部图片，
包括空标注及仅 crowd 图片；不通过“图像数量大致相同”冒充完整 COCO。

## 模型和运行协议

复现工具位于 `tools/full_accuracy_{recipe,data,detr,train,bind}.py`。
它们是同一份外部实验驱动，不是模型或平台实现的修改：

- 三条路径均调用原生 Runner 的前向、反向、更新、EMA、LR 和评估接口。
- 工作进程按种子、epoch、rank 初始化，移除原先与 UUID 相关的数据进程随机种子；
  模型 CPU/GPU RNG 与数据进程 RNG 分开保存。**这不等同于原样运行 UP 默认 CLI 的数据/RNG 循环。**
- 每条路径启用 native、`cudnn.benchmark=True`、deterministic algorithms=False、matmul/convolution TF32=False。
  允许原生波动，记录差异，不把一次运行当成统计等价证明。
- 分类学习率按本实验选择的名义参考全局 batch 512 线性缩放，warmup 保持相同名义样本预算。
  原 YAML 未指定 world size；512 是本轮显式实验选择，不是对上游硬件配置的断言。
- DETR 固定官方源码 `29901c51d7fe8712168b8d0d64351170bc0f83e0`，
  保留官方 Transformer、matcher、criterion、91 类槽及 no-object。
  只转换 UP 的类别编号、坐标与结果格式；保留相同 aspect-ratio grouping。
  数据增强沿用 UP 的既有检测输入流程，并非逐项复现官方 DETR 的数据增强配方。
  使用 ImageNet backbone 初始化加随机检测器，不从已训练 COCO 检测器起步。
  外部官方 DETR 内部 PyTorch 操作保持原样；这不证明每个操作均经过 UP 平台类别接口。
- 每 5 epochs 以及最终预算结束时验证，每 epoch 保存检查点。
  两个滚动槽保存模型、优化器、LR、EMA，以及各 rank 的 buffer、RNG、scaler；完整性校验成功后才发布恢复索引。
  不永久保留每个 epoch 的大权重，所有训练记录和失败尝试仍保留。
- AMP 跳过的优化器更新单独计数；不能把调度批次数一律称为成功执行的 `optimizer.step()` 次数。
- 训练内 eval 检查模型/EMA/scaler 未被更新，并恢复模块模式。
  独立推理使用新进程和推理用途，恢复各 rank 的 BN/EMA buffer；不将它们误替换为 rank 0 的 buffer。

## 已处理的启动问题

| 问题 | 处理及证据边界 |
| --- | --- |
| 官方下载源慢，原共享 COCO 无读取权限 | 按用户建议本地下载、校验后上传；不改共享目录权限 |
| 首次续传要求目标文件已存在 | 在本任务目录创建零长接收文件，再做可恢复传输；保留失败记录 |
| 新 C500 容器缺 `libGL.so.1` | 只在新任务容器安装匹配的 GL 包；不替换厂商 Torch |
| C500 默认 Python 缺 UP 依赖 | 使用已验证的 `/workspace/up-run/venv/bin/python`；保留导入失败，不记为训练失败 |
| 原版与新版 YOLO AMP 均出现 Half/Float 索引赋值错误 | 三条路径统一 FP32，不给模型添加 dtype 修补 |
| 两个 NVIDIA 版本的单卡 FP32 YOLO batch 64 均 OOM | 两卡各 32，保持全局 batch 64；不用隐藏的小 batch 冒充同配置 |
| 新 C500 容器没有厂商编译缓存/日志目录 | 只创建本任务容器内相应可写目录；保留初始化耗时，不改计算内核 |
| 首次修正解释器的启动被尚未释放的预检查锁拦截 | 未重复占用 GPU；保留锁失败记录，原进程退出后才重新启动 |
| DETR 在 epoch 内恢复时复用第 0 轮分组缓存 | 三条路径均在第 34 步起出现不同图片顺序；只修复外部续训驱动的缓存失效，并新增真实原版采样器回归测试；以新阶段重新验证，不覆盖旧结果 |

两边当前普通依赖匹配：NumPy 1.26.4、SciPy 1.16.0、Pillow 11.2.1、OpenCV 4.11.0.86、
pycocotools 2.0.10、ONNX 1.17.0、PyYAML 6.0.3、einops 0.8.1、easydict 1.13。
NVIDIA 为 Python 3.12.3 / Torch 2.10.0+cu126 / TorchVision 0.25.0+cu126；
C500 为 Python 3.12.11 / Torch 2.10.0+metax3.8.1.0 / TorchVision 0.25.0+metax3.8.1.0。
Python 补丁版本、厂商 SDK、GPU 和第三方内核不同，不能声称环境完全相同。

## 验证状态与复现入口

两边匹配虚拟环境的当前 **102 项 CPU 契约测试均通过**。新增采样器用例先在旧驱动复现失败，再在修复后通过。
无 Torch/CUDA 的独立环境可构建 wheel/sdist、安装并导入公共平台包。
[源码核对](validation/20261025/full-data/frozen-baseline-audit.json)确认：
原版 618 个相关文件与 `52aaeb9e` 无内容差异；新版 781 个运行文件与本地一致，仅归一化 CRLF/LF。

正式训练前，双卡预检查分别执行不间断训练、epoch 内中断/恢复、训练内 eval、独立分布式推理。
旧阶段的进程执行成功不代表恢复协议通过：DETR 的图片顺序审计已发现并保留上述失败；
新阶段对每个预检查批次记录输入 Tensor 校验值，同时检查跨路径输入一致及恢复前后顺序/增强一致。
预检查仅使用真实图片小视图，保留完整类别数；**短程成功或 AP 为零都不是完整精度结论**。
正式训练必须等两边预检查、逐图数据审计、公共初始权重与配置/输入核对完成，再冻结正式计划；不会拿预检查计划代替正式计划。

数据准备和审计：

```bash
python tools/prepare_imagenet_parquet.py --help
python tools/prepare_coco2017.py --help
python tools/full_accuracy_bind.py \
  --imagenet "$IMAGENET_ROOT" --coco "$COCO_ROOT" \
  --detr-source "$DETR_SOURCE" \
  --detr-source-sha256 954c6297faa72af4b59719a42c850d769d8276fcd99a7c704cb5e34f5e5eb5fb \
  --backbone "$BACKBONE" --anchors "$ANCHORS" --output data-binding.json
```

正式计划冻结后，每条路径分别执行以下命令；`LANE` 为 `original_nvidia`、`new_nvidia` 或 `new_metax`。
`SOURCE` 指向对应冻结源码，不能把新版源码当成原版路径。

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc-per-node=2 \
  tools/full_accuracy_train.py --source "$SOURCE" --data data-binding.json \
  --plan full-plan.json --model resnet50 --lane "$LANE" \
  --initial "$COMMON_INITIAL" --output "$RESULT/train"
# 经检查点完整性审计后，原命令增加 --resume 恢复；不换计划、种子、数据或输出目录。
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc-per-node=2 \
  tools/full_accuracy_train.py --source "$SOURCE" --data data-binding.json \
  --plan full-plan.json --model resnet50 --lane "$LANE" \
  --initial "$COMMON_INITIAL" --mode inference --checkpoint "$RESULT/train" \
  --output "$RESULT/inference"
```

按实际获配编号替换 `CUDA_VISIBLE_DEVICES`，C500 必须使用厂商环境。
正式计划、公共初始权重校验值、机器数据绑定索引、完成指标和失败记录会随实际完成状态追加；当前尚未产生完整训练精度表。
