# 完整 ImageNet / COCO 三路径单 epoch 对照

日期：2026-10-25。**状态：前三模型共 9 次单 epoch 对照中，C500 的 ResNet50、YOLOv5-S、ViT 共 3 次已完成训练、完整验证、独立推理及 CPU 指标复算。NVIDIA 仍等待旧作业首个检查点后接续，尚无三路径精度结论。DETR 暂缓。**

本轮与先前 CIFAR-10 / PennFudan 实验分别记录；旧多随机种子任务保持暂停。
不修改 `up/`、`up_platform/` 或原模型配置文件，不迁入 portable/v2，不放宽算子比较容差。

## 当前已审核的读数

以下是 **C500 单平台、第 1 个 epoch、原配方 EMA** 的读数。
两条 NVIDIA 路径尚未完成，不能据此判断新旧框架或跨厂商精度是否一致。

| 已完成模型 | 第 1 epoch EMA 指标 | 训练时间（分钟） | rank 0 峰值 allocated（GiB） |
| --- | --- | ---: | ---: |
| ResNet50 | Top-1 **0.2980%**，Top-5 **1.2140%** | 29.27 | 3.49 |
| YOLOv5-S | COCO bbox AP **0.7869**（0–100） | 10.09 | 12.87 |
| ViT-B/16 | Top-1 **3.3120%**，Top-5 **10.3860%** | 36.19 | 14.44 |

YOLO 原始 `bbox.AP` 为 `0.007869252330576044`，上表乘 100 显示 AP 点，不能把两种单位混用。
这些早期指标很低；本轮保留长配方的 warmup 和 EMA，且只有一轮，低值原因和跨平台差异仍待对照确认，不作收敛精度结论。
时间为 rank 0 训练循环，含取批次等待，不含模型初始化、检查点保存和独立推理；显存为 rank 0 的 PyTorch 分配器峰值，不是两卡总显存。

- ResNet50：10,010 个调度步，10,005 次成功优化器更新，原生 AMP 跳过 5 次。
- YOLOv5-S：1,849 个调度步及优化器更新。
- ViT-B/16：10,010 个调度步，10,006 次成功优化器更新，原生 AMP 跳过 4 次。
- 三模型均核对公共初始权重、冻结计划/源码、完整检查点与各 rank 文件哈希；训练内 eval 和新进程独立推理的完整预测文件逐字节相同。
- CPU 从保存预测重新计分，两种分类模型的验证集均为 50,000 张，路径/标签无缺失或重复；COCO 在全部 5,000 张上重新执行原版 `pycocotools` 评估，包含无预测图片。重算值与报告逐项一致。

原始指标、环境、成功更新数、预测与恢复校验值见 [部分结果汇总](validation/20261025/one-epoch/partial-metrics.json)，
[ResNet50 审核](validation/20261025/one-epoch/new_metax-resnet50-audit.json)、[YOLO 审核](validation/20261025/one-epoch/new_metax-yolov5s-audit.json)和 [ViT 审核](validation/20261025/one-epoch/new_metax-vit_b16-audit.json)。
其中 `comparisons` 明确标记为等待三条路径，不把部分读数标成对照通过。

C500 的三模型队列守卫已确认最后一次独立推理和检查点完整，结束父控制器并阻止 DETR 接续。
因此旧父状态文件可能仍显示最后一次 inference 的 `running`；完成依据是队列守卫、各模型结果和 CPU 审计，不能据此重复启动。
独立的[平台算子检查](MODEL_OPERATOR_PARITY_20261025.md)也已完成 C500 部分，不能把其局部误差直接解释为以上精度读数的原因。

## 比较什么

| 模型 | 数据 | 本轮执行预算 | 每路径调度步数 | 每卡 batch / 两卡全局 batch | 精度模式 |
| --- | --- | ---: | ---: | ---: | --- |
| UP ResNet50 bag-of-tricks | ImageNet-1K | 1 epoch | 10,010 | 64 / 128 | 原生 AMP FP16 |
| UP ViT-B/16 | ImageNet-1K | 1 epoch | 10,010 | 64 / 128 | 原生 AMP FP16 |
| UP YOLOv5-S | COCO2017 | 1 epoch | 1,849 | 32 / 64 | 原生 FP32 |
| 同一官方 DETR R50 适配器 | COCO2017 | 暂缓，保留原计划 | — | 2 / 4 | 原生 FP32 |

每个模型比较原版 UP/NVIDIA、新版 UP/NVIDIA、新版 UP/C500，种子均为 **131**。
原版和新版 NVIDIA 顺序使用同一对 RTX 4090；C500 使用两张卡，保持相同全局 batch。
每个模型只生成一次公共初始权重，再分发给三条路径。报告第 1 个 epoch 结束后的指标，保留原配置中启用的 EMA，不挑最好一次。
当前先交付前三个模型的 **9 次单 epoch 训练**及对应完整验证、恢复和独立推理，使用完整训练集和验证集。
报告原版/新版 NVIDIA 的差异、新版 NVIDIA/C500 的差异，以及实际耗时和显存；**不代表完成原配方训练或达到收敛精度**。
不等待 DETR 才汇总前三个模型，也不自动恢复 DETR。

这是用户确认的**对应 UP 模型**，不是逐项复现 model-verify：原版 UP 没有 YOLO26 或原生 DETR，
ViT 和 ResNet 的训练配方也不同。不能用本轮结果直接核对该仓库表中的精度。

### 预算变更与旧任务

64 步真实图片诊断中，双 4090 的 FP32 DETR 预热后中位耗时约 **3.21 秒/步**。
原计划为 ResNet50 200 epochs，其余三模型各 300 epochs。用户先明确四模型均只测 1 epoch，随后要求先不管 DETR、先得出前三模型的结论。
因此本轮只执行各原配方的第 1 轮，**保留原学习率、warmup、优化器、增强及 EMA 时间轴，不把长配方压缩成新的一轮配方**。
动态尺寸、初始化和数据吞吐影响很大，上述短测不作为完成时间承诺。

数据审计完成后，旧三模型长预算控制器已于 2026-10-25 09:52（北京时间）启动 NVIDIA ResNet50，C500 随后启动。
变更预算后，仅阻止本任务控制器启动后续长作业，让已有训练发布首个完整检查点，再向本任务 torchrun 发送 SIGINT 并释放设备。
旧计划、日志、检查点和停止原因单独保留；它们不会重新标成新阶段的完整结果。
新阶段从公共初始权重独立执行，输出在 `one-epoch/`，不覆盖旧 `full-cohort/`。

[新计划](validation/20261025/one-epoch/plan.json)的 SHA256 为
`5e640ab5765222062df3f5afa14c91bf4f88be19179b70f9f5b3b0b46c996426`。
旧长预算计划 SHA256 为 `9a3e53beba98af7bd2daff88acbb53baa12d4d5aec4574ee5d4b9fd5fdd27cd1`，保持原样。

四模型父计划及已启动训练保持原样；[当前执行选择](validation/20261025/one-epoch/selection.json)单独固定前三个模型，
共 **65,607 个调度步**，不把 AMP 跳过的批次算作成功更新。
两台机器已各启动一个队列守卫，等最后一个选定 ViT 的独立推理运行完，再结束本任务父控制器、释放设备锁，阻止后续 DETR 自动启动。
当前训练和推理不受这项队列选择打断；[部署回执](validation/20261025/one-epoch/selection-receipt.json)记录选择哈希与真实进程。
原控制器仍关联四模型父计划，因此本轮以选定三模型的完成回执及逐项结果审计判断完成，不能靠父控制器退出推断全部 12 次通过。

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
- 每 5 epochs 以及最终执行预算结束时验证；本轮因此在第 1 个 epoch 完整验证并保存检查点。
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

此前两边匹配虚拟环境的 **102 项 CPU 契约测试均通过**。新增采样器用例先在旧驱动复现失败，再在修复后通过。
单 epoch 阶段新增执行预算契约，两家均为 **105 项通过、零跳过**，见
[NVIDIA](validation/20261025/one-epoch/cpu-nvidia.xml)和 [C500](validation/20261025/one-epoch/cpu-metax.xml)。
只修改外部 `full_accuracy_recipe.py`、`full_accuracy_train.py` 的预算检查、循环终点和报告；
两家快照核对确认 `up/`、`up_platform/`、`configs/` 和已有编译二进制不变。
无 Torch/CUDA 的独立环境可构建 wheel/sdist、安装并导入公共平台包。
[源码核对](validation/20261025/full-data/frozen-baseline-audit.json)确认：
原版 618 个相关文件与 `52aaeb9e` 无内容差异；新版 781 个运行文件与本地一致，仅归一化 CRLF/LF。

正式训练前，双卡预检查分别执行不间断训练、epoch 内中断/恢复、训练内 eval、独立分布式推理。
旧阶段的进程执行成功不代表恢复协议通过：DETR 的图片顺序审计已发现并保留上述失败；
新阶段对每个预检查批次记录输入 Tensor 校验值，同时检查跨路径输入一致及恢复前后顺序/增强一致。
预检查仅使用真实图片小视图，保留完整类别数；**短程成功或 AP 为零都不是完整精度结论**。
两边预检查、逐图数据审计、公共初始权重与配置/输入核对已完成，随后冻结独立的单 epoch 正式计划；没有拿预检查计划代替正式计划。

2026-10-25 更新：[preflight-08 独立审计](validation/20261025/full-data/preflight-audit.json)已确认四模型、三条路径的 **48 项执行全部完成**。
每个 rank 的全部预检查输入在三条路径之间相同；中断恢复后的图片顺序与增强结果也与不间断运行逐批相同。
DETR 为每路径 36 步，其余模型各 8 步，另做相同预算的中断/恢复对照；合计 360 个调度步、348 次成功优化器更新。
ResNet50 的原生 AMP 在每条训练轨迹中均跳过 2 次更新，已如实计数。
这确认执行和恢复协议可用，**没有据此断言新旧最终权重数值等价或完整数据精度一致**。

### 冻结工具的换行格式

运行源码清单只归一化 CRLF/LF，工具计划则检查实际文件字节。
Windows 留存的部分旧工具是 CRLF；Linux 克隆后不能简单修改计划中的哈希来绕过检查。
历史 training-07 的[逐文件字节索引](validation/20261025/full-data/frozen-tool-bytes.json)对应 Git 提交 `7da1be8`，保留不变。
本轮 training-08 使用[新的逐文件字节索引](validation/20261025/one-epoch/frozen-tool-bytes.json)，同时记录原始哈希、LF 内容哈希和换行格式。
可从相同版本的 Git 内容还原到独立目录，原模型文件不需要修改：

```python
import hashlib, json
from pathlib import Path
specs = json.loads(Path("docs/platform/validation/20261025/one-epoch/frozen-tool-bytes.json").read_text())
target = Path("frozen-tools")
target.mkdir(exist_ok=False)
for name, spec in specs.items():
    raw = (Path("tools") / name).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == spec["lf_normalized_sha256"]
    raw = raw if spec["line_endings"] == "LF" else raw.replace(b"\n", b"\r\n")
    assert hashlib.sha256(raw).hexdigest() == spec["sha256"]
    (target / name).write_bytes(raw)
```

下例的 `FROZEN_TOOLS` 指向该目录。源快照中也采用这 13 个工具形成源码清单；
原版快照包含原版 `up/`、`configs/` 和 `tools/`，新版再包含 `up_platform/`，避免把其他实验脚本混入同一清单。

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

每条路径分别执行以下命令；`LANE` 为 `original_nvidia`、`new_nvidia` 或 `new_metax`。
`SOURCE` 指向对应冻结源码，不能把新版源码当成原版路径。
`ONE_EPOCH_PLAN` 指向上述公开单 epoch 计划；`execution_epochs: 1` 是正式执行预算，不能改用 `--probe-steps`。
该计划固定本轮数据绑定 SHA256；换机器复现需要先完成同样的逐图审计，再另存新的路径绑定及计划，不能伪造旧绑定哈希。

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc-per-node=2 \
  "$FROZEN_TOOLS/full_accuracy_train.py" --source "$SOURCE" --data data-binding.json \
  --plan "$ONE_EPOCH_PLAN" --model resnet50 --lane "$LANE" \
  --initial "$COMMON_INITIAL" --output "$RESULT/train"
# 经检查点完整性审计后，原命令增加 --resume 恢复；不换计划、种子、数据或输出目录。
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc-per-node=2 \
  "$FROZEN_TOOLS/full_accuracy_train.py" --source "$SOURCE" --data data-binding.json \
  --plan "$ONE_EPOCH_PLAN" --model resnet50 --lane "$LANE" \
  --initial "$COMMON_INITIAL" --mode inference --checkpoint "$RESULT/train" \
  --output "$RESULT/inference"
```

按实际获配编号替换 `CUDA_VISIBLE_DEVICES`，C500 必须使用厂商环境。
本轮只对 `resnet50`、`yolov5s`、`vit_b16` 替换 `--model`、对应初始权重和独立输出目录，步数必须符合上表。
执行前读取 `selection.json` 的 `models_to_run`；DETR 留待后续用户指示，不执行四模型父计划中的全部模型。
本轮 [证据索引](validation/20261025/one-epoch/index.json)记录计划、工具字节、源码及测试；完成指标和失败记录随真实进度追加。
当前只有上述已审核的 C500 部分读数，尚未产生三路径单 epoch 的完整指标表，不宣称新旧或跨厂商精度一致。
