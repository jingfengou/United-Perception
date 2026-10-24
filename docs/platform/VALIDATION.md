# 真机验证与复现

初轮日期：2026-10-23；问题跟进：2026-10-24。分支：`codex/up-platform-clean`。基线：UP `52aaeb9e`。

**结论：平台层和多项真机能力已实现，但本分支尚未达到 NVIDIA、沐曦“两家完整适配”。**
不能把算子通过、模型执行成功、模型数值等价、数据集精度四种结论混用。
未恢复旧多随机种子任务，未迁入 v1/v2/portable 数值策略，没有放宽容差。

最新变化见 [问题跟进](FOLLOWUP_20261024.md)：依赖错误处理已修正；沐曦显式 DNN 控制下四个模型的短程对照通过。
[Faster R-CNN 后续定位与修复](FRCNN_DETERMINISM_20261024.md) 已确认两家的 RoIAlign 原子反向非确定性；开启固定顺序反向后，两家同协议的训练/eval/恢复及独立推理对照通过。
[RoIAlign 后续性能优化](ROI_ALIGN_PERFORMANCE_20261024.md) 只改平台内核的线程映射；相同大规模输入在 C500 上快 3.72–3.90 倍、4090 上快 2.95–3.07 倍，同平台新旧算子与短程模型结果逐位一致。
[KITTI 评估器](KITTI_GEOMETRY_20261024.md) 经用户单独授权纠正原版几何错误，两家真机通过；这是有意改变错误指标值，不冒充与原版数值等价。

## 当前验收重点

本轮五模型、三条路径的单种子 native 真实数据完整预算已完成：**15 次训练、24,960 次更新**，最终指标、恢复及独立推理已审计，见 [三路径训练记录](ACCURACY_NATIVE_20261024.md)。
新版沐曦相对新版 NVIDIA 的绝对差为 ResNet 0.05、YOLO 0.46、ViT 1.80、Faster R-CNN 3.10、DETR 1.47 pp。
原版与新版 NVIDIA 初次单次对照的 YOLO 相差 +15.60 pp、ViT 相差 −4.15 pp；该记录不设通过线，不能宣称精度等价。

后续 [NVIDIA 同卡重复与算子定位](NVIDIA_NATIVE_REPEAT_20261024.md) 已完成 8 次完整训练及权重交换推理：确认原版 YOLO 的重复索引目标覆盖，以及两模型原生卷积权重梯度波动。四次跨代码权重推理的任务指标相同；两次同源重复仍不足以宣称长期精度统计等价。上面的旧单次结果保留，不被后续较高分数覆盖。

[原生确定性完整训练对照](NVIDIA_DETERMINISTIC_REPEAT_20261024.md) 已完成 8 次训练、14,960 次更新和 4 次权重交换推理。YOLO 四次 AP 均为 27.4236，ViT 四次 Top-1 均为 25.75%；各模型全部 loss、最终模型/优化器/调度器保存状态和指标一致。新版相对上一轮同卡 native 的平均训练耗时分别增加 2.82%、9.94%，详细显存和测量限制见报告。结论限于该固定预算/种子/硬件，不恢复旧多随机种子任务。

2026-10-24 按用户最新要求调整优先级：以沐曦相对 NVIDIA 的验证集精度差异为主要依据，多次运行的波动先记录，不单独作为跨厂商接入的阻塞项。
既有严格对照和失败证据保留，不能把新的验收重点解释为旧精度实验已通过。

- **主要指标**：相同模型和评估协议下的最终 Top-1、mAP 等任务指标，分别列出 NVIDIA、沐曦结果、有符号百分点差和绝对差。明显偏离时优先核对数据、配置、依赖环境及平台实现，不因浮点差异直接认定厂商内核有错。
- **对照条件**：固定数据划分、预处理、初始权重、种子、训练预算、优化器、精度模式及 checkpoint 选择规则；两家使用相同指标定义，记录厂商 Torch/SDK 的实际版本差异。
- **运行路径**：以正常 native 路径作为精度/性能对照主线，不为消除多次运行波动默认开启较慢的确定性实现。RoIAlign 固定顺序反向仍由显式 Torch 确定性开关选择，保留作需要重复性时的选项。
- **波动记录**：记录种子、权重和数据校验和、环境、确定性开关、每次实际指标；重复性与最终精度分别出具结论。大幅精度下降、非有限结果或接口语义错误仍需定位，不能统一归为正常波动。
- **精度通过线**：用户已确认先完整报告差异，暂不设通过线；保留全部实际结果，不据此自动宣称精度通过。算子原有 `rtol`/`atol` 不变；这些浮点容差不等同于 Top-1/mAP 的百分点差。
- **后续验证顺序**：同 checkpoint 的真实验证集推理用于检查推理差异；同起点、同预算的训练后评估用于检查训练影响。当前三步合成输入对照不能替代这两项，本次不启动或恢复旧多随机种子训练。

## 已有同平台兼容性验证协议

- 同平台比较原版与重构版，不把 NVIDIA 和沐曦互相当作逐位参考。
- 固定种子 131、CPU 初始权重、输入、原配置、优化器和 native 数值策略。
- 浮点比较固定 `rtol=1e-4, atol=1e-5`；非浮点、形状、dtype、参数名称严格检查。
- 保留原版独立重复进程。原版自身超容差时，该项不能被判为严格重构等价通过；它是重复性诊断，不单独判定跨厂商最终精度失败。
- 已归档的确定性模型阶段双方均启用 PyTorch 原生 deterministic 控制和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；没有替换 GEMM/DNN 算法。这描述已有实验，不要求后续正常训练强制开启。
- 源码冻结与命令在各阶段 `source.json`、`suite.json`；早期失败没有删掉或覆盖。

运行环境为 RTX 4090 与沐曦 C500，详细版本、依赖补丁见 [环境说明](ENVIRONMENT.md)。
厂商 PyTorch 小版本、稀疏库版本不同，因此不是跨厂商数值等价测试。

## 结果矩阵

| 范围 | NVIDIA | 沐曦 | 通过边界 |
| --- | --- | --- | --- |
| CPU 契约 | 65 项通过 | 54 项通过 | 隐藏 GPU；NVIDIA 为既有54项加3项重放、5项数值配置、3项状态哈希检查；沐曦未执行新增11项工具测试 |
| UP 专用算子 | 两种用途各 31 用例通过 | 两种用途各 31 用例通过 | 19 个操作的已列测试，不是全部 dtype/布局组合 |
| RoIAlign 确定性 AVG 反向 | 通过 | 通过 | FP32/64 独立 autograd 参考、数值梯度、状态、非默认 stream；局部性能代价另列 |
| 稀疏卷积 | 通过 | 通过 | SubM/Conv/Inverse、dense、反向、恢复 |
| UP SparseBasicBlock | 执行通过 | 执行通过 | 两次实际 UP block 执行，不是原版 block 对照 |
| MQBench | 通过 | 通过 | 校准、QAT 一步、状态恢复、ONNX 导出 |
| QDrop | 通过 | 通过 | 小网络 4 次重构迭代、恢复、直接/路由对照 |
| MSBench 2:4 | 通过 | 通过 | 真正的 mask、优化器更新、恢复；permutation=False |
| MSBench Amba | 阻塞 | 阻塞 | 原配方依赖的 FakeSparse 未找到 |
| 双卡 Torch 通信 | 通过 | 通过 | 集合通信、DDP、SyncBN、AMP/scaler、恢复；不是 linklink |
| 专用操作 ONNX | 7 用例通过 | 7 用例通过 | 图拓扑、symbolic 属性、initializer；不是部署引擎执行 |
| KITTI rotate_iou_eval | 修正后几何通过 | 修正后几何通过 | 训练/推理各验证；不与错误的旧参考宣称等价，见独立报告 |
| spring.linklink / 三个融合优化器 | 阻塞 | 阻塞 | 缺真实 SDK，没有普通实现替代 |
| 昇腾、寒武纪 | 模板 | 模板 | 没有真机验证，不标为支持 |

发布前追加的全任务 CLI 启动和原版 SegFormer attention 对照也在两家通过，
检查共享默认 Dropout、初始权重、RNG、GPU 前向与梯度。
这是修复导入时尚未选择平台的问题，不是新的模型精度实验。

算子、工具链与双卡证据：

- NVIDIA：[算子训练](validation/20261023/nvidia/results/frozen-stage12/native-training.json)、[算子推理](validation/20261023/nvidia/results/frozen-stage12/native-inference.json)、[工具链](validation/20261023/nvidia/results/frozen-stage12/toolchains/summary.json)、[QDrop](validation/20261023/nvidia/results/extra-toolchains-01/summary.json)、[Amba](validation/20261023/nvidia/results/extra-toolchains-02/summary.json)、[双卡](validation/20261023/nvidia/results/frozen-stage12/distributed/rank0.json)、[ONNX](validation/20261023/nvidia/results/exports-02/summary.json)。
- 沐曦：[算子训练](validation/20261023/metax/results/frozen-stage13/native-training.json)、[算子推理](validation/20261023/metax/results/frozen-stage13/native-inference.json)、[工具链](validation/20261023/metax/results/frozen-stage13/toolchains/summary.json)、[QDrop/Amba](validation/20261023/metax/results/extra-toolchains-01/summary.json)、[双卡](validation/20261023/metax/results/frozen-stage13/distributed/rank0.json)、[ONNX](validation/20261023/metax/results/exports-01/summary.json)。

## 模型兼容性

下表保留 **2026-10-23 初轮原生配置** 的真实结果，不用后来的不同启动环境覆盖失败。
2026-10-24 增加两个显式 MCDNN 控制开关后的对照在 [单独结果表](FOLLOWUP_20261024.md#显式-dnn-控制下的模型对照)。
再加入平台层 RoIAlign 修复后的最终 Faster R-CNN 结果见 [后续报告](FRCNN_DETERMINISM_20261024.md#真机验证)，不覆盖以下旧失败。

真实 UP `ModelHelper`、原配方网络、原损失和原优化器，batch=2、三次更新；每步训练内 eval，第二步后重建模型和优化器并恢复。
独立推理使用新进程与同一份原版训练 checkpoint，避免把训练漂移混入推理对照。
输入是合成数据，**不是完整 Runner、数据管线或 CIFAR/ImageNet/COCO/KITTI 精度实验**。

| 模型 | NVIDIA 训练/eval/恢复 | NVIDIA 独立推理 | 沐曦训练/eval/恢复 | 沐曦独立推理 |
| --- | --- | --- | --- | --- |
| ResNet18 | 通过 | 通过 | 未通过；原版重复也失败 | 未通过；原版重复也失败 |
| YOLOv5-S | 通过 | 通过 | 未通过；原版重复也失败 | 通过 |
| ViT-B16 | 通过 | 通过 | 未通过；原版重复也失败 | 通过 |
| Faster R-CNN R50 FPN | 未通过；原版重复也失败 | 通过 | 未通过；原版重复也失败 | 未通过；原版重复也失败 |
| PointPillar | 通过 | 通过 | 未通过；原版重复也失败 | 通过 |

上述最终阶段均执行成功，但“执行成功”不覆盖数值失败。当前证据不能把所有失败简单归因于平台重构或某个厂商 GEMM，也不能据此宣称不影响最终精度。
本轮保留 native；不为通过测试而引入旧数值修复、改模型算法或修改容差。

原版 UP 基线没有 DETR 模型/配方，本分支不添加新 DETR 算法后冒充“原版兼容”；Transformer 代表使用原版 ViT。

模型证据：[NVIDIA 四模型](validation/20261023/nvidia/results/frozen-stage11/models/summary.json)、[NVIDIA PointPillar](validation/20261023/nvidia/results/frozen-stage13/models/summary.json)、[沐曦四模型](validation/20261023/metax/results/frozen-stage11/models/summary.json)、[沐曦 PointPillar](validation/20261023/metax/results/frozen-stage14/models/summary.json)。

## 对照适配与失败记录

这些适配都显式记录，不能称为未经任何环境兼容处理的原版：

| 问题 | 处理 | 证据边界 |
| --- | --- | --- |
| Python/NumPy 已移除的别名 | 原版对照提供同义别名，重构代码使用同义内置类型 | 不改 dtype 或模型公式 |
| 沐曦 PSRoIPool scratch 非连续 | 平台仅为整数 scratch 加 contiguous；修后参考也明确加同一项 | 最初失败保留；不是原版未修改版本直接通过 |
| 沐曦 spconv inverse 参数位置、dense 布局差异 | 厂商层使用关键字，N...C 转 NC... | 原版布局参考也明确转换；梯度与恢复实测 |
| PointPillar 未注册 | 测试进程显式加载 det_3d | stage11 失败保留，未改模型 |
| SegFormer 默认 Dropout 在解析配置前创建 | 延迟解析并缓存同一实际模块 | 不包裹 Module，不改变参数键、共享身份或 RNG；startup-stage15 单独验证 |
| 原版点云导入即触发 Open3D/Numba-CUDA | 对照脚本延迟未使用的 PCD 读取和 KITTI JIT | stage13 沐曦导入失败保留；调用评估器时仍执行原算法/报原依赖错误 |
| MQBench/MSBench 现代 API 不匹配 | 固定源版本、兼容补丁、修改前后哈希 | 直接调用和路由调用使用同一依赖；不改策略 |
| Amba 缺少 USE_LINKLINK 定义 | 显式选择其原有 Torch dist 分支 | 随后仍因 FakeSparse 缺失而失败，不冒充 linklink |
| KITTI 编译候选无法匹配原版边界 | 当时沐曦公开接口拒绝，保留候选诊断；后获授权单独修正几何 | [旧失败](validation/20261023/metax/results/kitti-01.json) 与 [新指标验收](KITTI_GEOMETRY_20261024.md) 分开 |

初轮 KITTI 候选在 NVIDIA 的诊断也失败，当时 NVIDIA 使用原版 Numba；当前两家公开调用均使用单独验证的修正实现，原版 Numba 留作历史对照。
linklink 的公开库检索和所缺 API 单列在 [LINKLINK.md](LINKLINK.md)。
完整工具链验收还需要解决 Amba、linklink，以及扩充尚未执行的变体；本次完整 Runner 的单种子小数据训练已执行，精度差异仍需定位，不能代表标准大数据集收敛或完整训练 AMP 验收。NART、PCD/Open3D、2:4 permutation 和部署引擎未验证。
原版显式 CPU voxel 预处理不是新增 GPU 回退；可选 cumm 分支未做真机用例验收。

## 复现

先按 [环境说明](ENVIRONMENT.md) 准备厂商环境、固定依赖和本分支，显式编译对应扩展。
把原版归档到独立目录，不能用当前重构版作“原版”：

```bash
ROOT=/absolute/path/to/United-Perception
BASE=/absolute/path/to/up-original
mkdir -p "$BASE"
git -C "$ROOT" archive 52aaeb9ed27a490820696d7754d19f98cd6ecd11 up configs | tar -x -C "$BASE"
export PYTHONPATH="$ROOT:/path/to/dependencies/MQBench:/path/to/dependencies/msbench:${PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# 只使用实际分配的两张卡；输出目录必须是新的
python "$ROOT/tools/run_platform_validation.py" \
  --root "$ROOT" --baseline "$BASE" --vendor nvidia --devices 0,1 \
  --output /absolute/path/to/results/nvidia-new

# 沐曦在对应厂商容器内执行，不能使用 NVIDIA 的二进制或 wheel
python "$ROOT/tools/run_platform_validation.py" \
  --root "$ROOT" --baseline "$BASE" --vendor metax --devices 0,1 \
  --output /absolute/path/to/results/metax-new
```

该兼容性诊断入口不会自动跳过失败。针对当前已知失败，非零退出是预期的失败记录，不代表全套通过；也不能仅凭重复性诊断的退出码判定跨厂商数据集精度。
脚本的 `--phases` 可选择 `cpu native toolchains distributed models`；`--models` 可明确子集。
固定协议为三步兼容性，不能改文档称为完整训练预算。

在相同依赖环境、分配好的单卡上补测：

```bash
export CUDA_VISIBLE_DEVICES=0 DEFAULT_TASKS=cls:det
VENDOR=metax  # 或 nvidia
python "$ROOT/tools/compare_platform_exports.py" --vendor "$VENDOR" \
  --baseline "$BASE" --output /absolute/path/to/results/exports-new
python "$ROOT/tools/compare_platform_toolchains.py" --vendor "$VENDOR" \
  --cases qdrop amba --output /absolute/path/to/results/extra-new
```

Amba 需要先在固定 MSBench 依赖目录执行 `prepare_platform_dependencies.py --msbench-dist torch`；即使如此，已记录的 FakeSparse 缺口仍会导致失败。
测试目录不会被覆盖。源码、配置、输入/权重来源、原版重复、最终梯度/状态和 stdout 均保留。

## 证据与 CI

完整原始张量、checkpoint、源码归档和编译日志留在两台本任务目录，不进入 Git：

- NVIDIA：`/data2/oujingfeng/up-platform-clean-20261023`。
- 沐曦容器：`/workspace/up-run/up-platform-clean-20261023`；宿主绑定：`/mnt/afs/oujingfeng/up-hardware-metax-20261021/up-platform-clean-20261023`。
- 仓库保存小报告和 [NVIDIA 索引](validation/20261023/nvidia/index.json)、[沐曦索引](validation/20261023/metax/index.json)，列出原始文件路径、大小、SHA256、环境与完整源码清单。索引不是公开数据下载链接。
- [NVIDIA 发布审计](validation/20261023/nvidia/publication-final-audit.json)、[沐曦发布审计](validation/20261023/metax/publication-final-audit.json) 核对报告与待提交源码。删除未使用 import 的整理逐文件核验；两处延迟默认值改动由后续验证单独覆盖，没有未解释的运行时代码变化。
- 后续验证：[NVIDIA](validation/20261023/nvidia-startup/results/startup-stage15/summary.json)、[沐曦](validation/20261023/metax-startup/results/startup-stage15/summary.json)。各自独立索引保留最终源码与该阶段的文件校验，不覆盖早期失败。

`collect_platform_evidence.py` 打包小报告并计算大文件校验；`audit_platform_publication.py` 拒绝未解释的运行时代码变化。
阶段归档保留了各次工具版本，不能将早期失败目录或不同源码阶段混成同一次结果。

[CI](../../.github/workflows/platform.yml) 包含无 Torch/CUDA 安装、独立导入、模板/边界/文档漂移、Python 3.10/3.12 CPU 契约。
CPU CI 不代替真机验收；本地已执行公共 wheel/sdist 构建和隔离导入。

追加启动回归的复现入口为 `tools/validate_platform_startup.py --root "$ROOT" --baseline "$BASE" --vendor "$VENDOR" --output /absolute/path/to/results/startup-new`，使用相同环境及已分配的单卡。
