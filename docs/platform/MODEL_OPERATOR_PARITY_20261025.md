# 三模型平台算子跨厂商对照

日期：2026-10-25。**状态：98 个公共样本已完成两家重放和跨厂商比较。34 个训练样本中 19 个超出原容差，64 个训练内 eval/独立推理样本均在容差内；没有把执行成功当作数值全部通过。两家隐藏 GPU 的 CPU 契约各 114 项通过。**

本报告覆盖实际采样的 16 类操作；包含设备、专用操作、通信、工具链及所有未测成员的 **156 项完整账本**见[硬件接口逐项清单](OPERATION_AUDIT_20261025.md)。

目标模型为 ResNet50、YOLOv5-S、ViT-B/16，比较新版 UP 平台层在 NVIDIA 与沐曦上的结果。
模型训练现已按用户指示切换为[每条路径 500 steps](FIXED_STEPS_20261025.md)；DETR 暂缓。算子诊断使用独立进程，不修改 `up/`、`up_platform/`、模型算法、配方或原容差，
也不把观察器挂到精度训练中。旧[单 epoch 协议](IMAGENET_COCO_PARITY_20261025.md)与已有证据保留。

## 跨厂商结果

NVIDIA 在 9 次 500-step 队列结束后，完成 **18 个观察、重放和 CPU 比较子作业，全部退出码为 0**。
共同样本来自已冻结的 C500 捕获，保留相同输入、stride、权重、状态和精度上下文。
比较的 98 个样本包含 **16 类实际调用接口**，每类只采样本计划规定的形状布局变体。

| 模型 | 训练样本 / 超差 | 训练内 eval 样本 / 超差 | 独立推理样本 / 超差 |
| --- | ---: | ---: | ---: |
| ResNet50 | 10 / 6 | 10 / 0 | 10 / 0 |
| YOLOv5-S | 13 / 4 | 11 / 0 | 11 / 0 |
| ViT-B/16 | 11 / 9 | 11 / 0 | 11 / 0 |
| **合计** | **34 / 19** | **32 / 0** | **32 / 0** |

其中 17 个超差样本属于非随机操作，另 2 个是训练中启用的 Dropout。
“非随机操作”不代表其 native GPU 归约具有确定性。
64 个 eval/推理样本的全部比较字段均在原容差内，已保存 Tensor 的最大绝对差为 **9.05990601e-6**，并非所有字段逐位相同。

下表按“模型 × 接口”聚合实际超差样本；最大值来自同操作数重放，不能换算成任务精度百分点。
分类保持 AMP FP16、YOLO 保持 FP32；参数梯度通常为 FP32，输出/输入梯度的实际 dtype 逐项保存在证据中。

| 模型 / 接口 | 超差样本数 | 超差字段及其最大绝对差 |
| --- | ---: | --- |
| ResNet / Conv2d | 2 | FP16 前向 `0.00390625`；输入梯度 `0.001953125`；权重梯度 `2.5` |
| ResNet / BatchNorm2d | 2 | FP16 前向 `0.00390625`；输入梯度 `0.0009765625`；参数梯度 `0.0001344681` |
| ResNet / Linear | 1 | FP16 前向 `0.0078125`；输入梯度 `0.000244140625`；权重梯度 `0.001953125` |
| ResNet / log_softmax | 1 | 仅输入梯度 `3.05175781e-5` |
| YOLO / Conv2d | 2 | 仅权重梯度，分别 `0.0012359619`、`0.0015869141`；前向和输入梯度在容差内 |
| YOLO / BatchNorm2d | 2 | 仅参数梯度，最大 `0.0041999817`；前向和输入梯度在容差内 |
| ViT / Conv2d | 1 | FP16 前向 `0.0009765625`；输入梯度 `0.000244140625`；权重梯度 `0.09375` |
| ViT / LayerNorm | 1 | 仅参数梯度，最大 `3.43322754e-5` |
| ViT / Linear | 2 | FP16 前向 `0.001953125`；输入梯度 `0.00048828125`；权重梯度 `0.0625` |
| ViT / matmul | 2 | FP16 前向 `0.0078125`；输入梯度最大 `0.0078125` |
| ViT / Dropout | 2 | 前向分别 `5.73008347`、`0.02853608`；输入梯度 `0.54982823`，保留真实跨设备随机掩码差异 |
| ViT / log_softmax | 1 | 仅输入梯度 `6.103515625e-5` |

19 项的输入形状、stride、dtype、具体前向/梯度字段和最大误差见[逐项发现](validation/20261025/model-ops/nvidia-complete-02/findings.json)，
全部通过及失败样本见[原始比较摘要](validation/20261025/model-ops/nvidia-complete-02/summary.json)与[六份完整比较](validation/20261025/model-ops/nvidia-complete-02/comparisons)。
反向上游值由固定 CPU 数列生成后转至 GPU，**不是整网损失梯度**；例如卷积权重梯度的绝对差 `2.5` 不能直接用于评价任务精度。

### 平台入口与厂商底层分别表现如何

两家**各自全部 98 个前向输出均与直接调用本厂商 Torch 的输出逐位相同**。
因此本次公共输入上的跨厂商前向差异在直接厂商 Torch 中同样存在，未观察到平台封装额外改变这些前向结果。

反向另有原生波动：NVIDIA 有 2 个 YOLO Conv2d 权重梯度样本，C500 有 5 个 Conv2d 权重梯度样本，
在“平台 vs 直接 Torch”及同平台三次重复中都超容差；各厂商这两种检查的失败样本集合完全相同。
其余样本的直接参考和同平台重复未超差。C500 的原始 5 项读数保留在下文，不被跨厂商结果覆盖。

这些控制支持“差异包含厂商原生计算与跨设备随机流差异”，不能据此宣布每个反向差异都是内核错误，
也未证明某个算子解释了全部最终训练精度差。Dropout 的同 seed 跨厂商掩码不同本身不构成实现错误。
本轮没有改平台数值策略、模型算法或容差，也没有新增 CPU 模型代算。

## 测试边界

动态观察三模型实际调用的平台函数及层工厂创建的模块，覆盖一个真实训练批次的前向/反向、训练内 eval 和新进程独立推理。
保持完整每卡 batch、真实图片与标签、公共初始权重、seed 131 和 native 配置。
诊断不执行优化器更新，不计算数据集精度；训练内 eval 保留训练用途，独立推理明确选择推理用途。

下面是由当前模型结构确定的候选接口，最终覆盖清单以真机调用记录为准：

| 模型 | 重点候选平台操作 |
| --- | --- |
| ResNet50 | Conv2d、BatchNorm2d、ReLU、池化、Linear、分类后处理/损失使用的 log_softmax 等 |
| YOLOv5-S | Conv2d、BatchNorm2d、SiLU、池化、上采样、sigmoid、实际分类/目标损失和 NMS |
| ViT-B/16 | patch 卷积、Linear、matmul、LayerNorm、GELU、softmax、Dropout |

每个“平台操作 × 执行阶段”默认保存最先遇到的两个不同输入形状/布局样本，并记录全部调用数、观察到的变体数及未采样变体。
因此这是**实际模型路径中的代表算子数值验证**，不是全部形状、空输入、dtype 或全部 156 项接口验收。
未触发的检测、点云、稀疏、量化、剪枝等接口单列为未覆盖，不能因为三个模型跑通就标记通过。
调用记录中的 `setup/*` 只说明构造调用，数值覆盖以实际保存并成功重放的 `cases` 为准；仅构造而未执行的层不标为通过。
设备、AMP、通信、优化器等状态接口仍由既有契约和双卡训练证据分别说明，不冒充为本轮单算子数值重放。
ViT 的普通 Tensor 随机操作（例如 DropPath 内部随机采样）不因此获得平台算子通过结论。

## 怎样区分差异来源

| 比较 | 控制条件 | 能回答什么 |
| --- | --- | --- |
| NVIDIA 平台调用 vs NVIDIA 原生 Torch | 同操作、输入、权重/状态、上游梯度、精度上下文及外部种子 | 观察平台入口与直接厂商 Torch 的差异 |
| 沐曦平台调用 vs 沐曦原生 Torch | 同上 | 观察沐曦平台入口是否引入额外变化 |
| NVIDIA 平台调用 vs 沐曦平台调用 | 读取同一份捕获输入、权重/状态和确定生成的上游梯度 | 隔离模型上游误差累积后的算子差异 |
| 同平台三次重放 | 每次恢复初始状态并使用相同种子 | 记录原生重复波动，不强制逐位确定性 |

公共操作数先从较早完成的 C500 三模型路径捕获，然后原样分发给两家重放；两边仍分别使用自己的明确后端。
NVIDIA 也会观察本机三个模型的完整诊断批次，核对实际调用覆盖。输入来源与数值比较参考方不是同一个概念。
本轮缩短训练预算后沿用已授权的双 4090，不等待 A100/A800 迁移；保留原诊断计划与同一公共输入，
仅将外部等待条件改为 500-step 队列完成。旧等待器的退役另有记录，不重跑 C500 捕获来替换已有波动。

前向输出、所需输入/参数梯度、BN 等运行状态分别比较，输入 stride、storage offset、dtype 及实际 autocast 上下文随捕获保存。
对无需梯度的 NMS 等操作保持原契约，索引使用精确比较。没有对应 `torch_impl` 的 UP 专用算子会明确注明缺少直接 Torch 参考，
仍做跨厂商比较，不用同名但不同语义的 TorchVision 操作替代。

浮点比较固定 **rtol=1e-4、atol=1e-5**，没有放宽。相同外部种子不保证不同厂商生成相同 Dropout 掩码：
随机算子单独标记，保留真实输出与同平台重复结果，不跨厂商复制 RNG 状态，也不替换掩码以制造一致。
同平台原生非确定性也可能影响“平台 vs 直接 Torch”两次调用；不能只凭一个差值就断言路由错误。

## 运行与证据

工具为 `tools/platform_operator_cases.py` 和 `tools/probe_platform_model_ops.py`，诊断工具独立冻结，
不修改正在运行的 `training-08`、`snapshots-08` 或单 epoch 父计划。
[诊断计划](validation/20261025/model-ops/plan.json)固定工具字节、公共输入来源、模型、阶段、样本上限、重复次数和容差。
[早期证据索引](validation/20261025/model-ops/index.json)保留 CPU 测试、部署回执和当时的待验证状态；
[完成证据索引](validation/20261025/model-ops/nvidia-complete-02/publication-index.json)记录本轮最终采样、全部逐项结果和原始元数据包校验值。

每台机器仅在本机前三模型训练、完整验证和独立推理结束且释放设备锁后，才运行新增 GPU 诊断。
公共大文件先通过本地凭据转运，NVIDIA 上传也等精度训练结束，避免增加其训练磁盘 I/O。
不停止其他任务，不擅用新 GPU；大 Tensor、模型、原始日志留在证据目录，不上传到 Git。

使用相同厂商环境、已分配设备和冻结源码。`FROZEN_TOOLS` 指向 `training-08/tools`，
`SOURCE` 指向对应 `snapshots-08/new`，其余文件遵循单 epoch 报告的数据与权重校验。

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det
# C500 保存公共算子输入；NVIDIA 观察自身覆盖时增加 --coverage-only。
CUDA_VISIBLE_DEVICES="$DEVICES" python -m torch.distributed.run --standalone --nproc-per-node=2 \
  tools/probe_platform_model_ops.py capture --source "$SOURCE" --frozen-tools "$FROZEN_TOOLS" \
  --data "$DATA" --plan "$PLAN" --initial "$INITIAL" --vendor "$VENDOR" \
  --model resnet50 --purpose training --cases-per-op 2 --output "$CAPTURE"
# 独立进程以 --purpose inference 再观察一次；对 YOLO/ViT 更换模型、初始权重与输出目录。
CUDA_VISIBLE_DEVICES="$DEVICES" python tools/probe_platform_model_ops.py replay \
  --source "$SOURCE" --captures "$COMMON_CAPTURE" --vendor "$VENDOR" --repeats 3 --output "$REPLAY"
# 结果比对只读取保存的 CPU 证据，不执行 CPU 模型计算。
CUDA_VISIBLE_DEVICES='' python tools/probe_platform_model_ops.py compare \
  --reference "$NVIDIA_REPLAY" --actual "$METAX_REPLAY" --output comparison.json
```

`observation.json` 说明实际调用与未覆盖项；`replay.json` 说明同平台重复及直接厂商 Torch 对照；
`comparison.json` 报告跨厂商逐样本差异。未完成、空观察、文件损坏或缺少梯度/状态的结果不能被汇总为通过。
观察和重放含 CPU 证据拷贝及额外同步，其时间不用于宣称训练性能；未加观察器的 500-step 执行耗时及数据缓存限制见[训练报告](FIXED_STEPS_20261025.md#耗时与显存记录)。

## 较早完成的 C500 局部阶段

在本机三个模型训练、完整验证及独立推理结束后，按冻结计划完成三模型各自的训练用途观察/重放、推理用途观察/重放，共 **12 个子作业，退出码均为 0**。
这表示执行完成；下表仍保留实际数值失败，不将退出码当作数值通过。

| 模型 | 阶段 | 采样用例 | 与直接厂商 Torch 超差 | 同平台三次重放存在超差 |
| --- | --- | ---: | ---: | ---: |
| ResNet50 | 训练前向/反向 | 10 | 2 | 2 |
| ResNet50 | 训练内 eval | 10 | 0 | 0 |
| ResNet50 | 独立推理 | 10 | 0 | 0 |
| YOLOv5-S | 训练前向/反向 | 13 | 2 | 2 |
| YOLOv5-S | 训练内 eval | 11 | 0 | 0 |
| YOLOv5-S | 独立推理 | 11 | 0 | 0 |
| ViT-B/16 | 训练前向/反向 | 11 | 1 | 1 |
| ViT-B/16 | 训练内 eval | 11 | 0 | 0 |
| ViT-B/16 | 独立推理 | 11 | 0 | 0 |
| **合计** | | **98** | **5** | **5** |

全部 98 个样本都有直接厂商 Torch 参考。超差均位于 `Conv2d` 的 `parameter_gradients.weight`，其他比较字段在本次样本中未发现超差。
重放使用相同操作数、权重和精度上下文；反向的非均匀上游梯度由 CPU 固定生成后传至 GPU，**不是直接复用整网损失梯度**。
以下最大绝对差只描述该梯度张量，不是 Top-1/AP 的差值，也不能跨形状、梯度量级直接比较：

| 模型 | 权重形状 | 平台 vs 直接 Torch 最大绝对差 | 原生重复最大绝对差 |
| --- | --- | ---: | ---: |
| ResNet50 | `[64, 3, 7, 7]` | 3.5 | 3.5 |
| ResNet50 | `[64, 64, 1, 1]` | 1.875 | 2.5 |
| YOLOv5-S | `[32, 12, 3, 3]` | 0.0001296997 | 0.0001525879 |
| YOLOv5-S | `[64, 32, 3, 3]` | 0.0003051758 | 0.0004272461 |
| ViT-B/16 | `[768, 3, 16, 16]` | 0.0625 | 0.0625 |

同一路径的原生重复也在这 5 个样本上超差，说明这些观测包含原生非确定性，不能把“平台 vs 直接 Torch”的一次差值单独归因为平台路由。
这里没有修改内核、模型、数值策略或容差；更不能据此推断 5 个算子样本解释了全部训练精度差。

实际采样接口如下，详细调用数、所有已观察变体、未采样范围、输入/权重/结果校验值见 [C500 局部结果](validation/20261025/model-ops/metax-partial.json)。

| 模型 | 已采样的接口 |
| --- | --- |
| ResNet50 | Conv2d、BatchNorm2d、ReLU、MaxPool2d、AdaptiveAvgPool2d、Linear、log_softmax/softmax |
| YOLOv5-S | Conv2d、BatchNorm2d、SiLU、MaxPool2d、Upsample、sigmoid、binary_cross_entropy_with_logits |
| ViT-B/16 | Conv2d、Linear、matmul、LayerNorm、tanh、softmax/log_softmax、Dropout |

本次 YOLO 公共初始权重的一批输入没有捕获到 NMS 数值用例；ViT 激活实际观察到 `tanh`，不能把候选 GELU 名称当成已验证接口。
Dropout 有训练、eval、推理的捕获记录，同厂商重复/直接参考未超差；其中 eval/推理关闭随机丢弃，仍不能推导跨厂商训练掩码一致。
未触发接口、未采样变体及普通 Tensor 随机桥接仍未验证。

公共 Tensor 和 C500 重放结果已打包，大小 **29,680,189,440 字节**，SHA256 为
`162a9540cef722009e7c740a516611d542bbfd85660f0e7013996f91595afad7`。
该大文件已逐文件校验转运至 NVIDIA，并完成上述重放和跨厂商比较，原始 Tensor 继续保存在任务证据存储。
历史 `common_archive_ready` 只表示当时 C500 本地阶段完成，不能单凭这个回执宣称跨厂商通过。

最终 JSON 元数据包为 `operator-short-02-evidence.tar.gz`，**68,011 字节**，SHA256
`ee0a6787e5c3873421ad96cc47e64c39ca304c4b97fe79f04252c2335e612bfa`。
仓库发布其观察/重放/比较 JSON 和额外的字段解释，不发布大 Tensor；文本统一 LF，发布索引校验 LF 字节，原包 `index.json` 保留原始文件字节 SHA。
本次没有捕获的 NMS、未执行接口、未采样形状及独立的设备/通信/SDK 缺口仍按原验收状态记录，不能将 79/98 样本在容差内写成全部平台能力通过。
