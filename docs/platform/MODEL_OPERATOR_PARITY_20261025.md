# 三模型平台算子跨厂商对照

日期：2026-10-25。**状态：工具和任务已准备，两家隐藏 GPU 的 CPU 契约各 114 项通过；真机算子任务等待当前三模型训练完成，尚无本轮真机数值结论。**

目标模型为 ResNet50、YOLOv5-S、ViT-B/16，比较新版 UP 平台层在 NVIDIA 与沐曦上的结果。
现有 9 次单 epoch 精度对照继续运行；DETR 暂缓。算子诊断使用独立进程，不修改 `up/`、`up_platform/`、模型算法、配方或原容差，
也不把观察器挂到正式精度训练中。[单 epoch 协议](IMAGENET_COCO_PARITY_20261025.md)与本报告分别记录。

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
[证据索引](validation/20261025/model-ops/index.json)记录 CPU 测试、部署回执和本轮待验证状态。

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
观察和重放含 CPU 证据拷贝及额外同步，其时间不用于宣称训练性能；实际性能仍采用未加观察器的单 epoch 训练记录。

当前只确认工具与队列准备完毕。真机执行后再补充逐模型、逐接口、逐阶段的前向/梯度/状态差异表和限制。
