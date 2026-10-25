# 三模型、三路径的 500-step 对照

日期：2026-10-25。**状态：三个模型的两平台短程队列已启动；完整三路径结果与 98 个公共算子样本的 NVIDIA 重放尚待完成。**

用户将执行预算从单 epoch 改为每条路径 **500 个调度 steps**。
本阶段独立记录，不把旧的一轮结果、旧小数据结果或未完成训练混入本次结论。
DETR 和旧多随机种子任务继续暂停；没有修改模型、平台实现或算子容差。

## 本轮比较

| 模型 | 数据 | 原版 UP/NVIDIA | 新 UP/NVIDIA | 新 UP/C500 | 每卡 batch / 精度 |
| --- | --- | --- | --- | --- | --- |
| ResNet50 | ImageNet-1K | 500 steps | 500 steps | 500 steps | 64 / AMP FP16 |
| YOLOv5-S | COCO2017 | 500 steps | 500 steps | 500 steps | 32 / FP32 |
| ViT-B/16 | ImageNet-1K | 500 steps | 500 steps | 500 steps | 64 / AMP FP16 |

共 9 次训练、4,500 个调度批次；AMP 跳过与成功更新分别计数。
NVIDIA 两条路径顺序使用同一对 RTX 4090，C500 使用原已分配的两卡；本轮不等待 A100/A800 迁移。
均从同一份公共初始 Tensor 开始，seed 131、数据划分、预处理、优化器、LR 和 EMA 时间轴相同。
保留当前完整数据协议的 native 设置：benchmark 开、强制确定性关、TF32 关。
500 步以后分别执行完整 50,000/5,000 张验证、检查点保存和新进程独立推理。

不压缩原 200/300 epoch 的学习率、warmup 或 EMA 配方，也不把 500 步称为收敛精度。
分类与 YOLO 的精度模式不同，但同一模型的三条路径保持一致。
保留厂商 Torch 构建及 Python 补丁版本差异，详见[父协议](IMAGENET_COCO_PARITY_20261025.md)。

### 怎样定位差异

1. 核对两个 rank 的全部 500 步输入 SHA、学习率和初始权重，条件不匹配时拒绝把 loss 偏离解释为数值差异。
2. 分别报告新旧 NVIDIA、C500 与新 NVIDIA 的首次 loss 数值差异、首次超容差、超限步数和成功更新计数分叉。
3. 报告训练后的 Top-1/AP 及有符号百分点差；任务精度没有新增通过线。
4. 用[公共算子操作数](MODEL_OPERATOR_PARITY_20261025.md)比较前向、必要梯度和状态，并保留直接厂商 Torch 参考和同平台重复。

浮点比较继续使用 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。
loss 轨迹差异、独立算子误差和任务精度分别报告；出现第一个偏离点不等于证明其贡献了全部最终精度差。
随机算子不强制共享掩码，不用 CPU 代算模型来制造一致。

## 已有小数据实验究竟有什么差异

以下是**以前已完成的独立实验**，不是本轮 500-step 结果。
旧分类模型为 ResNet18；本轮完整 ImageNet 使用 ResNet50，两者不能直接对比分数。

旧 native 单种子完整固定预算：分类采用 CIFAR-10 的 10,000/2,000 子集、20 epochs；
YOLO 采用 PennFudan 的 120/50 划分、300 epochs。

| 模型 / 指标 | 原版 UP/NVIDIA | 新 UP/NVIDIA | 新 UP/C500 | C500 − 新 NVIDIA |
| --- | ---: | ---: | ---: | ---: |
| ResNet18 / Top-1 % | 64.90 | 66.40 | 66.45 | +0.05 pp |
| YOLOv5-S / AP×100 | 14.03 | 29.62 | 29.16 | −0.46 pp |
| ViT-B16 / Top-1 % | 26.70 | 22.55 | 24.35 | +1.80 pp |

完整数字、预算与全部 15 次五模型训练证据见[原始 native 报告](ACCURACY_NATIVE_20261024.md)。
这些单次 native 分数存在差异，不能不经同源重复就全部归因于平台重构。

随后在同一 NVIDIA 卡上开启原生严格确定性，YOLO 的原版/新版各两次 AP 均为 **27.42361143**，
ViT 的四次 Top-1 均为 **25.75%**，全部 loss 和最终模型/优化器/调度器状态也一致。
这是[该组 NVIDIA 确定性实验](NVIDIA_DETERMINISTIC_REPEAT_20261024.md)的结论，不是跨厂商训练保证。

同条件 C500 严格训练的 YOLO AP 为 **44.90611898**（相对新 NVIDIA **+17.48250755 pp**），
ViT Top-1 为 **19.35000038%**（**−6.39999962 pp**），因此不能说“开启确定性后跨厂商训练都一致”。
见[C500 严格训练](METAX_DETERMINISTIC_TRAIN_20261024.md)。本轮 500-step 仍使用 native。

同一份 NVIDIA 训练后权重直接在 C500 推理时，ResNet18、YOLO、ViT 的全部已报告任务指标都相同：
分别为 Top-1 **66.40000153%**、AP **29.62426402**、Top-1 **22.55000114%**。
原始输出 Tensor 并非全部满足原容差，见[同权重推理报告](MODEL_VERIFY_PROTOCOL_20261024.md)。
“同权重推理指标一致”与“各自训练得到相同精度”是两个独立问题。

## 设备相关操作已有的数值证据

以下观测来自明确标注的旧测试条件，不能冒充本轮全部算子已通过。

| 范围 | 已观察差异 | 证据边界 |
| --- | --- | --- |
| 设备迁移 | 相同 CPU Tensor 迁至两家后逻辑值相同；本次旧探针中 NVIDIA 为 NCHW、C500 为 channels-last | 直接厂商 Torch 也复现布局差异；布局可能影响后续内核选择 |
| 首个 Conv2d，旧小数据 FP32 诊断 | YOLO 前向最大绝对差 `1.37091e-6`，ViT 为 `2.26498e-6`，均在原容差内 | 只是该输入/层；显式对齐布局后的首层前向逐位相同，反向仍有差异 |
| ViT 训练 Dropout | `[64,17,768]` 的 835,584 个掩码位置中 8,793 个不同，约 1.05% | 同 seed 不保证跨设备 RNG 网格/消耗相同；纯 Torch 重放仍复现，不证明内核错误 |
| YOLO 中间 BN | 第 15 个叶子模块有 6 个元素超限，最大绝对差 `3.05176e-5` | 输入已经不同；用公共相同输入重放该 BN，前向最大差 `3.81470e-6`，在容差内 |
| 修正后的 KITTI 旋转 IoU | 两家 24 组保存数组逐位相同，跨卡最大绝对差 0 | 对独立 GEOS 的最大误差两家均 `9.5367432e-7`；不推广到其他算子 |
| 当前真实模型算子采样 | C500 98 用例中 5 个训练 Conv2d 权重梯度样本在直接 Torch/自身重复中超差 | 这 5 项是同平台观测；本轮 NVIDIA 公共输入重放尚未完成，不给出虚构的跨卡误差 |

卷积、BN、随机掩码的细节和不导入 UP 的独立重放见[训练差异定位](VENDOR_TRAINING_DIAGNOSIS_20261024.md)；
几何指标见[KITTI 报告](KITTI_GEOMETRY_20261024.md)。
设备绑定、内存、同步和通信主要验证接口语义与状态，不都适合用一个浮点差值表示。
专用操作、稀疏卷积、量化、剪枝、通信的逐项执行状态，以及 linklink/Amba 等缺口见[验收矩阵](VALIDATION.md#结果矩阵)。
不能将 98 个采样用例或几个模型跑通等同于全部 156 项接口及所有变体通过。

## 复现与审计

新的执行控制只设置预算，不修改被冻结的训练驱动和生产源码。
[短阶段计划](validation/20261025/fixed-500/plan.json) SHA256 为 `79441420ed7be26805d024635691d0a48d2e179cb50c077bacb54beb38926e2c`；
它引用父计划 `5e640ab5765222062df3f5afa14c91bf4f88be19179b70f9f5b3b0b46c996426`，
对应父协议中相同的数据绑定、源码、初始权重和 13 个冻结工具。

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det
# 在已分配的两卡上执行。所有路径使用同一个父计划及对应源码/初始权重。
CUDA_VISIBLE_DEVICES="$DEVICES" python -m torch.distributed.run --standalone --nproc-per-node=2 \
  "$FROZEN_TOOLS/full_accuracy_train.py" --source "$SOURCE" --data "$DATA" \
  --plan "$PARENT_PLAN" --model "$MODEL" --lane "$LANE" --initial "$INITIAL" \
  --probe-steps 500 --output "$RESULT/train"
CUDA_VISIBLE_DEVICES="$DEVICES" python -m torch.distributed.run --standalone --nproc-per-node=2 \
  "$FROZEN_TOOLS/full_accuracy_train.py" --source "$SOURCE" --data "$DATA" \
  --plan "$PARENT_PLAN" --model "$MODEL" --lane "$LANE" --initial "$INITIAL" \
  --probe-steps 500 --mode inference --checkpoint "$RESULT/train" --output "$RESULT/inference"
```

历史驱动把此类有限 steps 报告为 `preflight_completed`，并保留父轮次的 `expected_steps`。
本次验收依据是短计划、`identity.preflight_steps=500` 和 `progress.completed_steps=500`；
不能据状态名称宣称未训练，也不能将 500 步升级为完整 epoch。

任务目录布局为 `one-epoch-plan.json`、`full-data-binding.json` 及 `short-500-01/results/<lane>/<model>/{train,inference}`。
新进程推理完成后，用与本轮相同的 CPU 审计实现核对检查点/预测、完整验证集指标及每个 rank 的 500 条记录：

```bash
CUDA_VISIBLE_DEVICES='' python tools/audit_fixed_steps.py "$TASK_ROOT" nvidia
# 在 C500 的厂商环境改为 metax；只读取已保存的预测，不执行 CPU 模型前向/反向。
```

收集报告、`cpu-audits` 和 trace 后，汇总工具检查全部 9 次报告、500 步完整序列、输入和 LR：

```bash
python tools/summarize_fixed_steps.py --nvidia "$COLLECTED_NVIDIA" --metax "$COLLECTED_C500" \
  --parent-plan "$PARENT_PLAN" --output comparison.json
```

大 Tensor、数据、权重、原始日志保留在各机独立任务目录；仓库只发布必要工具、摘要及校验索引。
旧单 epoch 输出和停止原因保留，不删除失败、未完成或被新预算替代的证据。
