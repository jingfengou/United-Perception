# NVIDIA 原生确定性：原版与平台版完整训练对照

日期：2026-10-24。**状态：同卡完整训练运行中，尚无最终结论。**

本轮接续 [native 波动诊断](NVIDIA_NATIVE_REPEAT_20261024.md)，验证原版和新版同时启用原生确定性后是否一致。
不修改模型算法、目标匹配规则或训练配方，不放宽 `rtol=1e-4, atol=1e-5`，不恢复旧多随机种子任务。

## 冻结协议

完整条件见 [protocol.json](validation/20261024/deterministic-repeat/protocol.json)。

- 同一张 RTX 4090 物理 1 卡，UUID `GPU-a11f730c-a975-2bf0-b765-3c24d83b7104`，持有全程 GPU 文件锁；各作业串行执行。
- 原版 UP `52aaeb9e`，新版运行时 `b137c68`，继续使用上一轮不可变源码、相同数据文件和初始化 Tensor。
- seed 131；每模型执行原版01、新版01、新版02、原版02，两个重复均保留，共 8 次完整训练。
- ViT-B16：CIFAR-10 10,000/2,000，64px、batch64，20 epoch、3140 次更新。
- YOLOv5-S：PennFudan 120/50，320px、batch64，300 epoch、600 次更新。
- 共 14,960 次更新；每次执行训练内 eval、权重/优化器恢复、最终指标 CPU 复算、新进程独立推理，再做四次新旧权重交换推理。
- FP32、关闭 TF32、关闭 cuDNN benchmark；两个版本均使用同一套新实验工具，不编辑被冻结的 UP 或平台代码。
- 只采用最终 epoch，不重试挑选结果，不设新的精度通过线。逐步 loss、最终模型状态哈希及完整指标分别检查，不能仅凭分数相同声称计算一致。

## 原生确定性设置

在新进程启动、CUDA 初始化之前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，并使用：

```python
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

保留 PyTorch 默认的 `fill_uninitialized_memory=True` 并记录实际值。遇到未实现确定性的操作直接报错，保留失败证据，不关闭检查继续运行。
这些选项选择 Torch/cuDNN/cuBLAS 的原生算法，没有自写替代内核或 CPU 模型代算。
PyTorch 提供相关 [严格确定性接口](https://docs.pytorch.org/docs/2.10/generated/torch.use_deterministic_algorithms.html) 和 [复现说明](https://docs.pytorch.org/docs/2.10/notes/randomness.html)。

对 YOLO 的重复目标写入，本次只观察 PyTorch 原生确定性路径的行为，不定义新的 GT 取舍策略；不能把“写入可重复”视为原版冲突目标语义已经正确。

## 结果与性能口径

完整结果尚待全部训练、恢复及独立推理完成；不把首轮或少量步数一致标为完整通过。

性能参考为上一轮**同一物理卡**的 8 次 native 完整训练，核验两侧源码、数据、初始权重、全部输入批次、LR 和配方哈希一致。
按模型和原版/新版分别保留两次耗时、均值和变化百分比。
`training_wall_seconds` 包含 `runner.train()` 内的数据处理、GPU 训练、训练内 eval 和保存，不是单个 kernel 或纯 GPU 时间。
显存峰值从训练前重置，统计范围延续到最终 eval/恢复，分别报告 allocated/reserved；不得把它写成仅训练阶段峰值。
参考和本轮为先后执行的两组实验，尚未交错计时；小幅耗时差可能包含机器负载、缓存和温度变化，不能全部归因于确定性选项。

## 复现

服务端根目录：`/data2/oujingfeng/up-platform-clean-20261023/deterministic-repeat-20261024`。
数据和两套冻结源码仍在相邻的 `accuracy-20261024`；需要上一轮记录的真实数据与权重，不能用任意下载文件替代相同输入。

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det
export PYTHONPATH="$DEPS:$MQBENCH:$MSBENCH"
# 新目录 tools/ 包含本轮 accuracy_train.py、run_accuracy_cohort.py
# 其余公共实验工具沿用冻结 source-03/tools，不能修改原版或新版源码树
python tools/run_native_repeat.py --base "$FROZEN_ACCURACY" \
  --output "$NEW_DETERMINISTIC_TASK" --deterministic --launch
```

控制器本次固定物理 GPU1/UUID，并检查锁和当前计算进程；重复 launch/覆盖输出会拒绝。
CPU 配置与平台契约检查已通过 62 项。首次 CPU 测试因实验目录漏接原版配置适配器的源码引用而失败 1 项；补齐指向冻结源码的目录链接后全部通过，首份失败 XML 保留。
GPU 完成后将补齐结果、性能统计、源码和原始证据索引。
