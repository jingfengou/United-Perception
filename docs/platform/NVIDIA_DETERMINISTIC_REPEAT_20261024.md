# NVIDIA 原生确定性：原版与平台版完整训练对照

日期：2026-10-24。**状态：8 次完整训练、4 次权重交换推理、保存状态与指标审计全部完成。**

本轮接续 [native 波动诊断](NVIDIA_NATIVE_REPEAT_20261024.md)，验证原版和新版同时启用原生确定性后是否一致。
不修改模型算法、目标匹配规则或训练配方，不放宽 `rtol=1e-4, atol=1e-5`，不恢复旧多随机种子任务。

## 结果

**在本轮固定数据、初始权重、seed 131、软件和物理 GPU 下，原版和平台版的训练结果一致。**
两模型各自四次运行的全部 loss、每轮验证指标及最终模型参数/缓冲区内容完全相同。

| 模型 / 最终指标 | 原版01 | 原版02 | 新版01 | 新版02 | 新 − 原 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv5-S / AP×100 | 27.42361143 | 27.42361143 | 27.42361143 | 27.42361143 | 0 pp |
| ViT-B16 / Top-1 % | 25.75 | 25.75 | 25.75 | 25.75 | 0 pp |

- **8/8** 完成预算，共 **14,960** 次优化器更新；每模型四次的全部输入批次、初始 Tensor、配置和 LR 轨迹哈希匹配。
- YOLO 的 **600 步 loss**、ViT 的 **3140 步 loss**，在各自四次运行中逐位相同；原容差检查超限数为 **0**。
- **8/8** 最终检查点的模型、优化器和调度器保存状态，按组件检查值、Tensor 形状/dtype/字节和标量内容，各模型四次均相同。这里比较保存状态内容，不把 ZIP 文件整体校验和当作状态相等的必要条件。
- **8/8** 权重严格重载、最终预测 CPU 指标复算、新进程独立推理完成，全部已报告指标与各自最终训练内 eval 相同。
- **4/4** 权重交换推理完成：每模型的原版01权重交给新版、新版01权重交给原版，全部任务指标相同。YOLO 包含全部 12 项 COCO 指标；不冒充对全部 logits 的逐位比较。
- `up/`、`up_platform/`、`configs/` 均未修改；没有新的数值内核、梯度伪实现或 CPU 模型代算。确定性是本轮显式实验选项，框架默认配置继续使用 native。

精确结果、逐步比较及权重交换记录见 [summary.json](validation/20261024/deterministic-repeat/summary.json)、[结果 CSV](validation/20261024/deterministic-repeat/results.csv)；保存状态组件哈希见 [checkpoint-states.json](validation/20261024/deterministic-repeat/checkpoint-states.json)。

前一轮 native 原版 YOLO 为 24.62/26.78、新版为 19.46/25.84；ViT 原版为 19.30/22.20、新版为 24.05/25.45。
这些历史结果完整保留。切换确定性会选用固定的数值执行路径，当前结果不要求等于某一次 native 的最终分数；本轮不据此宣称提高了总体精度。
本次干预后新旧及各自重复均一致，结合原有独立算子重放，支持此前差异中存在原版/native 波动来源，未发现平台改造在这份协议中引入额外训练偏差。
各控制项对最终分数的贡献没有单独拆分，不能把全部差异只归给卷积、GEMM 或目标写入中的某一项。

## 冻结协议

完整条件见 [protocol.json](validation/20261024/deterministic-repeat/protocol.json)。

- 同一张 RTX 4090 物理 1 卡，UUID `GPU-a11f730c-a975-2bf0-b765-3c24d83b7104`，持有全程 GPU 文件锁；各作业串行执行。
- 原版 UP `52aaeb9e`，新版运行时 `b137c68`，继续使用上一轮不可变源码、相同数据文件和初始化 Tensor。
- seed 131；每模型执行原版01、新版01、新版02、原版02，两个重复均保留，共 8 次完整训练。
- ViT-B16：CIFAR-10 10,000/2,000，64px、batch64，20 epoch、3140 次更新。
- YOLOv5-S：PennFudan 120/50，320px、batch64，300 epoch、600 次更新。
- 共 14,960 次更新；每次执行训练内 eval、权重/优化器恢复、最终指标 CPU 复算、新进程独立推理，再做四次新旧权重交换推理。
- FP32、关闭 TF32、关闭 cuDNN benchmark；两个版本均使用同一套新实验工具，不编辑被冻结的 UP 或平台代码。
- Python 3.12.3、Torch `2.10.0+cu126`、TorchVision `0.25.0+cu126`；NumPy 1.26.4、SciPy 1.16.0、pycocotools 2.0.10，完整包清单在证据索引中。NVIDIA 驱动 580.178.04。
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

## 性能代价

下表为每条路径两次完整训练耗时的均值，单位秒；均值与变化见 [性能 CSV](validation/20261024/deterministic-repeat/performance.csv)，所有单次值保留在汇总 JSON 中。

| 模型 / 代码 | native 均值 | 确定性均值 | 观察到的耗时增加 |
| --- | ---: | ---: | ---: |
| YOLOv5-S / 原版 | 1422.69 | 1489.50 | +4.70% |
| YOLOv5-S / 新版 | 1442.37 | 1483.09 | +2.82% |
| ViT-B16 / 原版 | 426.40 | 502.71 | +17.90% |
| ViT-B16 / 新版 | 430.03 | 472.79 | +9.94% |

显存单位 MiB。原版、新版的两次均值在各自数值策略下相同，故按模型合并显示；完整值仍分别保留。

| 模型 | allocated：native → 确定性 | reserved：native → 确定性 |
| --- | ---: | ---: |
| YOLOv5-S | 5775.89 → 5775.66 | 6212 → 6692 |
| ViT-B16 | 3318.19 → 3365.78 | 3970 → 3982 |

YOLO 实际分配峰值接近，缓存预留峰值增加 **480 MiB**；ViT 分配峰值增加约 **47.59 MiB**、缓存预留峰值增加 **12 MiB**。
allocated 是 Torch 实际分配统计，reserved 是其缓存分配器预留统计，不能混为同一指标或整卡物理显存占用。

性能参考为上一轮**同一物理卡**的 8 次 native 完整训练，核验两侧源码、数据、初始权重、全部输入批次、LR 和配方哈希一致。
按模型和原版/新版分别保留两次耗时、均值和变化百分比。
`training_wall_seconds` 包含 `runner.train()` 内的数据处理、GPU 训练、训练内 eval 和保存，不是单个 kernel 或纯 GPU 时间。
显存峰值从训练前重置，统计范围延续到最终 eval/恢复，分别报告 allocated/reserved；不得把它写成仅训练阶段峰值。
参考和本轮为先后执行的两组实验，尚未交错计时；小幅耗时差可能包含机器负载、缓存和温度变化，不能全部归因于确定性选项。
因此，表中保留原版/新版的实测差异，但不能据此推断平台改造有固定比例的性能增益，也不把这些比例推广到所有分辨率、batch 或硬件。

## 审计与范围

[原始索引](validation/20261024/deterministic-repeat/index.json)、[打包回执](validation/20261024/deterministic-repeat/receipt.json)、[发布校验](validation/20261024/deterministic-repeat/publication-audit.json)、[八次摘要](validation/20261024/deterministic-repeat/runs)、[恢复/独立推理/指标审计](validation/20261024/deterministic-repeat/audits) 均保留。
服务端保存 **317 个原始文件，约 8.71 GB**；约 **4.44 MB** 的 `reports.tar.gz` 包含报告、预测、配置和复现工具，不含模型权重或图片。
本地校验归档和索引 SHA256，重新从原始报告生成统计，与服务端结果一致；旧 native 参考报告也随归档保留，便于复算性能比较。

本轮是 CIFAR-10/PennFudan 固定预算、单种子、同一 NVIDIA 卡的兼容性和重复性结论。
不代表 ImageNet/COCO 全量收敛、不同种子总体精度、AMP、分布式或其他设备均已验证。
YOLO 原版重复目标索引的语义问题继续记录；本轮没有改变冲突目标的取舍规则。

## 复现

服务端根目录：`/data2/oujingfeng/up-platform-clean-20261023/deterministic-repeat-20261024`。
数据和两套冻结源码仍在相邻的 `accuracy-20261024`；需要上一轮记录的真实数据与权重，不能用任意下载文件替代相同输入。

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det
export PYTHONPATH="$DEPS:$MQBENCH:$MSBENCH"
mkdir -p "$NEW_DETERMINISTIC_TASK/tools" "$NEW_DETERMINISTIC_TASK/tests/platform"
cp "$FROZEN_ACCURACY/source-03/tools/"*.py "$NEW_DETERMINISTIC_TASK/tools/"
cp tools/{accuracy_train,run_accuracy_cohort,run_native_repeat,probe_native_operators}.py \
  "$NEW_DETERMINISTIC_TASK/tools/"
cp tests/platform/*.py "$NEW_DETERMINISTIC_TASK/tests/platform/"
cp docs/platform/validation/20261024/deterministic-repeat/protocol.json \
  "$NEW_DETERMINISTIC_TASK/protocol.json"
# 修改的是新的公共实验工具副本，不能修改原版或新版源码树
python tools/run_native_repeat.py --base "$FROZEN_ACCURACY" \
  --output "$NEW_DETERMINISTIC_TASK" --deterministic --launch

# 全部 GPU 阶段结束后执行；CPU_RESULT 是通过的契约测试 XML
CUDA_VISIBLE_DEVICES= python tools/package_native_repeat.py \
  --root "$NEW_DETERMINISTIC_TASK" --base "$FROZEN_ACCURACY" \
  --output "$NEW_DETERMINISTIC_TASK/evidence-01" --cpu-result "$CPU_RESULT" \
  --deterministic --native-reference "$FROZEN_NATIVE_REPEAT"
```

控制器本次固定物理 GPU1/UUID，并检查锁和当前计算进程；重复 launch/覆盖输出会拒绝。
运行前 CPU 配置与平台契约检查通过 62 项，增加保存状态哈希审计测试后共 **65 项通过**，见 [最终 XML](validation/20261024/deterministic-repeat/cpu-03.xml)。
首次 CPU 测试因实验目录漏接原版配置适配器的源码引用而失败 1 项；补齐指向冻结源码的目录链接后通过，见 [首次失败](validation/20261024/deterministic-repeat/cpu.xml)、[62 项通过](validation/20261024/deterministic-repeat/cpu-02.xml)。没有删除失败，也没有 GPU 失败后重试挑选结果。
