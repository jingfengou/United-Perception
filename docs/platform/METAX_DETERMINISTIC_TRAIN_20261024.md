# 原生确定性开关下：沐曦与 NVIDIA 的 YOLO、ViT 完整训练对照

后续的前向、梯度、GPU 掩码及独立算子定位见 [训练差异诊断](VENDOR_TRAINING_DIAGNOSIS_20261024.md)。本页保留完整训练阶段的原始结果和当时的结论。

日期：2026-10-24。**状态：C500 两次完整训练、3,740 次更新、自身恢复/指标审计/独立推理及两次 NVIDIA 权重交叉推理全部完成。精度仍有明显差异，不判为跨厂商等价。**

本轮从相同初始权重分别训练，接续 [NVIDIA 确定性完整训练](NVIDIA_DETERMINISTIC_REPEAT_20261024.md)，
区别于上一轮 [同权重推理和零更新反向诊断](MODEL_VERIFY_PROTOCOL_20261024.md)。
不改模型算法、数据划分、配方或 `rtol=1e-4, atol=1e-5`；旧多随机种子任务保持暂停。

## 最终结果

精度单位为百分数：YOLO 是 COCO bbox AP@[.50:.95] × 100，ViT 是 Top-1 %；差值为有符号百分点。
全部采用最后一轮，NVIDIA 原版/新版各两次的结果已一致，C500 每模型本轮各执行一次。

| 模型 | 原版 UP / NVIDIA | 新版 UP / NVIDIA | 新版 UP / C500 | C500 − 新版 NVIDIA |
| --- | ---: | ---: | ---: | ---: |
| YOLOv5-S | 27.42361143 | 27.42361143 | 44.90611898 | +17.48250755 pp |
| ViT-B16 | 25.75 | 25.75 | 19.35000038 | −6.39999962 pp |

**开启相同原生严格开关后，两家的完整训练结果仍不一致。**
YOLO 的 C500 分数更高也不等于平台提高了模型精度；单次固定预算不能判断长期偏差或重复波动。
本轮不设精度通过线，ViT 的下降和 YOLO 的大幅变化均保留为需要继续定位的问题。

完整指标、每次评估、输入校验和及逐步 loss 统计见 [summary.json](validation/20261024/metax-deterministic-train/summary.json)，
主指标见 [results.csv](validation/20261024/metax-deterministic-train/results.csv)。

## 对照条件和数值审计

两模型均核对了相同初始文件和参数内容、数据 manifest、完整配方、训练驱动、第一批及**全部训练批次**的内容哈希，学习率的每步/每参数组轨迹完全相同。
两家相同初始权重的初始验证指标也相同。差异不能由本轮换了数据、配置、初始化或学习率来解释。

| 模型 | 首次 loss 数值差异 | 首次超原 loss 容差 | 超限步数 | 最大 loss 绝对差 |
| --- | ---: | ---: | ---: | ---: |
| YOLOv5-S | 第 1 步 | 第 4 步 | 592 / 600 | 0.87758350 |
| ViT-B16 | 第 1 步 | 第 1 步 | 3121 / 3140 | 0.14299965 |

loss 比较使用原公式 `abs(C500 − NVIDIA) <= 1e-5 + 1e-4 * abs(NVIDIA)`，没有修改容差；两模型最终权重内容也不同。
ViT 第一轮前向 loss 为 NVIDIA `2.3273837566`、C500 `2.3060355186`，差异已在首次优化器更新之前出现。
这说明不能把全部偏离仅解释为后续优化器累计误差，但尚未定位首个发生差异的算子。

两次 C500 训练均完整完成预算，检查点的模型/优化器/调度器状态有限，严格重载成功。
训练最终 eval、重载后的 eval、保存预测的独立 CPU 指标复算及新进程独立推理的全部任务指标一致。
这些是执行、恢复和审计结果，不把它们替代为跨厂商数值等价结论。

## 冻结条件

机器协议与校验和见 [protocol.json](validation/20261024/metax-deterministic-train/protocol.json)。

| 模型 | 数据 | 输入 / batch | 完整预算 | 指标 |
| --- | --- | --- | --- | --- |
| UP ViT-B16 | CIFAR-10：10,000 train / 2,000 test | 64px / 64，保留尾批 16 | 20 epoch，3140 次更新 | Top-1 % |
| UP YOLOv5-S SiLU | PennFudan：120 train / 50 holdout | 320px / 64，EMA off | 300 epoch，600 次更新 | COCO bbox AP×100 |

- C500 每模型执行一次，seed 131，共 3,740 次更新。仅报告最后一轮，不挑最好权重，不失败后自动重试。
- 复用 NVIDIA 每模型四次、共八次已完成的同种子严格确定性结果（原版/新版各两次）；本轮不重复占用 NVIDIA GPU。
- 两家初始 `initial.pt` 文件、UP 冻结源码、数据 manifest、配方和训练工具的 SHA256 匹配。
- Python/NumPy/Torch 的模型种子为 131；两家解释器 `PYTHONHASHSEED=0`，线程各 2。
- FP32、TF32 off、cuDNN benchmark off，启用 `torch.use_deterministic_algorithms(True, warn_only=False)`、cuDNN deterministic，启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。
- 保留 Torch 的 `fill_uninitialized_memory=True`；不增加旧 portable/v2 内核，也不改损失、随机掩码或目标覆盖规则。
- C500 使用原生厂商 Torch；开关启用和已证明逐位确定性分别报告。此前诊断已观察到开关启用后的微小梯度波动，本轮不预设跨卡结果必然一致。
- 每次完整训练包含训练内 eval、最终权重/优化器/调度器重载、最终预测 CPU 指标复算和新进程独立推理。任何失败均保留。
- 最终精度按百分点报告，不设新的通过线；逐步 loss 的原容差检查与任务精度是独立结论。

## 补充的同权重推理检查

ViT 完整训练得到 19.35% Top-1，与 NVIDIA 的 25.75% 有明显差距，因此增加独立的训练后检查，
见 [cross-weight-protocol.json](validation/20261024/metax-deterministic-train/cross-weight-protocol.json)。
这个补充协议在观察到 ViT 分数后、运行补充推理前冻结；原始训练协议、权重、预算和容差不变。

从已经完成的 NVIDIA `new-01` 最终检查点以 CPU 导出模型 Tensor，验证原检查点和模型状态哈希后，
将两模型相同的最终权重放到同一张 C500 上做完整留出集推理。模型权重选择没有使用 C500 分数。
补充阶段不增加优化器更新，也不修改模型，用于区分训练产生的权重差异与同权重推理的任务指标差异。
它只比较任务指标，不替代原始输出 Tensor 的容差验收。

| NVIDIA 最终权重 | 在 NVIDIA 上评估 | 原样加载到 C500 评估 | 差值 |
| --- | ---: | ---: | ---: |
| YOLOv5-S | 27.42361143 | 27.42361143 | 0 pp |
| ViT-B16 | 25.75 | 25.75 | 0 pp |

所有已报告指标相同，交叉预测也经独立 CPU 指标复算。
在同一 C500 评估设备上比较两家训练出的权重，仍得到 +17.4825 pp / −6.4000 pp 的差距。
因此，本轮明显差异与训练产生的权重有关；把相同 NVIDIA 权重移到 C500 推理，没有复现这些大幅精度变化。
这仍不能直接断言是平台路由、厂商某个算子、随机数或其他训练路径中的哪一项导致。

原生严格开关并不承诺跨平台得到同样的浮点执行和 GPU 随机序列，参见 [PyTorch 2.10 复现说明](https://docs.pytorch.org/docs/2.10/notes/randomness.html)。
ViT 原配方保留 `dropout=0.1`、`attention_dropout=0.1`、`drop_path=0.1`；本轮未关闭或统一这些随机掩码。
随机路径和训练算子仍需分别核对，不能据此预先认定某一个就是全部精度差的原因。
YOLO 已有的重复目标索引覆盖问题见 [之前的定位](NVIDIA_NATIVE_REPEAT_20261024.md)；本轮没有改变 GT 取舍规则，也没有重新定位两家各自的覆盖行为。

## 证据、验证与边界

C500 主实验在北京时间 2026-10-24 **21:38:20–22:19:04** 执行；补充交叉推理为 **22:20:15–22:21:38**。
同一物理 C500 GPU 2、PCI `0000:0e:00.0`，各阶段核对空闲状态并持有设备锁，全部 GPU 工作结束后释放，见 [释放回执](validation/20261024/metax-deterministic-train/lease-release.json)。
NVIDIA 参照是此前同一 RTX 4090 GPU 1、UUID `GPU-a11f730c-a975-2bf0-b765-3c24d83b7104` 的已完成记录，本轮只以 CPU 导出参照权重。

- C500 启动前 **72 项 CPU 契约测试通过**；新增的 3 项审计检查在本地通过，防止短 trace、近零 loss 容差和 warn-only 被误报为严格验证通过。
- 运行后重新校验全部数据、标签、预训练资源和冻结源码；实际训练驱动与 NVIDIA 的 `a604fc94…` 字节相同。
- `up/`、`up_platform/`、`configs/` 与 `b137c68` 相同；本轮只增加协议、结果审计工具和文档，没有引入新数值内核或 CPU 模型代算。
- [检查点审计](validation/20261024/metax-deterministic-train/metax-checkpoint-audit.json)、[C500 预检](validation/20261024/metax-deterministic-train/metax-preflight.json)、[NVIDIA 预检](validation/20261024/metax-deterministic-train/nvidia-preflight.json)、[两次训练与恢复/推理摘要](validation/20261024/metax-deterministic-train/runs) 均保留；推理摘要记录原始报告哈希和源码哈希，完整报告留在归档中。
- [原始文件索引](validation/20261024/metax-deterministic-train/evidence-index.json) 共 **155 项、约 3.64 GB**；[归档回执](validation/20261024/metax-deterministic-train/receipt.json) 的 `reports.tar.gz` 约 **1.24 MB**，包含原始报告、预测、日志、工具和测试。数据与权重留在证据目录，不加入 Git。
- [发布索引](validation/20261024/metax-deterministic-train/publication-index.json) 固定公共文件的 SHA256；NVIDIA 八份原始报告另存于已校验的 `reference.zip`。

本轮是单种子、固定小数据预算，不验证全量 ImageNet/COCO 收敛、C500 完整训练的重复方差、AMP 或多卡。
未通过此实验认定所有厂商内核都已确定，也不能把差距简单称为正常浮点误差。
本轮未在 C500 同卡重跑 native 作为干预对照，不能从与历史分数的差值判断确定性开关普遍提高或降低精度。

## 环境与复现方法

C500 使用 Python 3.12.11、Torch `2.10.0+metax3.8.1.0`、TorchVision `0.25.0+metax3.8.1.0`；
NVIDIA 参考使用 Python 3.12.3、Torch `2.10.0+cu126`、TorchVision `0.25.0+cu126`。
NumPy 1.26.4、SciPy 1.16.0、Pillow 11.2.1、OpenCV 4.11.0.86、pycocotools 2.0.10 匹配。
两家厂商 Torch/SDK 和 Python 补丁版本不同，完整环境分别记录，不称为逐字相同的二进制环境。

沿用 [真实数据准备与初始权重](ACCURACY_NATIVE_20261024.md) 和 [环境说明](ENVIRONMENT.md)。
本轮 C500 独立目录为 `/workspace/up-run/up-platform-clean-20261023/metax-deterministic-train-20261024`，
`BASE` 为相邻的 `accuracy-20261024`，其 `source-03`、`data/data.json` 和 `initial` 保持冻结。
准备同一份源码与输入时逐项核对协议中的 SHA256，不用重新随机初始化代替共同 `initial.pt`。

先确认实际获配的 C500 卡和无其他计算进程；本轮为物理 GPU 2，PCI `0000:0e:00.0`，全程持有该卡的共享任务锁。
在包含厂商 Torch 的既有容器环境运行，不替换为 NVIDIA wheel。
`TOOLS` 指本轮独立工具副本，七个关键工具与 NVIDIA 参考的文件 SHA256 相同。

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det
export PYTHONPATH="$DEPS:$MQBENCH:$MSBENCH"
unset MCDNN_USE_DETERMINISTIC_ALGO MCDNN_STABLE_RESULT_MOD
unset PYTORCH_DISABLE_CUDA_CUDNN_TF32 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python "$TOOLS/run_accuracy_cohort.py" \
  --source "$BASE/source-03" --data "$BASE/data/data.json" \
  --initial "$BASE/initial" --output "$NEW_RESULTS" \
  --lock "$GPU_TASK_LOCK" --lane new_metax --device 2 \
  --models vit_b16 yolov5s --deterministic
```

`NEW_RESULTS` 必须是新目录；`GPU_TASK_LOCK` 是与其他任务协商使用的同一设备锁。
本轮外层控制器持有全程设备锁，内层使用单独 worker 锁；直接执行上述命令时让该命令持有设备锁。
脚本启动实际训练前设置严格原生开关，保留每个模型的退出码、日志和失败；不自动重试。
训练内 eval 继续使用训练后端，后续独立推理为新进程和 inference 用途。

补充交叉推理使用上述相同环境，以及协议中已校验的 NVIDIA 模型权重文件：

```bash
CUDA_VISIBLE_DEVICES=2 python "$TOOLS/accuracy_train.py" \
  --source "$BASE/source-03" --data "$BASE/data/data.json" \
  --initial "$NVIDIA_FINAL_MODEL" --output "$NEW_CROSS_RESULTS" \
  --lane new_metax --model "$MODEL" --mode inference --deterministic
```

`MODEL` 分别为 `vit_b16`、`yolov5s`；`NVIDIA_FINAL_MODEL` 为本轮 `reference-weights` 中对应的模型 Tensor 文件，文件/状态校验和见补充协议。
本轮所有交叉推理也持有同一设备锁，只在两个完整训练结束后执行，不与训练并发。

两模型结束且审计完成后比较：

```bash
python tools/summarize_deterministic_vendors.py \
  --metax "$C500_TASK_ROOT" --reference "$NVIDIA_REFERENCE" \
  --protocol docs/platform/validation/20261024/metax-deterministic-train/protocol.json \
  --output "$NEW_SUMMARY"
```

`C500_TASK_ROOT/results` 包含两个模型的原始报告、最终预测、metric-audit 和独立推理报告。
`NVIDIA_REFERENCE/<model>/<variant>` 包含原版/新版各两次已完成训练的原始报告及审计；
不能用仅有最终分数的摘要 JSON 代替原始报告。参考文件校验和在协议中逐项固定。
也可以解压本轮已校验的 `reports.tar.gz` 与 `reference.zip` 后，在没有 GPU 的环境运行该汇总工具复算表格；它不执行模型。
