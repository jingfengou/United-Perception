# 五模型、三条路径的 native 真实数据训练

日期：2026-10-24。分支：`codex/up-platform-clean`。**状态：15 次完整训练和后续审计全部完成；精度差异如实报告，不作等价或通过结论。**

本轮按用户确认的 CIFAR-10/PennFudan 固定预算运行单种子 131，比较：

1. 原版 UP `52aaeb9e` / NVIDIA RTX 4090。
2. 新平台层 UP / NVIDIA RTX 4090。
3. 同一份新平台层 UP / 沐曦 C500。

先完整报告精度差异，不设通过线，不做 TOST，不把重复波动直接判为接入失败。
旧多随机种子任务继续暂停；旧报告和两步预检均不计入本轮完整训练。

## 最终结果

分类为 Top-1（%），检测为 COCO bbox AP@[.50:.95] × 100；后三列是**有符号的百分点差（pp）**，不是相对百分比。
所有数值来自最后一个 epoch，未挑选最好的一轮。

| 模型 | 原 UP/NVIDIA | 新 UP/NVIDIA | 新 UP/沐曦 | 新N − 原N | 沐曦 − 新N | 沐曦 − 原N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | 64.90 | 66.40 | 66.45 | +1.50 | +0.05 | +1.55 |
| YOLOv5-S | 14.03 | 29.62 | 29.16 | +15.60 | −0.46 | +15.14 |
| ViT-B16 | 26.70 | 22.55 | 24.35 | −4.15 | +1.80 | −2.35 |
| Faster R-CNN | 60.27 | 61.47 | 58.37 | +1.20 | −3.10 | −1.90 |
| 官方 DETR 适配器 | 77.65 | 79.23 | 77.75 | +1.58 | −1.47 | +0.10 |

完整精度和绝对差见 [summary.json](validation/20261024/accuracy/summary.json) 和 [results.csv](validation/20261024/accuracy/results.csv)。

这次执行证明五个模型的约定训练预算、训练内 eval、保存/重载和独立推理能够完成，**不能据此宣称平台重构不影响精度**。
尤其原版与新版 NVIDIA 的 YOLO 相差 15.60 pp、ViT 相差 −4.15 pp，仍需定位；不能因为某条路径分数更高就宣称改进，也不能未经同源重复就统称正常波动。
后续已启动 [NVIDIA 同卡重复与算子定位](NVIDIA_NATIVE_REPEAT_20261024.md)，捕获到原版、新版共有的原生卷积权重梯度波动；完整重复结果另列，不覆盖本表。
新版沐曦相对新版 NVIDIA 的绝对差依次为 0.05、0.46、1.80、3.10、1.47 pp。
用户暂未设通过线，因此这些差异均保留为观测结果，不自动判为通过或失败。

## 审计结果与差异边界

- **15/15** 完成全部预算，合计 **24,960** 次优化器更新；没有训练失败、自动重跑或缺失的正式结果。
- 五个模型在三条路径上的初始文件/实际参数、规范数据清单、配方、完整学习率轨迹、验证驱动相同；**首批及全部训练批次的哈希也相同**。
- **15/15** 保存的最终预测经 CPU 独立指标复算，与 Runner 指标完全一致。
- **15/15** checkpoint 的模型、优化器、调度器状态有限，模型参数严格重载一致；保存后重建优化器/调度器并加载其状态，再执行 eval。这里没有额外做“恢复后继续一步训练”的验证。
- **15/15** 新进程独立推理成功；每次重载和独立推理的全部已报告指标均与**各自**训练结束时相同。这不是三条路径共享同一个最终 checkpoint 的交换推理实验。
- 两机隐藏 GPU 的 CPU 契约测试各 **54 项通过、0 跳过**。训练前后源码一致，依赖文件、图片/标签、初始文件和原生扩展在收尾再次核验。
- 本轮没有修改 `up/`、`up_platform/` 或 `configs/` 中的 **781 个运行时源码/配置文件**，只增加验证工具、测试和报告。原版 NVIDIA 扩展另有 43 个 C++/CUDA 源文件与原版相符的记录。

原版与新版 NVIDIA 首次出现**非逐位相同的 loss**分别在 ResNet 第 3 步、YOLO 第 1 步、ViT 第 180 步、Faster R-CNN 第 3 步、DETR 第 3 步。
例如 YOLO 第一轮 loss 为 13.5158376694 与 13.5158205032，差约 −1.72e−5；ViT 首次可见 loss 差约 −2.38e−7。
这些只是训练轨迹观测，不是原算子容差失败，也不代表已定位第一个产生梯度差异的算子。
见 [native-diagnostics.json](validation/20261024/accuracy/native-diagnostics.json)。

本轮关闭确定性控制，每条路径只运行一次；两个 NVIDIA 队列分别在同型号 RTX 4090 的物理 0、1 卡执行。
相同 seed 不保证不同厂商内部 GPU 随机数、采样、归约顺序逐位相同；本轮没有为强行对齐它们改模型或把模型计算移到 CPU。
因此现在可以排除已核验的外部输入/预算不一致，**仍不能区分框架改动、物理卡/原生内核波动及训练轨迹放大各自的影响**。
没有用旧的合成输入或确定性阶段替代当前精度结果，也没有恢复旧多随机种子任务。

## 固定预算

| 模型 | 真实数据划分 | 输入/批量 | 完整预算 | 每条路径更新数 | 主要指标 |
| --- | --- | --- | --- | ---: | --- |
| ResNet18 | CIFAR-10：10,000 train / 2,000 test | 64px / batch64 | 20 epoch | 3140 | Top-1 |
| YOLOv5-S SiLU | PennFudan：120 train / 50 holdout | 320px / batch64，EMA off | 300 epoch | 600 | COCO bbox AP |
| ViT-B16 | 同一 CIFAR-10 子集 | 64px / batch64，12 blocks、width768、17 tokens | 20 epoch | 3140 | Top-1 |
| Faster R-CNN R50-FPN | 同一 PennFudan 划分 | 短边320、最大512 / batch2 | 12 epoch | 720 | COCO bbox AP |
| 官方 DETR R50 适配器 | 同一 PennFudan 划分 | 短边320、最大512 / batch2 | 12 epoch | 720 | COCO bbox AP |

合计 **15 次训练、24,960 次优化器更新**。每次使用最后一个 epoch，不选最好 checkpoint 或有利重跑。
这是完整执行约定的小数据预算，不代表 ImageNet/COCO 全量训练、标准分辨率或充分收敛。
尤其 ViT 的低分辨率短配方只作为回归；不能把其绝对精度当作标准 ViT 的性能。

详细冻结信息见 [plan.json](validation/20261024/accuracy/plan.json)。

## 相同条件与 native 定义

- 图片、标签、划分、anchors、预训练文件和官方 DETR 源码逐文件验证；两机的规范数据清单 SHA256 相同。
- 五份 CPU 初始模型状态只在 NVIDIA 原版生成一次，再原样复制到沐曦。每条训练严格加载同一文件，检查参数和 buffer 哈希；不只依赖“种子一样”。
- 训练配方复用既有预算，三个版本使用相同的增强、优化器、参数分组、学习率轨迹和评估定义。
- FP32，两种 TF32 均关闭，benchmark 关闭；Torch/cuDNN 强制确定性关闭。不设置两个 MCDNN 确定性变量或 CUBLAS workspace 控制，不带入 v1/v2/portable、梯度舍入或 CPU 模型代算。
- 两边分别保留厂商 Torch `2.10.0`、TorchVision `0.25.0` 构建。NumPy 1.26.4、SciPy 1.16.0、pycocotools 2.0.10、Pillow 11.2.1、OpenCV 4.11.0.86 一致；Python 补丁版和 SDK 差异单独记录。
- 原版入口只补 Python/NumPy 已移除的同义别名，使用真正的原版扩展；43 个原生源码文件与 `52aaeb9e` 相同，没有把修正后的 RoIAlign 扩展塞给原版。
- 新版使用已验证的各厂商扩展。确定性开关关闭，因此 RoIAlign 使用 native 原子反向，运行波动先记录。

DETR 在原版 UP 中没有模型/配方。这里给三个 Runner 加载**同一个外部官方模型适配器**，不是声称原版自带 DETR。
源码固定到 `facebookresearch/detr@29901c51d7fe8712168b8d0d64351170bc0f83e0`，复用同一 COCO 权重、person/no-object 分类头及骨干/其他参数分组。
本轮使用官方原生 `SetCriterion`，不带入旧实验的二维 CE 确定性改写；三条路径使用完全相同的实现。
该项覆盖 UP Runner、数据/损失适配与设备/训练接入，不表示已把官方 DETR 内部全部运算改写为 UP 的类别接口。

## 验证与记录

三条路径、五个模型的两步真实数据预检均完成，包含训练内 eval、恢复、CPU 指标复算及独立进程推理。
两机隐藏 GPU 的契约测试均为 54 项通过。预检不计入上述 24,960 次正式更新。

每次完整训练保留 loss/LR 轨迹、每步输入哈希、初始/最终参数哈希、最后 epoch 指标和完整 checkpoint。
权重、优化器和调度器保存后恢复；另外在新进程中进行独立推理。
保存训练结束时的预测快照，从相同 holdout 标签在 CPU 独立复算 Top-1/Top-5 或标准 COCOeval 指标。
这里的 CPU 是指标审计，不是替代 GPU 模型计算。

报告分别列出“新 NVIDIA − 原版 NVIDIA”“新沐曦 − 新 NVIDIA”“新沐曦 − 原版 NVIDIA”的百分点差及跨厂商绝对差。
同种子单次结果不能估计训练方差或证明统计等价；恢复/推理出现指标波动也单独保留，不挑有利的一次。
耗时是验证工作进程的观测值；两台机器同时运行多个队列，不将其当作独占环境的性能排名。
显存计数在训练前重置，覆盖之后的训练、最终 eval 及恢复，不能误称纯训练峰值。

两边均使用 FP32。NVIDIA 的 Torch/TorchVision 是 `2.10.0+cu126` / `0.25.0+cu126`，Python 3.12.3；
沐曦是 `2.10.0+metax3.8.1.0` / `0.25.0+metax3.8.1.0`，Python 3.12.11，MACA 3.8.1.3、driver 3.6.11。
普通依赖版本相同，厂商 Torch/SDK 和 Python 补丁版本差异保留在报告中。

每次的训练墙时、峰值显存保留在 summary 和原始报告，训练墙时包含 Runner 内的周期验证、保存及本次输入哈希记录开销。
这些不是隔离后的算子基准，也不能仅按这组时间推断厂商性能优劣。

## 证据索引

- [发布审计](validation/20261024/accuracy/publication-audit.json)：冻结源码、781 个运行时文件、共同输入及公开文件校验和。
- [NVIDIA 收集回执](validation/20261024/accuracy/nvidia/receipt.json)、[沐曦收集回执](validation/20261024/accuracy/metax/receipt.json)：完整数量、CPU 测试、原始报告包位置和 SHA256。
- [NVIDIA 原始文件索引](validation/20261024/accuracy/nvidia/index.json)、[沐曦原始文件索引](validation/20261024/accuracy/metax/index.json)：逐文件校验和、完整源映射、实际环境、训练后数据和扩展校验。
- [NVIDIA 逐次摘要](validation/20261024/accuracy/nvidia/runs)、[沐曦逐次摘要](validation/20261024/accuracy/metax/runs)：全部 15 次指标、初始化、权重、配置、轨迹索引。
- [NVIDIA 指标/推理审计](validation/20261024/accuracy/nvidia/audits)、[沐曦指标/推理审计](validation/20261024/accuracy/metax/audits)。推理文件精简了重复源码映射，保留原文件 SHA256，完整文件在原始包中。
- [原版扩展来源](validation/20261024/accuracy/original-binary-provenance.json)：原版 CUDA/C++ 源和实际二进制对应记录。

图片、checkpoint、完整日志、完整 loss/LR 曲线留在下面的原始证据目录；仓库保留约 1.2 MB 的摘要和索引。
源码/配方公开不等于已公开数据和权重。精确复现需要准备索引中的冻结数据和初始化文件，不能重新抽样后冒充相同实验。

## 原始目录与复现

两机均通过 `/tmp/up-clean-accuracy-20261024` 访问本任务；这是指向各自独立任务目录的符号链接。

- NVIDIA：`/data2/oujingfeng/up-platform-clean-20261023/accuracy-20261024`。
- 沐曦容器：`/workspace/up-run/up-platform-clean-20261023/accuracy-20261024`。
- `source-03` 是冻结的本次源码；`original` 是独立原版。原始失败和早期准备快照保留，不覆盖。
- `results/<lane>-gpu*/<model>/status.json` 给出真实训练进度，`report.json` 只有该次训练完成后才生成。
- `launch-*.json`、`execution.json` 保存控制器 PID、物理设备、命令和退出码；不要重复启动已有控制器。
- `evidence-01/reports.tar.gz` 包含原始报告、预测、标签、配置、命令和 CPU 测试 XML；不含图片、权重、二进制或凭据。下载后先核对回执中的 SHA256 再解包。
- 本次所有控制器已正常结束，GPU 文件锁随进程退出释放；未配置自动续跑，也未启用旧多种子控制器。

新环境先按 [环境说明](ENVIRONMENT.md) 准备厂商依赖，并构建自己的厂商扩展。
用 [accuracy_data.py](../../tools/accuracy_data.py) 验证并接入已冻结的数据，不能重新抽样后仍宣称使用本轮输入。
先以原版源码构建它自己的 NVIDIA 扩展，不要复制新版 `_C` 充当原版；对应二进制和源码哈希在上面的来源记录中。
本轮使用的五份初始化文件保留在原始目录的 `initial/<model>/initial.pt`。新实验可按下面的 initialize 命令各生成一次，再复制同一份文件并检查 SHA256；精确复核本轮应使用原始文件。
以下命令在已分配的单卡上执行；输出必须是新目录，`ROOT`/`BASE`/`DATA`/`INITIAL` 指向准备好的冻结文件：

```bash
# 新 UP/NVIDIA；原版改 --source "$BASE" --lane original_nvidia
# 沐曦改 --lane new_metax，并使用沐曦 Python/SDK/扩展
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det
export PYTHONPATH="$ROOT:$DEPS:$MQBENCH:$MSBENCH"
unset CUBLAS_WORKSPACE_CONFIG MCDNN_USE_DETERMINISTIC_ALGO MCDNN_STABLE_RESULT_MOD
unset PYTORCH_DISABLE_CUDA_CUDNN_TF32 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE

# 仅用于新实验准备；对五个模型各执行一次，随后所有路径共用这些文件
for model in resnet18 yolov5s vit_b16 faster_rcnn detr; do
  python "$ROOT/tools/accuracy_train.py" --source "$BASE" --data "$DATA" \
    --lane original_nvidia --model "$model" --mode initialize \
    --output "$INITIAL/$model"
done

python "$ROOT/tools/run_accuracy_cohort.py" --source "$ROOT" --data "$DATA" \
  --initial "$INITIAL" --lane new_nvidia --device 0 --lock /absolute/path/gpu0.lock \
  --output /absolute/path/new-run --models resnet18 yolov5s vit_b16 faster_rcnn detr

# 收集三条路径的全部报告和指标审计后，在 CPU 汇总；缺任何一项都会拒绝生成完整结论
python "$ROOT/tools/summarize_accuracy_cohort.py" --runs /collected/results \
  --output /collected/summary.json
```

控制器只清除子进程继承的确定性环境变量，不改共享环境。失败目录不会被覆盖或自动重跑。
第一次预检发现验证入口混用了 `cuda`/`cuda:0`；另一次独立推理发现原调度器保存了 NumPy 标量。
均已在验证工具中修正：显式使用相同设备标识，仅对本任务自己生成且校验过的完整 checkpoint 使用可信加载。
源码、模型算法和算子容差没有因此改变。一次 CPU 测试因未加入源码路径而无法导入，修正启动命令后两机 54 项通过；早期 XML 保留。
