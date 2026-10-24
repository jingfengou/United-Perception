# 五模型、三条路径的 native 真实数据训练

日期：2026-10-24。分支：`codex/up-platform-clean`。**状态：完整训练正在运行，尚无全部最终结果。**

本轮按用户确认的 CIFAR-10/PennFudan 固定预算运行单种子 131，比较：

1. 原版 UP `52aaeb9e` / NVIDIA RTX 4090。
2. 新平台层 UP / NVIDIA RTX 4090。
3. 同一份新平台层 UP / 沐曦 C500。

先完整报告精度差异，不设通过线，不做 TOST，不把重复波动直接判为接入失败。
旧多随机种子任务继续暂停；旧报告和两步预检均不计入本轮完整训练。

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

报告将分别列出“新 NVIDIA − 原版 NVIDIA”“新沐曦 − 新 NVIDIA”“新沐曦 − 原版 NVIDIA”的百分点差及跨厂商绝对差。
同种子单次结果不能估计训练方差或证明统计等价；恢复/推理出现指标波动也单独保留，不挑有利的一次。
耗时是验证工作进程的观测值；两台机器同时运行多个队列，不将其当作独占环境的性能排名。
显存计数在训练前重置，覆盖之后的训练、最终 eval 及恢复，不能误称纯训练峰值。

## 原始目录与复现

两机均通过 `/tmp/up-clean-accuracy-20261024` 访问本任务；这是指向各自独立任务目录的符号链接。

- NVIDIA：`/data2/oujingfeng/up-platform-clean-20261023/accuracy-20261024`。
- 沐曦容器：`/workspace/up-run/up-platform-clean-20261023/accuracy-20261024`。
- `source-03` 是冻结的本次源码；`original` 是独立原版。原始失败和早期准备快照保留，不覆盖。
- `results/<lane>-gpu*/<model>/status.json` 给出真实训练进度，`report.json` 只有该次训练完成后才生成。
- `launch-*.json`、`execution.json` 保存控制器 PID、物理设备、命令和退出码；不要重复启动已有控制器。

新环境先按 [环境说明](ENVIRONMENT.md) 准备厂商依赖，并构建自己的厂商扩展。
用 [accuracy_data.py](../../tools/accuracy_data.py) 验证并接入已冻结的数据，不能重新抽样后仍宣称使用本轮输入。
先在原版 NVIDIA 对每个模型运行 `accuracy_train.py --mode initialize` 生成共同初始文件，再复制同一份文件并检查 SHA256。
以下命令在已分配的单卡上执行；输出必须是新目录，`ROOT`/`BASE`/`DATA`/`INITIAL` 指向准备好的冻结文件：

```bash
# 新 UP/NVIDIA；原版改 --source "$BASE" --lane original_nvidia
# 沐曦改 --lane new_metax，并使用沐曦 Python/SDK/扩展
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det
export PYTHONPATH="$ROOT:$DEPS:$MQBENCH:$MSBENCH"
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
