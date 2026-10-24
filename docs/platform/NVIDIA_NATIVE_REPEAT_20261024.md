# NVIDIA 同卡 native 波动与平台改造对照

日期：2026-10-24。**状态：短程梯度与独立算子重放已完成，8 次完整训练仍在串行运行。**

本轮回答 [三路径精度记录](ACCURACY_NATIVE_20261024.md) 中 YOLO、ViT 的 NVIDIA 原新版差异。
不改模型算法，不放宽 `rtol=1e-4, atol=1e-5`，不恢复旧多随机种子任务。
所有训练仍是 seed 131；同源重复不是更换 seed，也不挑选有利的一次。
冻结要求见 [protocol.json](validation/20261024/native-repeat/protocol.json)。

## 已定位的操作

**原版和新版都存在原生 CUDA 卷积权重梯度波动；脱离 UP 也可复现。**

| 模型 | 本次首次捕获的不稳定调用 | 权重梯度形状 | native 独立重放 | 确定性控制重放 |
| --- | --- | --- | --- | --- |
| YOLOv5-S | 第 1 次 `aten.convolution_backward` | `[3,512,1,1]` | 9/9 次与第一次不逐位相同，最大差约 7.45e−8 | 9/9 次逐位相同 |
| ViT-B16 | 第 157 次 `aten.convolution_backward`，patch embedding | `[768,3,16,16]` | 9/9 次与第一次不逐位相同，最大差约 2.91e−11 | 9/9 次逐位相同 |

每个捕获都运行 10 次，表中 9 次是其余结果分别与第一次比较；不是 9 个训练 seed。
两个模型均对原版、新版各取到一个捕获，四组独立重放的 native/确定性结果一致。
这些局部差异均在既有算子容差内，不能把“不逐位相同”直接称为算子正确性失败。
原始重放记录：[YOLO 原版](validation/20261024/native-repeat/operators/yolo-operators-original)、[YOLO 新版](validation/20261024/native-repeat/operators/yolo-operators-new)、[ViT 原版](validation/20261024/native-repeat/operators/vit-operators-original)、[ViT 新版](validation/20261024/native-repeat/operators/vit-operators-new)。

捕获点的原版、新版参数完全相同：Tensor 的内容哈希、形状、dtype、stride 均一致。
重放进程只加载捕获参数并调用 `torch.ops.aten`，**没有导入 `up` 或 `up_platform`**；每次核验输入没有变化。
因此，这组不稳定性不需要平台路由代码就能出现，不能归为新平台独有的问题。
这也不等于已证明上轮的全部 AP/Top-1 差异都只有这一个来源；完整同卡重复和权重交换推理仍在完成中。

ViT 配方中 10,000 张训练图、batch64，第一轮第 157 批只有 16 张。
探针在前 156 次卷积反向的两次同参数调用中未捕获差异，第 157 次才捕获。
该次输入为 `[16,3,64,64]`、上游梯度为 `[16,768,4,4]`，与小 batch 对应。
这解释了为什么前 3 步严格测试一致，却仍不足以覆盖后续训练。
这里的“首次”限定于本次探针检查的操作和执行轨迹，不冒充对所有 CUDA 内核的穷尽证明。

原生确定性控制是单独的因果诊断：设置 Torch/cuDNN deterministic，并在新进程设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。
**完整训练仍关闭这些控制**，没有将更慢的确定性路径混入 native 精度表。
PyTorch 文档明确区分：关闭 benchmark 可以固定算法选择，但算法自身仍可能非确定；其 cuDNN 源码在 deterministic 模式会过滤标记为非确定性的引擎。
见 [PyTorch 2.10 说明](https://docs.pytorch.org/docs/2.10/notes/randomness.html#cuda-convolution-determinism) 和 [Conv_v8.cpp](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cudnn/Conv_v8.cpp#L131-L159)。
本轮尚未记录实际 cuDNN engine 编号，因此不把具体归约/原子内核的猜测写成已测结论。

## 同源短程对照

同一张卡、相同初始参数和三个真实训练 batch；每次均为独立进程。
CPU/GPU RNG 状态及模块 train/eval 状态相同。梯度快照会同步 GPU，故短程观察与完整训练分开记录。

| 对照 | YOLO 第 1/2/3 步超原容差梯度数（共197） | ViT 第 1/2/3 步（共152） |
| --- | --- | --- |
| 原版重复 | 0 / 0 / 56 | 0 / 0 / 0，全部逐位一致 |
| 新版重复 | 0 / 0 / 177 | 0 / 0 / 0，全部逐位一致 |
| 原版01 vs 新版01 | 0 / 189 / 197 | 0 / 0 / 0，全部逐位一致 |

原版 YOLO 本身的原容差失败保留；不为通过测试而放宽。
六组逐步记录见 [梯度对照](validation/20261024/native-repeat/gradients)。
叶子模块类型和配置一致；少量 UP 损失/激活的 forward 源码哈希不同，来自已声明的路由改写，不自动判为错误。
例如 YOLO 的 BCE 路由仍绑定同一个 PyTorch 函数。捕获原语的共同输入检查比单独比较源码哈希更直接。

## 完整同卡协议

- GPU：4090 服务器的物理 1 卡，UUID `GPU-a11f730c-a975-2bf0-b765-3c24d83b7104`；整个任务持有同一个 GPU 文件锁。
- 原版为 UP `52aaeb9e`，新版为 `b137c68` 的运行时，复用上轮不可变 `source-03`；本轮不编辑 `up/`、`up_platform/` 或配方。
- 数据、初始权重、普通依赖、FP32/TF32 设置及 CPU 线程数与上轮相同。Torch `2.10.0+cu126`、TorchVision `0.25.0+cu126`、Python 3.12.3。
- ViT：CIFAR-10 10,000/2,000，64px/batch64，20 epoch，3140 更新；YOLO：PennFudan 120/50，320px/batch64，300 epoch，600 更新。
- 每模型按 `原版01 → 新版01 → 新版02 → 原版02` 串行执行，共 **8 次完整训练、14,960 次更新**。短程和算子诊断的更新不计入此数。
- 保留最终 epoch，不重试、不挑最好值；分别报告两次原版、两次新版、各自范围和所有新旧差。不设精度通过线，不用两次重复声称统计等价。
- 每次保存后严格重载、CPU 指标复算、新进程独立推理。随后把原版01、新版01 的最终权重分别交给另一套运行时推理，分开检查训练轨迹和推理差异。
- GPU 作业不重叠。首个完整训练子任务正常结束后插入算子诊断；只延后队列父进程的下一次启动，未暂停或终止训练/eval 子进程，已自动恢复队列。完整调度收据保留。

完整指标表等待 8 次预算与交换推理齐全后发布；当前不能用一轮或短程结果冒充完整结论。

## 复现入口

服务端目录：`/data2/oujingfeng/up-platform-clean-20261023/native-repeat-20261024`。
共同数据/初始化/两套冻结源码在相邻的 `accuracy-20261024`，具体哈希沿用 [上轮冻结协议](validation/20261024/accuracy/plan.json)。
精确复现需要这些冻结资源；源码公开不等于图片和权重已公开。

```bash
# 只使用已分配的同一张物理卡；本次控制器固定检查上述 GPU1/UUID，不能在另一张卡冒充重现
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det
export PYTHONPATH="$DEPS:$MQBENCH:$MSBENCH"
mkdir "$NEW_TASK"
cp tools/trace_native_training.py "$NEW_TASK/"
python tools/run_native_repeat.py --base "$FROZEN_ACCURACY" --output "$NEW_TASK"

# 直接重放已捕获的底层操作；该路径不导入 UP
CUDA_VISIBLE_DEVICES=1 python tools/probe_native_operators.py \
  --capture /path/to/capture-0.pt --output /new/path/native-replay.json

# 确定性对照需要独立进程；不要把此环境混到完整 native 训练中
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python tools/probe_native_operators.py --capture /path/to/capture-0.pt \
  --deterministic --output /new/path/deterministic-replay.json
```

控制器要求新任务目录预先存在，且放入同一版 `trace_native_training.py`；重复 launch 和覆盖输出会拒绝。
公开算子探针的 CPU 测试覆盖“可变目标只更新一次、纯操作不改输入、输入指纹包含布局”。
两套路径合并的 CPU 契约与新增探针测试为 **57 项通过**，见 [XML](validation/20261024/native-repeat/cpu.xml)。一次 pytest 因同时使用符号路径和实际路径，把根目录选到 `/` 而发生权限错误；改用共同实际根目录后通过，失败 XML 没有删除。
