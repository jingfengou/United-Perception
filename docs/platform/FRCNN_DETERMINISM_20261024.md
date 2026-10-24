# Faster R-CNN 训练差异定位与修复

日期：2026-10-24。原版模型基线：UP `52aaeb9e`。修复前平台版本：`19a680a`。
本轮固定 seed=131、原配置、CPU 初始化权重和输入，比较阈值仍为 `rtol=1e-4, atol=1e-5`。
`up/` 和 `configs/` 未修改，旧多随机种子任务没有恢复。

## 根因

两台机器的首个差异都来自 **UP 原版自带的 `RoIAlign` AVG 反向 CUDA 内核**。
原版使用多个线程向同一输入像素执行 `atomicAdd`；加法顺序随 GPU 调度变化，浮点舍入结果也随之变化。
固定随机数种子无法固定这种累加顺序。原内核没有读取 Torch 的确定性开关，因此打开该开关后仍会执行原子累加。

新增的 pybind 边界跟踪同时观察 RoIAlign、IoU 和 NMS，检查可读输入及有效输出。
NMS 的未初始化 scratch 尾部不参与比较；RoIAlign 反向已有的梯度缓冲区属于可读输入。

| 修复前原版重复，调用编号从 0 开始 | NVIDIA RTX 4090 | 沐曦 C500 |
| --- | --- | --- |
| 首个位差异 | 1773：`native.roi_align.backward_avg_cuda` | 1765：同一操作 |
| 该操作输入 | 逐位一致 | 逐位一致 |
| 该次输出 | 有位差异，当时仍在容差内 | 有位差异，当时仍在容差内 |
| 首次超容差 | 1937：`aten.convolution_backward.default` | 1939：同一操作 |
| 卷积反向输入 | 已不同 | 已不同 |

此前只跟踪 ATen 时看到的卷积反向是误差放大后的观察点；补上原生扩展边界后，才能定位到更早的 RoIAlign。
调用编号只描述这次固定环境，不能用作所有模型/版本的固定索引。

将真实的首个差异输入单独交给原二进制，在**不导入 UP 模型或平台包**的进程里调用 8 次，后 7 次均出现位差异，两家相同。
NVIDIA 捕获为 `top_grad[1,256,7,7] / rois[1,5]`，沐曦为 `[3,256,7,7] / [3,5]`；输入梯度均为 `[2,256,16,16]`。
这是同平台重复测试，两家各自使用自己的捕获，没有把不同厂商输出当作逐位参考。

沐曦沿用此前已验证的两个显式 MCDNN 控制开关，以固定另外已知的卷积非确定性。
这些开关控制 DNN 库，不能控制 UP 自己的 RoIAlign 内核。相关环境限制见 [前次跟进](FOLLOWUP_20261024.md)。

## 平台层修复

修改集中于 [RoIAlign 内核](../../up_platform/ops/_native/csrc/roi_align/roi_align_kernel.cu)、
[绑定版本](../../up_platform/ops/_native/csrc/pybind.cpp) 和 [平台 autograd 入口](../../up_platform/ops/detection/_roi_align.py)。

1. Torch 确定性开关开启时，每个输入梯度像素只有一个写入线程，按固定的 RoI、bin、采样点及四个插值权重顺序累加。
2. 保留原采样、边界、平均池化和梯度公式，使用原 dtype；保留缓冲区中已有梯度。FP32 模型仍用 FP32 计算。
3. 开关关闭时继续使用原子路径。默认数值策略没有改成实验性 v1/v2/portable。
4. 前向和反向使用当前设备/current stream，避免非默认 stream 下与输入准备、梯度清零发生竞争。
5. FP64 原子反向原来每线程使用 74 个寄存器，1024 线程启动在 4090 上失败；FP64 的线程块改为 256。FP32 原子路径仍为 1024。
6. `roi_align.deterministic_backward_version=1` 标记新扩展。Python 入口在确定性反向中遇到旧二进制会明确要求重编译。

NVIDIA、沐曦复用这份源代码，分别使用自己的 SDK 编译，继续通过各厂商显式绑定加载各自二进制。
父类签名、旧 `.apply()`、checkpoint 参数键和 ONNX symbolic 保持原样。

## 真机验证

最终验收阶段为 `frcnn-fixed-03`。先前阶段及失败保留在证据索引中。

| 检查 | NVIDIA | 沐曦 |
| --- | --- | --- |
| 同一捕获，固定顺序反向 8 次 | 后 7 次逐位一致 | 后 7 次逐位一致 |
| 同一新二进制，关闭确定性开关 | 后 7 次均有位差异 | 后 7 次均有位差异 |
| 修后梯度对原捕获参考 | 原容差内 | 原容差内 |
| FP32/FP64 独立 CPU autograd 参考 | 通过 | 通过 |
| FP64 数值梯度检查 | 通过 | 通过 |
| aligned、固定/自适应采样、空 RoI | 通过 | 通过 |
| 微小/零面积/倒置/越界 RoI | 通过 | 通过 |
| 非连续上游梯度、已有梯度缓冲区 | 通过 | 通过 |
| 非默认 stream | 通过 | 通过 |

CPU 参考仅用于算子审计；训练没有 CPU 代算。
FP16/AMP、分布式训练、大规模 RoI 和真实数据集最终精度不在本轮验收范围。

模型协议保持为实际 UP ModelHelper、原网络/损失/优化器、batch=2、3 次更新，每步 eval，第 2 步后重建并恢复。
独立推理在新进程使用同一份原版 checkpoint。原版 Python 模型和路由版均明确使用同一修正扩展及相同开关；
这验证修复后的模型接口兼容性，不把修复前不稳定的原子结果改写成通过。

| 最终模型与回归 | NVIDIA | 沐曦 |
| --- | --- | --- |
| 原版 Python 模型重复，训练/eval/恢复 | 通过，1947 个比较项 | 通过，1947 个比较项 |
| 原版 Python 模型与路由版，训练/eval/恢复 | 通过，1947 个比较项 | 通过，1947 个比较项 |
| 原版重复及路由独立推理 | 通过，各 54 项 | 通过，各 54 项 |
| 原有专用算子，训练/推理 | 每种用途 31 用例，重复/路由均通过 | 每种用途 31 用例，重复/路由均通过 |
| CPU 契约 | 47 通过，0 跳过 | 47 通过，0 跳过 |

比较项包含张量与元数据。输入为固定合成数据，这些结果证明本协议下的兼容性和重复性，尚未证明真实数据集最终精度。

## 局部性能代价

实际捕获输入的热重放，包含相同梯度缓冲区清零，预热 20 次，再测 5 轮 × 100 次，表中为每轮均值的中位数：

| 平台内比较 | 原子路径 | 固定顺序路径 | 倍率 |
| --- | ---: | ---: | ---: |
| NVIDIA，1 个 RoI | 0.01590 ms | 0.04187 ms | 2.63 |
| 沐曦，3 个 RoI | 0.03162 ms | 0.56286 ms | 17.80 |

固定顺序实现有实际开销。不同厂商捕获的 RoI 数不同，此表只比较各自路径，不能用于跨卡性能排名；
这些数值也不能直接换算为整网吞吐损失，大规模 RoI 的性能尚未测量。
关闭 Torch 确定性开关时，FP32 保留原子实现，便于按需求选择。
性能脚本和逐轮测量保存在原始阶段的 `benchmark_roi.py`、`results/benchmark.json`，完整索引记录其 SHA256。

## 证据

- [NVIDIA 摘要](validation/20261024/frcnn/nvidia.json)、[沐曦摘要](validation/20261024/frcnn/metax.json) 包含原版重复跟踪、独立重放、CPU 数值参考、模型结果、环境开关及二进制校验。
- 最终归一化源码 SHA256：`0b8bd0e1e4e7cb67763878f16d009adec158c8a8530f168ece66921c71626204`，两家相同。
- NVIDIA 最终二进制：`aec7a4a918049ead35d0fdb61ff98af21ea6b29b484af269efd7c98f033f9869`。
- 沐曦最终二进制：`ad90d516639e87be520ba8f7c6c7684c119ffb40a81eb70289563f969266738c`。
- 完整原始张量、模型状态、源码、编译日志与文件索引留在两台任务目录；仓库保存摘要。原始文件不是公开下载链接。
- `frcnn-20261024` 保留修复前捕获；`frcnn-fixed-01` 保留 FP64 启动/测试布局失败；`fixed-02` 保留中间检查；`fixed-03` 增加已有梯度缓冲区契约并完成最终验收。
- 各阶段工具版本和源码摘要分别记录，没有以最终工具冒充早期工具版本。修复前独立扩展加载不依赖 UP 模型；算子仍是 UP 的编译内核，不是 TorchVision 的另一实现。

## 开启与复现

使用 [环境说明](ENVIRONMENT.md) 的两套已测环境，先重编译：

```bash
cd "$ROOT"
UP_BUILD_BACKEND="$VENDOR" python setup.py build_ext --inplace
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
# 沐曦的已测组合：
if [ "$VENDOR" = metax ]; then
  export MCDNN_USE_DETERMINISTIC_ALGO=1 MCDNN_STABLE_RESULT_MOD=1
fi

# 原版 BASE 必须是 52aaeb9e 的独立目录；OUT 必须是新的
python tools/validate_platform_roi_align.py --vendor "$VENDOR" --output "$OUT/operator"
python tools/compare_platform_models.py --root "$ROOT" --baseline "$BASE" \
  --vendor "$VENDOR" --models faster_rcnn --deterministic --output "$OUT/models"
```

`--deterministic` 在这个对照工具中调用 `torch.use_deterministic_algorithms(True)` 并设置卷积控制。
在自己的训练入口中，需于模型运行前显式执行这个 Torch API；只设置 `cudnn.deterministic` 不会选择 RoIAlign 固定顺序路径。
该开关的语义见 [PyTorch 2.10 文档](https://docs.pytorch.org/docs/2.10/generated/torch.use_deterministic_algorithms.html)。

重现旧问题时，在独立目录构建 `19a680a` 的厂商扩展，用当前诊断工具跟踪它：

```bash
# OLD 包含 19a680a 的 up、up_platform、configs、tools、setup.py 等
PYTHONPATH="$OLD:$DEPS:$MQBENCH:$MSBENCH" \
python "$ROOT/tools/diagnose_platform_operator.py" --root "$OLD" --baseline "$BASE" \
  --vendor "$VENDOR" --model faster_rcnn --variant original --native-ops --limit 2400 \
  --output "$OLDOUT/original"
PYTHONPATH="$OLD:$DEPS:$MQBENCH:$MSBENCH" \
python "$ROOT/tools/diagnose_platform_operator.py" --root "$OLD" --baseline "$BASE" \
  --vendor "$VENDOR" --model faster_rcnn --variant repeat --native-ops --limit 2400 \
  --reference "$OLDOUT/original" --output "$OLDOUT/repeat"

# 只加载自己产生且可信的 pt 和匹配厂商的二进制；每个命令是独立进程
python "$ROOT/tools/diagnose_platform_operator.py" --capture "$OLDOUT/repeat/first-bitwise.pt" \
  --native-binary "$OLD"/up_platform/backends/_"$VENDOR"_ops*.so \
  --repeats 8 --output "$OUT/replay-old.json"
python "$ROOT/tools/diagnose_platform_operator.py" --capture "$OLDOUT/repeat/first-bitwise.pt" \
  --native-binary "$ROOT"/up_platform/backends/_"$VENDOR"_ops*.so \
  --repeats 8 --output "$OUT/replay-fixed.json"
python "$ROOT/tools/diagnose_platform_operator.py" --capture "$OLDOUT/repeat/first-bitwise.pt" \
  --native-binary "$ROOT"/up_platform/backends/_"$VENDOR"_ops*.so --replay-atomic \
  --repeats 8 --output "$OUT/replay-retained-atomic.json"
```

达到跟踪上限仅代表该调用前缀已保存，不代表完整训练通过。完整模型比较由前面的 `compare_platform_models.py` 执行。
