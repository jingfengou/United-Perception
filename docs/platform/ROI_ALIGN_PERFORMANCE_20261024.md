# RoIAlign 确定性反向性能优化

日期：2026-10-24。基线：`4b8a2a8024754c36ea406b3a2a7d0314a9e6224f`；分支：`codex/up-platform-clean`。

**结论：相同输入下，约 1024 个 RoI 的确定性 AVG 反向在 C500 上快 3.72–3.90 倍，在 RTX 4090 上快 2.95–3.07 倍。**
两家各自的新旧梯度、Faster R-CNN 三步训练/训练内 eval/恢复及独立推理逐位一致。
比较容差仍为 `rtol=1e-4, atol=1e-5`；逐位一致指同平台优化前后，不指两厂商之间。
这次没有修改模型、配方和数值策略，没有恢复旧多随机种子任务，也不是数据集精度或收敛验收。

## 修改位置和原理

唯一运行时修改是 [平台层 RoIAlign 内核](../../up_platform/ops/_native/csrc/roi_align/roi_align_kernel.cu) 的
`ROIAlignAvgBackwardDeterministic` 线程索引：

| 项目 | 优化前 | 优化后 |
| --- | --- | --- |
| 逻辑线程顺序 | 相邻线程沿 NCHW 的 x 方向排列 | 相邻线程沿 channel 方向排列 |
| 写回地址 | `bottom_diff[index]` | 由逻辑坐标算出 NCHW `target` 后写回 |
| 单个元素的累加 | RoI → bin → sample → 四个插值角 | 完全相同 |
| 梯度所有权 | 每个元素一个线程 | 每个元素一个线程 |

对于实测的 256 通道 FPN，相邻线程处理相同空间位置的不同通道，RoI/bin/sample 的分支更一致。
物理布局仍是 NCHW，仍用原 dtype，仍从已有梯度缓冲区开始累加；没有原子加法或新的归约顺序。
线程块仍为 256，当前设备和 stream 的契约不变。两家分别使用自己的 SDK 编译同一份源码。

试过显式 FP32 `ceil` 特化和 128 线程块，均没有稳定的额外收益，最终没有采用；探针源码和结果留在证据目录。
Torch 确定性开关关闭时仍使用原来的原子路径。本报告比较的是**旧确定性实现与新确定性实现**，不把原子路径当成旧基线。

## 相同输入的正式扩展测量

使用完整生产扩展，旧、新二进制各在独立进程加载。输入先从旧版实际 UP Faster R-CNN 的三步训练捕获，
再把同一组 CPU 保存的张量复制到两台机器；8 个文件的 SHA256 全部相同。并非各平台分别随机生成一组近似输入。

前 6 组均为 FP32、batch=2、channels=256、pooled=7×7、adaptive sampling、`aligned=False`。
16×16 特征的 scale 为 0.125，32×32 为 0.25。
RoI 多于 64 时预热 3 次，测 5 轮×5 次；其余预热 20 次，测 5 轮×100 次。
表中是每轮均值的中位数，包含双方相同的梯度缓冲区清零；不是整网吞吐。

| case：RoI 数/特征边长 | 4090 旧 ms | 4090 新 ms | 加速比 | C500 旧 ms | C500 新 ms | 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 00：1 / 16 | 0.04186 | 0.03216 | 1.30 | 0.18769 | 0.13750 | 1.37 |
| 01：1023 / 32 | 19.84143 | 6.72397 | 2.95 | 172.82268 | 44.36936 | 3.90 |
| 02：3 / 16 | 0.10283 | 0.08308 | 1.24 | 0.63907 | 0.52141 | 1.23 |
| 03：1021 / 32 | 19.71466 | 6.42742 | 3.07 | 171.29272 | 45.47989 | 3.77 |
| 04：2 / 16 | 0.05333 | 0.05228 | 1.02 | 0.40087 | 0.34803 | 1.15 |
| 05：1022 / 32 | 19.93462 | 6.49477 | 3.07 | 172.66602 | 46.43779 | 3.72 |

另外两组用固定 CPU generator 构造，检查 channel 尾部、边界/退化 RoI 和非零梯度缓冲区：

| case | 契约 | 4090 旧→新 ms | C500 旧→新 ms |
| --- | --- | ---: | ---: |
| 06 | FP16、17 通道、aligned、sampling=2、已有梯度 0.25 | 0.02941→0.02442 | 0.09773→0.08545 |
| 07 | FP64、3 通道、非 aligned、adaptive、已有梯度 0.25 | 0.22236→0.17811 | 0.37305→0.37204 |

8 组均为旧、新各重复 3 次，结果全部逐位一致。C500 的小规模 FP64 基本没有提速，不能把大规模 FP32 的收益推广到所有形状。
FP16 这里只验证直接反向调用，不代表完整 AMP 训练通过。
所有逐轮计时、形状、dtype、输入校验和与结果校验和见 [NVIDIA 摘要](validation/20261024/roi-performance/nvidia.json) 和 [沐曦摘要](validation/20261024/roi-performance/metax.json)。

## 模型和契约回归

两家均执行同一 UP Faster R-CNN 配方、CPU 初始权重、种子 131、batch=2、三次更新，
每步训练内 eval，第二步后重建模型/优化器并恢复；独立推理各起新进程，加载同一份旧版训练 checkpoint。
原环境版本与厂商 DNN 控制保持一致，见 [环境说明](ENVIRONMENT.md)。

| 检查 | NVIDIA | 沐曦 |
| --- | --- | --- |
| 旧训练 vs 新训练，含梯度、eval、状态、恢复 | 1947 项通过且逐位一致 | 1947 项通过且逐位一致 |
| 新训练 vs 新训练独立重复 | 1947 项通过且逐位一致 | 1947 项通过且逐位一致 |
| 同 checkpoint 独立推理 | 54 项通过且逐位一致 | 54 项通过且逐位一致 |
| 独立 CPU/autograd 参考、数值梯度、空输入、已有梯度、非默认 stream | 通过 | 通过 |
| 完整扩展专用算子回归 | 训练/推理各 31 用例通过 | 训练/推理各 31 用例通过 |
| 隐藏 GPU 的 CPU 契约 | 49 项通过 | 49 项通过 |

两家各自用于对照的算子原始输出、训练/重复训练和独立推理 `.pt` 归档 SHA256 也全部相同，摘要保留了这些校验值。

额外记录 batch 迁移与 forward/backward 的 CUDA event 区间，排除优化器更新、eval、恢复及测试用 CPU 证据拷贝。
下表保留全部三步，包括冷启动；它不是端到端 step、训练吞吐或完整 Runner 测量。

| 平台/运行 | 第 1 步 ms | 第 2 步 ms | 第 3 步 ms |
| --- | ---: | ---: | ---: |
| 4090 旧 | 746.10 | 75.19 | 76.90 |
| 4090 新 | 722.17 | 63.24 | 67.49 |
| 4090 新重复 | 742.36 | 76.45 | 78.97 |
| C500 旧 | 3954.27 | 234.61 | 224.47 |
| C500 新 | 4480.15 | 111.06 | 97.36 |
| C500 新重复 | 3823.65 | 103.72 | 98.16 |

C500 后两步开销明显下降，但首步仍受冷启动影响。4090 的新版本重复存在计时波动，不能据此声称稳定的整网加速比。
确定性反向仍需扫描 RoI 和采样点，这次没有证明它与原子路径同速；首次修复的原子/确定性比较保留在 [历史报告](FRCNN_DETERMINISM_20261024.md)。

## 复现

使用 [已测环境](ENVIRONMENT.md)，准备三个独立目录：原版 `BASE` 为 `52aaeb9e`，旧平台 `OLD` 为 `4b8a2a8`，新平台 `NEW` 为本分支。
只把当前测试工具复制给 `OLD`，不改它的 `up/`、`up_platform/` 和 `configs/`。
以下为 Linux Bash 命令；`OUT` 必须是新目录，`DEPS`/`MQBENCH`/`MSBENCH` 指向已准备的固定依赖。

```bash
set -euo pipefail
VENDOR=metax  # NVIDIA 环境改为 nvidia，保留对应厂商 PyTorch
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det CUBLAS_WORKSPACE_CONFIG=:4096:8
if [ "$VENDOR" = metax ]; then
  export MCDNN_USE_DETERMINISTIC_ALGO=1 MCDNN_STABLE_RESULT_MOD=1
fi
for tool in compare_platform_models.py capture_roi_align_cases.py benchmark_roi_align.py compare_roi_align_models.py; do
  cp "$NEW/tools/$tool" "$OLD/tools/$tool"
done
for ROOT in "$OLD" "$NEW"; do
  (
    cd "$ROOT"
    UP_BUILD_BACKEND="$VENDOR" python setup.py build_ext --inplace
  )
done
export PYTHONPATH="$OLD:$DEPS:$MQBENCH:$MSBENCH"
python "$NEW/tools/capture_roi_align_cases.py" --root "$OLD" --baseline "$BASE" \
  --vendor "$VENDOR" --output "$OUT/captures"
python "$NEW/tools/prepare_roi_align_cases.py" --source "$OUT/captures" --output "$OUT/cases"
# 跨机器使用同一份 cases；传输完成、校验整个包后再解包，不要分别重新采样。
python "$NEW/tools/benchmark_roi_align.py" --cases "$OUT/cases" \
  --baseline-binary "$OLD"/up_platform/backends/_"$VENDOR"_ops*.so \
  --candidate-binary "$NEW"/up_platform/backends/_"$VENDOR"_ops*.so --output "$OUT/operator"
export PYTHONPATH="$NEW:$DEPS:$MQBENCH:$MSBENCH"
python "$NEW/tools/compare_roi_align_models.py" --baseline-root "$OLD" --candidate-root "$NEW" \
  --upstream-root "$BASE" --vendor "$VENDOR" --output "$OUT/models"
python "$NEW/tools/validate_platform_roi_align.py" --vendor "$VENDOR" --output "$OUT/oracle"
CUDA_VISIBLE_DEVICES= python -m pytest "$NEW/tests/platform" -q
```

`benchmark_roi_align.py` 在加载可信 `.pt` 前核对文件集合和每个文件的 SHA256；新旧完整扩展使用不同进程，避免 pybind 类型重复注册。
本次公开的 fixture 生成器也已在 CPU 重建出同样 8 个文件校验和。只加载自己生成或已审核的张量文件与厂商二进制。
更新源码后需要重编译才能获得提速；`deterministic_backward_version=1` 标记的是契约，不是本次性能版本，实际二进制以报告 SHA256 为准。

## 证据与边界

- [NVIDIA 完整索引](validation/20261024/roi-performance/nvidia-index.json)、[沐曦完整索引](validation/20261024/roi-performance/metax-index.json) 保留源码、旧/新二进制、固定种子 131 的全部结果和失败启动的 SHA256；大张量、checkpoint、日志留在服务器，不是公开下载链接。
- 任务目录分别为 `/data2/oujingfeng/up-platform-clean-20261023/roi-perf-20261024`、`/workspace/up-run/up-platform-clean-20261023/roi-perf-20261024`。
- 测试冻结源的归一化 SHA256 为 `667282988afa399a19f14f5776e5dee7962395d98dddfa975463e217031572d8`，两家相同。相对于旧参考，只有一个内核文件变化；`up/` 和 `configs/` 完全相同。
- 发布整理仅把基准工具的 SHA256 计算改为 Python 3.10 兼容的分块读取，并加入公开 CPU fixture 生成器；运行时与真机测试源一致。逐文件核对见 [发布审计](validation/20261024/roi-performance/publication-audit.json)。
- 沐曦一次输入包在传输完成前被解包，导致初始探针读取失败；原目录保留，完整包校验并解包到新目录后重测通过。NVIDIA 一次命令使用了错误二进制路径，未运行内核，记录保留。两者均未混入最终数值或性能结果。
- 没有扩展到所有形状、dtype、AMP、分布式、长程训练或数据集精度。原有 linklink/Amba 等缺口不因这次优化而变为通过，见 [验证状态](VALIDATION.md)。
