# KITTI 旋转框评估器修复

用户于 2026-10-24 明确授权：单独修正原版评估器的几何错误，记录数值变化，两家重新验证。
**这是指标语义修复，不是“重构前后逐数值等价”。不改模型、训练配方或比较容差。**

2026-10-25 已从固定原版源码重新执行验证：原版全部数组与旧参考最大差为 0；相同框样例仍得到 `0.33333337` 或 `0`，修正后及独立 GEOS 参考均为 `1`。见[新的 NVIDIA 现场复现与逐项补测](NVIDIA_INTERFACE_PARITY_20261025.md#kitti原版具体错在哪里)及[原始结果](validation/20261025/nvidia-interfaces/kitti-examples.json)。

## 问题与边界

原版 Numba CUDA 通过角点内外判断、线段相交、顶点排序求交面积。
有限精度下，共边/端点的闭合边界判定会漏掉顶点；完全相同的旋转框也可能只得到部分交面积。
固定原版参考的 71 对正面积相同框中，有 64 对自 IoU 不接近 1。
编译候选强行贴近其运算顺序、或关闭 FMA，都没有解决问题，也不能作为正确几何的参考。

旧参考、失败候选、编译日志和源码归档未覆盖。`_kitti_numba.py` 保持原样，仍可专门复现旧行为；
新的 NVIDIA 和沐曦公开接口均显式调用修正的 GPU 实现，不再让两个平台使用不同的评估定义。

## 修改范围

- `up_platform/ops/_native/csrc/kitti_iou/kitti_iou.cu`：在查询框局部坐标系中进行四个闭半平面的凸多边形裁剪，累加交集面积。避免对接近边界的顶点进行不稳定的排序/重复交点组合。
- 只在这个不可微的 **GPU 评估内核**中使用 double 中间值，返回仍为 float32；不是 FP64 模型训练或 CPU 回退。
- `criterion=-1` 仍为 IoU，`0` 仍除以 query 面积，`1` 仍除以 boxes 面积，其他整数仍返回交面积。保留 UP 的非通常分母约定、顺时针角度、行列顺序和 NumPy 返回。
- 闭合边界相接但无面积时返回 0，零宽/高框与任何框的重叠返回 0。负尺寸、非有限值或错误形状明确报错，不传播 NaN 作为有效指标。
- 两家均绑定自己编译的二进制。新增 `kitti_geometry_version=2` 标识，加载旧二进制会提示重建，不静默继续使用旧候选。
- `up/`、`configs/`、其他原生内核、损失、梯度和 checkpoint 结构均没有修改。

double 计算只影响此指标的评估开销。本轮没有测量端到端评估吞吐，不宣称没有性能代价。

## 验收方法

独立参考使用 Shapely 2.1.2 / GEOS 的 [多边形求交](https://shapely.readthedocs.io/en/2.1.2/reference/shapely.intersection.html)，
双精度、`grid_size=0`，不使用量化网格或放宽判定。
Shapely **只用于测试 oracle**，不进入平台运行时依赖和模型计算路径。
固定比较阈值仍为 `rtol=1e-4, atol=1e-5`。

每种用途测试 69,119 对框、4 个 criterion，共 276,476 个输出：

| 输入集 | 框对数 | 目的 |
| --- | ---: | --- |
| 原始参考输入 | 7,100 | 与历史数值变化逐项对照 |
| 解析边界/退化框 | 144 | 重合、共边、共点、包含、局部重叠、近相切、零面积 |
| 随机旋转框 | 18,750 | 独立几何参考 |
| 大坐标 | 18,750 | 中心平移到约一百万，仍按输入 float32 的实际数值计算 |
| 细长框 | 18,750 | 宽为 0.001，检查边界稳定性 |
| 非连续 NumPy 视图 | 5,625 | 输入布局转换契约 |

另外检查 10 个手算值、交换输入的对称关系及分母方向、3 种空形状。
每组 GPU 运算执行三次，后两次与第一次逐位比较；这些重复只验证此指标，不证明模型确定性。
`training` 和 `inference` 在独立进程各执行一遍。

## 真机结果

| 检查 | NVIDIA RTX 4090 | 沐曦 C500 |
| --- | --- | --- |
| 修正后几何，training / inference | 两种用途均通过 | 两种用途均通过 |
| 独立 GEOS 参考最大绝对差 | `9.5367432e-07` | `9.5367432e-07` |
| 解析值、对称关系、自 IoU、空输入 | 通过 | 通过 |
| 同输入三次重复 | 所测输出逐位相同 | 所测输出逐位相同 |
| 原有专用算子回归 | 每种用途 31 用例，原版重复/路由均通过 | 每种用途 31 用例，原版重复/路由均通过 |
| CPU 契约 | 42 通过、0 跳过 | 42 通过、0 跳过 |

两平台此指标的 24 组保存输出也逐位相同，最大绝对差 0；只针对上述输入，不扩展为整个模型的跨厂商等价保证。
沐曦原生算子回归沿用已记录的参考 PSRoIPool 整数 scratch 布局兼容项，没有额外变更。

证据：[NVIDIA](validation/20261024/kitti-nvidia.json)、[沐曦](validation/20261024/kitti-metax.json)、[跨平台输出核对](validation/20261024/kitti-cross-vendor.json)。
三份报告核对同一归一化源码 SHA256：`efa09357d1dab0e2721b6370b279a643e9fca342aa78da863a852b8e8f19c2bf`。
二进制、全部输出、构建日志和 CPU XML 均有 SHA256；完整 `source.json` 与索引留在各服务器独立的 `kitti-fix-01` 阶段。
早期定位和模型阶段 `followup-01` 没有被本次指标修正覆盖。

## 数值变化

固定原版参考 SHA256：`b1bace744f0f8f21037b4c55e7bd8c457d09163d835195f4a335e7157bd54bb3`。
其 7,100 个框对中，以下变化超过原固定容差；未修改旧参考来追求通过。

| criterion | 超容差变化的框对 | 最大绝对变化 |
| --- | ---: | ---: |
| -1，IoU | 64 | 1.0 |
| 0，交面积/query 面积 | 64 | 1.0 |
| 1，交面积/boxes 面积 | 64 | 1.0 |
| 2，交面积 | 64 | 13.39433956 |

这些框对均为对角线上的相同框。比如 `(4,4)` 的原版 IoU 约 0.33333337，修后为 1；`(14,14)` 从 0 变为 1。
修后所有 71 对正面积相同框的自 IoU 都满足 1 的原固定容差；其余框对在该容差内。
这可能改变 KITTI BEV/3D AP，**不能把修前、修后分数直接混为同一个评估版本**。
本轮没有运行完整 KITTI 数据集，没有声称最终 AP 提升或保持不变。

## 复现

使用 [环境说明](ENVIRONMENT.md) 中已测厂商环境及对应 GPU。旧多种子任务不用恢复。

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
export DEFAULT_TASKS=cls:det CUBLAS_WORKSPACE_CONFIG=:4096:8
VENDOR=nvidia  # 沐曦容器中改为 metax
cd "$ROOT"
UP_BUILD_BACKEND="$VENDOR" python setup.py build_ext --inplace

# 测试依赖装入独立目录，不替换厂商 Torch
python -m pip install --no-deps --target "$ROOT/test-deps" shapely==2.1.2
export PYTHONPATH="$ROOT:$ROOT/test-deps:${PYTHONPATH}"

# REF 使用保留的原版 kitti-reference.npz
python tools/validate_platform_kitti.py --vendor "$VENDOR" --purpose training \
  --reference "$REF" --output "$OUT/kitti-training"
python tools/validate_platform_kitti.py --vendor "$VENDOR" --purpose inference \
  --reference "$REF" --output "$OUT/kitti-inference"
```

没有原参考文件时，在 NVIDIA 环境中通过下面命令从未修改的 Numba 代码产生它。
该工具仍按**旧数值等价**比较，因本次有意修正，退出码 1 是预期；生成的参考文件不能覆盖已有证据。

```bash
python tools/compare_platform_kitti.py --vendor nvidia --write-reference \
  --reference "$REF" --output "$OUT/legacy-comparison.json"
```

旧工具的非零退出不是新几何验收。新几何验收只看 `validate_platform_kitti.py` 的 oracle、解析值、对称性和重复结果，
`legacy_changes` 中的 `passed: false` 明确代表“与旧数值不同”。
输出目录必须是新的；`values.npz` 保存所有实际 GPU 输出，`summary.json` 保存逐组检查、历史变化及源码/二进制/参考 SHA256。

检查两家保存的指标输出时，先核对各自 `summary.json` 的 `values_sha256` 和源码/输入摘要，再运行：

```python
import numpy as np
a = np.load('/path/to/nvidia/kitti-training/values.npz')
b = np.load('/path/to/metax/kitti-training/values.npz')
assert a.files == b.files
for name in a.files:
    assert a[name].shape == b[name].shape and a[name].dtype == b[name].dtype
    np.testing.assert_allclose(a[name], b[name], rtol=1e-4, atol=1e-5)
```

原生算子回归入口仍为 `tools/compare_platform_native.py --vendor "$VENDOR" --purpose training --baseline "$BASE" --output "$OUT/native-training.json"`，
另以 `--purpose inference` 和新输出文件验证推理；沐曦显式加 `--reference-psroi-layout-fix`。
`tools/collect_platform_evidence.py` 可为独立阶段计算原始文件和扩展二进制校验并打包小报告。
