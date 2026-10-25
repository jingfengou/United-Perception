# NVIDIA：156 项平台接口补测与差异记录

日期：2026-10-25。原版 UP：`52aaeb9ed27a490820696d7754d19f98cd6ecd11`。

**原完整清单含 156 项接口，其中 152 项已执行相应对照，4 项因真实依赖缺失而阻塞；没有未执行却填为零的接口。**
这是有限输入下的接口、梯度和状态检查，不是全部输入空间、所有可选 SDK 变体或模型收敛精度的证明。
本轮没有改 `up/`、`up_platform/`、`configs/`、模型算法或原容差，没有恢复 DETR、多随机种子或长程训练。

[156 项 CSV](validation/20261025/nvidia-interfaces/interfaces.csv) · [逐项摘要](validation/20261025/nvidia-interfaces/summary.json) · [完整小型元数据包](validation/20261025/nvidia-interfaces/metadata.tar.gz) · [原始证据索引](validation/20261025/nvidia-interfaces/evidence-index.json)

## GPU 相关口径（2026-10-26 整理）

用户要求重点关注 GPU 操作，本节从上述已保存的真机结果筛选范围，**没有新增 GPU 执行或改写旧结果**。
GPU 相关接口包括实际 GPU 计算，以及决定 GPU 行为的设备、AMP、DDP、通信和工具链控制；不能因为控制代码或返回元数据在 CPU 上，就把 GPU 运行时接口排除。

| GPU 相关结果 | 接口数 |
| --- | ---: |
| 所测数值或状态与参考一致 | 146 |
| MQBench 合法初始化非有限状态一致 | 1 |
| KITTI 几何错误有意修正 | 1 |
| 真实依赖阻塞 | 4 |
| 合计 | **152** |

因此 GPU 范围内 148 项已执行相应对照、4 项依赖阻塞。这里的 148 与此前全接口执行的 152 不同，后者还包含 4 个 CPU 接口。
专用扩展按 GPU 分支筛选后为 **16 个操作、每种用途 26 个配对用例，共 52 个**；下文历史表的 19 个操作/62 个用例包含 CPU 操作与分支。

[GPU 逐项 CSV](validation/20261025/nvidia-interfaces/gpu-interfaces-20261026.csv) · [机器可读范围、排除规则和原始证据 SHA256](validation/20261025/nvidia-interfaces/gpu-scope-20261026.json)

以下仅 CPU 计算的接口不计入 GPU 适配验收：

- `pointcloud.points_in_boxes_cpu`、`pointcloud.boxes_bev_iou_cpu`、`sparse.voxelize_cpu`。
- `detection.softer_nms`：虽然接受 GPU Tensor，但原版和新版都将数据转到 CPU 调用 `cpu_softer_nms`，再把结果放回输入设备；这不是新加的 CPU 回退，也不是 GPU 内核验收。

`detection.naive_nms` 保留 GPU 分支，剔除 CPU 输入的 4 条配对记录。总共排除 11 条 CPU 记录；GPU 范围保留 834 条匹配、2 条相同报错、18 条相同非有限状态、1 条几何修正及 6 条依赖阻塞记录。
原有 CPU 功能、133 项 CPU 软件检查及完整历史表保留，但不拿它们增加 GPU 通过数。
KITTI 返回 NumPy、GPU RNG 返回 CPU 状态字节，并不意味着这些操作在 CPU 上计算；分类按实际实现和用途，不按返回容器猜测。

下面继续保留原 **156 项全接口历史结果**，其数字没有因调整关注范围而被覆盖。

## 全接口历史结果

| 接口级结果 | 项数 | 含义 |
| --- | ---: | --- |
| 所测数值或状态与参考一致 | 150 | 浮点保持 `rtol=1e-4, atol=1e-5`；整数、键、类型及适用状态精确核对 |
| 初始非有限状态相同 | 1 | `quantization.prepare` 的未校准观察器含 `Inf/-Inf` 哨兵，位置、符号及其余有限字段相同；不冒充有限数值通过 |
| 有意修正原版几何错误 | 1 | `pointcloud.rotate_iou_eval`，下节给出原始代码现场复现与独立参考 |
| 缺依赖，未完成执行验收 | 4 | linklink 融合优化器，以及 Amba 的三个接口；没有拿替代或空操作冒充 |
| 合计 | **156** | 当前父类公开方法数；不是 GPU 内核数或全部输入变体数 |

详细记录有 **845 条容差/状态匹配、2 条相同报错、18 条相同非有限边界状态、1 条几何修正、6 条依赖阻塞**。
同一接口可能有多个 dtype、用途、布局或优化器名称；双卡的两 rank 分开记录，这些数目不是独立统计样本数。
每项普通对照执行参考、参考重复、平台、平台重复；没有按最高分挑选。

### 实际测到了哪些内容

| 范围 | 本轮实际检查 |
| --- | --- |
| 设备 18 项 | 设备/属性、绑定、Tensor/模型/嵌套容器迁移、同步、显存计数与活存储、GPU RNG 保存恢复、CPU RNG 不被 GPU seed 改动、卷积开关与环境 |
| 普通 Torch 类别 64 项 | 全部入口；FP32、非连续 FP32、FP64、autocast FP16；训练/推理分进程；另有空批次、适用的 in-place、构造初始化/RNG/参数身份、共享 Q/K/V 的 Attention |
| UP 专用操作 19 项 | 原版 Python/autograd **加载原版 `_C`**，新版加载自己的 `_nvidia_ops`；训练/推理各 31 用例，输入校验和逐项相同 |
| 训练/推理 | autocast、scaler 正常/溢出更新及恢复、SGD/AdamW 更新/恢复、裁剪、训练内 eval 的混合子模块模式和状态恢复、legacy ONNX 图；融合优化器缺 SDK |
| 通信 17 项及同步层 | 两张真实 GPU、NCCL；初始化/结束、组、同步/异步集合通信、对象通信、DDP 更新/恢复、SyncBN 前向/反向及运行统计 |
| 稀疏 11 项 | 三类卷积及反向、稀疏张量/序列、features/dense、replace_feature 的别名、类型/继承、CPU voxel 原有分支及 scratch 独立性 |
| 量化 7 项 | 实际 MQBench prepare、校准、QAT 更新、类型/BackendType、4 次 QDrop 重构、原有部署 ONNX 转换；初始化哨兵与校准后的状态分别记录 |
| 剪枝 10 项 | Ampere 2:4 的真实 mask、更新、初始化优化器、稀疏率、迭代值；可用 BaseScheduler/DefaultFakeSparse 的实际 mask 导出单列；原 Amba 缺口保留 |
| KITTI | 原始 Numba CUDA 重新执行 7,100 框对 × 4 种 criterion，与原始保存值及测试专用 GEOS 比较 |

稀疏 voxel 本来就是显式 CPU 接口；CPU 还用于保存结果和复算差值，不是把 GPU 模型转到 CPU 代算。
设备句柄、属性、RNG 和上下文按状态语义检查，不能统一解释成一个精度误差。
没有执行完整部署引擎、NART、剪枝 permutation=True 或所有 dtype/布局/形状组合。
RoIAlign 可选确定性反向的既有验收见[独立报告](FRCNN_DETERMINISM_20261024.md)；本轮专用扩展对照使用默认 native 路径。

### 测得的非零普通数值差

下表是本批用例中输出、适用梯度和状态字段的最大绝对差。没有超原容差，不等于任意输入都逐位一致。

| 接口 | 参考→新版最大绝对差 | 参考自身重复最大差 | 说明 |
| --- | ---: | ---: | --- |
| `deformable.deform_conv` | `4.76837158203125e-7` | `7.152557373046875e-7` | 原版与新版各自的编译扩展；差异出现在训练梯度 |
| `pooling.Upsample` | `1.1920928955078125e-7` | `1.1920928955078125e-7` | 直接 Torch 与平台；所测训练反向有低位波动 |
| `pooling.interpolate` | `1.1920928955078125e-7` | `1.1920928955078125e-7` | 同上；AMP 用例中另一次为 `5.960464477539063e-8` |

这些微小差异在参考自身重复中也观察到；本轮没有将其归因为某个厂商内部指令，也没有改实现消除波动。
其他有适用有限 Tensor 比较的接口，本批保存字段的最大差为 0；无数值字段的元数据接口仍写 `null`。
这不是宣称所有接口或模型训练永远逐位相同。

两种边界行为另外记录：

- `binary_cross_entropy` 在 autocast 内，两边都以相同错误拒绝执行；FP32 等其他用例正常比较。不关闭检查、不改用另一种损失。
- 8 个损失入口的空批次均值在两种用途下产生相同非有限结果；另有两组 MQBench prepare 初始化哨兵。共 18 条记录保留原始有限性检查失败，并额外核对非有限位置/符号及有限部分，没有放宽数值容差或称为有限数值通过。

## KITTI：原版具体错在哪里

**是原版这个评估器的部分几何边界计算有错，不是“UP 所有模型算法有错”。**

IoU 是交集面积除以并集面积。两个完全相同、面积大于零的框，自 IoU 应为 1。
本次从固定原版源码重新执行，得到：

| 相同框样例 | 原版 Numba CUDA | 新版 GPU 评估器 | 独立 GEOS 参考 |
| --- | ---: | ---: | ---: |
| `(4,4)` | `0.3333333730697632` | `1.0` | `1.0` |
| `(14,14)` | `0.0` | `1.0` | `1.0` |

完整框坐标、源码/旧参考 SHA256 及四种 criterion 结果见[现场复现记录](validation/20261025/nvidia-interfaces/kitti-examples.json)。
原版重新计算的全部数组与此前保存值最大差 **0**，不是引用另一套输入来解释旧问题。

原算法的角点内外判断、线段交点和排序在共边/端点等闭合边界上会漏掉交集顶点，导致相同旋转框的交面积不完整。
修正版在查询框局部坐标系做闭半平面的多边形裁剪，仅评估内核使用 double 中间值，输出仍为 float32。

在这批 7,100 对框中有 71 对正面积相同框，64 对旧自 IoU 有误。修后：

- `criterion=-1/0/1` 的新旧最大差均为 **1.0**；交面积 `criterion=2` 最大差 **13.394339561462402**，后者是面积单位。
- 四种模式的新实现与 GEOS 的本批最大差均为 **0**。
- 这是修正错误指标值；不是模型精度下降 1 个百分点，也不能推出 KITTI AP 提升多少。本轮没有完整 KITTI 数据集 AP。
- 旧、新评估版本的 KITTI 分数不能混用。模型前向、损失及参数没有因此修改。

之前更广的随机/退化/大坐标/细长框及两家真机证据保留在[KITTI 几何报告](KITTI_GEOMETRY_20261024.md)。

## 四项阻塞与变体限制

| 接口 | 实际缺口 |
| --- | --- |
| `training.fused_optimizer` | 缺 `spring.linklink`；三个原融合优化器均明确报缺依赖，未改用普通优化器 |
| `pruning.prune` | UP Amba 配方所需 `FakeSparse` 不在当前 MSBench 中 |
| `pruning.last_prune_step` | 同一 Amba 流程无法构造运行，不能用常量或另一个策略冒充 |
| `pruning.set_lr_scheduler` | 同上，未用假 scheduler 验收 |

`pruning.export` 的可用 BaseScheduler/DefaultFakeSparse 流程已实际导出 mask，并核对权重归零和状态；
它不能证明 Amba 导出已支持。普通 Torch SyncBN/AMP/通信的通过也不代表 linklink 专用变体通过。
上层仍有 11 处旧 linklink 导入待隔离，见[依赖与迁移缺口](LINKLINK.md)。

## 环境、原版边界与证据

- 同一台 RTX 4090；单卡对照在物理 GPU1 串行执行，多卡使用原授权 GPU0/1。UUID 与原任务一致；图形进程保留。
- Python 3.12.3、Torch `2.10.0+cu126`、TorchVision `0.25.0+cu126`；现有固定 MQBench/MSBench/spconv 依赖复用，未修改共享安装。
- native：cuDNN benchmark=True、deterministic=False、TF32=False；同样的种子 131 与对照控制。
- 原版 `_C` SHA256：`d49bb62be6f70cb8003f7ef2c94999c562e6215181f80920135f00db1340c59a`。
- 新版 `_nvidia_ops` SHA256：`ac30ea811dd4999890dbb4eee285327d9627bf78507847cac0fcedc5a7fadd0a`。
- 原版 618 个运行源文件逐字节归一化后匹配上游；新版 781 个运行源文件匹配当前仓库。只统一 CRLF/LF，两个 README 不计运行源，见[源文件核对](validation/20261025/nvidia-interfaces/runtime-source-match.json)和[完整源清单](validation/20261025/nvidia-interfaces/runtime-manifest.json)。
- 最终 CPU 契约与分析测试 **133 通过、0 失败、0 跳过**，见 [XML](validation/20261025/nvidia-interfaces/cpu.xml)。

发布前只删除了测试工具的 6 个未使用导入；相对最终 CPU 快照，函数体 AST 不变，[工具源核对](validation/20261025/nvidia-interfaces/tool-source-match.json)保留前后 SHA256。各 GPU 阶段仍使用各自归档的工具，不用当前文件冒充早期执行文件。

参考类型分开保留：专用操作使用原版 Python/二进制；普通计算使用原 UP 依赖的独立 Torch API；
可选工具链使用其真实依赖直接调用；新增设备/验证上下文辅助方法检查明确的状态契约。
不能把后三种全部改称“每一个原 UP 函数原封不动执行”。第三方库既有环境兼容项见[环境说明](ENVIRONMENT.md)，本轮没有再次修改它们。

所有初次失败保留：[阶段记录](validation/20261025/nvidia-interfaces/phase-history.json)。包括操作数设备标注错误、测试树打包缺文件、元数据类型参考错误、导出测试配置参数错误，以及反复销毁/初始化组时复用 TCPStore 的测试组合问题。
最终 NCCL 对照使用每个生命周期独立的 FileStore rendezvous、明确设备绑定和组生命周期，两个 rank 全部完成；改动仅在测试驱动，平台实现未改。
错误或未完成阶段没有计入以上通过数。

完整可读小型报告、每条结果、旧失败及 CPU XML 已打入 [metadata.tar.gz](validation/20261025/nvidia-interfaces/metadata.tar.gz)，
SHA256 与文件数见[回执](validation/20261025/nvidia-interfaces/metadata-receipt.json)。大 Tensor、原日志、二进制仍在任务证据目录，索引保留大小/校验和，不随 Git 重复上传。
诊断包含 CPU 拷贝、同步及重复执行，耗时不能当作算子或框架性能基准。

## 复现

需要匹配的 NVIDIA 环境、原版独立目录及其 `_C`，以及新版独立编译的 `_nvidia_ops`；不能给参考端也加载新版扩展。
以下 `devices` 必须是实际授权设备。输出均选择新目录，保留失败，不能原地覆盖已有阶段。

```bash
export PYTHONPATH="$NEW:$NEW/tools:$DEPS:$MQBENCH:$MSBENCH:$TEST_DEPS"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
export DEFAULT_TASKS=cls:det CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=1

# 完整普通接口、训练/推理与边界行为；每次显式指定用途。
for purpose in training inference; do
  python "$NEW/tools/audit_platform_interfaces.py" --group basic \
    --purpose "$purpose" --output "$OUT/basic-$purpose"
  python "$NEW/tools/audit_platform_interfaces.py" --group additional \
    --purpose "$purpose" --output "$OUT/additional-$purpose"
done
python "$NEW/tools/audit_platform_interfaces.py" --group device --purpose training --output "$OUT/device"
python "$NEW/tools/audit_platform_interfaces.py" --group training --purpose training --output "$OUT/training"
python "$NEW/tools/audit_platform_interfaces.py" --group inference --purpose inference --output "$OUT/inference"

# 原、新专用二进制分进程运行；输入 hash 必须相同。
for purpose in training inference; do
  for route in original routed; do
    python "$NEW/tools/audit_platform_native_binaries.py" --baseline "$ORIGINAL" \
      --purpose "$purpose" --route "$route" --output "$OUT/native/$purpose-$route"
  done
done
python "$NEW/tools/audit_platform_native_binaries.py" --results "$OUT/native" --output "$OUT/native-comparison.json"

# 可选包必须真实运行；查看 summary 的 errors/entries，不用退出码代表通过。
for case in sparse quantization qdrop pruning base_export amba; do
  for route in direct routed; do
    python "$NEW/tools/audit_platform_optional.py" --case "$case" --route "$route" \
      --output "$OUT/optional/$case-$route"
  done
done
python "$NEW/tools/audit_platform_optional.py" --results "$OUT/optional" --output "$OUT/optional-comparison.json"

# 仅在两卡均实际授权时执行；每次生命周期使用独立 rendezvous。
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$NEW/tools/audit_platform_collectives.py" --output "$OUT/distributed"
CUDA_VISIBLE_DEVICES='' python -m pytest "$NEW/tests/platform" -q

# REF 是保留的 kitti-reference.npz。
python "$NEW/tools/audit_platform_kitti_examples.py" --baseline "$ORIGINAL" \
  --reference "$REF" --output "$OUT/kitti"
```

没有旧 `REF` 时，可用 `tools/compare_platform_kitti.py --vendor nvidia --write-reference --reference "$REF" --output "$OUT/legacy-difference.json"`
从保留的原版 Numba 函数生成相同 seed 的参考。该工具随后比较旧、新指标，因有意修正而非零退出是预期记录，不要改旧参考让它通过。

原任务所有冻结阶段路径与哈希保留在元数据包和索引中。对该原始证据目录的完整 CPU 汇总命令为：

```bash
CUDA_VISIBLE_DEVICES='' python tools/summarize_interface_audit.py \
  --evidence-root /path/to/imagenet-coco-parity-20261025 \
  --interfaces docs/platform/interfaces.json --output /new/path/to/analysis
```

该汇总器读取本报告列出的冻结阶段结构，不把任意一次新运行冒充本批证据。
