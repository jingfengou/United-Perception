# YOLO、ViT 跨厂商训练差异定位

日期：2026-10-24。平台运行时代码、模型、配方和容差均未修改，旧多随机种子任务继续暂停。

**ViT 已定位到原生 Dropout 的跨设备随机数映射差异；YOLO 已定位到原生卷积的微小数值差异、逐层传播及反向归约差异。两条路径均有不导入 UP / up_platform 的 PyTorch 重放证据。**
这不等于已修复最终精度差，也不等于沐曦硬件或所有相关内核存在缺陷。

## 首个差异在哪里

区分“数值不再逐位相同”和“超出原容差”：所有浮点检查仍使用 `rtol=1e-4, atol=1e-5`。
索引从 0 开始；本次记录的是实际模型的叶子模块边界，随后对完整操作做独立前向/反向重放。

| 检查 | ViT-B16 | YOLOv5-S |
| --- | --- | --- |
| 同一批输入的逻辑值 | 完全相同 | 完全相同 |
| 首次数值不逐位相同 | 索引 0，`backbone.embedding` 的 Conv2d | 索引 1，`backbone.focus.conv_block.conv` 的 Conv2d；此前 Space2Depth 完全相同 |
| 该卷积最大绝对差 | 2.26498e-6，全部在容差内 | 1.37091e-6，全部在容差内 |
| 首次前向超容差 | 索引 1，`backbone.dropout` | 索引 14，即第 15 个模块的 BatchNorm |
| 该处超限 | 8,793 个元素；最大绝对差 4.36260 | 6 个元素；最大绝对差 3.05176e-5 |
| 首步参数梯度跨卡检查 | 152 / 152 个 Tensor 超限 | 73 / 197 个 Tensor 超限 |
| NVIDIA 原版与新版首步梯度 | 152 / 152 个 Tensor 逐位相同 | 197 / 197 个 Tensor 逐位相同 |

YOLO 的完整名称是 `backbone.csp_modules.0.1.cbh_blocks.0.conv_block2.bn1`。
前 14 个叶子模块均在原容差内；该 BN 后的 SiLU 又回到容差内，不能把一个首次超限点解释为后面每个操作都失败。
YOLO 本次连续检查了前 32 个叶子模块，ViT 对首次超限前的两个模块作完整检查，另捕获全部随机模块的实际掩码。

逐项数据见 [首步前向和梯度](validation/20261024/vendor-training-diagnosis/comparisons)、[YOLO 前 32 层](validation/20261024/vendor-training-diagnosis/yolo32-comparison.json)。

## ViT：相同种子没有产生相同掩码

从 autograd 保存的 `NativeDropoutBackward0._saved_result1` 读取真实布尔掩码，DropPath 则读取乘法保存的掩码。
不从含零的激活值反推掩码，也不重新抽样；73 次随机模块调用均记录了掩码，且都可重构原输出。

首个 Dropout 的输入为 `[64,17,768]`，两家输入仍全部在原容差内，但 835,584 个掩码位置中有 **8,793 个不同**。
进入该操作前，实际 GPU 生成器状态相同：种子 `294499921`、计数偏移 0。
这个种子是 UP 原有 `131 ** 4` 规则的结果，两边遵循同一规则，并非平台层换了种子。

| 原生 Dropout 的网格/计数观测 | RTX 4090 | C500 |
| --- | ---: | ---: |
| 多处理器数量 | 128 | 104 |
| 每多处理器最大线程数 | 1536 | 2048 |
| 按公开 CUDA 实现计算的网格块数 | 768 | 832 |
| 每块 256 线程、每线程 4 元素时的首轮覆盖量 | 786,432 | 851,968 |
| 本次实际生成器偏移变化 | 0 → 8 | 0 → 4 |

掩码在前 786,432 个位置全部相同，首次不同在下标 786,443，全部差异都在后面的区间；实际计数增量也与表中网格策略吻合。
公开 [PyTorch 2.10 Dropout CUDA 源码](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/Dropout.cu#L364-L393)
根据设备属性限制网格，线程索引参与 Philox 随机流，网格还影响计数增量。
这解释了相同种子为何可以在同一设备重复，却不能保证不同设备采用相同掩码。
沐曦内部内核源码未审阅；其本次行为与上述策略吻合，不宣称两份厂商源码逐行相同。

第一次计数分叉之后，本次模型捕获的 73 次掩码哈希均不同。
独立小实验中重新设置种子，较小的 attention / DropPath 形状反而可以相同；不是所有随机操作天生都不兼容，而是形状、网格及前面消耗的随机流共同决定结果。

将相同输入、权重及各自保存的 RNG 状态交给**纯 PyTorch Dropout**，两家仍复现不同掩码、输出和输入梯度；每家自身 10 次重放完全一致。
因此该差异无需 UP 平台路由也会出现，原生确定性开关没有提供跨设备一致的随机序列。
它是已证实的训练路径分叉点，但本轮没有通过更换掩码重新完成训练，不能把全部 6.4 pp 差距量化归因于这一项。

## YOLO：原生布局与浮点计算，不能归为 Dropout

同一 CPU 图像通过厂商 PyTorch 的 `.to('cuda')` 和 `.cuda()`，在 NVIDIA 上保持 NCHW，在本次 C500 环境变为 channels-last，逻辑值完全不变。
不导入 UP 的设备拷贝实验也有相同行为；平台设备接口只是调用 `.to(...)`，没有自行增加布局转换。

首个卷积的输入值和权重相同，其原生输出已有容差内的小差异。
在独立重放中显式对齐输入及权重的 NVIDIA 布局后，**两模型首个卷积的前向输出都逐位相同**；卷积反向仍存在小差异。
这是一项单操作诊断，没有将布局强制改写加入模型或平台默认路径。

YOLO 前 32 个叶子模块的观测显示这些小差异向后传播，索引 14 的 BN 首次有 6 个值超限。
**它的输入已经不同，不能据此指认该 BN 自身错误。**
把同一份 NVIDIA 输入、参数、状态交给两家纯 PyTorch 重放该 BN，其前向和输入梯度均在原容差内，前向最大差 3.81470e-6；
其缩放参数梯度仍有 3 / 32 个元素超限（原生布局），匹配布局后为 5 / 32。

独立重放首个 BN 也复现缩放参数梯度跨卡超限，十次自身重复则稳定。
这里使用**共同固定上游梯度**隔离归约行为，数值大小不能冒充模型原始 loss 的梯度。
补充的离线 CPU FP64 参考显示，NVIDIA 和 C500 的 FP32 缩放梯度都存在归约误差并有元素超该比较容差；
因此这些样本说明 FP32 归约路径不同，不能直接断言 NVIDIA 每个值都是精确真值、只有沐曦算错。
FP64 仅用于离线诊断，没有用于模型计算。

此外，本轮没有观测到 YOLO 前五步前向中的 GPU 随机调用，两家 RNG 状态也未在前向推进。
选出的 6 个“存在重复索引、值张量非恒定”的真实目标写入，用共同 NVIDIA 操数各重放 10 次，两家均与参考逐位一致。
这是排除本轮这些写入案例的证据；不否定旧 native 模式中的重复覆盖问题，也不证明其他输入永远稳定。

## 平台层和底层分别得到了什么结论

- **平台迁移检查**：同一 NVIDIA 卡，原版/新版首步全部 349 个参数梯度逐位相同，历史严格模式完整训练的 loss、状态和最终指标也相同。当前没有发现平台改写模型计算的证据。
- **厂商实现检查**：不加载 UP 的完整操作重放仍有 Dropout 掩码差异，以及卷积/BN 的原生浮点与反向归约差异，相关差异属于厂商 PyTorch / 运行库执行路径。
- **重复性检查**：C500 的卷积权重梯度在十次重放中仍有非逐位波动；真实模型“有叶子观测/无叶子观测”首步梯度比较全部在原容差内，波动继续记录。
- **尚未解决**：尚未证明单独修改某一个算子就能消除最终 YOLO +17.4825 pp、ViT −6.4000 pp；未开展新的完整预算或多种子精度训练，也未加入新的数值修复。

观测影响也单独核对：NVIDIA 两模型的五步控制 loss 与既有完整训练逐位相同，C500 ViT 的最大差为 2.38419e-7。
**C500 YOLO 第五步控制 loss 与既有完整训练相差 0.00857353，超出原容差**；本轮不同五步前缀之间也有差异，不能宣称整段轨迹已被原样复现。
首步 loss 与原始完整训练相同，首步有/无叶子观测的梯度也都在原容差内；同输入独立算子重放的结论与完整训练精度归因分开保留。
目前不能进一步区分同步观测对具体内核调度的影响，以及该环境原生重复波动各自贡献了多少。

已有完整训练数字保留在 [上一阶段报告](METAX_DETERMINISTIC_TRAIN_20261024.md)，本轮不产生新的最终精度分数。
[PyTorch 复现说明](https://docs.pytorch.org/docs/2.10/notes/randomness.html) 也区分单一平台的重复性与跨平台一致性；开关启用不能代替实测。

## 条件、证据及复现

沿用上一阶段相同数据、初始文件、权重内容、配置、种子和 native strict 数值策略。
NVIDIA 为 Torch 2.10.0+cu126 / Python 3.12.3，C500 为 Torch 2.10.0+metax3.8.1.0 / Python 3.12.11；依赖版本及差异见 [协议](validation/20261024/vendor-training-diagnosis/protocol.json)。
两家运行后的源码、数据和资源校验均通过，设备锁已释放。CPU 契约测试各 **81 项通过**。

观测器在完整 worker 的初始 eval 之后接入，不替换模型、输出、梯度、掩码或优化器。
总共 48 次诊断性优化器更新，仅执行同种子的短前缀；不能称为完整训练或多种子实验。
CPU 拷贝会同步 GPU，本轮耗时不用于性能结论。
首次 C500 小实验因缺少 USER/LOGNAME 启动失败，首次 NVIDIA CPU 测试因 PYTHONPATH 未设置而收集失败；纠正启动环境后以新记录重试，失败证据保留。
审计时补全了随机调用计数器对“仅有 device 参数、没有 Tensor 输入”的创建操作的覆盖，新增回归测试，并在两家重新执行五步 YOLO 前缀；旧捕获和工具版本均保留。

按 [数据及初始权重准备](ACCURACY_NATIVE_20261024.md)、[环境说明](ENVIRONMENT.md) 获取相同资源，并选择已经分配的设备。
诊断工具放在冻结源码目录之外：冻结目录的全量 manifest 含当时的 tools/tests，本次新增工具不会覆盖它们。
当前 `up/`、`up_platform/`、`configs/` 与 `b137c68` 相同，三个生产目录合计 781 个文件的归一化摘要也单独记录。

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONHASHSEED=0 DEFAULT_TASKS=cls:det CUBLAS_WORKSPACE_CONFIG=:4096:8
unset MCDNN_USE_DETERMINISTIC_ALGO MCDNN_STABLE_RESULT_MOD
unset PYTORCH_DISABLE_CUDA_CUDNN_TF32 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
# 容器 UID 无 passwd 记录时，设置实际账号的 USER / LOGNAME。
# REPO=当前仓库，SOURCE=冻结生产源，DATA=同一 data.json，INITIAL=同一初始权重。
MODEL=vit_b16        # 或 yolov5s
LANE=new_metax      # NVIDIA 为 new_nvidia，原版为 original_nvidia
python "$REPO/tools/observe_vendor_prefix.py" \
  --source "$SOURCE" --frozen-tools "$REPO/tools" --data "$DATA" \
  --initial "$INITIAL" --output "$NEW_OUTPUT" --lane "$LANE" \
  --model "$MODEL" --steps 1 --leaves 8
# 控制前缀改为 --steps 5 --leaves 0；YOLO 深层观测为 --steps 1 --leaves 32。
```

两平台均使用同一模型初始文件；原版 NVIDIA 的 SOURCE 指向原版 UP，不指向新版。
单独测试 RNG / 设备拷贝：`probe_vendor_randomness.py --output 新目录`、`probe_vendor_transfer.py --output 新文件`。
同机原版/新版或两家捕获可用 `compare_vendor_prefix.py --reference 参考目录 --actual 待比目录 --leaves 8 --output 新文件` 比较；
只检查梯度时加 `--gradients-only`。
YOLO 前 32 层可先用 `compare_vendor_forward.py --captures NVIDIA目录 --leaves 32 --output 参考.pt` 打包，再加 `--reference 参考.pt` 检查 C500 捕获。

对实际模块作独立重放：

```bash
python "$REPO/tools/replay_vendor_leaf.py" \
  --capture "$COMMON_NVIDIA_LEAF" --own-rng "$THIS_VENDOR_LEAF" \
  --layout native --repeats 10 --output "$NEW_REPLAY"
# 再以新输出目录执行 --layout reference，显式匹配参考输入和权重布局。
# 不把 NVIDIA RNG 状态导入 C500：--own-rng 必须是当前厂商自己的捕获。
```

首个卷积/Dropout 的捕获编号分别为 ViT 0/1，YOLO 首个卷积/BN 为 1/2，首次超限 BN 为深层捕获 14。
完整操作重放使用固定共同上游梯度、恢复每次模块状态，不复用已经修改的 BN running state。
目标写入用 `observe_vendor_writes.py` 捕获，再用 `replay_vendor_writes.py --reference NVIDIA写入目录 --repeats 10 --output 新目录` 重放；后者不导入 UP。

机器可读结果、证据位置和校验索引在 [本轮证据目录](validation/20261024/vendor-training-diagnosis)。
逐次梯度/掩码统计、计数器分析和离线 FP64 检查的原始审计脚本归档在 [audit-scripts](validation/20261024/vendor-training-diagnosis/audit-scripts)，不属于平台运行时。
它们按本次相对目录读取捕获；在证据副本根目录放置脚本及当前 `tools/` 后执行，不覆盖原始目录。
脚本归档仅统一 CRLF/LF；[脚本索引](validation/20261024/vendor-training-diagnosis/audit-scripts/index.json) 同时保留执行版本与归档版本的哈希。
大 Tensor、检查点、全部失败日志和执行工具保留在两家服务器的 `up-platform-clean-20261023/vendor-training-diagnosis-20261024` 目录，未加入 Git，也不是公开下载数据集。
