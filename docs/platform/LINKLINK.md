# linklink 依赖缺口

状态：**阻塞，未通过验收**。核查日期：2026-10-23。

本分支仍以原版 UP 的 `spring.linklink` API 为目标，不将 PyTorch DDP、
普通 BatchNorm 或 SGD/AdamW 冒充 linklink 实现。

## 这里的 SDK 到底是什么

SDK 是软件开发工具包的统称。**本问题里更准确的名称是“缺少 `spring.linklink` 的真实实现库及其匹配依赖”**，并非缺少 NVIDIA 驱动、CUDA，或沐曦驱动、MACA。

linklink 是原版 UP 部分分支额外调用的通信与训练加速库：上层有 Python 接口，底层需要真实的通信、GPU/设备计算和训练状态实现。UP 保留了调用它的代码，但原版仓库和默认安装流程没有提供本次所需的完整依赖。
仅有几个同名 Python 函数、空壳包或使用普通 Torch 改写的兼容层，不能证明已经提供了原版的全部功能。

| 功能 | 当前已有的普通 Torch 路径 | 仍缺少的 linklink 路径 |
| --- | --- | --- |
| 多卡通信 | `torch.distributed`、DDP | linklink 初始化、分组、集合通信及异步语义 |
| SyncBN | Torch SyncBN | `link.nn.SyncBatchNorm2d`、`SyncBNFunc`、var mode/group |
| 混合精度 | Torch autocast、GradScaler | `linklink.fp16` 模块注册、初始化及原有状态语义 |
| 优化器 | `torch.optim` 的原有普通优化器 | 三个 linklink 融合优化器、master weights 和 reload/恢复 |

因此，已验证的普通 Torch 路径可以训练和推理；缺库不代表全部多卡、SyncBN 或混合精度都不可用。选用 linklink 专用配置或仍直接引用它的旧分支时，才涉及本项缺口。

## 要补齐，需要提供什么

可以提供以下两种材料之一，外加版本和 API 说明：

1. **可安装的完整包**：例如含实际本地库/设备扩展的 Linux wheel，以及必须配套的动态库和依赖。不能只是 `spring.linklink` 这个模块名。
2. **完整源码与构建方法**：Python 部分、本地/设备实现、头文件、构建脚本和依赖版本，能够构建出本项目所需的完整功能。

还需要明确：

- **对应环境**：Python、PyTorch、操作系统/编译 ABI，以及 NVIDIA 的 CUDA 环境或沐曦的 MACA 环境。当前验证使用 Python 3.12、厂商 PyTorch 2.10；给出其他版本时需另建匹配环境并重新验收。
- **覆盖的 API**：下方列出的通信、SyncBN、FP16 和三个融合优化器必须真实存在，尤其需要状态保存/恢复与 master-weight 行为。
- **支持的平台**：只有 NVIDIA 构建不意味着可以在 C500 上运行；沐曦需要相应兼容构建或源码移植。不能拿 NVIDIA 本地库替代沐曦实现。
- **可执行参考**：原维护方的测试、示例或一个可运行环境，用于核对语义，避免函数同名但行为不同。

不需要先找到这个库才能继续做平台边界隔离。**接口定义和 11 处调用迁移可以先完成，缺库时由平台后端明确报不支持；真实功能执行和验收再依赖上述材料。** 当前仍存在的上层直接导入是另一个未完成的迁移项，不能只用 SDK 缺失解释。

## 已核查

- 原版 UP 的 `requirements.txt` 和 `easy_setup.sh` 不安装 `spring.linklink`。
- 当前 NVIDIA 和沐曦验证解释器均没有这个 SDK。
- [DeCLIP 的 linklink](https://github.com/Sense-GVT/DeCLIP/blob/main/linklink/__init__.py)
  是基于 `torch.distributed` 的接口改写，不包含 UP 所需的完整 FP16 和融合优化器。
- [TFMQ-DM 的 linklink](https://github.com/ModelTC/TFMQ-DM/blob/main/linklink/__init__.py)
  也转发到 `torch.distributed`；其 `SyncBatchNorm2d` 明确使用普通 BatchNorm2d。
- [PyPI 同名包](https://pypi.org/project/linklink/) 用于 URL 拼接，与 GPU SDK 无关。

上述公开代码均不能作为本计划的等价依赖，因此没有安装或复制进项目。
本次检索未找到可验证的完整 SDK，不等于证明任何渠道都不存在。

## 待交付的真实依赖

需要对应 Python/PyTorch/设备版本的 `spring.linklink` 安装包或源码，并执行：

1. initialize/finalize、rank/group、同步与异步集合通信。
2. SyncBatchNorm2d、SyncBNFunc、var mode、分组状态和反向。
3. FP16 模块注册、转换及状态恢复。
4. FusedFP16SGD、FusedFP16AdamW、FusedZeroFP16AdamW 更新与 master-weight 恢复。

PyTorch `dist` 路径单独实现和验证，不能用于把本表的状态改为通过。
