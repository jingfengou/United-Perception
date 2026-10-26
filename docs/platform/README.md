# 平台层设计

以原版 UP `52aaeb9ed27a490820696d7754d19f98cd6ecd11` 重建，参考 LightX2V `19c19614b067f1746ac24350dfd097e1c0b48591` 的设备、算子类别和训练后端。
同仓命名空间 `up_platform` 随 UP 一起安装，不反向导入 `up`。

## 目录

```text
up/                         原版模型、损失、数据、配方和业务编排
up_platform/
  base.py                   DeviceBackend、PlatformBackend
  factory.py                显式选择、插件注册、固定用途
  contracts.py              契约和能力状态
  backends/                 设备子类、各类别的显式组合
  ops/<类别>/
    base.py                 具名接口及契约
    nvidia.py / metax.py     厂商绑定，允许显式引用共享实现
    ascend.py / cambricon.py 未启用的逐项模板
    torch_impl.py            可共用的 PyTorch 调用
  ops/_native/              扩展构建、加载与内核源码
  training/                 AMP、DDP/FSDP2、优化器、训练验证状态
  inference/                推理 AMP、原有 ONNX 导出
  templates/                SDK/autograd 示例
tests/platform/             无 UP 依赖的接口及路由测试
tools/*platform*             审计、构建、模板生成与复现工具
docs/platform/              接口参考、接入教程和环境指南
tests/platform/fixtures/    接口契约及边界检查的固定测试数据
```

## 调用规则

`UP → 类别接口 → 当前厂商文件中的绑定 → 厂商 PyTorch / 共享实现 / SDK`。

- 每家文件明确声明实现，没有“找不到就尝试另一家”的全局回退。
- `nvidia/cuda` 与 `metax/cuda` 身份不同；分别编译 `_nvidia_ops`、`_metax_ops`，不混用二进制。
- 共享源码由两家工具链分别编译；源码共享不意味着沐曦加载 NVIDIA 二进制。
- 用途模块按需加载，配置后固定；另一用途使用新进程，不按 `.eval()`、`requires_grad` 或 `no_grad()` 猜测。
- 普通导入不初始化设备、不加载扩展。旧 `up.extensions` 路径及 `.apply()` 仍是兼容门面。
- 基础算术、索引、reshape、张量存储、autograd、模块系统继续由 PyTorch 负责，属于设备桥接前提。
- 原版显式 CPU 数据处理/评估/序列化保留；新增缺失操作不得静默 CPU 代算。
- 原版 CPU RoIAlign 导出占位行为保留，但不能用于证明 CPU 数值正确。

## 训练与推理

| 入口 | 用途 | 约束 |
| --- | --- | --- |
| train | training | 原配方决定优化器分组、学习率、损失和训练步骤 |
| 训练内 eval、train --evaluate | training | 临时 eval/no_grad，恢复进入前每个子模块及 EMA 模式 |
| 独立图片/视频推理 | inference | 普通 PyTorch 模型，不自动替换部署引擎 |
| 原有导出 | inference | 保留 legacy ONNX symbolic |

NMS 等本来不可微的训练期操作仍不可微。适配不改 checkpoint 键、初始化、参数身份、量化规则或剪枝策略。

FSDP2 是显式选择的训练策略，分片发生在优化器创建前，保留原模型参数名称。其保存恢复、EMA 和验证流程见 [FSDP2 指南](FSDP.md)。DDP 默认行为不变。

## 开发与检查

### GPU 接口复测

在已安装对应厂商 PyTorch 的环境中执行；NVIDIA 将下面的 `metax` 改为 `nvidia`。每次使用新的输出目录和会合目录。

```bash
# 常用算子的前向、所需梯度和状态；独立推理将 purpose 改为 inference
python tools/audit_platform_interfaces.py --backend metax \
  --group basic --purpose training --output dist/metax-basic-training

# 空输入、原地操作和层构造
python tools/audit_platform_interfaces.py --backend metax \
  --group additional --purpose training --output dist/metax-additional-training

# 双卡通信、SyncBN 和 DDP；会合目录放在支持文件锁的本地存储
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  tools/audit_platform_collectives.py --backend metax \
  --rendezvous-dir /dev/shm/up-platform-audit --output dist/metax-distributed
```

`--group device` 检查设备管理，`--group training` 检查 AMP/优化器等，`--group inference --purpose inference` 检查推理执行接口。可选工具链使用 [audit_platform_optional.py](../../tools/audit_platform_optional.py)。整套带 `job.json` 执行记录的结果可由 [summarize_vendor_interfaces.py](../../tools/summarize_vendor_interfaces.py) 复算并汇总，原始失败和显式补测记录一并保留。

### 软件检查

- [在线交互教程](https://jingfengou.github.io/United-Perception/)：改动概览、各部分功能与实现、运行示例和新厂商扩展；[下载 HTML](https://github.com/jingfengou/United-Perception/raw/refs/heads/codex/up-platform-clean/docs/platform/TUTORIAL.html) 可离线阅读。
- [完整接口](INTERFACES.md)：由父类生成签名、返回、梯度、状态、设备、布局、空输入和导出要求。
- [GPU 接口代码对照表](GPU_COVERAGE.md)：逐项定位 UP 引用、NVIDIA/沐曦实现、评测代码及真机状态。
- [固定测试数据](../../tests/platform/fixtures)：保存机器契约、原版调用清单和当前调用快照；它们是 CI 输入，不是运行报告。
- [环境与依赖](ENVIRONMENT.md)：安装顺序、厂商版本及可选 SDK。
- [厂商接入](ADDING_VENDOR.md)：生成模板、绑定实现和注册。

```bash
python tools/platform_inventory.py
python tools/platform_contracts.py
python tools/platform_call_sites.py
python tools/platform_coverage.py
python tools/check_platform_boundaries.py
python -m pytest tests/platform -q
```

`capabilities()` 的已注册、依赖可发现、已执行、已验证分别记录，不把导入成功当成通过。
厂商接入还需在目标设备执行所用模型、算子和第三方工具链；环境缺口见 [安装指南](ENVIRONMENT.md)。

设备和算子是训练、推理共用的能力。模型直接调用 `get_platform().category(...)`；执行流程通过 `get_platform().execution` 获取当前用途的 AMP、DDP/FSDP2 等能力。基础张量操作和 autograd 继续由厂商 PyTorch 提供。

实验报告和日志输出到 `dist/`；`docs/platform/validation/` 及旧阶段报告只在本地保留，并由 Git/Docker 忽略。文档链接检查只扫描已跟踪文件，并拒绝指向本地未跟踪文件的链接。
