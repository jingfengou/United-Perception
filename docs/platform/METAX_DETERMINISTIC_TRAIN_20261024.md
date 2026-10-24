# 沐曦原生严格确定性：YOLO、ViT 完整训练对照

日期：2026-10-24。协议已冻结，结果待完整训练与审计后追加。

本轮从相同初始权重分别训练，接续 [NVIDIA 确定性完整训练](NVIDIA_DETERMINISTIC_REPEAT_20261024.md)，
区别于上一轮 [同权重推理和零更新反向诊断](MODEL_VERIFY_PROTOCOL_20261024.md)。
不改模型算法、数据划分、配方或 `rtol=1e-4, atol=1e-5`；旧多随机种子任务保持暂停。

## 冻结条件

机器协议与校验和见 [protocol.json](validation/20261024/metax-deterministic-train/protocol.json)。

| 模型 | 数据 | 输入 / batch | 完整预算 | 指标 |
| --- | --- | --- | --- | --- |
| UP ViT-B16 | CIFAR-10：10,000 train / 2,000 test | 64px / 64，保留尾批 16 | 20 epoch，3140 次更新 | Top-1 % |
| UP YOLOv5-S SiLU | PennFudan：120 train / 50 holdout | 320px / 64，EMA off | 300 epoch，600 次更新 | COCO bbox AP×100 |

- C500 每模型执行一次，seed 131，共 3,740 次更新。仅报告最后一轮，不挑最好权重，不失败后自动重试。
- 复用 NVIDIA 已完成的四次同种子严格确定性结果（原版/新版各两次）；本轮不重复占用 NVIDIA GPU。
- 两家初始 `initial.pt` 文件、UP 冻结源码、数据 manifest、配方和训练工具的 SHA256 匹配。
- Python/NumPy/Torch 的模型种子为 131；两家解释器 `PYTHONHASHSEED=0`，线程各 2。
- FP32、TF32 off、cuDNN benchmark off，启用 `torch.use_deterministic_algorithms(True, warn_only=False)`、cuDNN deterministic，启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。
- 保留 Torch 的 `fill_uninitialized_memory=True`；不增加旧 portable/v2 内核，也不改损失、随机掩码或目标覆盖规则。
- C500 使用原生厂商 Torch；开关启用和已证明逐位确定性分别报告。此前诊断已观察到开关启用后的微小梯度波动，本轮不预设跨卡结果必然一致。
- 每次完整训练包含训练内 eval、最终权重/优化器/调度器重载、最终预测 CPU 指标复算和新进程独立推理。任何失败均保留。
- 最终精度按百分点报告，不设新的通过线；逐步 loss 的原容差检查与任务精度是独立结论。
