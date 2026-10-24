# 参考 model-verify：固定权重推理与训练重复性诊断

日期：2026-10-24。协议在运行结果产生前冻结，真机结果待追加。

## 先澄清比较的对象

参考仓库固定在 [120d9af31fb](https://github.com/xumy1995/model-verify/tree/120d9af31fb022ccec71f1060eee892491341ca1)。
首页主要比较相同预训练权重的推理指标，不能据此推断两家独立训练会得到相同权重或逐位相同的输出。
分类日志只打印两位小数；检测汇总的 [comparison](https://github.com/xumy1995/model-verify/blob/120d9af31fb022ccec71f1060eee892491341ca1/generate_summary.py#L15-L22)
直接用 0–1 的 AP 差显示百分号，没有乘 100。例如 DETR AP 0.414 → 0.413 是约 **−0.1 个百分点**，显示为 `−0.00 %`；它不是零差异。
源日志本身也经过取整，不能恢复未公布的小数或原始 logits。

直接 Hugging Face 推理脚本使用 `eval()`、`inference_mode()` 和固定验证顺序，没有统一启用严格确定性。
训练脚本的设置各不相同，公布的训练指标也有差异。YOLO 使用的 Ultralytics 8.4.115
[种子初始化](https://github.com/ultralytics/ultralytics/blob/v8.4.115/ultralytics/utils/torch_utils.py#L614-L635)
启用 `torch.use_deterministic_algorithms(True, warn_only=True)`、cuDNN deterministic 及 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。
`warn_only=True` 允许没有确定性实现的操作继续执行，因此配置项不是确定性验收结果。

## 本轮冻结的测试

完整条件见 [protocol.json](validation/20261024/model-verify-protocol/protocol.json)。

- 四模型为 UP ResNet18、YOLOv5-S、ViT-B16，以及已有的官方 DETR R50 UP 适配器；保留原算法。
- 统一选用上一轮 `new_nvidia` native 队列的四个最终检查点，不按分数挑权重。
- 只在 NVIDIA 准备一次 CPU 权重与真实数据批次，校验后传给 C500；每模型用独立进程准备，防止原版 UP 全局标志在模型之间泄漏。
- 推理使用 CIFAR-10 的全部 2,000 个测试样本或 PennFudan 的全部 50 个留出样本；两平台输入 Tensor、顺序、权重完全相同。
- native、Ultralytics 风格两种配置，各执行两次新进程推理，并在每个进程内重复首批 10 次；记录完整任务指标与输出 Tensor。
- 训练诊断分别用 native、Ultralytics 风格、strict，固定首批、第二批及 epoch 尾批，逐次恢复模型参数/BN 缓冲区和 RNG，重复前向与反向 10 次；**不执行优化器更新**。
- strict 遇到未实现确定性的操作就保留失败，不关闭检查重跑。警告、执行完成、逐位一致、原容差通过和任务精度相同分别报告。
- seed 131，FP32、TF32 off、cuDNN benchmark off；`rtol=1e-4, atol=1e-5`。不加入旧 portable/v2 内核，不恢复旧多随机种子任务。

借鉴的是固定权重/输入、推理状态控制及 warn-only 开关，不是照抄参考仓库所有默认值。
这次数据集、模型大小及 NVIDIA 型号不同，不与其 A100/ImageNet/COCO 绝对速度或精度直接排名。
PyTorch 的[复现说明](https://docs.pytorch.org/docs/2.10/notes/randomness.html)也明确没有跨平台、跨版本逐位复现保证。

## 工具

`tools/audit_model_verify.py` 从固定版本的已公开日志提取指标，显式区分百分数与 0–1 比率，再换算百分点。
`tools/verify_checkpoint_protocol.py` 提供 `prepare`、`inference`、`gradients` 三个阶段；结果、警告和异常写入独立目录，拒绝覆盖。
模型算法、平台运行时和配置文件在本轮保持不变。
