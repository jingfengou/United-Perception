# 参考 model-verify：固定权重推理与训练重复性诊断

日期：2026-10-24。**状态：两家共 56 个预定案例、证据和指标审计完成；54 个执行成功，2 个 strict DETR 明确失败。执行成功不等于数值通过。**

**同权重推理的任务指标一致；跨卡输出 Tensor 并非全部满足原容差；C500 的训练梯度仍有重复波动。**
本轮没有改模型、平台默认数值策略或原有容差，没有恢复旧多随机种子任务，也没有执行新的优化器更新。

后续已单独完成 [C500 YOLO、ViT 的原生严格开关完整训练](METAX_DETERMINISTIC_TRAIN_20261024.md)：仍存在明显跨厂商训练精度差。该后续结果与本报告的同权重推理结论分别记录。

## 四模型同权重推理结果

NVIDIA 为同一张 RTX 4090，沐曦为同一张 C500；两家加载完全相同的 NVIDIA native 最终检查点和 CPU 输入 Tensor。
分类用 CIFAR-10 的 2,000 个测试样本，检测用 PennFudan 的 50 个留出样本。
native、Ultralytics 风格各跑两个新进程，**每模型八次完整推理的全部已报告指标相同**，不是仅四舍五入后相同。

| 模型 / 指标 | NVIDIA | C500 | C500 − NVIDIA |
| --- | ---: | ---: | ---: |
| ResNet18 / Top-1 % | 66.40000153 | 66.40000153 | 0 pp |
| YOLOv5-S / bbox AP×100 | 29.62426402 | 29.62426402 | 0 pp |
| 官方 DETR R50 适配器 / bbox AP×100 | 79.22607413 | 79.22607413 | 0 pp |
| UP ViT-B16 / Top-1 % | 22.55000114 | 22.55000114 | 0 pp |

这回答了与参考仓库类似的“相同权重拿到另一张卡做推理”问题。
之前 [分别训练得到的结果](ACCURACY_NATIVE_20261024.md) 保留；两种实验比较的权重不同，不能互相覆盖。

### 输出 Tensor 与同卡重复

同平台两次新进程的全部已保存输出逐位一致，native 与 Ultralytics 风格的输出也逐位一致。
32 次推理进程内共 **320 次首批重复**均逐位一致。这只描述本轮有限输入和执行方式，不是所有输入下的确定性保证。

跨平台的原始输出比较使用既有 `rtol=1e-4, atol=1e-5`，没有重排框、舍弃失败 Tensor 或修改容差。
下面 native 和 Ultralytics 风格的结果相同；最大绝对差按已保存 Tensor 统计，混有 logits、分数或像素坐标单位，**不是精度下降的百分点**。

| 模型 | 跨卡逐位相同 | 原容差 | 最大绝对差 | 超容差 Tensor 数 |
| --- | --- | --- | ---: | ---: |
| ResNet18 | 否 | 通过 | 8.1062e−6 | 0 |
| YOLOv5-S | 否 | 未通过 | 1.8311e−4 | 1 |
| DETR 适配器 | 否 | 未通过 | 0.1329956 | 13 |
| ViT-B16 | 否 | 未通过 | 3.7313e−5 | 2 |

因此，本轮可以说任务精度指标相同，不能说全部计算结果数值等价。
本轮没有对跨卡输出差异另做首个算子捕获，不能仅凭这些差值认定 GEMM、平台路由或某个厂商内核存在错误。

## 训练反向诊断结果

每个训练批次先固定同一模型参数、BN 缓冲区和 RNG，再做 10 次真实 UP loss/backward；首轮作为参考，其余 9 轮比较。
ResNet、DETR、ViT 各取首批、第二批、epoch 尾批，共 27 次比较；YOLO 只有两批，共 18 次比较。
所有成功案例合计 540 次梯度比较；**优化器更新为 0**，不能据此推断完整训练的最终精度。

下表每格为 **逐位相同次数 / 原容差通过次数 / 总比较次数**。原容差不变。

| 模型 | 配置 | NVIDIA | C500 |
| --- | --- | ---: | ---: |
| ResNet18 | native | 0 / 27 / 27 | 0 / 27 / 27 |
| ResNet18 | Ultralytics 风格 | 27 / 27 / 27 | 0 / 27 / 27 |
| ResNet18 | strict | 27 / 27 / 27 | 0 / 27 / 27 |
| YOLOv5-S | native | 0 / 4 / 18 | 0 / 0 / 18 |
| YOLOv5-S | Ultralytics 风格 | 18 / 18 / 18 | 0 / 18 / 18 |
| YOLOv5-S | strict | 18 / 18 / 18 | 0 / 18 / 18 |
| DETR 适配器 | native | 0 / 27 / 27 | 0 / 27 / 27 |
| DETR 适配器 | Ultralytics 风格 | 27 / 27 / 27 | 0 / 27 / 27 |
| DETR 适配器 | strict | NLL2D 前向失败 | NLL2D 前向失败 |
| ViT-B16 | native | 18 / 27 / 27 | 0 / 27 / 27 |
| ViT-B16 | Ultralytics 风格 | 27 / 27 / 27 | 0 / 27 / 27 |
| ViT-B16 | strict | 27 / 27 / 27 | 0 / 27 / 27 |

- NVIDIA 的 warn-only 四模型在本轮样本中逐位重复；DETR 仍报告 NLL2D 没有确定性实现的警告。有限重复成功不能替代严格实现保证。
- C500 的 warn-only 梯度全部在原容差内，但都没有达到逐位重复。YOLO 最大重复梯度差由 native 的约 `3.0924e−2` 降为约 `6.8545e−7`；这不是完整训练精度或性能改善的结论。
- C500 strict 的 ResNet、YOLO、ViT 没报错，但仍有微小重复梯度差；strict 开关本身不能证明这套厂商栈的全部内核可重复。
- 两家 strict DETR 均在官方 `F.cross_entropy(src_logits.transpose(1, 2), ...)` 调到 `nll_loss2d_forward_out_cuda_template` 时失败。保留官方 CE 布局、警告和异常，没有展平损失输入来绕过检查。

这套配置可以作为减少部分波动的选项，不能称为“参考仓库已经保证所有任务无随机性”的方法。
本轮也没有重新逐算子定位所有残余梯度波动；不把它们全部归给 GEMM。

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

参考仓库[训练表](https://github.com/xumy1995/model-verify/blob/120d9af31fb022ccec71f1060eee892491341ca1/README.md#训练结果汇总)本身也不是零差异：
ResNet50 Top-1 为 80.09% / 80.37%，DETR AP 为 0.411 / 0.408，YOLO26n 独立验证 mAP 为 0.3765 / 0.3821，ViT 最佳 EMA Top-1 为 80.932% / 80.954%（NVIDIA / C500）。
这些是对方公布的数据，不是本轮复跑；其训练预算、最佳/最终权重选择和模型版本不同，不能与下面 UP 的绝对分数直接比较。
全部 24 项推理精度的日志提取与换算保存在 [reference-audit.json](validation/20261024/model-verify-protocol/reference-audit.json)。

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
输出比较只覆盖 UP 已暴露的 `logits`、`scores`、`preds`、`dt_bboxes`；DETR 保留原适配器输出的框和分数，不冒充比较了每个中间层。
每批输出与梯度都同步复制到 CPU 作审计，结论只适用于该执行方式，不保证其他异步并发安排下的重复性。

## 工具

`tools/audit_model_verify.py` 从固定版本的已公开日志提取指标，显式区分百分数与 0–1 比率，再换算百分点。
`tools/verify_checkpoint_protocol.py` 提供 `prepare`、`inference`、`gradients` 三个阶段；结果、警告和异常写入独立目录，拒绝覆盖。
模型算法、平台运行时和配置文件在本轮保持不变。

## 环境、审计与失败证据

| 项目 | NVIDIA | 沐曦 |
| --- | --- | --- |
| 物理设备 | RTX 4090，GPU 1，UUID `GPU-a11f730c-a975-2bf0-b765-3c24d83b7104` | C500，GPU 2，PCI `0000:0e:00.0` |
| Python | 3.12.3 | 3.12.11 |
| Torch | 2.10.0+cu126 | 2.10.0+metax3.8.1.0 |
| TorchVision | 0.25.0+cu126 | 0.25.0+metax3.8.1.0 |
| CPU 契约 | 72 项通过 | 72 项通过 |

两家使用相同 FP32 配方与普通依赖，但厂商 Torch/SDK 和 Python 补丁版本不同，不能声称完整环境逐字相同。
运行前核对设备身份、无计算进程及单 GPU 文件锁；各主机串行执行。
NVIDIA 矩阵为 12:46:48–12:58:09 UTC，C500 为 12:52:22–13:15:00 UTC。
矩阵墙钟包含构建、CPU 审计、序列化和数据验证，**不作为训练速度对比**。
JSON 中另有首批 CUDA-event 前向探针，也不能与参考仓库 A100/全量数据吞吐直接排名。

审计确认：两家共同权重、输入与配置匹配；验证驱动与 Git `3f57d38` 的字节一致；
全部 `up/`、`up_platform/`、`configs/` 与本轮之前的 `b137c68` 相同。
运行后再次验证全部数据、标签、初始预训练资源及 DETR 源码，并核对 NVIDIA 55 个、C500 76 个原始 Tensor 文件的 SHA256。

- [完整统计](validation/20261024/model-verify-protocol/summary.json) / [结果 CSV](validation/20261024/model-verify-protocol/results.csv)：包含全部预定案例、警告、失败、环境和输出比较，未挑选最好结果。
- [发布校验索引](validation/20261024/model-verify-protocol/publication-index.json)：协议、工具版本及小型证据的校验和。
- [NVIDIA 证据索引](validation/20261024/model-verify-protocol/nvidia/evidence-index.json) / [Tensor 索引](validation/20261024/model-verify-protocol/nvidia/full-tensor-index.json)；[C500 证据索引](validation/20261024/model-verify-protocol/metax/evidence-index.json) / [Tensor 索引](validation/20261024/model-verify-protocol/metax/full-tensor-index.json)。
- 原始报告、日志及完整推理输出分别打包，压缩后约 30 MB；权重、输入和大梯度 Tensor 留在两家任务证据目录，仓库只保留摘要与索引。

初始验证工具重放固定批次时漏掉 UP 的 `before_data` 日志计时钩子，产生 8 个日志错误后中止第 9 个案例。
这批证据保留在原目录，见 [harness-attempt.json](validation/20261024/model-verify-protocol/harness-attempt.json)；不计入上述 56 个案例。
`3f57d38` 仅修正验证工具钩子，另建 `matrix-v02` 完整执行全部预定案例，没有覆盖失败或修改平台实现。

本轮不验证全量 ImageNet/COCO 收敛、独立训练后的精度等价、AMP、多卡或任意输入上的确定性。
没有切换 CPU 模型代算、修改损失/目标覆盖规则或放宽容差。

## 复现方法

先按 [环境](ENVIRONMENT.md) 和 [数据准备](ACCURACY_NATIVE_20261024.md) 建立两套厂商环境；保留厂商 Torch，不能把 NVIDIA wheel 装到 C500。
本轮共同依赖为 NumPy 1.26.4、SciPy 1.16.0、Pillow 11.2.1、OpenCV 4.11.0.86、pycocotools 2.0.10，实际完整环境随结果保存。

1. 审计固定版本的参考仓库：

   ```bash
   git clone https://github.com/xumy1995/model-verify reference
   git -C reference checkout 120d9af31fb022ccec71f1060eee892491341ca1
   python tools/audit_model_verify.py --reference reference --output reference-audit.json
   ```

2. 使用 [checkpoint-plan.json](validation/20261024/model-verify-protocol/checkpoint-plan.json) 中同一队列的全部四个最终权重，保持校验和不变，只调整文件路径。
   从 NVIDIA 环境按模型分别启动新进程准备输入；`SOURCE` 是冻结的 UP 源码根目录，`DATA` 是相同逻辑路径的数据 manifest。
   所需 DETR 源码/权重、标签、anchors 均由该 manifest 校验，不联网下载另一版本。

   ```bash
   export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=131 DEFAULT_TASKS=cls:det
   unset CUBLAS_WORKSPACE_CONFIG MCDNN_USE_DETERMINISTIC_ALGO MCDNN_STABLE_RESULT_MOD
   unset PYTORCH_DISABLE_CUDA_CUDNN_TF32 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
   for model in resnet18 yolov5s detr vit_b16; do
     python tools/verify_checkpoint_protocol.py --phase prepare --vendor nvidia \
       --model "$model" --source "$SOURCE" --data "$DATA" \
       --assets "$ASSETS" --output "$ASSETS/build/$model" --checkpoint-plan checkpoint-plan.json
   done
   ```

3. 将 `ASSETS` 下四个模型的 `weights.pt`、`validation.pt`、`training.pt` 和 `manifest.json` 原样传到 C500。
   本轮 [assets-transfer.json](validation/20261024/model-verify-protocol/assets-transfer.json) 给出全部 16 项及归档校验和。不能各自重新预处理输入。
   权重和输入体积约 1.10 GB，保留在证据存储，不加入 Git。

4. 每家在已分配的卡上串行运行；`VENDOR` 为 `nvidia` 或 `metax`，`CUDA_VISIBLE_DEVICES` 仅指定获配的物理卡。
   每次命令都是新进程。保留上述清理后的环境，native 不设置 BLAS workspace；另两种在进程启动前设置。

   ```bash
   for profile in native ultralytics; do
     if [ "$profile" = native ]; then unset CUBLAS_WORKSPACE_CONFIG;
     else export CUBLAS_WORKSPACE_CONFIG=:4096:8; fi
     for model in resnet18 yolov5s detr vit_b16; do
       for repeat in 01 02; do
         python tools/verify_checkpoint_protocol.py --phase inference \
           --vendor "$VENDOR" --profile "$profile" --model "$model" --trials 10 \
           --source "$SOURCE" --data "$DATA" --assets "$ASSETS" \
           --output "$RESULTS/inference-$model-$profile-$repeat"
       done
     done
   done
   for profile in native ultralytics strict; do
     if [ "$profile" = native ]; then unset CUBLAS_WORKSPACE_CONFIG;
     else export CUBLAS_WORKSPACE_CONFIG=:4096:8; fi
     for model in resnet18 yolov5s detr vit_b16; do
       python tools/verify_checkpoint_protocol.py --phase gradients \
         --vendor "$VENDOR" --profile "$profile" --model "$model" --trials 10 \
         --source "$SOURCE" --data "$DATA" --assets "$ASSETS" \
         --output "$RESULTS/gradients-$model-$profile"
     done
   done
   ```

   诊断失败时保存非零退出码、`report.json` 和 `failure.txt`，继续其他预定案例；不要把失败改为成功或覆盖目录。
   本轮另用单设备锁和控制器记录全部命令、退出码、GPU 身份及源码哈希。模型使用原有 API，不复制参考仓库的模型实现。
