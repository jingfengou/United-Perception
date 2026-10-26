# FSDP2 训练

FSDP2 将模型参数、梯度和优化器状态分散到多张卡。UP 通过统一训练接口调用当前厂商的 PyTorch；NVIDIA 与沐曦使用同一套公共实现。

## 配置和运行

要求厂商 PyTorch **2.10+** 提供 FSDP2、DTensor 和分布式通信，同一作业至少两张卡。默认训练仍使用 DDP。
在已有模型 YAML 的 `runtime` 下增加配置，其余模型、损失、优化器分组和训练配方保持原样：

```yaml
runtime:
  platform:
    backend: metax                 # NVIDIA 改为 nvidia
  distributed:
    strategy: fsdp2
    shard_modules: [backbone, head] # 真实子模块路径；应按模型结构填写
    reshard_after_forward: true
```

`shard_modules` 支持 `backbone.blocks.0` 这类完整路径，平台从内向外分片，最后处理根模型。省略列表时仅对整个模型分片，通信重叠较少。路径填写错误直接报错。

在对应硬件上，沿用[通用镜像运行示例](QUICKSTART_UNIFIED.md)准备数据、配置和工作目录：

```bash
export UP_WORK="$HOME/up-demo"
export UP_DATA="/datasets/coco2017"
export VENDOR=metax                 # NVIDIA 使用 nvidia

# 两张卡训练；--backend dist 表示使用 PyTorch 通信，与厂商和 FSDP 策略分开
UP_GPU=2,3 ./up_docker "$VENDOR" train --config model.yaml \
  --backend dist --launch pytorch --fork-method spawn --ng 2

# 验证使用同一训练平台路径
UP_GPU=2,3 ./up_docker "$VENDOR" eval --config model.yaml \
  --backend dist --launch pytorch --fork-method spawn --ng 2 \
  --opts saver.auto_resume False saver.resume_model checkpoints/ckpt_latest.pth

# 恢复训练：相同设备数、模型配置和优化器分组
UP_GPU=2,3 ./up_docker "$VENDOR" train --config model.yaml \
  --backend dist --launch pytorch --fork-method spawn --ng 2 \
  --opts saver.auto_resume False saver.resume_model checkpoints/ckpt_latest.pth

# 独立推理读取普通完整权重，不启动 FSDP，也不需要两张卡
UP_GPU=2 ./up_docker "$VENDOR" inference --config model.yaml \
  --ckpt checkpoints/ckpt_latest.pth --img_path images/ \
  --vis_dir predictions --launch pytorch --ng 1
```

以上显卡编号是示例，NVIDIA 机器通常使用 `UP_GPU=0,1`。原生环境使用对应环境的 `python -m up train`，具体安装和启动方式见[环境指南](ENVIRONMENT.md)。修改源码后需重新构建镜像。

## 平台层负责什么

| 接口 | 行为 |
| --- | --- |
| `wrap_fsdp` | 同步初始参数，将参数整理为 FSDP2 所需的连续布局，按设备和模块路径分片；之后再创建优化器 |
| `sharded_grad_scaler` | 在 GPU 本地梯度分片上取消 AMP 缩放；任一张卡溢出时，所有卡一起跳过更新 |
| `full_state_dict` | 所有 rank 参与收集，主 rank 得到完整 CPU 权重和按参数名保存的优化器状态 |
| `load_full_state_dict` | 将完整 checkpoint 恢复为当前设备的分片，保留参数组；兼容原有按参数编号保存的优化器状态 |
| `validation` | 训练内临时收集完整参数，允许各 rank 验证批次数不同；退出后恢复分片和每个模块原有的训练状态 |

具体签名见[训练接口](INTERFACES.md#trainingwrap_fsdp)。两家在 `training/nvidia.py`、`training/metax.py` 显式绑定公共实现；新增厂商可复用，但其 PyTorch 必须真实支持 FSDP2、DTensor 和所需集合通信，否则补齐厂商实现或明确报不支持。

`get_platform().require(['training.wrap_fsdp'])` 可检查注册与 API 依赖；真机执行另行验收。

## 保存、验证和限制

- 权重名称保持原模型名称，完整 checkpoint 可供普通模型推理。优化器在分片后创建；UP 仍决定参数组、学习率、损失和训练步骤。
- 支持 base runner 与 base ModelHelper、标准 SGD/Adam/AdamW、AMP、梯度裁剪、EMA、普通保存和最佳权重保存。保存需要所有 rank 参与，只有主 rank 写文件。
- 训练恢复还保留各 rank 的 BN 等 buffer、CPU/设备/Python/NumPy 随机状态和 scaler，并记录“已更新、尚未推进学习率”的保存边界，避免恢复时重复该批次。更改设备数时应通过 `pretrain_model` 仅加载权重，再开始新训练。
- 验证临时需要每张卡容纳模型及已启用 EMA 的完整参数；checkpoint 收集需要主 rank 有足够 CPU 内存。这版没有增加分片文件存储、CPU 参数卸载或跨节点混合分片配置。
- NAS、量化/剪枝等专用 runner、`async_norm` 和信号触发的 `auto_checkpoint` 尚未接入此路径，选择这些组合会报错。
- FSDP 会改变并行归约顺序，性能和数值需按模型实测；原生算子的波动仍按原容差检查。

## 验证命令

在已安装对应厂商依赖、可独占两卡的环境中执行：

```bash
export VENDOR=metax
export CUDA_VISIBLE_DEVICES=2,3
export DEFAULT_TASKS=cls:det

python -m pytest tests/platform -q

python -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/platform/fsdp_smoke.py --vendor "$VENDOR" --output /tmp/up-fsdp-fp32
python tests/platform/fsdp_smoke.py --vendor "$VENDOR" --purpose inference --output /tmp/up-fsdp-fp32

python -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/platform/fsdp_smoke.py --vendor "$VENDOR" --amp --output /tmp/up-fsdp-amp
python tests/platform/fsdp_smoke.py --vendor "$VENDOR" --purpose inference --amp --output /tmp/up-fsdp-amp

python -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/platform/fsdp_up_runner.py --vendor "$VENDOR" --output /tmp/up-fsdp-runner
python tests/platform/fsdp_up_runner.py --vendor "$VENDOR" --purpose inference --output /tmp/up-fsdp-runner
```

每次使用新的输出目录。前一组检查 DDP/FSDP 的结果、全局梯度裁剪、EMA、验证、恢复和单 rank AMP 溢出；后一组使用真实 UP BaseRunner、ResNet18 和合成批次检查完整生命周期。它们检验功能与恢复，不是数据集精度训练或性能基准。

已验证以下单机双卡环境：

| 设备与 PyTorch | 小网络 FP32/AMP 与 DDP 对照 | UP ResNet18 AMP + EMA |
| --- | --- | --- |
| RTX 4090 × 2，`2.10.0+cu126` | 通过 | 训练、训练内 eval、恢复、独立推理通过 |
| 沐曦 C500 × 2，`2.10.0+metax3.8.1.0` | 通过 | 训练、训练内 eval、恢复、独立推理通过 |

数值检查沿用 `rtol=1e-4, atol=1e-5`。两家还验证了单 rank 溢出时全体跳步，以及验证批次数不等时正常退出。原始日志和失败阶段保存在仓库外，公开代码包含上述复现工具。

实现依据：[PyTorch FSDP2](https://docs.pytorch.org/docs/2.10/distributed.fsdp.fully_shard.html)、[分布式状态字典](https://docs.pytorch.org/docs/2.10/distributed.checkpoint.html)。
