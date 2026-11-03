# 三家厂商如何分工

先打开 [GPU 核查 HTML](GPU_PORTABILITY.html)。页面可按厂商、原版/瘦身版、函数名和处理方式筛选；展开条目可看到源码、签名、依据和验收要求。

## 哪些放进类里

`up/runtime/backend.py::RuntimeBackend` 集中处理设备、显存、同步、RNG、AMP 入口、通信后端和计算库策略。厂商设备 API 兼容时直接复用；不兼容时在这里增加厂商分支或子类，并在 `setup_runtime()` 的工厂位置选择它。

卷积、Linear、Attention 里的普通 Torch 运算继续使用原模型代码。RoIAlign、DeformConv 等专用操作按类别放在 `up/platform/`，提供完整前向、反向和状态，再接到 `up/extensions/python/` 的原入口。不要把全部模型数学和 CUDA 内核塞进一个大类。

## 三位负责人分别交付什么

| 负责人 | 首先确认环境 | 设备/训练接口 | 专用操作 |
|---|---|---|---|
| 沐曦 C500 | MACA、厂商 Torch/TorchVision、驱动版本；CUDA 兼容 API 实际绑定库 | 复用 `torch.cuda` 设备 API；确认 AMP、mcDNN 控制项、mcCL 通信和状态恢复 | 复用已有 Torch IoU/IoF、NMS、Sigmoid Focal 路线并核对变体；其余专用操作移植或绑定沐曦实现 |
| 寒武纪 MLU590 | 匹配的 Torch、torch_mlu、CNToolkit、CNNL、CNCL | 用 `torch.mlu`；核 AMP 与 CNCL；单独核实 CNNL 控制项 | 相同共享路线先验证；专用操作绑定厂商库或实现 MLU 内核及所需反向 |
| 昇腾 910B | 匹配的 Torch、torch_npu、CANN、驱动、HCCL | 用 `torch.npu`；核 AMP 与 HCCL；计算库策略按 CANN 实际能力实现 | 相同共享路线先验证；专用操作绑定适配的 NPU 操作或 SDK 内核及所需反向 |

HTML 中的“需单独实现”是接入任务：先检查厂商库是否已有**相同契约**，有则绑定，不要求重写已有内核。不能只对上操作名字。

## 设备接口清单

当前代码直接使用的设备服务如下；名字兼容就继承厂商 Torch，不另写一层转发。

```text
is_available(), device_count(), current_device(), set_device(index)
get_device_properties(index).total_memory
memory_allocated(device=None), max_memory_allocated(), reset_peak_memory_stats()
empty_cache(), synchronize()
manual_seed(seed), get_rng_state(), set_rng_state(state)
amp.autocast(...), amp.GradScaler(...)
```

平台类额外提供 `device`、`dist_backend`、`memory_available(device, needed_memory, reserve_bytes)`、`configure_dnn(options, legacy_nocudnn=None)`。返回设备、byte 单位显存、bool 可用性；错误带厂商和缺失能力。设备 RNG 只在同厂商/匹配版本中恢复，不互抄跨厂商状态字节。

通信继续使用 `torch.distributed` / DDP / SyncBatchNorm，确认所选 NCCL 兼容 API、CNCL 或 HCCL 真正支持使用的操作、dtype、进程组和异步句柄；不是更换通信字符串就算验收完成。历史 spring/linklink 不在本轮任务中。

## 专用操作怎么实现

HTML 的 16 项 GPU 原生操作逐项列出原签名、源码和反向要求。Softer NMS 原本在 CPU 计算，单列为扩展打包/加载问题，不安排 GPU 内核移植。交付每项时附：

1. **完整操作入口**：输入/输出的 shape、dtype、布局、设备、空输入、坐标和排序规则；与原接口一致。
2. **训练契约**：原操作需要的输入/参数梯度，保存的索引、采样或映射状态；原来不支持反向的操作保持原约定。
3. **接线位置**：原 `up/extensions/python/<操作>.py` 的包装/`.apply()` 入口，按 `get_runtime().platform` 选择完整厂商实现。仅改 `_C` 加载路径不够，包装中的 CUDA 分配和 `is_cuda` 断言也要处理。
4. **导出**：保留原 `symbolic()` 的节点和属性；不支持导出时单独报错。
5. **测试**：固定权重、输入和上游梯度；比较前向、所需反向、状态、空输入与非连续布局。保留 `rtol=1e-4, atol=1e-5`；索引/标志精确比较。原生重复有波动时同时记录，不据此放宽容差。

特别注意：UP Native NMS 与 TorchVision NMS 的坐标契约不同；UP IoU 的 GPU 内核是 `+1` 闭区间；RoIAlign 的采样和 pooling mode 不能换成默认不同的同名算子。

原版 RoIAlign 的 `MAX` Python 分支存在，但其内核未在 pybind 中导出；本轮只把实际导出的 `AVG` 作为接入要求，不要求厂商补一个原版也没有可用的变体。

## 通过 config 选择

在**已安装正确厂商 Torch/SDK 的环境**中，只选一个平台：

```yaml
runtime:
  platform: metax_cuda  # nvidia / metax_cuda / cambricon_mlu / ascend_npu
```

`runtime.device` 可省略，系统由平台决定设备类型；若保留旧字段，应与平台一致：CUDA、MLU、NPU 分别写 `cuda`、`mlu`、`npu`。

```bash
# --opts 放最后；训练内 eval 仍使用同一个运行时
python -m up train --launch pytorch --ng 1 --config configs/det/yolox/yolox_tiny.yaml --opts runtime.platform metax_cuda
python -m up train --evaluate --launch pytorch --ng 1 --config configs/det/yolox/yolox_tiny.yaml --opts runtime.platform metax_cuda
python -m up inference --launch pytorch --ng 1 --config configs/det/yolox/yolox_tiny.yaml -i /data/images -c /data/model.pth -v /data/predictions --opts runtime.platform metax_cuda
```

路径与模型参数按自己的数据设置；将 `metax_cuda` 换成另外三个值即可选厂商。config 只选当前环境的后端，不安装或切换 Torch wheel。

可选 `runtime.dnn` 支持 `enabled / benchmark / deterministic / allow_tf32` 的布尔设置。目前仅接到 CUDA 兼容 PyTorch API；C500 需确认安装版本中这些设置对 mcDNN 的实际效果。MLU/NPU 默认使用厂商原生设置，显式请求未接入的控制项会报错。公开历史 CNNL 源码对部分同名设置只发警告，不应当作有效开关。

## 验收顺序

环境与真实加载库 → 设备/AMP/状态 → 原配方实际用到的标准 Torch 前后向 → 专用操作 → 双卡通信/SyncBN → 训练、训练内 eval、保存恢复、独立推理。每项给版本、命令、输入、输出误差和状态；本轮源码核查与无 GPU 测试不填成真机通过。

本地软件检查：`python -m unittest discover -s tests -v`。源码扫描脚本是 `tools/audit_gpu_portability.py`；输入是三个固定提交的源码快照，输出 JSON 和逐调用位置 CSV。
