# 三家厂商 GPU 函数交付清单

这份文件直接交给 C500、MLU590、910B 的负责人。每项都有函数接口、作用、输入输出、反向要求、交付位置和 NVIDIA 参考实现。

## 先确定分工

| 工作 | 厂商负责人交付 | UP 维护者负责 |
|---|---|---|
| 设备、AMP、通信 | 验证已有厂商 API；遇到不兼容项提供适配器 | 在 RuntimeBackend 中选择适配器，保留训练/推理调用 |
| 专用 GPU 操作 | 按下面的 16 个方法交付完整操作及测试；可调用厂商已有库 | 接到原包装/.apply 入口，维护配置与导出兼容 |
| 卷积、Linear、普通激活等标准 Torch | 验证原配方实际用到的 dtype/shape，提供异常清单 | 保留原模型算法，不重复包装这些操作 |

16 个是本仓库保留的专用 GPU 操作入口，不是要求每家重写全部 PyTorch。标为点云可选的条目，只有恢复该任务时才需要接入。

## 实现文件放在哪里

```text
up/platform/<vendor>/
  __init__.py      # 导出 NativeOps；普通导入不初始化设备
  native_ops.py    # class NativeOps：实现下面同名方法，内部含所需 autograd
  device.py        # 只有设备/AMP API 不兼容时才新增适配器
  csrc/            # 可选：厂商 SDK 内核与编译代码
tests/<vendor>/   # 前向、梯度、状态、布局、边界和实际运行测试
```

`<vendor>` 分别为 `metax_cuda`、`cambricon_mlu`、`ascend_npu`。这些是**交付路径约定**，当前还没有注册这三套 NativeOps 类；现有共享实现和 NVIDIA 原扩展继续按原路运行。

复制 [完整接口模板](templates/native_ops.py) 后，将类命名为 `NativeOps`。未实现的方法明确抛出 NotImplementedError，允许按模型先交付一部分。模板本身不会让算子自动可用。

**接入必须发生在原 `Function.apply()` 之前。** 厂商方法返回带所需梯度图的完整操作；不要在旧 `Function.forward()` 内套另一个完整 autograd Function，这会进入 no_grad 前向并切断预期的图。框架侧可在外层模块/兼容入口选厂商；直接使用旧 `.apply()` 的路径也要一起接入。

## 三家分别做什么

| 厂商 | config 值 | 设备/AMP API | 通信 | 交付目录 |
|---|---|---|---|---|
| 沐曦 C500 | `metax_cuda` | `torch.cuda`，厂商 Torch | mcCL 的 NCCL 兼容 API | `up/platform/metax_cuda/` |
| 寒武纪 MLU590 | `cambricon_mlu` | `torch.mlu`，torch_mlu | CNCL | `up/platform/cambricon_mlu/` |
| 昇腾 910B | `ascend_npu` | `torch.npu`，torch_npu | HCCL | `up/platform/ascend_npu/` |

C500 的 CUDA 兼容服务优先复用；MLU/NPU 的主入口已映射到 torch.mlu/torch.npu。三家都先运行下面的接口检查，有真实差异才写适配器。厂商环境需在启动前安装好，config 不替换 Torch。

按负责人下载任务表：[C500](assignments/metax_cuda.csv) · [MLU590](assignments/cambricon_mlu.csv) · [910B](assignments/ascend_npu.csv)。每张表都包含具体签名和 NVIDIA 源码链接。

## A. 设备、AMP、通信接口：兼容则复用

这里列 UP 实际需要的调用形式。完整标准库签名由所安装的 Torch 版本决定，不要求重写 Torch 整套接口。设备总量和已分配显存使用 byte。

| 接口 | 返回 | 做什么 | NVIDIA 对应函数/实现 | 修改位置 |
|---|---|---|---|---|
| `is_available()` | bool | 当前设备和运行库能否使用；不把“导入成功”当卡可用。 | [torch.cuda.is_available](https://docs.pytorch.org/docs/2.10/cuda.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/cuda/__init__.py) | RuntimeBackend.module；API 不兼容才在 up/platform/{vendor}/device.py 写适配器 |
| `device_count()` | int | 当前进程可见设备数，遵守可见卡设置。 | [torch.cuda.device_count](https://docs.pytorch.org/docs/2.10/cuda.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/cuda/__init__.py) | RuntimeBackend.module；API 不兼容才在 up/platform/{vendor}/device.py 写适配器 |
| `current_device()` | int | 当前进程绑定的逻辑卡号。 | [torch.cuda.current_device](https://docs.pytorch.org/docs/2.10/cuda.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/cuda/__init__.py) | RuntimeBackend.module；API 不兼容才在 up/platform/{vendor}/device.py 写适配器 |
| `set_device(index)` | None | 绑定本 rank 的设备；不改其他 rank 的卡。 | [torch.cuda.set_device](https://docs.pytorch.org/docs/2.10/cuda.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/cuda/__init__.py) | RuntimeBackend.module；API 不兼容才在 up/platform/{vendor}/device.py 写适配器 |
| `get_device_properties(index)` | 对象，含 total_memory:int | 显存总量按 byte 返回；不要返回 MB。 | [torch.cuda.get_device_properties](https://docs.pytorch.org/docs/2.10/cuda.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/cuda/__init__.py) | RuntimeBackend.module；API 不兼容才在 up/platform/{vendor}/device.py 写适配器 |
| `memory_allocated(device=None)` | int，byte | 当前 allocator 已分配显存，支持查询指定设备。 | [torch.cuda.memory_allocated](https://docs.pytorch.org/docs/2.10/cuda.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/cuda/memory.py) | RuntimeBackend.module；API 不兼容才在 up/platform/{vendor}/device.py 写适配器 |
| `max_memory_allocated(device=None)` | int，byte | 自最近重置以来的显存峰值。 | [torch.cuda.max_memory_allocated](https://docs.pytorch.org/docs/2.10/cuda.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/cuda/memory.py) | RuntimeBackend.module；API 不兼容才在 up/platform/{vendor}/device.py 写适配器 |
| `reset_peak_memory_stats(device=None)` | None | 重置峰值统计，不释放正在使用的张量。 | [torch.cuda.reset_peak_memory_stats](https://docs.pytorch.org/docs/2.10/cuda.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/cuda/memory.py) | RuntimeBackend.module；API 不兼容才在 up/platform/{vendor}/device.py 写适配器 |
| `empty_cache()` | None | 释放 allocator 可释放缓存，不删除活跃 Tensor。 | [torch.cuda.empty_cache](https://docs.pytorch.org/docs/2.10/cuda.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/cuda/memory.py) | RuntimeBackend.module；API 不兼容才在 up/platform/{vendor}/device.py 写适配器 |
| `synchronize(device=None)` | None | 等待指定/当前设备的工作完成。 | [torch.cuda.synchronize](https://docs.pytorch.org/docs/2.10/cuda.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/cuda/__init__.py) | RuntimeBackend.module；API 不兼容才在 up/platform/{vendor}/device.py 写适配器 |
| `manual_seed(seed)` | None | 按 UP 原种子规则设置当前设备 RNG。 | [torch.cuda.manual_seed](https://docs.pytorch.org/docs/2.10/cuda.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/cuda/__init__.py) | RuntimeBackend.module；API 不兼容才在 up/platform/{vendor}/device.py 写适配器 |
| `get_rng_state(device=None)` | 可保存的状态对象 | 取出当前设备 RNG 状态，用于 checkpoint/重计算。 | [torch.cuda.get_rng_state](https://docs.pytorch.org/docs/2.10/cuda.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/cuda/__init__.py) | RuntimeBackend.module；API 不兼容才在 up/platform/{vendor}/device.py 写适配器 |
| `set_rng_state(state, device=None)` | None | 恢复同厂商兼容版本保存的 RNG 状态。 | [torch.cuda.set_rng_state](https://docs.pytorch.org/docs/2.10/cuda.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/cuda/__init__.py) | RuntimeBackend.module；API 不兼容才在 up/platform/{vendor}/device.py 写适配器 |
| `autocast(enabled=True)` | 上下文管理器 | 前向按厂商 AMP 策略选 dtype；退出时恢复上下文。原匹配中的 enabled=False 仍强制该段关闭 AMP。 | [torch.cuda.amp.autocast](https://docs.pytorch.org/docs/2.10/notes/amp_examples.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/amp/autocast_mode.py) | RuntimeBackend.amp；工厂返回厂商 AMP，名字/行为不兼容才写适配器 |
| `GradScaler(enabled=True)` | scaler 对象 | 创建缩放器；当前 UP 使用 enabled 参数，不要求复制所有版本的额外参数。 | [torch.cuda.amp.GradScaler](https://docs.pytorch.org/docs/2.10/notes/amp_examples.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/amp/grad_scaler.py) | RuntimeBackend.amp；工厂返回厂商 AMP，名字/行为不兼容才写适配器 |
| `scale(loss)` | Tensor，保留计算图 | 缩放 loss，允许随后 .backward()。 | [scaler.scale](https://docs.pytorch.org/docs/2.10/notes/amp_examples.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/amp/grad_scaler.py) | RuntimeBackend.amp；工厂返回厂商 AMP，名字/行为不兼容才写适配器 |
| `unscale_(optimizer)` | None | 按需在梯度裁剪前原地还原梯度；同一步不能重复还原。 | [scaler.unscale_](https://docs.pytorch.org/docs/2.10/notes/amp_examples.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/amp/grad_scaler.py) | RuntimeBackend.amp；工厂返回厂商 AMP，名字/行为不兼容才写适配器 |
| `step(optimizer)` | 原 optimizer.step 的返回/跳过时 None | 检查溢出，正常才更新参数；不改优化算法或参数组。 | [scaler.step](https://docs.pytorch.org/docs/2.10/notes/amp_examples.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/amp/grad_scaler.py) | RuntimeBackend.amp；工厂返回厂商 AMP，名字/行为不兼容才写适配器 |
| `update()` | None | 按原规则更新下一步 scale。 | [scaler.update](https://docs.pytorch.org/docs/2.10/notes/amp_examples.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/amp/grad_scaler.py) | RuntimeBackend.amp；工厂返回厂商 AMP，名字/行为不兼容才写适配器 |
| `state_dict()` | dict | 保存 scale 和相关计数。 | [scaler.state_dict](https://docs.pytorch.org/docs/2.10/notes/amp_examples.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/amp/grad_scaler.py) | RuntimeBackend.amp；工厂返回厂商 AMP，名字/行为不兼容才写适配器 |
| `load_state_dict(state_dict)` | None | 恢复训练，不重置已保存的缩放状态。 | [scaler.load_state_dict](https://docs.pytorch.org/docs/2.10/notes/amp_examples.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/amp/grad_scaler.py) | RuntimeBackend.amp；工厂返回厂商 AMP，名字/行为不兼容才写适配器 |
| `init_process_group(backend, rank, world_size, init_method=None, timeout=None)` | 初始化默认组 | 列出 UP 使用的参数；未指定 timeout 时沿用匹配 Torch 的默认值。先绑设备，再初始化厂商组。 | [torch.distributed.init_process_group](https://docs.pytorch.org/docs/2.10/distributed.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py) | RuntimeBackend.dist_backend + up/utils/env/{launch,dist_helper}.py；SyncBN 见 up/utils/model/bn_helper.py |
| `new_group(ranks=None, backend=None)` | ProcessGroup | 支持 UP 的局部 rank 分组；对象通信保留 Gloo CPU 组。 | [torch.distributed.new_group](https://docs.pytorch.org/docs/2.10/distributed.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py) | RuntimeBackend.dist_backend + up/utils/env/{launch,dist_helper}.py；SyncBN 见 up/utils/model/bn_helper.py |
| `all_reduce(tensor, op=SUM, group=None, async_op=False)` | None / Work | 原地归约，维持 op、dtype、组和异步句柄语义。 | [torch.distributed.all_reduce](https://docs.pytorch.org/docs/2.10/distributed.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py) | RuntimeBackend.dist_backend + up/utils/env/{launch,dist_helper}.py；SyncBN 见 up/utils/model/bn_helper.py |
| `broadcast(tensor, src, group=None, async_op=False)` | None / Work | 从指定 rank 广播，原地更新接收 Tensor。 | [torch.distributed.broadcast](https://docs.pytorch.org/docs/2.10/distributed.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py) | RuntimeBackend.dist_backend + up/utils/env/{launch,dist_helper}.py；SyncBN 见 up/utils/model/bn_helper.py |
| `all_gather(tensor_list, tensor, group=None, async_op=False)` | None / Work | 按 rank 顺序收集；保留列表内容和 dtype。 | [torch.distributed.all_gather](https://docs.pytorch.org/docs/2.10/distributed.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py) | RuntimeBackend.dist_backend + up/utils/env/{launch,dist_helper}.py；SyncBN 见 up/utils/model/bn_helper.py |
| `gather(tensor, gather_list=None, dst=0, group=None, async_op=False)` | None / Work | 只在目标 rank 填充 gather_list。 | [torch.distributed.gather](https://docs.pytorch.org/docs/2.10/distributed.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py) | RuntimeBackend.dist_backend + up/utils/env/{launch,dist_helper}.py；SyncBN 见 up/utils/model/bn_helper.py |
| `barrier(group=None, async_op=False)` | None / Work | 只同步目标组。 | [torch.distributed.barrier](https://docs.pytorch.org/docs/2.10/distributed.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py) | RuntimeBackend.dist_backend + up/utils/env/{launch,dist_helper}.py；SyncBN 见 up/utils/model/bn_helper.py |
| `destroy_process_group(group=None)` | None | 正常退出并释放本任务的通信资源。 | [torch.distributed.destroy_process_group](https://docs.pytorch.org/docs/2.10/distributed.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py) | RuntimeBackend.dist_backend + up/utils/env/{launch,dist_helper}.py；SyncBN 见 up/utils/model/bn_helper.py |
| `DistributedDataParallel(module, device_ids=None, find_unused_parameters=False, broadcast_buffers=True)` | Module | 列出 UP 的主要参数，完整签名沿用匹配 Torch；核多 rank 梯度同步、unused 参数和恢复。 | [torch.nn.parallel.DistributedDataParallel](https://docs.pytorch.org/docs/2.10/distributed.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/nn/parallel/distributed.py) | RuntimeBackend.dist_backend + up/utils/env/{launch,dist_helper}.py；SyncBN 见 up/utils/model/bn_helper.py |
| `SyncBatchNorm.forward(input)` | Tensor，与输入布局一致 | 训练跨 rank 同步统计并提供反向；eval 不更新统计。核 UP 的 TaskSyncBN 所用 PyTorch 私有 SyncBatchNorm.apply 版本兼容性。 | [torch.nn.SyncBatchNorm.forward / torch.nn.modules._functions.SyncBatchNorm.apply](https://docs.pytorch.org/docs/2.10/distributed.html) · [实现源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/nn/modules/batchnorm.py) | RuntimeBackend.dist_backend + up/utils/env/{launch,dist_helper}.py；SyncBN 见 up/utils/model/bn_helper.py |
| `memory_available(device, needed_memory, reserve_bytes=536870912)` | bool | 按 byte 比较 total_memory - memory_allocated - reserve_bytes；保留原 UP 的显存估计和比较规则。 | [RuntimeBackend.memory_available → torch.cuda.get_device_properties / memory_allocated](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/runtime/backend.py#L51) · [实现源码](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/runtime/backend.py#L51) | RuntimeBackend.memory_available；底层名称不兼容时写厂商 device.py 适配器 |
| `configure_dnn(options=None, legacy_nocudnn=None)` | None；未支持项抛 NotImplementedError | options 允许 enabled/benchmark/deterministic/allow_tf32 的 bool；只接真实有效的厂商控制项。None 保留当前默认；legacy_nocudnn 处理原 CLI。 | [RuntimeBackend.configure_dnn → torch.backends.cudnn](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/runtime/backend.py#L59) · [实现源码](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/runtime/backend.py#L59) | RuntimeBackend.configure_dnn；有厂商差异时由 device.py 提供策略实现 |

另外，现有平台类提供 `device`、`amp`、`dist_backend`、`set_device(local_rank)`、`memory_available(device, needed_memory, reserve_bytes)`、`configure_dnn(options, legacy_nocudnn=None)`；参考 [RuntimeBackend](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/runtime/backend.py)。

DNN 策略逐项处理 `enabled`、`benchmark`、`deterministic`、`allow_tf32`：NVIDIA 参考 `torch.backends.cudnn`；C500 核 mcDNN 的兼容控制项；MLU/910B 核 CNNL/CANN 实际能力。有对应控制项才接入，未提供的项明确报不支持；默认沿厂商原生策略。

AMP 的调用顺序保持 `autocast → scale(loss).backward() → 必要时 unscale_/裁剪 → step → update`，缩放器随 checkpoint 保存恢复。参考 [PyTorch AMP 示例](https://docs.pytorch.org/docs/2.10/notes/amp_examples.html)。

通信还需核 rank/world_size、组状态和异步 Work.wait()；不重写参数更新算法。TaskSyncBN 使用了 PyTorch 私有 SyncBatchNorm.apply，升级 Torch 时应单独检查签名与行为。

## B. 专用 GPU 操作：逐方法交付

以下 16 个方法属于 NativeOps。公共输入 Tensor 应在同一目标设备；索引输出和辅助状态也放在正确设备，不写死第0张卡。先完成 FP32 的 NVIDIA 配对；AMP/dtype 按原可执行配置补齐，不能仅凭 kernel 的模板声明认定整个 Python 包装支持某个 dtype。

有梯度要求的方法必须返回可反向的完整操作；内部可用厂商已有 autograd 操作，或自写 torch.autograd.Function 并保存状态。NMS 等不要求梯度；RoIPointPool3d 的原反向未实现。

| 方法 | 作用 | 当前任务状态 | NVIDIA Python 入口 |
|---|---|---|---|
| [`roi_align`](#roi-align) | RoIAlign 区域特征采样 | 需要厂商绑定或移植；有原生包装；当前瘦身模型未见直接调用，使用 RoI 模型时接入。 | [up/extensions/python/roi_align.py:30 · RoIAlignFunction.forward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/roi_align.py#L30) |
| [`psroi_align`](#psroi-align) | PSRoIAlign 位置敏感采样 | 需要厂商绑定或移植；有原生包装；按所选 RoI 配置接入。 | [up/extensions/python/psroi_align.py:20 · PSRoIAlignFunction.forward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/psroi_align.py#L20) |
| [`psroi_pool`](#psroi-pool) | PSRoIPool 位置敏感池化 | 需要厂商绑定或移植；有原生包装；按所选 RoI 配置接入。 | [up/extensions/python/psroi_pool.py:25 · PSRoIPoolFunction.forward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/psroi_pool.py#L25) |
| [`deform_conv`](#deform-conv) | DeformConv V1 可变形卷积 | 需要厂商绑定或移植；瘦身版保留 DCN 调用；配置使用 DCN 时需要。 | [up/extensions/python/deformable_conv.py:31 · DeformableConvFunction.forward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/deformable_conv.py#L31) |
| [`naive_nms`](#naive-nms) | UP Native NMS | 需要厂商绑定或移植；原扩展 API 保留；当前常用 NMS 路线另有共享 Torch 实现。 | [up/extensions/python/nms.py:9 · naive_nms](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/nms.py#L9) |
| [`sigmoid_focal`](#sigmoid-focal) | Sigmoid Focal Loss | 共享路线可复用，公开入口/变体待验；检测 Focal 配置可触发；已有部分共享实现。 | [up/extensions/python/focal_loss.py:11 · SigmoidFocalLossFunction.forward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/focal_loss.py#L11) |
| [`softmax_focal`](#softmax-focal) | Softmax Focal Loss | 需要厂商绑定或移植；原包装/损失配置保留，选择 Softmax Focal 时接入。 | [up/extensions/python/focal_loss.py:82 · SoftmaxFocalLossFunction.forward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/focal_loss.py#L82) |
| [`cross_focal`](#cross-focal) | Cross Sigmoid Focal Loss | 需要厂商绑定或移植；原包装/损失配置保留，选择 Cross Focal 时接入。 | [up/extensions/python/cross_focal_loss.py:12 · CrossSigmoidFocalLossFunction.forward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/cross_focal_loss.py#L12) |
| [`overlap`](#overlap) | 二维框 IoU / IoF / IoS | 共享路线可复用，公开入口/变体待验；瘦身检测匹配使用 IoU/IoF；已有部分共享实现。 | [up/extensions/python/iou_overlap.py:5 · gpu_iou_overlap](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/iou_overlap.py#L5) |
| [`bev_iou`](#bev-iou) | 旋转框 BEV IoU | 需要厂商绑定或移植；点云模型已从瘦身任务移除，包装保留；恢复点云任务时接入。 | [up/extensions/python/iou3d_nms_utils.py:31 · boxes_iou_bev](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/iou3d_nms_utils.py#L31) |
| [`iou3d`](#iou3d) | 三维框 IoU | 需要厂商绑定或移植；点云可选包装，按需接入。 | [up/extensions/python/iou3d_nms_utils.py:48 · boxes_iou3d_gpu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/iou3d_nms_utils.py#L48) |
| [`nms3d`](#nms3d) | 旋转 3D NMS | 需要厂商绑定或移植；点云可选包装，按需接入。 | [up/extensions/python/iou3d_nms_utils.py:82 · nms_gpu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/iou3d_nms_utils.py#L82) |
| [`nms3d_normal`](#nms3d-normal) | 普通 3D NMS | 需要厂商绑定或移植；点云可选包装，按需接入。 | [up/extensions/python/iou3d_nms_utils.py:100 · nms_normal_gpu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/iou3d_nms_utils.py#L100) |
| [`points_in_boxes`](#points-in-boxes) | 点属于哪个三维框 | 需要厂商绑定或移植；点云可选包装，按需接入。 | [up/extensions/python/roiaware_pool3d_utils.py:34 · points_in_boxes_gpu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/roiaware_pool3d_utils.py#L34) |
| [`roiaware`](#roiaware) | RoIAwarePool3d 体素池化 | 需要厂商绑定或移植；点云可选包装，按需接入。 | [up/extensions/python/roiaware_pool3d_utils.py:63 · RoIAwarePool3dFunction.forward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/roiaware_pool3d_utils.py#L63) |
| [`roipoint`](#roipoint) | RoIPointPool3d 区域点采样 | 需要厂商绑定或移植；点云可选包装，按需接入。 | [up/extensions/python/roipoint_pool3d_utils.py:32 · RoIPointPool3dFunction.forward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/roipoint_pool3d_utils.py#L32) |

### roi-align

**RoIAlign 区域特征采样**

厂商需要交付的方法：

```python
roi_align(features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, pooled_mode, *, aligned)
```

对应原 NVIDIA 的完整入口：

```python
RoIAlignFunction.apply(features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, pooled_mode)
```

- **输入与含义：** features[N,C,H,W]；rois[K,5]=[batch_idx,x1,y1,x2,y2]；pooled_h/pooled_w 为输出尺寸；spatial_scale 为坐标缩放；sampling_ratio 为采样数；pooled_mode="AVG"；aligned 为原 ALIGNED_FLAG.aligned。
- **返回：** Tensor[K,C,pooled_h,pooled_w]；与 features 同设备、同浮点 dtype。
- **反向与状态：** 对 features 求梯度；rois 和标量参数返回 None。保存 rois、特征尺寸和采样参数。
- **必须保留：** 输入按原包装转为 contiguous，rois 转成 features.dtype。空输出直接返回零尺寸 Tensor。MAX 的 Python 分支存在，但原 pybind 未导出 MAX 内核；本轮不要求补 MAX。
- **实现位置：** `up/platform/<vendor>/native_ops.py::NativeOps.roi_align`。
- **使用范围：** 有原生包装；当前瘦身模型未见直接调用，使用 RoI 模型时接入。

NVIDIA 实现链：

1. Python / autograd：[up/extensions/python/roi_align.py:30 · RoIAlignFunction.forward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/roi_align.py#L30)；[up/extensions/python/roi_align.py:58 · RoIAlignFunction.backward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/roi_align.py#L58)；[up/extensions/python/roi_align.py:18 · RoIAlignFunction.symbolic](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/roi_align.py#L18)
2. 扩展导出：[up/extensions/csrc/pybind.cpp:36 · up.extensions._C.roi_align.forward_avg_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L36)；[up/extensions/csrc/pybind.cpp:37 · up.extensions._C.roi_align.backward_avg_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L37)
3. C++ 实现：[up/extensions/csrc/roi_align/roi_align_cuda.cpp:5 · roi_align_avg_forward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/roi_align/roi_align_cuda.cpp#L5)；[up/extensions/csrc/roi_align/roi_align_cuda.cpp:39 · roi_align_avg_backward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/roi_align/roi_align_cuda.cpp#L39)
4. CUDA 算法：[up/extensions/csrc/roi_align/roi_align_kernel.cu:1 · roi_align_kernel.cu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/roi_align/roi_align_kernel.cu#L1)

`aligned` 是新增交付接口的显式上下文参数，由 UP 维护者从原全局标志传入；厂商代码不自行读取 UP 的全局业务变量。

### psroi-align

**PSRoIAlign 位置敏感采样**

厂商需要交付的方法：

```python
psroi_align(features, rois, group_size, spatial_scale, output_dim, sampling_ratio)
```

对应原 NVIDIA 的完整入口：

```python
PSRoIAlignFunction.apply(features, rois, group_size, spatial_scale, output_dim, sampling_ratio)
```

- **输入与含义：** features[N,C,H,W]；rois[K,5]；group_size、spatial_scale、output_dim、sampling_ratio。外层按通道数计算 output_dim。
- **返回：** Tensor[K,output_dim,group_size,group_size]，同特征设备/dtype；内部 mapping_channel 为 int32、同设备。
- **反向与状态：** 只对 features 求梯度；保存 rois、mapping_channel、输入尺寸和采样参数。
- **必须保留：** 保持位置到通道的映射，不替换成普通 RoIAlign。原包装没有专门的零 ROI 分支，空输入行为需与 NVIDIA 参考实测对齐。
- **实现位置：** `up/platform/<vendor>/native_ops.py::NativeOps.psroi_align`。
- **使用范围：** 有原生包装；按所选 RoI 配置接入。

NVIDIA 实现链：

1. Python / autograd：[up/extensions/python/psroi_align.py:20 · PSRoIAlignFunction.forward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/psroi_align.py#L20)；[up/extensions/python/psroi_align.py:47 · PSRoIAlignFunction.backward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/psroi_align.py#L47)；[up/extensions/python/psroi_align.py:9 · PSRoIAlignFunction.symbolic](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/psroi_align.py#L9)
2. 扩展导出：[up/extensions/csrc/pybind.cpp:45 · up.extensions._C.psroi_align.forward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L45)；[up/extensions/csrc/pybind.cpp:46 · up.extensions._C.psroi_align.backward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L46)
3. C++ 实现：[up/extensions/csrc/psroi_align/psroi_align_cuda.cpp:5 · psroi_align_forward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/psroi_align/psroi_align_cuda.cpp#L5)；[up/extensions/csrc/psroi_align/psroi_align_cuda.cpp:47 · psroi_align_backward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/psroi_align/psroi_align_cuda.cpp#L47)
4. CUDA 算法：[up/extensions/csrc/psroi_align/psroi_align_kernel.cu:1 · psroi_align_kernel.cu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/psroi_align/psroi_align_kernel.cu#L1)

### psroi-pool

**PSRoIPool 位置敏感池化**

厂商需要交付的方法：

```python
psroi_pool(features, rois, group_size, spatial_scale, output_dim)
```

对应原 NVIDIA 的完整入口：

```python
PSRoIPoolFunction.apply(features, rois, group_size, spatial_scale, output_dim)
```

- **输入与含义：** features[N,C,H,W]；rois[K,5]；group_size、spatial_scale、output_dim。
- **返回：** Tensor[K,output_dim,group_size,group_size]，同设备/dtype；内部 mapping_channel 为 int32。
- **反向与状态：** 只对 features 求梯度；保存 rois、mapping_channel 和输入尺寸。
- **必须保留：** 按原池化 bin 边界、取整和位置通道映射实现；与 PSRoIAlign 分开。空输入先测 NVIDIA 参考。
- **实现位置：** `up/platform/<vendor>/native_ops.py::NativeOps.psroi_pool`。
- **使用范围：** 有原生包装；按所选 RoI 配置接入。

NVIDIA 实现链：

1. Python / autograd：[up/extensions/python/psroi_pool.py:25 · PSRoIPoolFunction.forward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/psroi_pool.py#L25)；[up/extensions/python/psroi_pool.py:53 · PSRoIPoolFunction.backward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/psroi_pool.py#L53)；[up/extensions/python/psroi_pool.py:15 · PSRoIPoolFunction.symbolic](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/psroi_pool.py#L15)
2. 扩展导出：[up/extensions/csrc/pybind.cpp:52 · up.extensions._C.psroi_pooling.forward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L52)；[up/extensions/csrc/pybind.cpp:53 · up.extensions._C.psroi_pooling.backward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L53)
3. C++ 实现：[up/extensions/csrc/psroi_pooling/psroi_pooling_cuda.cpp:5 · psroi_pooling_forward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/psroi_pooling/psroi_pooling_cuda.cpp#L5)；[up/extensions/csrc/psroi_pooling/psroi_pooling_cuda.cpp:46 · psroi_pooling_backward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/psroi_pooling/psroi_pooling_cuda.cpp#L46)
4. CUDA 算法：[up/extensions/csrc/psroi_pooling/psroi_pooling_kernel.cu:1 · psroi_pooling_kernel.cu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/psroi_pooling/psroi_pooling_kernel.cu#L1)

### deform-conv

**DeformConv V1 可变形卷积**

厂商需要交付的方法：

```python
deform_conv(input, offset, weight, out_channels, kernel_size, stride, pad, dilations, groups, deformable_groups)
```

对应原 NVIDIA 的完整入口：

```python
DeformableConvFunction.apply(input, offset, weight, out_channels, kernel_size, stride, pad, dilations, groups, deformable_groups)
```

- **输入与含义：** input[N,Cin,H,W]、offset[N,2*deformable_groups*kh*kw,Hout,Wout]、weight[Cout,Cin/groups,kh,kw]；out_channels、kernel_size、stride、pad、dilations、groups、deformable_groups，二维参数沿用原 tuple。
- **返回：** Tensor[N,Cout,Hout,Wout]；Hout/Wout 按原 _output_size 公式；无 bias 参数。
- **反向与状态：** 按 needs_input_grad 提供 input、offset、weight 三者梯度；保存必要 columns/ones 等缓冲区或厂商等效状态。
- **必须保留：** 这是 DCNv1，不是增加 modulation mask 的 DCNv2。保留参数键、采样顺序、groups 与 deformable_groups；小输入仍按原公式报错。
- **实现位置：** `up/platform/<vendor>/native_ops.py::NativeOps.deform_conv`。
- **使用范围：** 瘦身版保留 DCN 调用；配置使用 DCN 时需要。

NVIDIA 实现链：

1. Python / autograd：[up/extensions/python/deformable_conv.py:31 · DeformableConvFunction.forward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/deformable_conv.py#L31)；[up/extensions/python/deformable_conv.py:69 · DeformableConvFunction.backward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/deformable_conv.py#L69)；[up/extensions/python/deformable_conv.py:21 · DeformableConvFunction.symbolic](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/deformable_conv.py#L21)
2. 扩展导出：[up/extensions/csrc/pybind.cpp:26 · up.extensions._C.deform_conv_v1.forward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L26)；[up/extensions/csrc/pybind.cpp:23 · up.extensions._C.deform_conv_v1.backward_input_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L23)；[up/extensions/csrc/pybind.cpp:20 · up.extensions._C.deform_conv_v1.backward_parameters_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L20)
3. C++ 实现：[up/extensions/csrc/deform_conv/deform_conv_cuda.cpp:7 · deform_conv_forward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/deform_conv/deform_conv_cuda.cpp#L7)；[up/extensions/csrc/deform_conv/deform_conv_cuda.cpp:93 · deform_conv_backward_input_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/deform_conv/deform_conv_cuda.cpp#L93)；[up/extensions/csrc/deform_conv/deform_conv_cuda.cpp:179 · deform_conv_backward_parameters_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/deform_conv/deform_conv_cuda.cpp#L179)
4. CUDA 算法：[up/extensions/csrc/deform_conv/deform_conv_cuda_kernel.cu:1 · deform_conv_cuda_kernel.cu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/deform_conv/deform_conv_cuda_kernel.cu#L1)

### naive-nms

**UP Native NMS**

厂商需要交付的方法：

```python
naive_nms(dets, thresh, *, offset)
```

对应原 NVIDIA 的完整入口：

```python
naive_nms(dets, thresh)
```

- **输入与含义：** dets[N,5]=[x1,y1,x2,y2,score]，按调用方约定降序排列；thresh 为 IoU 阈值；offset 为原 ALIGNED_FLAG.offset（0/1）。
- **返回：** 保留索引 Tensor[K]，int64，返回输入 dets 所在设备。
- **反向与状态：** 无梯度要求。
- **必须保留：** 保留 > 阈值判断、坐标 offset 和索引顺序。原实现包含 CPU 位掩码筛选；这不是要求新增 CPU 模型代算。与连续坐标的 TorchVision NMS 路线分开。
- **实现位置：** `up/platform/<vendor>/native_ops.py::NativeOps.naive_nms`。
- **使用范围：** 原扩展 API 保留；当前常用 NMS 路线另有共享 Torch 实现。

NVIDIA 实现链：

1. Python / autograd：[up/extensions/python/nms.py:9 · naive_nms](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/nms.py#L9)
2. 扩展导出：[up/extensions/csrc/pybind.cpp:59 · up.extensions._C.naive_nms.gpu_nms](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L59)
3. C++ 实现：[up/extensions/csrc/nms/nms_cuda.cpp:19 · gpu_nms](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/nms/nms_cuda.cpp#L19)
4. CUDA 算法：[up/extensions/csrc/nms/nms_kernel.cu:1 · nms_kernel.cu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/nms/nms_kernel.cu#L1)

`offset` 是新增交付接口的显式上下文参数，由 UP 维护者从原全局标志传入；厂商代码不自行读取 UP 的全局业务变量。

### sigmoid-focal

**Sigmoid Focal Loss**

厂商需要交付的方法：

```python
sigmoid_focal(preds, targets, weight_pos, gamma, alpha, num_classes, reduction)
```

对应原 NVIDIA 的完整入口：

```python
SigmoidFocalLossFunction.apply(preds, targets, weight_pos, gamma, alpha, num_classes, reduction)
```

- **输入与含义：** preds[N,C] 为 logits；targets[N] 为 int32；背景0、前景1..C、忽略-1；weight_pos 为单元素 normalizer Tensor；gamma、alpha、num_classes=C、reduction。
- **返回：** reduction="none" 返回 [N,C]，其他值返回 [1] 的求和结果；原 Python 包装输出 float32。
- **反向与状态：** 只对 preds 求梯度，乘上传入 grad_output；targets/weight_pos/标量无梯度。
- **必须保留：** 分母 max(weight_pos,1)。已有 up.platform.ops.sigmoid_focal_loss 可复用，但原生公开入口、极值、dtype 和 reduction 需对照验收。
- **实现位置：** `up/platform/<vendor>/native_ops.py::NativeOps.sigmoid_focal`。
- **使用范围：** 检测 Focal 配置可触发；已有部分共享实现。

NVIDIA 实现链：

1. Python / autograd：[up/extensions/python/focal_loss.py:11 · SigmoidFocalLossFunction.forward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/focal_loss.py#L11)；[up/extensions/python/focal_loss.py:53 · SigmoidFocalLossFunction.backward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/focal_loss.py#L53)
2. 扩展导出：[up/extensions/csrc/pybind.cpp:65 · up.extensions._C.focal_loss.sigmoid_forward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L65)；[up/extensions/csrc/pybind.cpp:68 · up.extensions._C.focal_loss.sigmoid_backward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L68)
3. C++ 实现：[up/extensions/csrc/focal_loss/focal_loss_cuda.cpp:5 · focal_loss_sigmoid_forward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/focal_loss/focal_loss_cuda.cpp#L5)；[up/extensions/csrc/focal_loss/focal_loss_cuda.cpp:26 · focal_loss_sigmoid_backward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/focal_loss/focal_loss_cuda.cpp#L26)
4. CUDA 算法：[up/extensions/csrc/focal_loss/focal_loss_sigmoid_kernel.cu:1 · focal_loss_sigmoid_kernel.cu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/focal_loss/focal_loss_sigmoid_kernel.cu#L1)

已有共享参考：`up.platform.ops.sigmoid_focal_loss`。可复用其中适用变体，并补公共入口、设备/dtype 和反向契约测试。

### softmax-focal

**Softmax Focal Loss**

厂商需要交付的方法：

```python
softmax_focal(preds, targets, weight_pos, gamma, alpha, num_classes)
```

对应原 NVIDIA 的完整入口：

```python
SoftmaxFocalLossFunction.apply(preds, targets, weight_pos, gamma, alpha, num_classes)
```

- **输入与含义：** preds[N,C] logits；targets[N] int32，类别0是背景、负值忽略；weight_pos、gamma、alpha、num_classes=C。
- **返回：** float32 Tensor[1]，求和 loss；该接口没有 reduction 参数。
- **反向与状态：** 只对 preds 求梯度；保存 softmax priors、targets、normalizer，原实现还用 [N] 缓冲区。
- **必须保留：** 先做逐行稳定 softmax；背景与前景使用不同 alpha 权重；分母 max(weight_pos,1)。不能用 Sigmoid Focal 替代。
- **实现位置：** `up/platform/<vendor>/native_ops.py::NativeOps.softmax_focal`。
- **使用范围：** 原包装/损失配置保留，选择 Softmax Focal 时接入。

NVIDIA 实现链：

1. Python / autograd：[up/extensions/python/focal_loss.py:82 · SoftmaxFocalLossFunction.forward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/focal_loss.py#L82)；[up/extensions/python/focal_loss.py:124 · SoftmaxFocalLossFunction.backward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/focal_loss.py#L124)
2. 扩展导出：[up/extensions/csrc/pybind.cpp:71 · up.extensions._C.focal_loss.softmax_forward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L71)；[up/extensions/csrc/pybind.cpp:74 · up.extensions._C.focal_loss.softmax_backward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L74)
3. C++ 实现：[up/extensions/csrc/focal_loss/focal_loss_cuda.cpp:47 · focal_loss_softmax_forward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/focal_loss/focal_loss_cuda.cpp#L47)；[up/extensions/csrc/focal_loss/focal_loss_cuda.cpp:70 · focal_loss_softmax_backward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/focal_loss/focal_loss_cuda.cpp#L70)
4. CUDA 算法：[up/extensions/csrc/focal_loss/focal_loss_softmax_kernel.cu:1 · focal_loss_softmax_kernel.cu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/focal_loss/focal_loss_softmax_kernel.cu#L1)

### cross-focal

**Cross Sigmoid Focal Loss**

厂商需要交付的方法：

```python
cross_focal(preds, targets, weight_pos, gamma, alpha, num_classes, gt_class_to_avoid, reduction)
```

对应原 NVIDIA 的完整入口：

```python
CrossSigmoidFocalLossFunction.apply(preds, targets, weight_pos, gamma, alpha, num_classes, gt_class_to_avoid, reduction)
```

- **输入与含义：** preds/targets/weight_pos/gamma/alpha/num_classes 同 Sigmoid Focal；另有 gt_class_to_avoid：int32 打包类别屏蔽表，格式按 kernel 的 neg_map 解码；保留 reduction 参数。
- **返回：** 原实现实际始终返回 float32 Tensor[1]：reduction 分支位于提前 return 之后，当前不可达。
- **反向与状态：** 只对 preds 求梯度；保存 targets、normalizer 和屏蔽表。
- **必须保留：** 照原 neg_map 规则屏蔽对应负类项。此次交付保持原实际输出；如修 reduction 行为，另列语义修复。
- **实现位置：** `up/platform/<vendor>/native_ops.py::NativeOps.cross_focal`。
- **使用范围：** 原包装/损失配置保留，选择 Cross Focal 时接入。

NVIDIA 实现链：

1. Python / autograd：[up/extensions/python/cross_focal_loss.py:12 · CrossSigmoidFocalLossFunction.forward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/cross_focal_loss.py#L12)；[up/extensions/python/cross_focal_loss.py:54 · CrossSigmoidFocalLossFunction.backward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/cross_focal_loss.py#L54)
2. 扩展导出：[up/extensions/csrc/pybind.cpp:80 · up.extensions._C.cross_focal_loss.cross_sigmoid_forward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L80)；[up/extensions/csrc/pybind.cpp:83 · up.extensions._C.cross_focal_loss.cross_sigmoid_backward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L83)
3. C++ 实现：[up/extensions/csrc/cross_focal_loss/cross_focal_loss_cuda.cpp:6 · cross_focal_loss_sigmoid_forward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/cross_focal_loss/cross_focal_loss_cuda.cpp#L6)；[up/extensions/csrc/cross_focal_loss/cross_focal_loss_cuda.cpp:30 · cross_focal_loss_sigmoid_backward_cuda](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/cross_focal_loss/cross_focal_loss_cuda.cpp#L30)
4. CUDA 算法：[up/extensions/csrc/cross_focal_loss/cross_focal_loss_sigmoid_kernel.cu:1 · cross_focal_loss_sigmoid_kernel.cu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/cross_focal_loss/cross_focal_loss_sigmoid_kernel.cu#L1)

### overlap

**二维框 IoU / IoF / IoS**

厂商需要交付的方法：

```python
overlap(b1, b2, mode='IoU')
```

对应原 NVIDIA 的完整入口：

```python
gpu_iou_overlap(b1, b2, mode='IoU')
```

- **输入与含义：** b1[N,>=4]、b2[M,>=4]，只取前4列 x1/y1/x2/y2；mode="IoU" / "IoF" / "IoS"。
- **返回：** 非空输入返回 [N,M]，同 b1 设备/dtype；任一输入为空时原接口返回 [0]，不是 [N,M]。
- **反向与状态：** 原扩展没有注册几何反向；不用给该匹配/评估入口新增梯度。
- **必须保留：** +1 闭区间；IoU 分母为并集，IoF 为第一组面积，IoS 为第二组面积；各分母至少1。共享 box_overlap 目前只覆盖 IoU/IoF，不能把 IoS 传给它冒充实现。
- **实现位置：** `up/platform/<vendor>/native_ops.py::NativeOps.overlap`。
- **使用范围：** 瘦身检测匹配使用 IoU/IoF；已有部分共享实现。

NVIDIA 实现链：

1. Python / autograd：[up/extensions/python/iou_overlap.py:5 · gpu_iou_overlap](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/iou_overlap.py#L5)
2. 扩展导出：[up/extensions/csrc/pybind.cpp:90 · up.extensions._C.overlaps.iou](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L90)
3. C++ 实现：[up/extensions/csrc/iou_overlap/iou_overlap.cpp:5 · gpu_iou_overlaps](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/iou_overlap/iou_overlap.cpp#L5)
4. CUDA 算法：[up/extensions/csrc/iou_overlap/iou_overlap_kernel.cu:1 · iou_overlap_kernel.cu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/iou_overlap/iou_overlap_kernel.cu#L1)

已有共享参考：`up.platform.ops.box_overlap`。可复用其中适用变体，并补公共入口、设备/dtype 和反向契约测试。

### bev-iou

**旋转框 BEV IoU**

厂商需要交付的方法：

```python
bev_iou(boxes_a, boxes_b)
```

对应原 NVIDIA 的完整入口：

```python
boxes_iou_bev(boxes_a, boxes_b)
```

- **输入与含义：** boxes_a[N,7]、boxes_b[M,7]，float32；[x,y,z,dx,dy,dz,heading]，heading 为原旋转约定。
- **返回：** float32 Tensor[N,M]，应分配在输入设备，去掉原硬编码默认 CUDA 卡。
- **反向与状态：** 原 BEV 扩展无注册反向。
- **必须保留：** 保持旋转方向、框中心/尺寸、几何边界与分母规则；空输入先运行 NVIDIA 参考确认。
- **实现位置：** `up/platform/<vendor>/native_ops.py::NativeOps.bev_iou`。
- **使用范围：** 点云模型已从瘦身任务移除，包装保留；恢复点云任务时接入。

NVIDIA 实现链：

1. Python / autograd：[up/extensions/python/iou3d_nms_utils.py:31 · boxes_iou_bev](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/iou3d_nms_utils.py#L31)
2. 扩展导出：[up/extensions/csrc/pybind.cpp:123 · up.extensions._C.iou3d_nms.boxes_iou_bev_gpu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L123)
3. C++ 实现：[up/extensions/csrc/iou3d_nms/iou3d_nms.cpp:70 · boxes_iou_bev_gpu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/iou3d_nms/iou3d_nms.cpp#L70)
4. CUDA 算法：[up/extensions/csrc/iou3d_nms/iou3d_nms_kernel.cu:1 · iou3d_nms_kernel.cu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/iou3d_nms/iou3d_nms_kernel.cu#L1)

### iou3d

**三维框 IoU**

厂商需要交付的方法：

```python
iou3d(boxes_a, boxes_b)
```

对应原 NVIDIA 的完整入口：

```python
boxes_iou3d_gpu(boxes_a, boxes_b)
```

- **输入与含义：** boxes_a[N,7]、boxes_b[M,7]，float32；同 BEV 框格式。
- **返回：** float32 Tensor[N,M]；BEV 交集面积 × 高度交集 / max(体积并集,1e-6)。
- **反向与状态：** 无完整自定义几何反向。原 BEV 缓冲区不求导，但外围高度/体积 Torch 计算可产生部分梯度；保持该图结构，不直接换成具有不同完整梯度的厂商 IoU3D。
- **必须保留：** 保留当前 Python 高度/体积组合，只替换 BEV overlap 内核最稳妥。主要用于 no_grad 匹配/评估；若训练配置需要梯度，单测原图行为。
- **实现位置：** `up/platform/<vendor>/native_ops.py::NativeOps.iou3d`。
- **使用范围：** 点云可选包装，按需接入。

NVIDIA 实现链：

1. Python / autograd：[up/extensions/python/iou3d_nms_utils.py:48 · boxes_iou3d_gpu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/iou3d_nms_utils.py#L48)
2. 扩展导出：[up/extensions/csrc/pybind.cpp:122 · up.extensions._C.iou3d_nms.boxes_overlap_bev_gpu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L122)
3. C++ 实现：[up/extensions/csrc/iou3d_nms/iou3d_nms.cpp:49 · boxes_overlap_bev_gpu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/iou3d_nms/iou3d_nms.cpp#L49)
4. CUDA 算法：[up/extensions/csrc/iou3d_nms/iou3d_nms_kernel.cu:1 · iou3d_nms_kernel.cu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/iou3d_nms/iou3d_nms_kernel.cu#L1)

### nms3d

**旋转 3D NMS**

厂商需要交付的方法：

```python
nms3d(boxes, scores, thresh, pre_maxsize=None, **kwargs)
```

对应原 NVIDIA 的完整入口：

```python
nms_gpu(boxes, scores, thresh, pre_maxsize=None, **kwargs)
```

- **输入与含义：** boxes[N,7] float32、scores[N]、thresh、pre_maxsize=None；**kwargs 原函数不使用。
- **返回：** (indices[K], None)，indices 为输入设备上的 int64；必须映射回原输入索引。
- **反向与状态：** 无梯度要求。
- **必须保留：** 先按 scores 降序，再按 pre_maxsize 截断；保持旋转框重叠和排序规则，输出的 None 不能删。
- **实现位置：** `up/platform/<vendor>/native_ops.py::NativeOps.nms3d`。
- **使用范围：** 点云可选包装，按需接入。

NVIDIA 实现链：

1. Python / autograd：[up/extensions/python/iou3d_nms_utils.py:82 · nms_gpu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/iou3d_nms_utils.py#L82)
2. 扩展导出：[up/extensions/csrc/pybind.cpp:124 · up.extensions._C.iou3d_nms.nms_gpu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L124)
3. C++ 实现：[up/extensions/csrc/iou3d_nms/iou3d_nms.cpp:90 · nms_gpu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/iou3d_nms/iou3d_nms.cpp#L90)
4. CUDA 算法：[up/extensions/csrc/iou3d_nms/iou3d_nms_kernel.cu:1 · iou3d_nms_kernel.cu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/iou3d_nms/iou3d_nms_kernel.cu#L1)

### nms3d-normal

**普通 3D NMS**

厂商需要交付的方法：

```python
nms3d_normal(boxes, scores, thresh, **kwargs)
```

对应原 NVIDIA 的完整入口：

```python
nms_normal_gpu(boxes, scores, thresh, **kwargs)
```

- **输入与含义：** boxes[N,7] float32、scores[N]、thresh；**kwargs 原函数不使用。
- **返回：** (indices[K], None)，indices 为输入设备上的 int64。
- **反向与状态：** 无梯度要求。
- **必须保留：** 按该 kernel 的轴对齐重叠规则实现，不复用旋转 NMS 的角度计算。
- **实现位置：** `up/platform/<vendor>/native_ops.py::NativeOps.nms3d_normal`。
- **使用范围：** 点云可选包装，按需接入。

NVIDIA 实现链：

1. Python / autograd：[up/extensions/python/iou3d_nms_utils.py:100 · nms_normal_gpu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/iou3d_nms_utils.py#L100)
2. 扩展导出：[up/extensions/csrc/pybind.cpp:125 · up.extensions._C.iou3d_nms.nms_normal_gpu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L125)
3. C++ 实现：[up/extensions/csrc/iou3d_nms/iou3d_nms.cpp:139 · nms_normal_gpu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/iou3d_nms/iou3d_nms.cpp#L139)
4. CUDA 算法：[up/extensions/csrc/iou3d_nms/iou3d_nms_kernel.cu:1 · iou3d_nms_kernel.cu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/iou3d_nms/iou3d_nms_kernel.cu#L1)

### points-in-boxes

**点属于哪个三维框**

厂商需要交付的方法：

```python
points_in_boxes(points, boxes)
```

对应原 NVIDIA 的完整入口：

```python
points_in_boxes_gpu(points, boxes)
```

- **输入与含义：** points[B,M,3]、boxes[B,T,7]，float32、contiguous，batch 数一致。
- **返回：** int32 Tensor[B,M]，同设备；未命中为 -1。
- **反向与状态：** 无梯度要求。
- **必须保留：** 保持边界判断、旋转坐标转换、多个框命中时原 kernel 的选择顺序。
- **实现位置：** `up/platform/<vendor>/native_ops.py::NativeOps.points_in_boxes`。
- **使用范围：** 点云可选包装，按需接入。

NVIDIA 实现链：

1. Python / autograd：[up/extensions/python/roiaware_pool3d_utils.py:34 · points_in_boxes_gpu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/roiaware_pool3d_utils.py#L34)
2. 扩展导出：[up/extensions/csrc/pybind.cpp:111 · up.extensions._C.roiaware_pool3d.points_in_boxes_gpu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L111)
3. C++ 实现：[up/extensions/csrc/roiaware_pool3d/roiaware_pool3d.cpp:98 · points_in_boxes_gpu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/roiaware_pool3d/roiaware_pool3d.cpp#L98)
4. CUDA 算法：[up/extensions/csrc/roiaware_pool3d/roiaware_pool3d_kernel.cu:1 · roiaware_pool3d_kernel.cu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/roiaware_pool3d/roiaware_pool3d_kernel.cu#L1)

### roiaware

**RoIAwarePool3d 体素池化**

厂商需要交付的方法：

```python
roiaware(rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method)
```

对应原 NVIDIA 的完整入口：

```python
RoIAwarePool3dFunction.apply(rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method)
```

- **输入与含义：** rois[N,7]、pts[P,3]、pts_feature[P,C] float32；out_size 为 int 或 (x,y,z)；max_pts_each_voxel；pool_method="max" / "avg"。
- **返回：** float32 Tensor[N,out_x,out_y,out_z,C]，同设备。
- **反向与状态：** 只对 pts_feature 返回 [P,C] 梯度；保存 int32 点索引、argmax、pool_method 和输入尺寸。
- **必须保留：** 保留体素坐标、每体素最大点数、max/avg 的空体素与截断规则。
- **实现位置：** `up/platform/<vendor>/native_ops.py::NativeOps.roiaware`。
- **使用范围：** 点云可选包装，按需接入。

NVIDIA 实现链：

1. Python / autograd：[up/extensions/python/roiaware_pool3d_utils.py:63 · RoIAwarePool3dFunction.forward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/roiaware_pool3d_utils.py#L63)；[up/extensions/python/roiaware_pool3d_utils.py:102 · RoIAwarePool3dFunction.backward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/roiaware_pool3d_utils.py#L102)
2. 扩展导出：[up/extensions/csrc/pybind.cpp:109 · up.extensions._C.roiaware_pool3d.forward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L109)；[up/extensions/csrc/pybind.cpp:110 · up.extensions._C.roiaware_pool3d.backward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L110)
3. C++ 实现：[up/extensions/csrc/roiaware_pool3d/roiaware_pool3d.cpp:29 · roiaware_pool3d_gpu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/roiaware_pool3d/roiaware_pool3d.cpp#L29)；[up/extensions/csrc/roiaware_pool3d/roiaware_pool3d.cpp:68 · roiaware_pool3d_gpu_backward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/roiaware_pool3d/roiaware_pool3d.cpp#L68)
4. CUDA 算法：[up/extensions/csrc/roiaware_pool3d/roiaware_pool3d_kernel.cu:1 · roiaware_pool3d_kernel.cu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/roiaware_pool3d/roiaware_pool3d_kernel.cu#L1)

### roipoint

**RoIPointPool3d 区域点采样**

厂商需要交付的方法：

```python
roipoint(points, point_features, boxes3d, pool_extra_width, num_sampled_points=512)
```

对应原 NVIDIA 的完整入口：

```python
RoIPointPool3dFunction.apply(points, point_features, boxes3d, pool_extra_width, num_sampled_points=512)
```

- **输入与含义：** points[B,N,3]、point_features[B,N,C]、boxes3d[B,M,7] float32；pool_extra_width；num_sampled_points=512。
- **返回：** (pooled[B,M,S,3+C], empty_flag[B,M])；S 为采样数；pooled float32，empty_flag int32，同设备。
- **反向与状态：** 原版 backward 未实现；不要宣称该操作支持训练反向，需要反向时明确报不支持。
- **必须保留：** 先按原 enlarge_box3d 扩框；保留点顺序、不足采样数时的重复策略和空框标志。
- **实现位置：** `up/platform/<vendor>/native_ops.py::NativeOps.roipoint`。
- **使用范围：** 点云可选包装，按需接入。

NVIDIA 实现链：

1. Python / autograd：[up/extensions/python/roipoint_pool3d_utils.py:32 · RoIPointPool3dFunction.forward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/roipoint_pool3d_utils.py#L32)；[up/extensions/python/roipoint_pool3d_utils.py:61 · RoIPointPool3dFunction.backward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/roipoint_pool3d_utils.py#L61)
2. 扩展导出：[up/extensions/csrc/pybind.cpp:117 · up.extensions._C.roipoint_pool3d.forward](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/pybind.cpp#L117)
3. C++ 实现：[up/extensions/csrc/roipoint_pool3d/roipoint_pool3d.cpp:23 · roipool3d_gpu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/roipoint_pool3d/roipoint_pool3d.cpp#L23)
4. CUDA 算法：[up/extensions/csrc/roipoint_pool3d/roipoint_pool3d_kernel.cu:1 · roipoint_pool3d_kernel.cu](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/csrc/roipoint_pool3d/roipoint_pool3d_kernel.cu#L1)

## C. 已有共享路线，不要求从零写

- `box_overlap(b1, b2, mode="IoU")`：[up/platform/ops.py:7 · box_overlap](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/platform/ops.py#L7)
- `nms(boxes, scores, threshold)`：[up/platform/ops.py:19 · nms](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/platform/ops.py#L19)
- `sigmoid_focal_loss(input, target, normalizer, gamma, alpha, num_channels, reduction)`：[up/platform/ops.py:42 · sigmoid_focal_loss](https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/platform/ops.py#L42)

这里的 nms 是业务中的 TorchVision 连续坐标路线；UP Native NMS 另有 offset 参数，不能直接拿前者替换后者。IoU 共享函数目前不覆盖 IoS。

## D. 验收与交付

每个方法交付：实现文件 + 实际依赖版本 + 固定输入/权重/上游梯度 + NVIDIA 配对结果 + 本卡重复结果。比较输出、所需梯度、dtype、shape、索引、状态和恢复；浮点保持 `rtol=1e-4, atol=1e-5`，整数/标志精确比较。

空输入、并列分数、非连续布局和极值按对应原 NVIDIA 入口先建立参考；原版也报错/未定义的变体单列，不自行换一种输出后标为兼容。RoIAlign MAX、Cross Focal reduction、IoU3D 部分梯度和 RoIPoint 反向的原行为已在各条目写明。

NVIDIA/C500/MLU/NPU 支持声明按函数实测记录；模板和导入成功只表示接口已定义。Softer NMS 在 CPU 执行，spring/linklink 属历史路线，都不列入这16项 GPU 内核任务。

可选 GPU 内核放厂商 csrc，构建只针对该厂商；保留原 `up/extensions/csrc`。现有 setup.py 仅为 NVIDIA 构建 _C，厂商新增内核须提供自己的构建/安装命令。ONNX symbolic 的原节点和属性保持；不支持的导出单独报错。

最后跑训练 → 训练内 eval → checkpoint 恢复 → 独立推理，以及双卡 DDP/SyncBN。未使用的点云操作可暂缓，状态写清楚；不改模型算法和训练配方。
