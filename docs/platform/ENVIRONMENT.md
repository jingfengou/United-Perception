# 环境与依赖

运行 UP 需要同时匹配显卡驱动、厂商 PyTorch/SDK、UP 依赖和所用模型的可选工具链。NVIDIA 与沐曦的 PyTorch 分开安装，不要覆盖到同一个 `site-packages`。

使用通用镜像时按 [完整示例](QUICKSTART_UNIFIED.md) 操作；需要原生 Conda/venv 或补充可选依赖时参考本页。

## 参考软件组合

| 组件 | NVIDIA | 沐曦 C500 |
| --- | --- | --- |
| Python | 3.12 | 3.12 |
| PyTorch | 2.10.0+cu126 | 2.10.0+metax3.8.1.0 |
| TorchVision | 0.25.0+cu126 | 0.25.0+metax3.8.1.0 |
| 扩展工具链 | 与 PyTorch 匹配的 CUDA；通用镜像为 12.6 | 厂商 MACA 3.8.1 系列 |
| 稀疏卷积 | spconv-cu126 2.3.8、cumm-cu126 0.7.11 | 厂商 spconv 2.1.0+metax3.8.1.0torch2.10 |

这是项目使用的软件组合；更换 Python、Torch 或 SDK 后，应重新编译扩展并执行所需模型和算子的测试。宿主机内核驱动由厂商安装流程提供。

## 原生环境安装

1. 为每家建立独立容器、Conda 或 venv 环境，先安装厂商 PyTorch/TorchVision。
2. 保留厂商包，安装 UP 的普通依赖；原版 `requirements.txt` 含历史版本，不宜直接覆盖 Python 3.12 环境。
3. 按所用模型安装 spconv、MQBench、MSBench 等可选依赖。
4. 安装 UP，再使用当前厂商 SDK 显式编译扩展。

```bash
python -c "import torch; print(torch.__version__)"
python -m pip install -r docs/platform/requirements-common.txt
python -m pip install --no-deps -e .

# NVIDIA 环境
UP_BUILD_BACKEND=nvidia python setup.py build_ext --inplace

# 沐曦环境，替代上面一条构建命令
UP_BUILD_BACKEND=metax python setup.py build_ext --inplace
```

[requirements-common.txt](requirements-common.txt) 只固定普通依赖子集，不包含厂商 Torch/SDK 及所有模型的可选包。沐曦不能安装 NVIDIA 的 Torch 或 spconv wheel。

扩展分别生成 `_nvidia_ops`、`_metax_ops`，不能跨厂商混用。更新内核源码后必须重新编译；仅更新 Python 文件不能替代扩展升级。

## 选择后端

原生命令继续使用 UP 参数，例如：

```bash
export DEFAULT_TASKS=cls:det
python -m up train --config model.yaml --launch pytorch --ng 1 \
  --opts runtime.platform.backend metax

python -m up train --evaluate --config model.yaml --launch pytorch --ng 1 \
  --opts runtime.platform.backend metax

python -m up inference --config model.yaml --img_path images/ --ckpt model.pth \
  --launch pytorch --ng 1 --opts runtime.platform.backend metax
```

NVIDIA 将 `metax` 换成 `nvidia`。`--backend dist` 使用 PyTorch 通信，训练策略可选 DDP 或 FSDP2，与显卡厂商分别配置。

`spring/linklink` 仅保留原版兼容入口，不属于当前平台接入、依赖检查和验收范围。

`DEFAULT_TASKS` 选择 UP 任务模块；分类/二维检测使用 `cls:det`，点云可使用 `det_3d` 并补齐其依赖。未设置时 UP 会导入全部任务，可能要求不使用的可选包。

## 可选工具链与 SDK

MQBench、MSBench 可按以下固定源码准备。依赖目录放在项目之外：

```bash
git clone https://github.com/ModelTC/MQBench.git MQBench
git -C MQBench checkout 00f477d58ef67c1c1bbe71d14312731fd3a4960d
git clone https://github.com/ModelTC/msbench.git msbench
git -C msbench checkout b00632036c8a03cacff6ae94814195e98f1361f8
python /path/to/UP/tools/prepare_platform_dependencies.py --root "$PWD"
export PYTHONPATH="/path/to/UP:$PWD/MQBench:$PWD/msbench:$PYTHONPATH"
```

兼容脚本处理对应版本的导入、导出入口和已有类 re-export，不更换量化或剪枝策略。使用其他版本时应重新核对兼容性。

- **Amba 剪枝**：所需 MSBench `FakeSparse` 类缺失，相关配方尚不能运行。
- **其他工具链**：点云读取、NART 和外部部署引擎按各自模型配置安装，ONNX 导出不等于部署引擎已支持。

厂商 wheel、私有 SDK 和第三方源码不随本仓库分发。

## 随机性与数值设置

默认沿用原生数值策略。需要确定性时，在训练脚本中显式调用 `torch.use_deterministic_algorithms(True)`，并在进程启动前设置对应厂商环境变量。

`up_docker` 会传入用户显式设置的 `PYTHONHASHSEED`、`CUBLAS_WORKSPACE_CONFIG`；沐曦还会传入 `MCDNN_USE_DETERMINISTIC_ALGO`、`MCDNN_STABLE_RESULT_MOD`。这些变量的可用范围应按厂商 SDK 版本确认。

同平台或跨厂商对照须使用相同输入、权重、种子、配置和数值开关；厂商 RNG 和浮点实现仍需在目标模型上检查。
