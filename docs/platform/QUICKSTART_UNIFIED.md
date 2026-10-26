# 一个镜像，选择 NVIDIA 或沐曦运行 UP

下面用 **YOLOv5-S + COCO 2017** 串起构建、训练、验证和图片推理。命令在对应服务器的 **Linux Bash** 中执行。
两家使用同一镜像和模型配置，`./up_docker nvidia ...` 或 `./up_docker metax ...` 选择厂商。

## 1. 构建通用镜像

宿主机需要 Docker 和对应显卡驱动。NVIDIA 还需配置 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)；沐曦需能访问 `/dev/dri`、`/dev/mxcd`，参见 [C500 容器说明](https://developer.metax-tech.com/api/client/document/preview/462/C500_UserGuide_CN.html)。
当前账号应有运行 Docker 的权限；需要 sudo 时使用 `sudo -E ./up_docker ...`，保留下文已导出的目录变量。

```bash
git clone --branch codex/up-docker-launcher \
  https://github.com/jingfengou/United-Perception.git
cd United-Perception

docker build --progress=plain \
  -f docker/unified/Dockerfile \
  -t up-unified:local .
```

构建分别安装两家 PyTorch、配置 SDK 并编译专用扩展，不需要 GPU。需要能访问厂商镜像、软件包和系统包源。
默认 NVIDIA 扩展覆盖 SM80/SM89，即 A100/A800、RTX 4090；其他架构通过 `--build-arg NVIDIA_ARCH_LIST=...` 指定。

构建一次即可分发给两台服务器：

```bash
# 构建机器导出；把 tar 文件传到另一台服务器
docker save -o up-unified.tar up-unified:local

# 另一台服务器导入
docker load -i up-unified.tar
```

每台机器仍需本厂商驱动。镜像不包含宿主机内核驱动，也不负责跨服务器调度。
已有 `up-unified:local` 时，直接用新脚本，不必为宿主机启动脚本重新构建镜像。

## 2. 准备数据和工作目录

先准备 COCO 2017，保持官方图片和标注布局：

```text
/datasets/coco2017/
  train2017/
  val2017/
  annotations/
    instances_train2017.json
    instances_val2017.json
```

在仓库根目录执行，只需把 `UP_DATA` 改为本机数据目录：

```bash
export UP_DATA="/datasets/coco2017"
export UP_WORK="$HOME/up-demo"
export UP_GPU=0

mkdir -p "$UP_WORK/images" "$UP_WORK/outputs"

# 只在配置副本中替换数据路径；模型结构、损失和优化器保持原样
sed 's|coco/|/data/|g' \
  configs/det/yolov5/yolov5_small_silu.yaml > "$UP_WORK/model.yaml"
cp configs/det/yolov5/anchors.json "$UP_WORK/anchors.json"
cp "$UP_DATA/val2017/000000000139.jpg" "$UP_WORK/images/"
```

| 宿主机 | 容器内 | 用途 |
| --- | --- | --- |
| `$UP_WORK` | `/work`，读写 | 配置、anchors、待推理图片、权重和结果 |
| `$UP_DATA` | `/data`，只读 | 训练和验证数据 |

图片、标注、权重不打进镜像。后续在仓库根目录调用 `./up_docker`，工作目录由 `UP_WORK` 指定。
两台服务器各自设置本机目录，使用同一份配置副本；不要指望一台机器上的路径自动出现在另一台上。

## 3. 检查环境

在 NVIDIA 服务器执行：

```bash
./up_docker nvidia check
```

在沐曦服务器执行：

```bash
./up_docker metax check
```

`check` 检查所选 PyTorch、GPU 矩阵前后向、专用扩展和动态库。`check --no-device` 只检查包。
查看当前环境的 PyTorch 版本也可以直接运行：

```bash
./up_docker metax python -c 'import torch; print(torch.__version__)'
```

## 4. 一条命令训练

先执行 1 个 epoch 确认流程。正式训练时删除 `trainer.max_epoch`、`trainer.test_freq`、`trainer.save_freq` 这三项覆盖，即沿用原配方。
这个例子不压缩学习率或 warmup 时间轴，也不是新的单轮精度配方。

NVIDIA：

```bash
./up_docker nvidia train \
  --config model.yaml --launch pytorch --ng 1 \
  --opts \
  trainer.max_epoch 1 trainer.test_freq 1 trainer.save_freq 1 \
  saver.save_dir outputs/nvidia/checkpoints \
  saver.results_dir outputs/nvidia/eval
```

沐曦：

```bash
./up_docker metax train \
  --config model.yaml --launch pytorch --ng 1 \
  --opts \
  trainer.max_epoch 1 trainer.test_freq 1 trainer.save_freq 1 \
  saver.save_dir outputs/metax/checkpoints \
  saver.results_dir outputs/metax/eval
```

只在对应厂商服务器运行对应命令。权重分别保存为：

```text
$UP_WORK/outputs/nvidia/checkpoints/ckpt_latest.pth
$UP_WORK/outputs/metax/checkpoints/ckpt_latest.pth
```

原配方开启自动恢复；重复执行会查找该厂商输出目录的检查点。新实验换一个输出目录，避免接着旧实验训练。
`ckpt_latest.pth` 可能是指向容器内 `/work/...` 的链接，下面的容器命令可直接读取。要在宿主机把权重复制到另一台机器，请复制对应的实际 `ckpt_e*.pth` 文件，而不是这个链接。
显存不足时按实际训练目标配置 batch 和卡数；两家对照实验须保持这些设置相同。

## 5. 验证训练权重

先选择当前服务器的厂商，然后运行同一条命令：

```bash
export VENDOR=metax  # NVIDIA 服务器改为 nvidia

./up_docker "$VENDOR" eval \
  --config model.yaml --launch pytorch --ng 1 \
  --opts \
  saver.auto_resume False \
  saver.resume_model "outputs/$VENDOR/checkpoints/ckpt_latest.pth" \
  saver.save_dir "outputs/$VENDOR/checkpoints" \
  saver.results_dir "outputs/$VENDOR/eval"
```

`eval` 是原有 `train --evaluate` 的简写：使用训练平台适配，在 COCO 验证集评估，不执行训练更新。
训练循环中的定期验证同样使用训练路径。

## 6. 独立图片推理

NVIDIA：

```bash
./up_docker nvidia inference \
  --config model.yaml \
  --img_path images/ \
  --ckpt outputs/nvidia/checkpoints/ckpt_latest.pth \
  --vis_dir outputs/nvidia/predictions \
  --launch pytorch --ng 1
```

沐曦：

```bash
./up_docker metax inference \
  --config model.yaml \
  --img_path images/ \
  --ckpt outputs/metax/checkpoints/ckpt_latest.pth \
  --vis_dir outputs/metax/predictions \
  --launch pytorch --ng 1
```

使用独立推理平台适配，可视化结果保存在 `$UP_WORK/outputs/<厂商>/predictions/`。
容器退出后工作目录里的文件保留。比较同权重跨卡推理时，应把同一个检查点复制到两台机器，并让两条命令指向这份权重。

## 可调整的启动参数

`./up_docker --help` 显示完整用法。其余参数直接传给 UP，`--opts` 仍放在 UP 参数最后。

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `UP_WORK` | 当前目录 | 已存在的工作目录，挂载到 `/work` |
| `UP_DATA` | 不挂载 | 可选数据目录，只读挂载到 `/data` |
| `UP_GPU` | `0` | 宿主机卡号；多卡如 `0,1`，同时显式设置 `--ng 2` |
| `UP_IMAGE` | `up-unified:local` | 使用的通用镜像标签 |
| `DEFAULT_TASKS` | `cls:det` | 分类和二维检测；其他任务需选择模块并安装其可选依赖 |

多卡启动示例：

```bash
UP_GPU=0,1 ./up_docker nvidia train \
  --config model.yaml --launch pytorch --ng 2 \
  --opts saver.save_dir outputs/nvidia-2gpu/checkpoints \
  saver.results_dir outputs/nvidia-2gpu/eval
```

脚本只负责 Docker 设备映射、目录和参数；镜像内 `up-run` 选择隔离的 Python、SDK、扩展及 `runtime.platform.backend`。
它不会自动开启 AMP、确定性或 CPU 代算。用户显式设置的 `PYTHONHASHSEED`、`CUBLAS_WORKSPACE_CONFIG` 会传入容器；沐曦还会传入显式设置的 `MCDNN_USE_DETERMINISTIC_ALGO`、`MCDNN_STABLE_RESULT_MOD`。

镜像结构、环境验证范围和数值记录见 [通用环境说明](UNIFIED_ENVIRONMENTS.md)。
新脚本的测试与实际启动记录见 [启动器验证记录](validation/20261026/docker-launcher/summary.json)。

## 本次验证范围

| 检查 | 结果 |
| --- | --- |
| NVIDIA、沐曦服务器上的 Linux 启动器测试 | 两台各 15 项通过，覆盖参数、挂载、厂商选择、环境隔离、验证入口及错误返回 |
| C500 实际 Docker 启动 | GPU 检查、ResNet18 两步训练、训练内验证、单独 `eval`、权重恢复及图片推理通过；验证和推理未改写检查点 |
| NVIDIA Docker GPU 启动 | 4090 尚未安装 Docker，本轮只验证启动参数和缺依赖报错；此前目录模式 GPU 记录见通用环境说明 |

C500 使用 4 张固定生成图片做命令流程检查；本页的完整 COCO 训练未在此次启动脚本验证中重跑，也没有恢复旧多随机种子任务。
软件测试使用 `python3 -m unittest discover -s tests/platform -p 'test_*launcher.py' -v`，不需要 GPU。
