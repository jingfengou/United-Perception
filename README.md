# United Perception

<img src="./up-logo.png" width="50%" alt="United Perception" />

A focused version of [ModelTC/United-Perception](https://github.com/ModelTC/United-Perception)
for YOLOX, ResNet, ViT and DETR. Training uses PyTorch DistributedDataParallel (DDP).
The original UP model interfaces, data pipeline, Runner and native C++/CUDA operators are retained.

| Family | Example configuration | Task |
| --- | --- | --- |
| YOLOX | `configs/det/yolox/yolox_tiny.yaml` | Object detection |
| ResNet | `configs/cls/resnet/res18.yaml` | Image classification |
| ViT | `configs/cls/vision_transformer/vit_b16_224.yaml` | Image classification |
| DETR | `configs/det/detr/detr_r50.yaml` | Object detection |

DETR is integrated from [facebookresearch/detr at 29901c5](https://github.com/facebookresearch/detr/tree/29901c51d7fe8712168b8d0d64351170bc0f83e0).
Its detector, Transformer, Hungarian matching and detection losses are included with the original license.
This configuration is an integration example, not a reproduced COCO accuracy result.
YOLOX's shared YOLOv5 building blocks and augmentation helpers remain as dependencies.

## Installation

Use Linux with an NVIDIA GPU, a CUDA toolkit and a compatible C++ compiler.
Install PyTorch and torchvision for that environment first. The retained legacy dependencies
are used with Python 3.7, PyTorch 1.13.1 and torchvision 0.14.1.
Build environments are installed separately for each target machine.

```bash
git clone --branch codex/up-four-models-ddp-20261029 https://github.com/jingfengou/United-Perception.git
cd United-Perception
# Select the CUDA toolkit/compiler in your environment before building.
bash easy_setup.sh
python -m up --help
```

Native sources live in `up_platform/ops/_native/csrc` and build into
`up_platform.ops._native._C`. The existing Python operator API remains under `up.extensions`.
PyTorch's standard layers and tensor operations are called directly.

## Training and evaluation

Set the dataset paths and batch sizes in the selected YAML configuration.
Classification uses image lists containing `relative/image/path label`, with zero-based labels.
Detection uses COCO-format annotations. UP detection class counts include background;
DETR translates foreground labels to its zero-based labels internally.
DETR inputs use RGB, ImageNet normalization and top-left batch padding; keep its resize
transform unpadded so the image dimensions also define the attention mask.

```bash
# One GPU; outputs go to the current working directory unless configured otherwise.
bash scripts/train.sh 1 configs/cls/resnet/res18.yaml

# Two GPUs, with one DDP process per GPU.
bash scripts/train.sh 2 configs/det/yolox/yolox_tiny.yaml

# torchrun also supplies rank and rendezvous settings, including multi-node jobs.
torchrun --standalone --nproc_per_node=2 -m up train --config configs/det/detr/detr_r50.yaml

# Load a saved checkpoint and evaluate the configured validation dataset.
bash scripts/test.sh 1 configs/cls/resnet/res18.yaml --opts saver.resume_model /path/to/checkpoint.pth
```

Use `CUDA_VISIBLE_DEVICES` to choose physical GPUs. Dataset, checkpoint, log and result directories
should be outside the source checkout. Set `runtime.runner.kwargs.work_dir` and the YAML `saver`
paths accordingly, or run the Python command from an output directory with the source on `PYTHONPATH`.

## Image inference

```bash
bash scripts/inference.sh configs/det/yolox/yolox_tiny.yaml /path/to/image.jpg /path/to/checkpoint.pth --vis_dir /path/to/output
```

Inference loads the selected model and checkpoint and saves visualizations. The same command
supports the classification configurations. Dataset paths, class count and preprocessing must
match the checkpoint used for inference.

## License

UP is distributed under the [Apache 2.0 license](LICENSE).
The included DETR files retain their [original license](up/tasks/det/plugins/detr/official/LICENSE).
