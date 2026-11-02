## United Perception

<img src=./up-logo.png width=50% />


This branch retains YOLOX, ResNet and ViT, and adds DETR. Public commands are
`train` and `inference`; execution uses PyTorch, with DDP selected automatically
when the requested world size exceeds one. Original shell scripts and their
environment/Slurm examples are retained unchanged.

Configurations: `configs/det/yolox/`, `configs/cls/resnet/`,
`configs/cls/vision_transformer/`, and `configs/det/detr/detr_r50.yaml`.
DETR is adapted from [facebookresearch/detr at 29901c5](https://github.com/facebookresearch/detr/tree/29901c51d7fe8712168b8d0d64351170bc0f83e0),
with its [license](up/tasks/det/plugins/detr/official/LICENSE).

## 文档

[up 官方文档](https://modeltc-up.readthedocs.io/en/latest/index.html)
