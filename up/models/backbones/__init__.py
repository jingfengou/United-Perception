# 公共 backbone 注册保留 ResNet/ViT；YOLOX 的专属 backbone 由对应插件注册。
from .resnet import (resnet101,  # noqa
                     resnet152,
                     resnet18,
                     resnet34,
                     resnet50,
                     resnet_custom)


from .vision_transformer import (  # noqa
    vit_base_patch32_224, vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch14_224,
    deit_tiny_patch16_224, deit_small_patch16_224, deit_base_patch16_224
)





from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY

imported_vars = list(globals().items())

for var_name, var in imported_vars:
    if callable(var):
        MODULE_ZOO_REGISTRY.register(var_name, var)
