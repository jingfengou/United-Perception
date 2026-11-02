from .resnet import (resnet18, resnet34, resnet50, resnet101, resnet152, resnet_custom)  # noqa: F401
from .vision_transformer import (  # noqa: F401
    vit_base_patch32_224, vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch14_224,
)
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY

for name, factory in list(globals().items()):
    if callable(factory):
        MODULE_ZOO_REGISTRY.register(name, factory)
