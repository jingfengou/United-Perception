"""Shared data-view adapters for full ImageNet/COCO parity, outside UP models."""


COCO_CATEGORY_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
    39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90)


def full_coco_index(coco, expected_images):
    """Include empty/crowd-only images; never infer image membership from anns."""
    if sorted(coco.cats) != list(COCO_CATEGORY_IDS):
        raise ValueError('Full COCO requires the original 80 category IDs')
    ids = sorted(coco.imgs)
    if len(ids) != expected_images:
        raise ValueError('Unexpected full COCO image count: '+str(len(ids)))
    ratios = [float(coco.imgs[key]['height'])/coco.imgs[key]['width'] for key in ids]
    return ids, ratios


def register_full_coco():
    from up.tasks.det.data.datasets.coco_dataset import CocoDataset
    from up.utils.general.registry_factory import DATASET_REGISTRY

    @DATASET_REGISTRY.register('full_accuracy_coco')
    class FullCocoDataset(CocoDataset):
        def __init__(self, expected_images, **kwargs):
            super().__init__(**kwargs)
            self.img_ids, self.aspect_ratios = full_coco_index(self.coco, expected_images)

