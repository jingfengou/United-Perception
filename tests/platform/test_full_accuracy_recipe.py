"""Resource edits must not replace the native models or omit full data."""
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('full_accuracy_recipe', ROOT/'tools/full_accuracy_recipe.py')
recipes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipes)


class TestFullRecipes(unittest.TestCase):
    def data(self):
        return {'imagenet': {'train': {'meta': 'train.txt'}, 'test': {'meta': 'val.txt'}, 'image_dir': 'images'},
            'coco': {split: {'meta': split+'.json', 'image_dir': split,
                            'images': 118287 if split == 'train' else 5000} for split in ('train', 'test')},
            'detr': {'source': 'official-pinned'}, 'yolov5_anchors': {'path': 'anchors.json'}}

    def test_complete_budgets_and_full_class_counts(self):
        for model, expected in (('resnet50', 200), ('vit_b16', 300), ('yolov5s', 300), ('detr', 300)):
            with self.subTest(model=model):
                config = recipes.recipe(ROOT, self.data(), model, batch_size=64, world_size=1)
                self.assertEqual(config['trainer']['max_epoch'], expected)
                self.assertEqual(config['num_classes'], 1000 if model in ('resnet50', 'vit_b16') else 81)
                if model in ('yolov5s', 'detr'):
                    self.assertEqual(config['dataset']['test']['dataset']['kwargs']['expected_images'], 5000)
                    self.assertEqual(config['dataset']['train']['dataset']['kwargs']['expected_images'], 118287)
                self.assertFalse(config['saver']['auto_resume'])
                if model == 'detr':
                    self.assertEqual(config['dataset']['batch_sampler']['type'], 'aspect_ratio_group')

    def test_native_vit_size_and_stochastic_layers_are_retained(self):
        config = recipes.recipe(ROOT, self.data(), 'vit_b16', batch_size=64, world_size=8)
        self.assertEqual(config['trainer']['optimizer']['kwargs']['lr'], .00075)
        self.assertEqual(config['net'][0]['kwargs']['drop_path'], .1)
        self.assertEqual(config['dataset']['train']['dataset']['kwargs']['transformer'][0]['kwargs']['size'], 224)

    def test_native_yolo_bn_constraint_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, 'BS>=64'):
            recipes.recipe(ROOT, self.data(), 'yolov5s', batch_size=8, world_size=1)
        config = recipes.recipe(ROOT, self.data(), 'yolov5s', batch_size=8, world_size=8)
        self.assertFalse(config['runtime']['fp16'])
        self.assertEqual(config['dataset']['batch_sampler']['kwargs']['batch_size'], 8)

