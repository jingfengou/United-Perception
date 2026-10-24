"""Full-dataset membership and official category convention must not drift."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


spec = importlib.util.spec_from_file_location('full_accuracy_data',
    Path(__file__).resolve().parents[2] / 'tools/full_accuracy_data.py')
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


class TestFullCocoData(unittest.TestCase):
    def test_unannotated_images_remain_in_the_split(self):
        coco = SimpleNamespace(cats={key: {} for key in data.COCO_CATEGORY_IDS},
            imgs={9: {'height': 20, 'width': 10}, 2: {'height': 10, 'width': 20}},
            anns={1: {'image_id': 9}})
        ids, ratios = data.full_coco_index(coco, expected_images=2)
        self.assertEqual(ids, [2, 9])
        self.assertEqual(ratios, [.5, 2.])
        with self.assertRaisesRegex(ValueError, 'count'):
            data.full_coco_index(coco, expected_images=3)

    def test_categories_are_original_discrete_ids_not_filtered_hf_indices(self):
        coco = SimpleNamespace(cats={key: {} for key in range(80)}, imgs={})
        with self.assertRaisesRegex(ValueError, 'category IDs'):
            data.full_coco_index(coco, expected_images=0)
        self.assertEqual(len(set(data.COCO_CATEGORY_IDS)), 80)
        self.assertEqual(data.COCO_CATEGORY_IDS[-1], 90)
