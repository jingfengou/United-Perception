import copy
import importlib
import math
from pathlib import Path

import pytest


@pytest.fixture
def recipes(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'tools'))
    return importlib.import_module('accuracy_recipe')


@pytest.fixture
def data():
    result = {name: {split: {'meta': '/fixed/'+name+'/'+split, 'image_dir': '/fixed/'+name+'/images'}
                    for split in ('train', 'test')} for name in ('cifar10', 'pennfudan')}
    result.update(yolov5_anchors={'path': '/fixed/anchors.json'}, resnet50_weights={'path': '/fixed/resnet50.pt'},
        detr={'source': '/fixed/detr', 'checkpoint': '/fixed/detr.pt', 'source_sha256': 'source',
              'checkpoint_sha256': 'weights', 'revision': '29901c51d7fe8712168b8d0d64351170bc0f83e0'})
    return result


@pytest.mark.parametrize('model', ['resnet18', 'yolov5s', 'vit_b16', 'faster_rcnn', 'detr'])
def test_budget_matches_full_sampler_epochs(recipes, data, model):
    before = copy.deepcopy(data)
    root = Path(__file__).resolve().parents[2]
    config = recipes.recipe(root, data, model)
    dataset = config['dataset']
    train = dataset['train']
    batch = train.get('batch_sampler', dataset.get('batch_sampler'))['kwargs']['batch_size']
    images = 10000 if model in ('resnet18', 'vit_b16') else 120
    assert math.ceil(images/batch)*config['trainer']['max_epoch'] == recipes.BUDGETS[model]['steps']
    assert config['runtime']['random_seed'] == 131
    assert data == before
    if model == 'detr':
        assert config['trainer']['lr_scheduler']['warmup_iter'] == 0
        assert config['trainer']['optimizer']['special_param_group'][0]['lr'] == 1e-5
    elif model == 'faster_rcnn':
        assert config['trainer']['lr_scheduler']['warmup_iter'] == 60
