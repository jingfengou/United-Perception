"""Frozen small-data recipes reused from the 2026-10-21 accuracy budget.

Validation recipes only: no production model or training algorithm changes.
"""
import copy
from pathlib import Path


BUDGETS = {
    'resnet18': {'epochs': 20, 'steps': 3140, 'metric': 'top1', 'scale': 1.0},
    'yolov5s': {'epochs': 300, 'steps': 600, 'metric': 'bbox.AP', 'scale': 100.0},
    'vit_b16': {'epochs': 20, 'steps': 3140, 'metric': 'top1', 'scale': 1.0},
    'faster_rcnn': {'epochs': 12, 'steps': 720, 'metric': 'bbox.AP', 'scale': 100.0},
    'detr': {'epochs': 12, 'steps': 720, 'metric': 'bbox.AP', 'scale': 100.0},
}
CONFIGS = {
    'resnet18': 'configs/cls/resnet/res18.yaml',
    'yolov5s': 'configs/det/yolov5/yolov5_small_silu.yaml',
    'vit_b16': 'configs/cls/vision_transformer/vit_b16_224.yaml',
    'faster_rcnn': 'configs/det/faster_rcnn/faster_rcnn_r50_fpn_1x.yaml',
    'detr': 'configs/det/faster_rcnn/faster_rcnn_r50_fpn_1x.yaml',
}


def recipe(root, data, model, seed=131):
    import yaml
    config = yaml.safe_load((Path(root)/CONFIGS[model]).read_text())
    epochs = BUDGETS[model]['epochs']
    classification = model in ('resnet18', 'vit_b16')
    selected = data['cifar10' if classification else 'pennfudan']
    config.setdefault('runtime', {}).update(device='cuda', random_seed=seed, fp16=False)
    config['args'] = {'backend': 'dist', 'display': 50, 'no_running_config': True}
    config['hooks'] = []
    config['trainer'].update(max_epoch=epochs, test_freq=10 if model == 'yolov5s' else 1,
                             save_freq=epochs, only_save_latest=False)
    config['saver'] = {'auto_resume': False, 'save_result': True}
    config['num_classes'] = 10 if classification else 2
    for module in config['net']:
        if 'num_classes' in module.get('kwargs', {}):
            module['kwargs']['num_classes'] = config['num_classes']
    for split in ('train', 'test'):
        dataset = config['dataset'][split]
        options = dataset['dataset']['kwargs']
        options['meta_file'] = selected[split]['meta']
        options['image_reader']['kwargs']['image_dir'] = selected[split]['image_dir']
        if not classification and split == 'test':
            options['evaluator']['kwargs']['gt_file'] = selected[split]['meta']
        for transform in options['transformer']:
            values = transform.setdefault('kwargs', {}) if model in ('vit_b16', 'yolov5s') else transform.get('kwargs', {})
            kind = transform['type']
            if classification and kind in ('torch_random_resized_crop', 'torch_center_crop'):
                values['size'] = 64
            elif classification and kind == 'torch_resize':
                values['size'] = 72
            elif model == 'vit_b16' and kind == 'torch_auto_augmentation':
                values['size'] = 64
            elif model == 'yolov5s' and kind == 'mosaic':
                values['tar_size'] = 320
            elif model == 'yolov5s' and kind == 'random_perspective':
                values['border'] = [-160, -160]
            elif model == 'yolov5s' and kind == 'keep_ar_resize_max':
                values['max_size'] = 320
            elif not classification and model != 'yolov5s' and kind == 'keep_ar_resize':
                values.update(scales=[320], max_size=512)
        if classification:
            dataset['batch_sampler']['kwargs']['batch_size'] = 64
            dataset['dataloader']['kwargs'].update(num_workers=0, pin_memory=True)
    if classification:
        config['net'][2]['kwargs']['cls_loss']['kwargs']['num_classes'] = 10
    if model == 'resnet18':
        scheduler = config['trainer']['lr_scheduler']
        scheduler['warmup_iter'] = 0
        scheduler['kwargs']['milestones'] = [12, 18]
        config['trainer']['optimizer']['kwargs']['lr'] = .05
    elif model == 'vit_b16':
        config['net'][0]['kwargs']['image_size'] = 64
        config['dataset']['train']['dataloader']['kwargs']['batch_fn']['kwargs']['num_classes'] = 10
        scheduler = config['trainer']['lr_scheduler']
        scheduler.update(warmup_iter=157, warmup_ratio=.01)
        scheduler['kwargs'].update(T_max=epochs, warm_epoch=1)
    elif model == 'yolov5s':
        config['ema']['enable'] = False
        config['dataset']['batch_sampler']['kwargs']['batch_size'] = 64
        config['dataset']['dataloader']['kwargs'].update(num_workers=0, pin_memory=True)
        config['net'][-1]['kwargs']['cfg']['anchor_generator']['kwargs']['base_anchors_file'] = data['yolov5_anchors']['path']
        scheduler = config['trainer']['lr_scheduler']
        scheduler.update(warmup_iter=0, warmup_epochs=3, warmup_total_epochs=epochs)
        scheduler['kwargs']['T_max'] = epochs
    else:
        config['dataset']['dataloader']['kwargs'].update(num_workers=0, pin_memory=True)
        scheduler = config['trainer']['lr_scheduler']
        scheduler['warmup_iter'] = 60
        scheduler['kwargs']['milestones'] = [8, 11]
        config['trainer']['optimizer']['kwargs']['lr'] = .0025
        config['saver']['pretrain_model'] = data['resnet50_weights']['path']
        if model == 'detr':
            scheduler['warmup_iter'] = 0
            config['saver'].pop('pretrain_model')
            config['net'] = [{'name': 'detr', 'type': 'accuracy_detr.DetrForUP',
                'kwargs': copy.deepcopy(data['detr'])}]
            config['trainer']['optimizer'] = {'type': 'AdamW', 'kwargs': {'lr': 1e-4, 'weight_decay': 1e-4},
                'special_param_group': [{'key': 'detector.backbone', 'lr': 1e-5}]}
            scheduler['warmup_register_type'] = 'no_scale_lr'
            config['hooks'] = [{'type': 'grad_clipper', 'kwargs': {'mode': 'pre_defined', 'max_norm': .1, 'norm_type': 2}}]
    return config
