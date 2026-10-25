"""Full-data UP recipes; path/resource edits are shared by all comparison lanes.

These are corresponding UP models, not exact model-verify reproductions.
The effective world size and batches must be frozen in the plan before training.
"""
import copy
import math
from pathlib import Path


RECIPES = {
    'resnet50': {'config': 'configs/cls/resnet/res50_200e_bag_of_tricks.yaml',
                 'epochs': 200, 'precision': 'amp_fp16', 'metric': 'top1'},
    'vit_b16': {'config': 'configs/cls/vision_transformer/vit_b16_224.yaml',
                'epochs': 300, 'precision': 'amp_fp16', 'metric': 'top1'},
    'yolov5s': {'config': 'configs/det/yolov5/yolov5_small_silu.yaml',
                'epochs': 300, 'precision': 'fp32', 'metric': 'bbox.AP'},
    'detr': {'config': 'configs/det/faster_rcnn/faster_rcnn_r50_fpn_1x.yaml',
             'epochs': 300, 'precision': 'fp32', 'metric': 'bbox.AP'},
}


def execution_budget(model, settings, epoch_steps, native_max_iter):
    """Stop at a frozen epoch boundary without compressing the native recipe."""
    recipe_epochs = RECIPES[model]['epochs']
    epochs = settings.get('execution_epochs', recipe_epochs)
    if type(epochs) is not int or not 1 <= epochs <= recipe_epochs:
        raise ValueError('Execution epochs must be an integer within the native recipe')
    if epoch_steps < 1 or native_max_iter != recipe_epochs*epoch_steps:
        raise ValueError('Native UP recipe step budget changed')
    expected = epochs*epoch_steps
    if settings['expected_steps'] != expected:
        raise ValueError('Execution step budget differs from frozen plan')
    return {'recipe_epochs': recipe_epochs, 'execution_epochs': epochs,
            'steps_per_epoch': epoch_steps, 'expected_steps': expected}


def recipe(source, data, model, *, batch_size, world_size, workers=4, seed=131):
    import yaml
    if min(batch_size, world_size) < 1 or workers < 1:
        raise ValueError('Positive batch/world size and worker processes required')
    config = yaml.safe_load((Path(source)/RECIPES[model]['config']).read_text())
    epochs = RECIPES[model]['epochs']
    global_batch = batch_size*world_size
    classification = model in ('resnet50', 'vit_b16')
    config.setdefault('runtime', {}).update(device='cuda', random_seed=seed,
        fp16=RECIPES[model]['precision'] == 'amp_fp16')
    config['args'] = {'backend': 'dist', 'display': 100, 'no_running_config': True}
    config['hooks'] = []
    config['trainer'].update(max_epoch=epochs, test_freq=5, save_freq=5,
                             only_save_latest=True)
    config['saver'] = {'auto_resume': False, 'save_result': True, 'memory_friendly': True}
    for split in ('train', 'test'):
        block = config['dataset'][split]
        options = block['dataset']['kwargs']
        if classification:
            selected = data['imagenet']
            options['meta_file'] = selected[split]['meta']
            options['image_reader']['kwargs']['image_dir'] = selected['image_dir']
        else:
            selected = data['coco'][split]
            block['dataset']['type'] = 'full_accuracy_coco'
            options.update(meta_file=selected['meta'], expected_images=selected['images'])
            options['image_reader']['kwargs']['image_dir'] = selected['image_dir']
            if split == 'test':
                options['evaluator']['kwargs']['gt_file'] = selected['meta']
        sampler = block.get('batch_sampler', config['dataset'].get('batch_sampler'))
        sampler['kwargs']['batch_size'] = batch_size
        loader = block.get('dataloader', config['dataset'].get('dataloader'))
        loader['kwargs'].update(num_workers=workers, pin_memory=True)
        # No UUID worker_init: a dedicated per-epoch generator is installed by
        # the shared driver, independent of the model's CPU/GPU RNG.
        loader['kwargs'].pop('worker_init', None)
    if classification:
        # This comparison explicitly chooses a nominal reference BS512 for LR
        # scaling. The original YAML files do not themselves prescribe a world
        # size; this is a documented experiment choice, not an upstream claim.
        scale = global_batch/512
        config['trainer']['optimizer']['kwargs']['lr'] *= scale
        scheduler = config['trainer']['lr_scheduler']
        scheduler['warmup_iter'] = math.ceil(scheduler['warmup_iter']/scale)
        if 'warmup_iter' in scheduler['kwargs']:
            scheduler['kwargs']['warmup_iter'] = scheduler['warmup_iter']
    elif model == 'yolov5s':
        if global_batch < 64:
            raise ValueError('Native UP YOLO requires global BS>=64 with solo BN; do not silently change BN')
        config['net'][-1]['kwargs']['cfg']['anchor_generator']['kwargs']['base_anchors_file'] = data['yolov5_anchors']['path']
        config['trainer']['lr_scheduler']['warmup_total_epochs'] = epochs
    else:
        config['saver'].pop('pretrain_model', None)
        config['net'] = [{'name': 'detr', 'type': 'full_accuracy_detr.FullCocoDetrForUP',
                          'kwargs': copy.deepcopy(data['detr'])}]
        config['trainer']['optimizer'] = {'type': 'AdamW', 'kwargs': {'lr': 1e-4, 'weight_decay': 1e-4},
            'special_param_group': [{'key': 'detector.backbone', 'lr': 1e-5}]}
        config['trainer']['lr_scheduler'] = {'warmup_iter': 0, 'warmup_register_type': 'no_scale_lr',
            'type': 'MultiStepLR', 'kwargs': {'milestones': [200], 'gamma': .1}}
        config['hooks'] = [{'type': 'grad_clipper',
            'kwargs': {'mode': 'pre_defined', 'max_norm': .1, 'norm_type': 2}}]
        # Retain UP's native aspect-ratio grouping to bound padding on 24-GiB
        # devices. The identical sampler is used in all three DETR lanes.
    return config

