def get_task_from_cfg(cfg):
    cfg['runtime'] = cfg.setdefault('runtime', {})
    task_info = ''
    if 'task_names' in cfg['runtime']:
        task_names = cfg['runtime']['task_names']
        if isinstance(task_names, list):
            for task_name in task_names:
                task_info = task_info + str(task_name) + '_'
            task_info = task_info[:-1]
        else:
            task_info = str(task_names)
    else:
        task_dict = {
            'coco': 'det',
            'custom': 'det',
            'lvis': 'det',
            'cls': 'cls',
            'seg': 'seg',
            'keypoint': 'kp',
            'kitti': '3d',
            'unknown': 'unknown'
        }
        dataset_type = cfg['dataset'].get('train', {}).get('dataset', {}).get('type', 'unknown')
        task_info = task_dict.get(dataset_type, 'unknown')
    return task_info
