"""Verify the existing CIFAR/PennFudan split and bind it to a common logical path."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def verify(data):
    results = {}
    for name in ('cifar10', 'pennfudan'):
        for split in ('train', 'test'):
            entry = data[name][split]
            if digest(entry['meta']) != entry['meta_sha256']:
                raise ValueError('Changed labels: '+name+'/'+split)
            if name == 'cifar10':
                rows = [line.rsplit(' ', 1) for line in Path(entry['meta']).read_text().splitlines()]
                hashes = [[filename, digest(Path(entry['image_dir'])/filename), int(label)] for filename, label in rows]
            else:
                rows = json.loads(Path(entry['meta']).read_text())['images']
                hashes = [[row['file_name'], digest(Path(entry['image_dir'])/row['file_name']),
                    digest(Path(entry['mask_dir'])/(Path(row['file_name']).stem+'_mask.png'))] for row in rows]
            value = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
            if len(hashes) != entry['count'] or value != entry['image_manifest_sha256']:
                raise ValueError('Changed image cohort: '+name+'/'+split)
            results[name+'/'+split] = {'count': len(hashes), 'sha256': value}
    for name, key, expected in (('resnet50_weights', 'path', 'sha256'), ('yolov5_anchors', 'path', 'sha256'),
                                ('detr', 'checkpoint', 'checkpoint_sha256')):
        if digest(data[name][key]) != data[name][expected]:
            raise ValueError('Changed weights/anchors: '+name)
        results[name] = data[name][expected]
    source = Path(data['detr']['source'])
    files = {str(path.relative_to(source)): digest(path) for path in sorted(source.rglob('*.py'))}
    value = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if value != data['detr']['source_sha256']:
        raise ValueError('Changed official DETR source')
    results['detr_source'] = value
    return results


def prepare(source, output):
    old = json.loads(source.read_text())
    verified = verify(old)
    output.mkdir(parents=True, exist_ok=False)
    data = {'split_seed': old['split_seed']}
    for name in ('cifar10', 'pennfudan'):
        data[name] = {}
        for split in ('train', 'test'):
            entry = old[name][split]
            folder = output/name/split
            folder.mkdir(parents=True)
            meta = folder/Path(entry['meta']).name
            shutil.copyfile(entry['meta'], meta)
            (folder/'images').symlink_to(Path(entry['image_dir']).resolve(), target_is_directory=True)
            selected = {key: entry[key] for key in ('count', 'meta_sha256', 'image_manifest_sha256')}
            selected.update(meta=str(meta), image_dir=str(folder/'images'))
            if name == 'pennfudan':
                (folder/'masks').symlink_to(Path(entry['mask_dir']).resolve(), target_is_directory=True)
                selected['mask_dir'] = str(folder/'masks')
            data[name][split] = selected
    for name, key, expected in (('resnet50_weights', 'path', 'sha256'), ('yolov5_anchors', 'path', 'sha256'),
                                ('detr', 'checkpoint', 'checkpoint_sha256')):
        entry = old[name]
        target = output/(name+Path(entry[key]).suffix)
        target.symlink_to(Path(entry[key]).resolve())
        data[name] = {key: str(target), expected: entry[expected]}
    target = output/'official-detr'
    target.symlink_to(Path(old['detr']['source']).resolve(), target_is_directory=True)
    data['detr'].update(source=str(target), source_sha256=old['detr']['source_sha256'], revision=old['detr']['revision'])
    assert verify(data) == verified
    (output/'data.json').write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')
    (output/'provenance.json').write_text(json.dumps({'source_manifest': str(source),
        'source_manifest_sha256': digest(source), 'verified': verified}, indent=2)+'\n')
    return verified


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = prepare(args.source, args.output) if args.output else verify(json.loads(args.source.read_text()))
    print(json.dumps(result, indent=2))
