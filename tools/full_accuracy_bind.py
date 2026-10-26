"""Audit full dataset bytes once, then create machine-local benchmark bindings.

READY metadata alone is not integrity evidence. ImageNet images are checked
against their per-image SHA256 manifests; COCO files against official ZIP CRC
and sizes. The resulting identity contains no machine-specific absolute paths.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import zipfile
import zlib

from accuracy_data import digest
from full_accuracy_data import COCO_CATEGORY_IDS
from prepare_coco2017 import ARCHIVES, checksum


def verify_imagenet_shard(root, manifest, *, decode_images=False):
    count, image_bytes = 0, 0
    with manifest.open() as rows:
        for line in rows:
            row = json.loads(line)
            path = (root/'images'/row['path']).resolve()
            if not path.is_relative_to((root/'images').resolve()):
                raise ValueError('ImageNet image path escapes dataset')
            if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
                raise ValueError('ImageNet encoded bytes differ: '+row['path'])
            if decode_images:
                from PIL import Image
                with Image.open(path) as image:
                    image.convert('RGB').load()
            count += 1
            image_bytes += row['bytes']
    return count, image_bytes


def audit_imagenet(root):
    ready = json.loads((root/'READY.json').read_text())
    if ready['label_base'] != 0 or ready['classes'] != 1000 or not ready['image_bytes_unchanged']:
        raise ValueError('Unexpected ImageNet representation')
    result = {}
    for name, split, expected in (('train', 'train', 1281167), ('test', 'validation', 50000)):
        record = ready['splits'][split]
        meta = root/record['meta']
        if record['count'] != expected or digest(meta) != record['meta_sha256']:
            raise ValueError('Full ImageNet labels/membership differ')
        images, image_bytes, hashes = 0, 0, {}
        for shard in record['shards']:
            path = root/'shards'/split/shard['name']
            if digest(path) != shard['manifest_sha256']:
                raise ValueError('ImageNet image manifest differs: '+shard['name'])
            count, size = verify_imagenet_shard(root, path, decode_images=True)
            images += count
            image_bytes += size
            hashes[shard['name']] = shard['manifest_sha256']
            print(json.dumps({'audit': 'imagenet/'+name, 'verified_images': images, 'expected_images': expected}), flush=True)
        if images != expected:
            raise ValueError('Incomplete full ImageNet image audit')
        result['imagenet/'+name] = {'images': images, 'image_bytes': image_bytes,
            'meta_sha256': record['meta_sha256'],
            'image_manifests_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}
    return result


def audit_coco(root):
    json.loads((root/'READY.json').read_text())  # Must have completed original preparation.
    archive_hashes = {}
    for name, md5 in ARCHIVES:
        path = root/'archives'/Path(name).name
        if checksum(path, 'md5') != md5:
            raise ValueError('COCO original archive differs: '+path.name)
        archive_hashes[path.name] = digest(path)
    result = {}
    for name, split, expected in (('train', 'train2017', 118287), ('test', 'val2017', 5000)):
        annotation = root/'annotations'/('instances_'+split+'.json')
        data = json.loads(annotation.read_text())
        if len(data['images']) != expected or sorted(item['id'] for item in data['categories']) != list(COCO_CATEGORY_IDS):
            raise ValueError('Unexpected full COCO image/category membership')
        ids = {row['file_name'] for row in data['images']}
        verified, image_bytes = set(), 0
        with zipfile.ZipFile(root/'archives'/(split+'.zip')) as archive:
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                path = (root/entry.filename).resolve()
                if not path.is_relative_to((root/split).resolve()) or path.name not in ids:
                    raise ValueError('Unexpected COCO image path: '+entry.filename)
                crc = 0
                with path.open('rb') as image:
                    for block in iter(lambda: image.read(4*1024*1024), b''):
                        crc = zlib.crc32(block, crc)
                if path.stat().st_size != entry.file_size or crc != entry.CRC:
                    raise ValueError('COCO encoded image differs: '+entry.filename)
                from PIL import Image
                with Image.open(path) as image:
                    image.convert('RGB').load()
                verified.add(path.name)
                image_bytes += entry.file_size
        if verified != ids:
            raise ValueError('COCO image audit incomplete')
        result['coco/'+name] = {'images': expected, 'image_bytes': image_bytes,
            'meta_sha256': digest(annotation), 'image_archive_sha256': archive_hashes[split+'.zip'],
            'annotation_archive_sha256': archive_hashes['annotations_trainval2017.zip']}
        print(json.dumps({'audit': 'coco/'+name, 'verified_images': len(verified)}), flush=True)
    return result


def bind(args):
    began = time.monotonic()
    args.imagenet, args.coco = args.imagenet.resolve(), args.coco.resolve()
    splits = dict(audit_imagenet(args.imagenet), **audit_coco(args.coco))
    source_files = {str(path.relative_to(args.detr_source)): digest(path)
                    for path in sorted(args.detr_source.rglob('*.py'))}
    source_sha = hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest()
    if source_sha != args.detr_source_sha256:
        raise ValueError('Pinned official DETR source differs')
    anchors, backbone = digest(args.anchors), digest(args.backbone)
    identity = {'scope': 'full ImageNet-1K and COCO2017, all original image bytes audited',
        'splits': splits, 'all_image_bytes_verified': True, 'all_images_rgb_decoded': True, 'detr_source_sha256': source_sha,
        'backbone_sha256': backbone, 'anchors_sha256': anchors}
    data = {'identity': identity, 'imagenet': {'image_dir': str(args.imagenet/'images'),
        'train': {'meta': str(args.imagenet/'train.txt'), 'images': 1281167},
        'test': {'meta': str(args.imagenet/'val.txt'), 'images': 50000}},
        'coco': {key: {'meta': str(args.coco/'annotations'/('instances_'+split+'.json')),
                       'image_dir': str(args.coco/split), 'images': count}
                 for key, split, count in (('train', 'train2017', 118287), ('test', 'val2017', 5000))},
        'yolov5_anchors': {'path': str(args.anchors), 'sha256': anchors},
        'detr': {'source': str(args.detr_source), 'source_sha256': source_sha,
            'revision': '29901c51d7fe8712168b8d0d64351170bc0f83e0',
            'checkpoint': str(args.backbone), 'checkpoint_sha256': backbone},
        'audit': {'created_utc': datetime.now(timezone.utc).isoformat(),
                  'seconds': time.monotonic()-began, 'tool_sha256': digest(Path(__file__))}}
    with args.output.open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('imagenet', 'coco', 'detr-source', 'backbone', 'anchors', 'output'):
        parser.add_argument('--'+name, type=lambda value: Path(value).absolute(), required=True)
    parser.add_argument('--detr-source-sha256', required=True)
    bind(parser.parse_args())
