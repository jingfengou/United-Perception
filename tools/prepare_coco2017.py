"""Fetch complete COCO2017 from its public S3 bucket, verify and safely extract.

No filtered HF detection subset is substituted for the 118,287/5,000 images.
Downloads resume in .partial files; failed files/logs are retained. Existing
complete archives must match their recorded original-object MD5.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile


ARCHIVES = (
    ('annotations/annotations_trainval2017.zip', 'f4bbac642086de4f52a3fdda2de5fa2c'),
    ('zips/val2017.zip', '442b8da7639aecaf257c1dceb8ba8c80'),
    ('zips/train2017.zip', 'cced6f7f71b7629ddf16f17bbcfab6b2'),
)
ORIGIN = 'https://s3.amazonaws.com/images.cocodataset.org/'


def checksum(path, name='sha256'):
    result = hashlib.new(name)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def download(url, destination, md5):
    if destination.exists():
        if checksum(destination, 'md5') != md5:
            raise ValueError('Existing archive differs: ' + str(destination))
        return
    partial = destination.with_suffix(destination.suffix + '.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={'Range': 'bytes=%d-' % offset} if offset else {})
    with urllib.request.urlopen(request, timeout=60) as response:
        if offset and (response.status != 206 or not response.headers.get('Content-Range', '').startswith('bytes %d-' % offset)):
            raise ValueError('Server did not honor the verified resume offset')
        expected = offset + int(response.headers['Content-Length'])
        updated = time.monotonic()
        with partial.open('ab' if offset else 'xb') as stream:
            for block in iter(lambda: response.read1(4 * 1024 * 1024), b''):
                stream.write(block)
                offset += len(block)
                if time.monotonic() - updated > 10:
                    print(json.dumps({'archive': destination.name, 'downloaded': offset, 'expected': expected}), flush=True)
                    updated = time.monotonic()
            stream.flush()
            os.fsync(stream.fileno())
    if partial.stat().st_size != expected or checksum(partial, 'md5') != md5:
        raise ValueError('Archive size/MD5 verification failed; partial retained')
    partial.rename(destination)


def extract(archive, output):
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            target = (output / info.filename).resolve()
            if not target.is_relative_to(output.resolve()):
                raise ValueError('Archive path escapes destination')
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Never overwrite a prior extraction; verify it against ZIP CRC.
            if target.exists():
                import zlib
                crc = 0
                with target.open('rb') as stream:
                    for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                        crc = zlib.crc32(block, crc)
                if target.stat().st_size != info.file_size or crc != info.CRC:
                    raise ValueError('Previously extracted file changed: ' + str(target))
            else:
                # A killed extraction must not leave a truncated final image.
                temporary = target.with_suffix(target.suffix + '.extracting')
                with bundle.open(info) as source, temporary.open('wb') as destination:
                    shutil.copyfileobj(source, destination, 4 * 1024 * 1024)
                temporary.rename(target)


def prepare(output):
    output = output.resolve()
    if (output / 'READY.json').exists():
        raise FileExistsError('Already complete; verify/reuse rather than restart')
    archives = output / 'archives'
    archives.mkdir(parents=True, exist_ok=True)
    receipts = []
    for name, md5 in ARCHIVES:
        path = archives / Path(name).name
        download(ORIGIN + name, path, md5)
        extract(path, output)
        receipts.append({'name': path.name, 'url': ORIGIN + name, 'md5': md5,
                         'sha256': checksum(path), 'bytes': path.stat().st_size})
        print(json.dumps({'archive_complete': path.name, 'sha256': receipts[-1]['sha256']}), flush=True)
    splits = {}
    for split, expected in (('train2017', 118287), ('val2017', 5000)):
        annotation = output / 'annotations' / ('instances_' + split + '.json')
        data = json.loads(annotation.read_text())
        if len(data['images']) != expected or len(data['categories']) != 80:
            raise ValueError('Unexpected standard COCO split/category count')
        for image in data['images']:
            if not (output / split / image['file_name']).is_file():
                raise ValueError('Missing COCO image ' + image['file_name'])
        splits[split] = {'images': expected, 'annotations': len(data['annotations']),
                         'annotation_sha256': checksum(annotation)}
    record = {'dataset': 'COCO2017', 'archives': receipts, 'splits': splits,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'filtered_subset': False}
    with (output / 'READY.json').open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output)
