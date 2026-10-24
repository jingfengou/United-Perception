"""Record native device-copy layout policy without importing UP or platform."""
import argparse
import json
from pathlib import Path
import sys

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    # Exact values are irrelevant here; this only observes native transfer layout.
    source = torch.arange(2 * 3 * 64 * 64, dtype=torch.float32).reshape(2, 3, 64, 64)
    records = {}
    for name, operation in (('to', lambda: source.to('cuda:0')), ('cuda', lambda: source.cuda(0)),
                            ('explicit_contiguous', lambda: source.to('cuda:0').contiguous())):
        value = operation()
        records[name] = {'stride': list(value.stride()), 'contiguous': value.is_contiguous(),
                         'channels_last': value.is_contiguous(memory_format=torch.channels_last),
                         'values_exact': torch.equal(source, value.cpu())}
    report = {'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
              'source_stride': list(source.stride()), 'copies': records,
              'up_imported': 'up' in sys.modules, 'up_platform_imported': 'up_platform' in sys.modules}
    a.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
