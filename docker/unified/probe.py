"""Audit the selected environment without treating an import as GPU validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", required=True, choices=("nvidia", "metax"))
    parser.add_argument("--no-device", action="store_true", help="Package inspection only; no GPU validation")
    args = parser.parse_args()
    import torch
    import torchvision
    from up_platform import configure_platform
    from up_platform.ops._native.loader import load_native
    result = {"vendor": args.vendor, "python": sys.executable, "torch": torch.__version__,
              "torchvision": torchvision.__version__, "torch_path": torch.__file__,
              "device_executed": not args.no_device,
              "environment": {key: os.environ.get(key) for key in (
                  "PATH", "LD_LIBRARY_PATH", "CUDA_HOME", "MACA_PATH", "TORCH_EXTENSIONS_DIR")}}
    if not args.no_device:
        if not torch.cuda.is_available():
            raise RuntimeError("Selected vendor has no accessible device")
        configure_platform({"backend": args.vendor}, purpose="training")
        extension = load_native(args.vendor)
        x = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4).requires_grad_()
        y = x @ x.T
        y.sum().backward()
        torch.cuda.synchronize()
        result.update(device_name=torch.cuda.get_device_name(), output=y.detach().cpu().tolist(),
                      gradient=x.grad.cpu().tolist(), extension=str(extension.__file__),
                      extension_sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest())
    maps = Path("/proc/self/maps")
    result["loaded_vendor_libraries"] = sorted({line.split()[-1] for line in maps.read_text().splitlines()
        if '/' in line and any(word in line.lower() for word in ('libcuda', 'libcublas', 'libcudnn', 'maca', 'mxdriver', 'mcdnn', 'mcblas'))}) if maps.exists() else []
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
