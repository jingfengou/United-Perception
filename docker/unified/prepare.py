"""Build both environments without a GPU; run inside the MACA development image."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from launch import environment, VENDOR_CHECK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "source"
    manifest = {}
    for vendor in ("metax", "nvidia"):
        prefix = root / "envs" / vendor
        command = [sys.executable, "-m", "venv"]
        if vendor == "metax":
            command.append("--system-site-packages")
        subprocess.run([*command, str(prefix)], check=True)
        python, env = environment(root, vendor)
        env.update(CUDA_VISIBLE_DEVICES="", MAX_JOBS="2", PIP_DISABLE_PIP_VERSION_CHECK="1",
                   TORCH_CUDA_ARCH_LIST=os.environ.get("UP_NVIDIA_ARCH_LIST", "8.0;8.9") if vendor == "nvidia" else "8.0")
        def run(arguments, **kwargs):
            options = dict(env=env, check=True)
            options.update(kwargs)
            subprocess.run([python, *map(str, arguments)], **options)
        if vendor == "nvidia":
            run(["-m", "pip", "install", "--no-cache-dir", "torch==2.10.0+cu126", "torchvision==0.25.0+cu126",
                 "--index-url", "https://download.pytorch.org/whl/cu126"])
        run(["-c", VENDOR_CHECK, vendor])
        constraints = root / (vendor + "-constraints.txt")
        result = subprocess.check_output([python, "-c", "from importlib.metadata import version; "
            "print('\\n'.join(p+'=='+version(p) for p in ('torch','torchvision')))"], env=env, text=True)
        constraints.write_text(result)
        run(["-m", "pip", "install", "--no-cache-dir", "-c", constraints,
             "-r", source / "docker/unified/requirements.txt"])
        # An inherited Ninja package does not install a command in this venv.
        run(["-m", "pip", "install", "--no-cache-dir", "--ignore-installed", "--no-deps", "ninja==1.11.1.4"])
        if vendor == "nvidia":
            run(["-m", "pip", "install", "--no-cache-dir", "-c", constraints,
                 "spconv-cu126==2.3.8", "cumm-cu126==0.7.11"])
        work = root / "build" / vendor
        shutil.copytree(source, work, ignore=shutil.ignore_patterns(".git", "dist", "build", "__pycache__", "*.so"))
        wheels = root / "wheels" / vendor
        wheels.mkdir(parents=True)
        build_env = dict(env, UP_BUILD_BACKEND=vendor)
        build_env.pop("PYTHONSAFEPATH", None)  # setup.py imports this trusted source checkout.
        run(["setup.py", "bdist_wheel", "--dist-dir", wheels], cwd=work, env=build_env)
        wheel, = wheels.glob("*.whl")
        run(["-m", "pip", "install", "--no-deps", wheel])
        run([root / "probe.py", "--vendor", vendor, "--no-device"])
        manifest[vendor] = {"python": python, "constraints": result.splitlines(), "wheel": wheel.name,
                            "gpu_execution": False}
    (root / "build-manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
