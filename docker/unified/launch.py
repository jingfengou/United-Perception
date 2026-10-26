"""Select an installed vendor environment, then replace this process with UP.

This module imports only the standard library. It never installs packages,
detects a vendor from the device name, or changes model/numerical settings.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
VENDOR_CHECK = """import sys, torch
vendor = sys.argv[1]
metax = bool(getattr(torch._utils_internal, 'USE_MACA', False)) or 'metax' in torch.__version__
if metax != (vendor == 'metax') or (vendor == 'nvidia' and (torch.version.hip or not torch.version.cuda)):
    raise SystemExit('Wrong PyTorch build for ' + vendor + ': ' + torch.__version__)
"""


def environment(root, vendor, inherited=None):
    root = Path(root).resolve()
    profiles = json.loads((root / "profiles.json").read_text())
    profile = profiles[vendor]
    expand = lambda value: value.format(root=str(root))
    env = dict(os.environ if inherited is None else inherited)
    for key in list(env):
        if key.startswith(("CONDA", "VIRTUAL_ENV", "MACA", "MCDNN", "MCBLAS")) or key in (
            "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "PYTHONPLATLIBDIR",
            "LD_LIBRARY_PATH", "LD_PRELOAD", "CUDA_HOME", "CUDA_PATH", "UP_BUILD_BACKEND",
            "LIBRARY_PATH", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "PKG_CONFIG_PATH", "CMAKE_PREFIX_PATH",
        ):
            # Keep explicit same-vendor numerical/device controls; never set new ones.
            if vendor == "metax" and key.startswith(("MCDNN", "MCBLAS", "MACA_VISIBLE")):
                continue
            env.pop(key)
    env.update({key: expand(value) for key, value in profile["variables"].items()})
    env["PATH"] = ":".join(map(expand, profile["bin"])) + ":" + SYSTEM_PATH
    env["LD_LIBRARY_PATH"] = ":".join(map(expand, profile["libraries"]))
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONSAFEPATH"] = "1"
    env["UP_SELECTED_VENDOR"] = vendor
    cache = Path(env.get("UP_CACHE_ROOT", f"/tmp/up-cache-{getattr(os, 'getuid', lambda: 0)()}")) / vendor
    for key, suffix in (("TORCH_EXTENSIONS_DIR", "extensions"), ("TRITON_CACHE_DIR", "triton"),
                        ("XDG_CACHE_HOME", "xdg"), ("MPLCONFIGDIR", "matplotlib")):
        env[key] = str(cache / suffix)
    return expand(profile["python"]), env


def up_arguments(arguments, vendor):
    arguments = list(arguments)
    key = "runtime.platform.backend"
    if key in arguments:
        index = arguments.index(key)
        if index + 1 == len(arguments) or arguments[index + 1] != vendor:
            raise ValueError("UP backend must match the selected environment: " + vendor)
        return arguments
    if any(arg.startswith("--opts=") or arg.startswith(key + "=") for arg in arguments):
        raise ValueError("Use --opts key value pairs; the launcher sets runtime.platform.backend")
    if "--opts" not in arguments:
        arguments.append("--opts")
    arguments.extend([key, vendor])
    return arguments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/opt/up"), help="Installed dual-environment bundle")
    parser.add_argument("vendor", choices=("nvidia", "metax"))
    parser.add_argument("command", choices=("train", "inference", "check", "python"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        python, env = environment(args.root, args.vendor)
        if not Path(python).is_file():
            raise ValueError("Environment not installed: " + python)
        subprocess.run([python, "-c", VENDOR_CHECK, args.vendor], env=env, check=True)
        if args.command == "check":
            command = [python, str(args.root / "probe.py"), "--vendor", args.vendor, *args.arguments]
        elif args.command == "python":
            command = [python, *args.arguments]
        else:
            command = [python, "-m", "up", args.command, *up_arguments(args.arguments, args.vendor)]
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"up-run: {error}\n")
    os.execve(python, command, env)


if __name__ == "__main__":
    main()
