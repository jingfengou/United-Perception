"""Environment separation and real subprocess dispatch, independent of Torch."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2] / "docker/unified"
spec = importlib.util.spec_from_file_location("vendor_launch", SOURCE / "launch.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class VendorLauncherTests(unittest.TestCase):
    def test_sdk_and_python_pollution_is_removed(self):
        for vendor in ("nvidia", "metax"):
            python, env = launch.environment(SOURCE, vendor, {
                "PYTHONPATH": "/foreign/torch", "PYTHONHOME": "/foreign", "LD_LIBRARY_PATH": "/foreign/sdk",
                "LD_PRELOAD": "/foreign/lib.so", "CUDA_HOME": "/foreign/cuda", "MACA_PATH": "/foreign/maca",
                "PATH": "/foreign/bin", "CONDA_PREFIX": "/foreign/env", "CUDA_VISIBLE_DEVICES": "2",
                "PYTHONHASHSEED": "131", "CPATH": "/foreign/include",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "MCDNN_STABLE_RESULT_MOD": "1",
            })
            self.assertIn(f"envs/{vendor}/bin/python", python.replace("\\", "/"))
            self.assertNotIn("foreign", repr(env))
            self.assertNotIn("PYTHONPATH", env)
            self.assertNotIn("LD_PRELOAD", env)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(env["PYTHONHASHSEED"], "131")
            self.assertEqual(env["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            self.assertEqual("MACA_PATH" in env, vendor == "metax")
            self.assertEqual("CUDA_HOME" in env, vendor == "nvidia")
            self.assertEqual("MCDNN_STABLE_RESULT_MOD" in env, vendor == "metax")
            self.assertIn(vendor, env["TORCH_EXTENSIONS_DIR"])
            if vendor == "nvidia":
                self.assertNotIn("sdk/cuda/lib64", env["LD_LIBRARY_PATH"])

    def test_preserve_opts_and_reject_vendor_conflict(self):
        before = ["--config", "model.yaml", "--opts", "trainer.max_epoch", "1"]
        after = launch.up_arguments(before, "metax")
        self.assertEqual(after, before + ["runtime.platform.backend", "metax"])
        self.assertEqual(before[-1], "1")
        self.assertEqual(launch.up_arguments(after, "metax"), after)
        with self.assertRaises(ValueError):
            launch.up_arguments(after, "nvidia")

    def test_add_opts_and_reject_ambiguous_syntax(self):
        self.assertEqual(launch.up_arguments(["--config", "a.yaml"], "nvidia")[-3:],
                         ["--opts", "runtime.platform.backend", "nvidia"])
        with self.assertRaises(ValueError):
            launch.up_arguments(["--opts=runtime.platform.backend=metax"], "metax")

    @unittest.skipIf(os.name == "nt", "POSIX execve dispatch is exercised on Linux CI and both servers")
    def test_dispatch_in_a_fresh_vendor_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = json.loads((SOURCE / "profiles.json").read_text())
            for vendor in ("nvidia", "metax"):
                profiles[vendor]["python"] = sys.executable
            (root / "profiles.json").write_text(json.dumps(profiles))
            # A stand-in torch only for the child import guard; no GPU claims.
            fake = root / "modules"
            fake.mkdir()
            (fake / "torch.py").write_text("import os\n__version__=os.environ['FAKE_TORCH_VERSION']\n"
                "class _utils_internal: USE_MACA=False\n"
                "class version: hip=None; cuda=None if '+cpu' in __version__ else '12.6'\n")
            executable = root / "python"
            # Use a small Python wrapper instead of shell interpolation of -c contents.
            executable.write_text(f"#!{sys.executable}\nimport os, sys\nos.environ['PYTHONPATH']={str(fake)!r}\n"
                                  f"os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n")
            executable.chmod(0o755)
            for vendor in profiles:
                profiles[vendor]["python"] = str(executable)
            (root / "profiles.json").write_text(json.dumps(profiles))
            for vendor, version in (("nvidia", "2.10.0+cu126"), ("metax", "2.10.0+metax3.8.1.0")):
                result = subprocess.run([sys.executable, str(SOURCE / "launch.py"), "--root", str(root),
                    vendor, "python", "-c", "import os; print(os.environ['UP_SELECTED_VENDOR'])"],
                    env=dict(os.environ, FAKE_TORCH_VERSION=version), text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), vendor)
            for vendor, version in (("nvidia", "2.10.0+cpu"), ("nvidia", "2.10.0+metax3.8.1.0"),
                                    ("metax", "2.10.0+cu126")):
                result = subprocess.run([sys.executable, str(SOURCE / "launch.py"), "--root", str(root),
                    vendor, "python", "-c", "raise AssertionError('must not execute')"],
                    env=dict(os.environ, FAKE_TORCH_VERSION=version), text=True, capture_output=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("Wrong PyTorch build", result.stderr)


if __name__ == "__main__":
    unittest.main()
