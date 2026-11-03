"""Python install by default; build original NVIDIA kernels only on request."""
import os
from pathlib import Path
import re
from setuptools import find_packages, setup


def extensions():
    platform = os.environ.get('UP_BUILD_PLATFORM', 'none')
    if platform == 'none':
        return [], {}
    if platform != 'nvidia':
        raise ValueError('UP_BUILD_PLATFORM accepts none or nvidia; vendor SDKs build separately')
    # Import build tools only for an explicit native build; no GPU probe.
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
    if CUDA_HOME is None:
        raise RuntimeError('NVIDIA build needs a CUDA toolkit and matching NVIDIA PyTorch')
    source = Path(__file__).parent / 'up' / 'extensions' / 'csrc'
    files = sorted(str(p) for pattern in ('*.cpp', '*.cu') for p in source.rglob(pattern))
    module = CUDAExtension('up.extensions._C', files, include_dirs=[str(source)],
                           define_macros=[('WITH_CUDA', None)],
                           extra_compile_args={'cxx': [], 'nvcc': [
                               '-DCUDA_HAS_FP16=1', '-D__CUDA_NO_HALF_OPERATORS__',
                               '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__']})
    return [module], {'build_ext': BuildExtension}


ext_modules, cmdclass = extensions()
version = re.search(r'__version__ = "(.*?)"', Path('up/__init__.py').read_text(encoding='utf-8')).group(1)
setup(name='UP', version=version, packages=find_packages(), python_requires='>=3.10',
      description='UP with config-selected accelerator runtimes and five model entries',
      ext_modules=ext_modules, cmdclass=cmdclass, include_package_data=True,
      package_data={'up.tasks.det.plugins.detr.official': ['LICENSE']},
      extras_require={'yolo11': ['ultralytics==8.3.225'],
                      'onnx': ['onnx>=1.16,<2', 'onnxruntime>=1.18,<2']})
