Accelerator platforms
=====================

Select the installed accelerator through the model configuration:

.. code-block:: yaml

   runtime:
     platform: nvidia

Omitting ``platform`` preserves the NVIDIA default. The supported selections are:

================  ===================  ==============  ========================
platform          PyTorch device       Communication   Required vendor extension
================  ===================  ==============  ========================
nvidia            cuda                 nccl            none
ascend_npu        npu                  hccl            torch_npu
cambricon_mlu     mlu                  cncl            torch_mlu
metax_cuda        cuda                 nccl API        mcPyTorch environment
================  ===================  ==============  ========================

Device type and communication backend are selected together. ``--backend dist``
retains its original meaning: the PyTorch execution path. It does not select a
vendor or force DDP. A single worker uses the ordinary model; multiple workers
use PyTorch DistributedDataParallel during training. Non-NVIDIA workers use
``spawn`` to avoid inheriting initialized accelerator state through ``fork``.

Environment and installation
----------------------------

Use a separate SDK/PyTorch environment for each vendor. Install a matching
driver, runtime, torch, torchvision and vendor extension before installing UP.
Do not install NVIDIA PyTorch into an existing vendor environment.

The original ``requirements.txt`` and ``easy_setup.sh`` remain available for the
legacy NVIDIA environment. Their old pinned packages are not an installation
recipe for modern vendor images. For a modern SDK environment, install the
application dependencies separately, keeping the vendor's PyTorch packages:

.. code-block:: bash

   python -m pip install -r requirements/runtime.txt

NVIDIA retains the original native build:

.. code-block:: bash

   # Set PATH, LD_LIBRARY_PATH and TORCH_CUDA_ARCH_LIST for your compiler/GPU.
   python setup.py build_ext -i
   python -m pip install --no-deps --no-build-isolation -e .

For Ascend, MLU or MetaX, choose the platform when installing. This skips the
NVIDIA extension build; it does not remove its C++/CUDA sources:

.. code-block:: bash

   UP_BUILD_PLATFORM=metax_cuda python -m pip install --no-deps --no-build-isolation -e .
   # Substitute ascend_npu or cambricon_mlu for the corresponding environment.

``UP_BUILD_PLATFORM`` is an installation setting. Runtime selection comes from
the YAML configuration, including normal ``--opts runtime.platform ...`` overrides.
The configuration cannot install a driver or make an incompatible torch wheel work.

The LightX2V platform Dockerfiles provide examples of separate vendor
environments: `Ascend 910B <https://github.com/ModelTC/LightX2V/blob/main/dockerfiles/platforms/Dockerfile_ascend_910b>`_
and `Cambricon MLU590 <https://github.com/ModelTC/LightX2V/blob/main/dockerfiles/platforms/Dockerfile_cambricon_mlu590>`_.
Their video-model dependencies are not UP requirements. Match the SDK and
framework versions using the vendor's compatibility matrix.

Training and inference
----------------------

Use the existing model configuration and replace only ``runtime.platform``.
Configure dataset paths and checkpoint paths as for the original UP:

.. code-block:: bash

   python -m up train --launch pytorch --ng 1 --config configs/det/yolox/yolox_tiny.yaml
   python -m up train --launch pytorch --ng 2 --config configs/det/yolox/yolox_tiny.yaml
   python -m up inference --launch pytorch --ng 1 --config configs/det/yolox/yolox_tiny.yaml \
       -i /path/to/images -c /path/to/checkpoint.pth -v /path/to/results

The existing ordinary/distributed shell scripts are preserved. Slurm resource
names and scheduling remain site-specific. The inference command retains the
original behavior: multiple workers do not automatically shard input images or
wrap the inference model in DDP. Use one worker for ordinary image inference.

Operators and scope
-------------------

ResNet, ViT, YOLOX and DETR keep their PyTorch layers, optimizers and mathematical
definitions. Tensor movement uses ``Tensor.to(device)``; intermediate tensors
inherit the input device. Only device services and missing native operations
need adaptation.

NVIDIA continues to use UP's compiled extension and torchvision NMS. Other
platforms use same-device PyTorch implementations of box IoU/IoF, NMS and
sigmoid focal loss. Accelerator IoU/IoF matches the original CUDA kernel's
inclusive-coordinate (+1) convention; CPU and aligned paths retain the existing
Python formulas. The reference NMS is sequential and may be slower than a
vendor kernel. No inference/training speed equivalence is implied.

All original native sources remain in ``up/extensions/csrc``. Optional native
operations outside these model paths, including deformable convolution, RoI
pooling, 3D operations, softmax focal loss and cross sigmoid focal loss, have not
been ported to non-NVIDIA platforms. Calling an unported native operation raises
an error instead of loading an incompatible NVIDIA binary.

DETR's original SciPy Hungarian assignment stays on the CPU by design. YOLOX's
existing CPU assignment fallback is restricted to out-of-memory errors; other
runtime errors are propagated. Distributed Python objects use a Gloo CPU group
by default, while model gradients use NCCL/HCCL/CNCL. Explicit accelerator
groups serialize object bytes as int32 for backend compatibility.
