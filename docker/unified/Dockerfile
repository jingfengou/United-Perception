# syntax=docker/dockerfile:1
ARG CUDA_IMAGE=nvcr.io/nvidia/cuda:12.6.3-devel-ubuntu24.04
ARG METAX_IMAGE=cr.metax-tech.com/public-library/maca-pytorch:3.8.1.2-torch2.10-py312-ubuntu24.04-amd64
FROM ${CUDA_IMAGE} AS cuda
FROM ${METAX_IMAGE}
USER root
COPY --from=cuda /usr/local/cuda-12.6 /opt/up/sdk/cuda
RUN ln -s /opt/maca /opt/up/sdk/maca && ln -s /opt/mxdriver /opt/up/sdk/mxdriver
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY . /opt/up/source
COPY docker/unified/launch.py docker/unified/probe.py docker/unified/profiles.json /opt/up/
ARG NVIDIA_ARCH_LIST=8.0;8.9
ARG UP_PIP_INDEX_URL=https://pypi.org/simple
ENV UP_NVIDIA_ARCH_LIST=${NVIDIA_ARCH_LIST}
RUN PIP_INDEX_URL=${UP_PIP_INDEX_URL} /opt/conda/bin/python /opt/up/source/docker/unified/prepare.py --root /opt/up
RUN printf '#!/bin/sh\nexec /opt/conda/bin/python -I /opt/up/launch.py "$@"\n' > /usr/local/bin/up-run \
    && chmod 755 /usr/local/bin/up-run \
    && mkdir -p /.metax/shadercache /mxlog/umd \
    && chmod 1777 /.metax/shadercache /mxlog/umd
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=""
WORKDIR /opt/up/source
ENTRYPOINT ["up-run"]
CMD ["--help"]
