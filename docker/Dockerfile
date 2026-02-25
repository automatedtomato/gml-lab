ARG CUDA="11.8.0"
ARG UBUNTU="22.04"

FROM nvidia/cuda:${CUDA}-cudnn8-devel-ubuntu${UBUNTU}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget vim graphviz \
    bash-completion build-essential \
    python3-pip python3.10-dev python3.10-distutils \
    libgl1 libsm6 libxext6 libglib2.0-0 libxrender1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /usr/local/bin/uv

WORKDIR /gml-lab

COPY docker/install/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# To install requred mmcv for mmdet, cuda, torch required version, it is necessary to use mim, not pip.
RUN mim install mmcv==2.1.0

RUN pip install -U "setuptools>=64.0"

COPY pyproject.toml .
