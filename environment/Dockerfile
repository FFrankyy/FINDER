# hash:sha256:cdeb275e7f16e6ee9324818a5174fa0c4bc4016c7e465bc9911d4ad258dca810
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc=4:7.4.0-1ubuntu2.3 \
        python-pip=9.0.1-2.3~ubuntu1.18.04.1 \
        python-setuptools=39.0.1-2 \
        python-wheel=0.30.0-0.2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    cython==0.29.13 \
    networkx==2.3 \
    numpy==1.17.3 \
    pandas==0.25.2 \
    scipy==1.3.1 \
    tensorflow-gpu==1.14.0 \
    tqdm==4.36.1
