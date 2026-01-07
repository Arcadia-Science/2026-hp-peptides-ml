# hash:sha256:79046a305236047a2f16c11657f8c3e0c064e4c8e4305a10929253342d04f880
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        jupyter==1.0.0 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    e3nn==0.4.4 \
    matplotlib==3.3.4 \
    numpy==1.20.1 \
    torch==1.10.0 \
    torch-geometric==2.0.2

COPY postInstall /
RUN /postInstall
