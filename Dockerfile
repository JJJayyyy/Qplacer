FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Installs system dependencies.
RUN apt-get update \
        && apt-get install -y \
            flex \
            libcairo2-dev \
            libboost-all-dev \
            libkrb5-dev \
            libgl1-mesa-glx

ENV QISKIT_METAL_HEADLESS=1

# Installs system dependencies from conda.
RUN conda create -y --name qplacer_env python=3.9 pip -c conda-forge
RUN conda init bash
RUN conda install -y -c conda-forge bison

# Installs cmake.
ADD https://cmake.org/files/v3.21/cmake-3.21.0-linux-x86_64.sh /cmake-3.21.0-linux-x86_64.sh
RUN mkdir /opt/cmake \
        && sh /cmake-3.21.0-linux-x86_64.sh --prefix=/opt/cmake --skip-license \
        && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
        && cmake --version