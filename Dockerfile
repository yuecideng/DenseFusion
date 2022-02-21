FROM nvidia/cudagl:10.2-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

# Essentials: developer tools, build tools, OpenBLAS
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils git curl vim unzip openssh-client wget \
    build-essential cmake \
    libopenblas-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# Python 3.5
RUN apt-get update && apt-get install -y --no-install-recommends python3.6 python3.6-dev python3-pip python3-tk && \
    pip3 install --no-cache-dir --upgrade pip setuptools && \
    echo "alias python='python3'" >> /root/.bash_aliases && \
    echo "alias pip='pip3'" >> /root/.bash_aliases

# RUN curl https://bootstrap.pypa.io/pip/3.5/get-pip.py --output get-pip.py

# Science libraries and other common packages
RUN pip3 --no-cache-dir install -i https://mirrors.aliyun.com/pypi/simple/ \
    numpy scipy pyyaml cffi pyyaml matplotlib Cython requests opencv-python

# Tensorflow
RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ torch==1.6.0 torchvision==0.7.0

# Expose port for TensorBoard
EXPOSE 6006

# cd to home on login
RUN echo "cd /root/dense_fusion" >> /root/.bashrc
