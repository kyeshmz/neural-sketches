FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
LABEL maintainer="kkshmz"
# docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 --ipc=host -it --name=bigan-container5 bigan /bin/bash
#apt-get update && apt-get install -y ibxrender-dev libxext6 libglib2.0-0 libsm6
#pip install opencv-python requests tqdm scikit-image scipy tensorflow-hub opencv-python 
#https://github.com/porjo/youtubeuploader

# Python 2.7 or 3.6 is supported by Ubuntu Bionic out of the box
ARG python=3.6
ENV PYTHON_VERSION=${python}

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ENV TENSORFLOW_VERSION=1.15.0 \
    CUDNN_VERSION=7.6.0.64-1+cuda10.0 \
    NCCL_VERSION=2.4.7-1+cuda10.0 \
    MXNET_VERSION=1.5.0 \
    PYTHON_VERSION=${python}
ENV LANG en_US.UTF-8 

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        g++-4.8 \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libcudnn7=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        python3.6 \
        python3.6-dev \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
        libblas3 \
        libblas-dev \
        libglib2.0-0 \
        libsm6 \ 
        libxext6 \
        libxrender-dev \
        python3-distutils \
        mediainfo \
        zip \
        unzip

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN  pip install tensorflow-gpu==${TENSORFLOW_VERSION} \
        tensorflow-hub \
        scikit-image \
        scipy \
        requests \
        tqdm \
        Pillow \
        opencv-python \
        keras \
        future \
        typing \
        numpy \
        jupyter \
        jupyterlab \
        h5py && \
        apt -y autoremove && \
        apt -y autoclean && \
        rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*


# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]
# --------------------------------   ANYTHING THAT RELATED TO OTHER CODE UNDER HERE -----------------------------------


RUN jupyter serverextension enable --py jupyterlab && \
    touch ~/.hushlogin && \
    echo "alias jl='jupyter lab --port=8000 --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' '" >> ~/.bashrc

WORKDIR /home
EXPOSE 20
CMD ["/usr/sbin/sshd", "-D"]