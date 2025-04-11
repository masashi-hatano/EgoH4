FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update
RUN export DEBIAN_FRONTEND="noninteractive"
RUN apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm ffmpeg\
    xz-utils tk-dev libffi-dev liblzma-dev python3 python3-pip python3-distutils apt-utils\
    openssl git bzip2 vim  bash-completion screen

# self-update pip before installing python packages
RUN python3 -m pip install --upgrade pip

# pytorch with GPU support
RUN python3 -m pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
RUN pip install fvcore
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt200/download.html

# clean up
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Set the default working directory
WORKDIR /workspace