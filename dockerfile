FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
################################
# Install apt-get Requirements #
################################
ENV LANG C.UTF-8
ENV APT_INSTALL="apt-get install -y --no-install-recommends"
ENV PIP_INSTALL="python -m pip --no-cache-dir install --upgrade --default-timeout 100"
RUN rm -rf /var/lib/apt/lists/* \
    /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
    apt-utils build-essential ca-certificates dpkg-dev pkg-config software-properties-common \
    cifs-utils openssh-server nfs-common net-tools iputils-ping iproute2 locales htop tzdata \
    tar wget git swig vim curl tmux zip unzip rar unrar sudo zsh cmake \
    libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev libglfw3 \
    libgl1-mesa-glx libncurses5-dev libncursesw5-dev openmpi-bin openmpi-common libopenmpi-dev

################
# Set Timezone #
################
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen en_US.UTF-8 && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8 && \
    echo "Asia/Shanghai" > /etc/timezone && \
    rm -f /etc/localtime && \
    rm -rf /usr/share/zoneinfo/UTC && \
    dpkg-reconfigure --frontend=noninteractive tzdata

#########
# NVTOP #
#########
RUN git clone https://github.com/Syllo/nvtop.git && \
    mkdir -p nvtop/build && cd nvtop/build && \
    cmake .. && \
    make && \
    make install

###################
# python packages #
###################
RUN conda install ruamel.yaml mpi4py -y
RUN ${PIP_INSTALL} gym easydict imageio opencv_contrib_python \
    pillow scikit-image scikit-video tensorboard matplotlib wandb

##################
# Apt auto clean #
##################
RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache/pip