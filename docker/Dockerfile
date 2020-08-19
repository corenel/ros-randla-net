FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
MAINTAINER Yusu Pan <xxdsox@gmail.com>

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Remove NVIDIA APT REPO
RUN rm /etc/apt/sources.list.d/nvidia-ml.list && apt clean

# Install dependencies for development
RUN apt-get update && \
    apt-get install -q -y --no-install-recommends \
    sed gawk rsync git curl wget tmux zsh vim htop iotop iftop \
    python3 python3-pip python3-dev python3-setuptools

# Install python packages
COPY ./requirements.txt /
RUN python3 -m pip install -U pip --user
RUN python3 -m pip install -U -r requirements.txt

# Fix libstdc++6
RUN echo "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu xenial main\ndeb-src http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu xenial main" >> /etc/apt/sources.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 60C317803A41BA51845E371A1E9377A2BA9EF27F
RUN apt-get update && \
    apt-get install -q -y libstdc++6

# Install patched python-pcl
RUN apt-get update && \
    apt-get install -q -y libpcl-dev
#RUN git clone https://github.com/strawlab/python-pcl.git
#COPY ./vtk.patch python-pcl
#RUN cd python-pcl && python3 -m pip install -e .

# Clean
RUN apt-get clean && \
    rm -rf \
    /tmp/* \
    /var/lib/apt/lists/* \
    /var/tmp/*

# setup entrypoint
COPY ./entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]

CMD ["/bin/bash"]
