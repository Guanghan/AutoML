FROM nvidia/cuda:10.1-cudnn7-devel
ENV DEBIAN_FRONTEND noninteractive

MAINTAINER Guanghan Ning "guanghan.ning@jd.com"

RUN apt-get update
RUN apt-get install -y python3-opencv ca-certificates python3-dev git wget vim ssh redis-server sudo
RUN apt-get update
RUN apt-get install -y iputils-ping htop

# link python: python = python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# instal pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
  python get-pip.py && \
  rm get-pip.py

RUN pip --no-cache-dir install torch torchvision -f https://download.pytorch.org/whl/cu100/torch_stable.html

# install pip packages from requirement
COPY requirements.txt /home/requirements.txt
RUN pip --no-cache-dir install -r /home/requirements.txt

COPY ../AutoML /home/AutoML-IU

# set WorkingDir
WORKDIR /home/AutoML-IU

ENTRYPOINT bash
