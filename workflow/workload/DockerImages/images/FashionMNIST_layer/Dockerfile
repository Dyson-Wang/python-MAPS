FROM ubuntu:18.04

WORKDIR /root

RUN \
    apt-get update && apt-get install -y \
    autoconf \
    build-essential \
    libtool \
    time \
    bc \
    python3 \
    python3-pip \
    wget

RUN \
    pip3 install torch
    
COPY assets assets
COPY main.py main.py
