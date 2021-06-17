FROM ubuntu:latest
WORKDIR /app
COPY . .
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y \
    python3 python3-pip curl unrar unzip ffmpeg libsm6 libxext6
RUN pip3 install --no-cache --upgrade pip setuptools
RUN pip3 install --no-cache -r requirements.txt
RUN curl -L http://www.atarimania.com/roms/Roms.rar > /tmp/Roms.rar
WORKDIR /tmp
RUN unrar x Roms.rar
RUN unzip ROMS.zip
RUN python3 -m atari_py.import_roms /tmp/ROMS
WORKDIR /app
RUN rm -rf /tmp
