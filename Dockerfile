FROM tensorflow/tensorflow:2.6.0-gpu

# Update the CUDA Linux GPG Repository Key
RUN apt-key del 7fa2af80
RUN apt-get install -y wget && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
  dpkg -i cuda-keyring_1.0-1_all.deb && \
  rm /etc/apt/sources.list.d/cuda.list && \
  apt-get update

# System packages.
RUN apt-get update && apt-get install -y \
  ffmpeg \
  libgl1-mesa-dev \
  python3-pip \
  unrar \
  wget \
  && apt-get clean

# MuJoCo.
ENV MUJOCO_GL egl
RUN mkdir -p /root/.mujoco && \
  wget -nv https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip && \
  unzip mujoco.zip -d /root/.mujoco && \
  rm mujoco.zip

# Python packages.
COPY requirements.txt ./
RUN pip3 install --upgrade pip && \
  pip3 install -r requirements.txt --no-cache-dir && \
  rm requirements.txt

# Atari ROMS.
RUN wget -L -nv http://www.atarimania.com/roms/Roms.rar && \
  unrar x Roms.rar && \
  python3 -m atari_py.import_roms ROMS && \
  rm -rf "Roms.rar" "ROMS" "HC ROMS"

# MuJoCo key.
RUN wget -P /root/.mujoco https://www.roboti.us/file/mjkey.txt

# DreamerV2.
ENV TF_XLA_FLAGS --tf_xla_auto_jit=2
COPY . /app
WORKDIR /app
CMD [ \
  "python3", "dreamerv2/train.py", \
  "--logdir", "/logdir/$(date +%Y%m%d-%H%M%S)", \
  "--configs", "defaults", "atari", \
  "--task", "atari_pong" \
]
