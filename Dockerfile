# docker build -t dreamer2 .
# docker run --gpus all -it -p 6006:6006 -v `pwd`:/tf/host dreamer2

FROM tensorflow/tensorflow:2.3.1-gpu-jupyter

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y tmux git build-essential cmake python3 python3-pip libgl1-mesa-dev ffmpeg
# libgl1-mesa-dev is needed according to https://stackoverflow.com/questions/63977422/error-trying-to-import-cv2opencv-python-package

RUN pip3 install tensorflow-gpu==2.3.1 tensorflow_probability==0.11.1 pandas matplotlib ruamel.yaml

RUN pip3 install 'gym[atari]'

COPY . /tf/dreamerv2

WORKDIR /tf/dreamerv2

ENTRYPOINT python3 dreamer.py --logdir /tf/host/logdir/atari_pong/dreamerv2/1 --configs defaults atari --task atari_pong
