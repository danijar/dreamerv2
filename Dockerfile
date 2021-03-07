FROM tensorflow/tensorflow:2.3.1-gpu-jupyter

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git build-essential cmake python3 python3-pip libgl1-mesa-dev ffmpeg
# libgl1-mesa-dev is needed according to https://stackoverflow.com/questions/63977422/error-trying-to-import-cv2opencv-python-package

RUN pip3 install tensorflow-gpu==2.3.1 tensorflow_probability==0.11.1 pandas matplotlib ruamel.yaml

RUN pip3 install 'gym[atari]'

RUN git clone https://github.com/danijar/dreamerv2.git
