FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# for depth anything
RUN apt-get update
RUN apt install sudo
RUN apt install -y zip
RUN apt install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
RUN apt install zstd
# only for development
RUN apt update && apt install -y eog nano
RUN apt install -y meshlab

ENV ZED_SDK_INSTALLER=ZED_SDK_Tegra_L4T35.3_v4.1.0.zstd.run
RUN wget --quiet -O ${ZED_SDK_INSTALLER} https://download.stereolabs.com/zedsdk/4.1/l4t35.2/jetsons
RUN chmod +x ${ZED_SDK_INSTALLER} && ./${ZED_SDK_INSTALLER} -- silent

RUN cd /root/ && git clone https://github.com/isl-org/Open3D.git
RUN cd /root/Open3D/ && bash util/install_deps_ubuntu.sh assume-yes
RUN cd /root/Open3D/ && mkdir build
RUN cd /root/Open3D/build && cmake -DPYTHON3_ROOT=/usr/bin/ -DBUILD_PYTHON_MODULE=OFF ..
RUN cd /root/Open3D/build && make

RUN cd /root && mkdir disparity-view/
RUN cd /root/disparity-view
WORKDIR /root/disparity-view
RUN mkdir disparity_view/
RUN mkdir ./scripts
COPY disparity_view/* disparity_view/
COPY scripts/* ./scripts/
COPY pyproject.toml Makefile *.py ./
RUN python3 -m pip install .[dev]

