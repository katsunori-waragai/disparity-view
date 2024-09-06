#!/bin/bash
xhost +
export GIT_ROOT=$(cd $(dirname $0)/.. ; pwd)
docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY \
	-v ${GIT_ROOT}/depth-anything-zed/weights:/root/Depth-Anything-for-Jetson-Orin/weights \
	-v ${GIT_ROOT}/depth-anything-zed/data:/root/Depth-Anything-for-Jetson-Orin/data \
	-v ${GIT_ROOT}/depth-anything-zed/test:/root/Depth-Anything-for-Jetson-Orin/test \
	--device /dev/bus/usb \
	--device /dev/video0:/dev/video0:mwr \
	-v /tmp/.X11-unix/:/tmp/.X11-unix depth-view:100
 
