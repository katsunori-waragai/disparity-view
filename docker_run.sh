#!/bin/bash
xhost +
export GIT_ROOT=$(cd $(dirname $0)/.. ; pwd)
docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY \
	--device /dev/bus/usb \
	--device /dev/video0:/dev/video0:mwr \
	-v ${GIT_ROOT}/disparity-view/test:/root/disparity-view/test \
	-v /tmp/.X11-unix/:/tmp/.X11-unix disparity-view:100
 
