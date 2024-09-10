#!/bin/bash
xhost +
docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY \
	--device /dev/bus/usb \
	--device /dev/video0:/dev/video0:mwr \
	-v /tmp/.X11-unix/:/tmp/.X11-unix disparity-view:100
 
