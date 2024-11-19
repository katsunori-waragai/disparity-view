#!/bin/bash
python3 -m pip uninstall --yes opencv-python-headless opencv-contrib-python opencv-python
python3 -m pip install opencv-python==3.4.18.65
