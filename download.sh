#!/bin/bash

#retrieval
mkdir -p data/networks/
wget https://imperialcollegelondon.box.com/shared/static/fznpeayct6btel2og2wjjgvqw0ziqnk4.pth -O data/networks/resnet101-solar-best.pth

#patch
mkdir -p solar_local/weights/
wget https://imperialcollegelondon.box.com/shared/static/4djweum6gs30os243zqzplhafxlys31z.pth -O solar_local/weights/local-solar-345-liberty.pth