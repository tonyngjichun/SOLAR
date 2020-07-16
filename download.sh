#!/bin/bash

# global model
mkdir -p data/networks/
wget -nc https://imperialcollegelondon.box.com/shared/static/fznpeayct6btel2og2wjjgvqw0ziqnk4.pth -O data/networks/resnet101-solar-best.pth

# local model
mkdir -p solar_local/weights/
wget -nc https://imperialcollegelondon.box.com/shared/static/4djweum6gs30os243zqzplhafxlys31z.pth -O solar_local/weights/local-solar-345-liberty.pth

# 1-mil distractor vecs
wget -nc https://imperialcollegelondon.box.com/shared/static/e9z542xirf1awck2iwr8yutubn85srnk.pt -O resnet101-solar-best.pth_vecs_revisitop1m.pt