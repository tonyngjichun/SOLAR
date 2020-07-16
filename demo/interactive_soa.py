import argparse
import os
import time
import pickle
import pdb
from tqdm import tqdm
import math

import cv2

import numpy as np

from PIL import Image
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn.functional as F

from solar_global.networks.imageretrievalnet import init_network
from solar_global.datasets.datahelpers import default_loader 
from solar_global.utils.plots import draw_soa_map


MODEL = 'data/networks/resnet101-solar-best.pth'

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", default="assets/interactive_demo.jpg", help="Path to the image")
args = parser.parse_args()

def nothing(x):
    pass


# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
 

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
 
		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)

# loading network
# pretrained networks (downloaded automatically)
print(">> Loading network:\n>>>> '{}'".format(MODEL))
state = torch.load(MODEL)

# parsing net params from meta
# architecture, pooling, mean, std required
# the rest has default values, in case that is doesnt exist
net_params = {}
net_params['architecture'] = state['meta']['architecture']
net_params['pooling'] = state['meta']['pooling']
net_params['local_whitening'] = state['meta'].get('local_whitening', False)
net_params['regional'] = state['meta'].get('regional', False)
net_params['whitening'] = state['meta'].get('whitening', False)
net_params['mean'] = state['meta']['mean']
net_params['std'] = state['meta']['std']
net_params['pretrained'] = False
net_params['pretrained_type'] = None
net_params['mode'] = 'draw'
net_params['soa'] = state['meta']['soa'] 
net_params['soa_layers'] = state['meta']['soa_layers']
net = init_network(net_params) 
net.load_state_dict(state['state_dict'])


print(">>>> loaded network: ")
print(net.meta_repr())


net = net.cuda() 

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args.image)
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)


while True: 
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(20) #& 0xFF
 
	# if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()
	# if the 'c' key is pressed, break from the loop       
    elif key == ord("q"):
        print("Exit")
        cv2.destroyAllWindows
        break
    if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
        break
 
    # if there are two reference points, then crop the region of interest
    # from the image and display it
    if len(refPt) == 2:
        # display soa
        soa = draw_soa_map(default_loader(args.image), net, refPt)
        cv2.imshow("Second order attention", soa)
        cv2.waitKey(20)

# close all open windows
cv2.destroyAllWindows()