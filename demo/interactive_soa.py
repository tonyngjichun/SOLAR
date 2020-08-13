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
from solar_global.utils.networks import load_network
from solar_global.utils.plots import draw_soa_map


MODEL = 'resnet101-solar-best.pth'
IMSIZE = 1024

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
net = load_network(network_name=MODEL)

print(">>>> loaded network: ")
print(net.meta_repr())

# moving network to gpu and eval mode
net.cuda() 
net.eval() 

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args.image)
h, w = image.shape[0], image.shape[1]
if (h <= w):
    resize = (int(w * IMSIZE/h), IMSIZE)
else:
    resize = (IMSIZE, int(h * IMSIZE/w))
image = cv2.resize(image, resize)
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