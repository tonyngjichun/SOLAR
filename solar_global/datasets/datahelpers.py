import os
from PIL import Image, ImageFile
import numpy as np
import sys
import subprocess

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms

def cid2filename(cid, prefix):
    """
    Creates a training image path out of its CID name

    Arguments
    ---------
    cid      : name of the image
    prefix   : root directory where images are saved

    Returns
    -------
    filename : full image filename
    """
    return os.path.join(prefix, cid[-2:], cid[-4:-2], cid[-6:-4], cid)

def pil_loader(path):
    # to avoid crashing for truncated (corrupted images)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def imcrop(img, params):
    img = transforms.functional.crop(img, *params)
    return img

def imthumbnail(img, imsize):
    img.thumbnail((imsize, imsize), Image.ANTIALIAS)
    return img

def imresize(img, imsize):
    img = transforms.Resize(imsize)(img)
    return img

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def collate_tuples(batch):
    if len(batch) == 1:
        return [batch[0][0]], [batch[0][1]]
    return [batch[i][0] for i in range(len(batch))], [batch[i][1] for i in range(len(batch))]

def unnormalise(rgb):
    '''
    Reverse the ImageNet normalisation on a batch of rgb images
    '''
    device = rgb.device
    mean = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
    rgb = (rgb * std + mean)
    rgb = torch.clamp(rgb, 0., 1.)

    return rgb
