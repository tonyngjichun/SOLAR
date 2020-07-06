import argparse
import os
import time
import pickle
import pdb

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torch.utils.model_zoo import load_url
import torch.nn.functional as F
from torchvision import transforms

from cirtorch.networks.imageretrievalnet import init_network, extract_ss, extract_ms
from cirtorch.utils.general import get_data_root, htime

"""
This script and the module `cirtorch' is based on and modified from github.com/filipradenovic/cnnimageretrieval-pytorch, which has an MIT license. For more details please refer to https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/LICENSE
"""

PRETRAINED = {
    'rSfM120k-tl-resnet50-gem-w'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'rSfM120k-tl-resnet101-gem-w'          : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'rSfM120k-tl-resnet152-gem-w'          : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet152-gem-w-f39cada.pth',
    'gl18-tl-resnet50-gem-w'                     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'gl18-tl-resnet101-gem-w'                     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
    'gl18-tl-resnet152-gem-w'                     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet152-gem-w-21278d5.pth',
}

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Example')
parser.add_argument('--network', '-n', metavar='NETWORK', default='solar-best.pth', 
                    help="network to be evaluated: " +
                        " | ".join(PRETRAINED.keys()))
parser.add_argument('--image-path', '-impath', dest='image_path', type=str, default='try.jpg')
parser.add_argument('--image-size', '-imsize', dest='image_size', default=1024, type=int, metavar='N',
                    help="maximum size of longer image side used for testing (default: 1024)")
parser.add_argument('--multiscale', '-ms', metavar='MULTISCALE', default='[1]',
                    help="use multiscale vectors for testing, " +
                    " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")

def main():
    args = parser.parse_args()

    # loading network
    # pretrained networks (downloaded automatically)
    print(">> Loading network:\n>>>> '{}'".format(args.network))
    state = torch.load(os.path.join(get_data_root(), 'networks', args.network))

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
    net_params['mode'] = 'test'
    net_params['self_attn'] = True
    net_params['sa_layers'] = '45'
    net = init_network(net_params) 
    net.load_state_dict(state['state_dict'])


    print(">>>> loaded network: ")
    print(net.meta_repr())

    # setting up the multi-scale parameters
    ms = list(eval(args.multiscale))
    print(">>>> Evaluating scales: {}".format(ms))

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()
    
    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    img = Image.open(args.image_path).convert('RGB')
    _input = transform(img)
    _input = _input.unsqueeze_(0).cuda()

    with torch.no_grad():
        if len(ms) == 1 and ms[0] == 1:
            desc = extract_ss(net, _input)
        else:
            desc = extract_ms(net, _input, ms, msp=1)

        # get last feature map and p
        p = net.pool.p
        fmap = net.features(_input.cuda())

        # Draw attention map
        attn = torch.pow(fmap.clamp(min=1e-6), p)
        attn = torch.sum(attn, 1, keepdim=True)
        attn = F.interpolate(attn, 
                    size=(_input.shape[2], _input.shape[3]), mode='bilinear')
        attn = (attn - attn.min()) / (attn.max() - attn.min())
        attn = attn.squeeze_().cpu()
    
        fig = plt.figure()
        ax_img = fig.add_subplot(221)
        ax_attn = fig.add_subplot(222)
        ax_desc = fig.add_subplot(212)
        
        for ax in [ax_img, ax_attn]:
            ax.set_xticks([])
            ax.set_yticks([])
    
        ax_img.imshow(img)
        ax_attn.imshow(img)
        ax_attn.imshow(attn, cmap='jet', alpha=.5)
        ax_desc.plot(desc.tolist())
    
        plt.tight_layout

        plt.show()

if __name__ == '__main__':
    main()
