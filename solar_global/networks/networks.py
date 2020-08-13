import os
import copy
import time
import numpy as np

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
import torch.utils.model_zoo as model_zoo
from torchvision import transforms, models, datasets

from solar_global.utils.general import get_data_root

# for some models, we have imported features (convolutions) from caffe because the image retrieval performance is higher for them
FEATURES = {
    'vgg16'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-vgg16-features-d369c8e.pth',
    'resnet50'  : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet50-features-ac468af.pth',
    'resnet101' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet101-features-10a101d.pth',
    'resnet152' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet152-features-1011020.pth',
}

PRETRAINED = {
    'SfM120k-vgg16-gem'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'SfM120k-resnet101-gem' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
    # new networks with whitening learned end-to-end
    'SfM120k-tl-resnet50-gem-w'  : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'SfM120k-tl-resnet101-gem-w' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'SfM120k-tl-resnet152-gem-w' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet152-gem-w-f39cada.pth',
    'gl18-tl-resnet50-gem-w'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'gl18-tl-resnet101-gem-w'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
    'gl18-tl-resnet152-gem-w'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet152-gem-w-21278d5.pth'
}

####################################################################################################
########################################## Functions ###############################################
####################################################################################################


## Kaiming weight initialisation
def weights_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass
        #nn.init.kaiming_normal_(module.weight.data)
        #nn.init.constant_(module.bias.data, 0.0)

def constant_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.constant_(module.weight.data, 0.0)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass
        #nn.init.kaiming_normal_(module.weight.data)

def extract_features_from_e2e(model):
    state_dict_features = OrderedDict()
    for key, value in model['state_dict'].items():
        if key.startswith('features'):
            state_dict_features[key[9:]] =  value

    return state_dict_features



####################################################################################################
########################################## Networks ###############################################
####################################################################################################

class ResNet(nn.Module):
    """ 
    """ 
    def __init__(self, base_model):
        super(ResNet, self).__init__()

        feat_in = base_model.fc.in_features

        self.base_model = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        x = self.base_model(x)
        return x


class SOABlock(nn.Module):
    def __init__(self, in_ch, k):
        super(SOABlock, self).__init__()

        self.in_ch = in_ch
        self.out_ch = in_ch
        self.mid_ch = in_ch // k

        print('Num channels:  in    out    mid')
        print('               {:>4d}  {:>4d}  {:>4d}'.format(self.in_ch, self.out_ch, self.mid_ch))

        self.f = nn.Sequential(
                nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)),
                nn.BatchNorm2d(self.mid_ch),
                nn.ReLU())
        self.g = nn.Sequential(
                nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)),
                nn.BatchNorm2d(self.mid_ch),
                nn.ReLU())
        self.h = nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1))
        self.v =nn.Conv2d(self.mid_ch, self.out_ch, (1, 1), (1, 1))

        self.softmax = nn.Softmax(dim=-1)

        for conv in [self.f, self.g, self.h]:    #, self.v]:
            conv.apply(weights_init)
            #conv.apply(constant_init)

        self.v.apply(constant_init)


    def forward(self, x, vis_mode=False):
        B, C, H, W = x.shape

        f_x = self.f(x).view(B, self.mid_ch, H * W) # B * mid_ch * N, where N = H*W
        g_x = self.g(x).view(B, self.mid_ch, H * W) # B * mid_ch * N, where N = H*W
        h_x = self.h(x).view(B, self.mid_ch, H * W) # B * mid_ch * N, where N = H*W

        z = torch.bmm(f_x.permute(0, 2, 1), g_x) # B * N * N, where N = H*W

        if vis_mode:
            # for visualisation only
            attn = self.softmax((self.mid_ch ** -.75) * z)
        else:
            attn = self.softmax((self.mid_ch ** -.50) * z)

        z = torch.bmm(attn, h_x.permute(0, 2, 1)) # B * N * mid_ch, where N = H*W
        z = z.permute(0, 2, 1).view(B, self.mid_ch, H, W) # B * mid_ch * H * W

        z = self.v(z)
        z = z + x

        return z, attn


class ResNetSOAs(nn.Module):
    def __init__(self, architecture='resnet101', pretrained_type='gl18', soa_layers='45', mode='train'):
        super(ResNetSOAs, self).__init__()

        base_model = vars(models)[architecture](pretrained=True)
        last_feat_in = base_model.inplanes
        base_model = nn.Sequential(*list(base_model.children())[:-2])

        if pretrained_type=='caffenet' and architecture in FEATURES:
            print(">> {}: for '{}' custom pretrained features '{}' are used"
                .format(os.path.basename(__file__), architecture, os.path.basename(FEATURES[architecture])))
            model_dir = os.path.join(get_data_root(), 'networks')
            base_model.load_state_dict(model_zoo.load_url(FEATURES[architecture], model_dir=model_dir))
        elif pretrained_type in ['SfM120k', 'gl18'] and architecture in FEATURES:
            pretrained_name = pretrained_type + '-tl-' + architecture + '-gem-w'
            print(">> {}: for '{}' custom pretrained features '{}' are used"
                .format(os.path.basename(__file__), architecture, os.path.basename(PRETRAINED[pretrained_name])))
            model_dir = os.path.join(get_data_root(), 'networks')
            base_model.load_state_dict(extract_features_from_e2e(model_zoo.load_url(PRETRAINED[pretrained_name], model_dir=model_dir)))

        res_blocks = list(base_model.children())

        self.conv1 = nn.Sequential(*res_blocks[0:2])
        self.conv2_x = nn.Sequential(*res_blocks[2:5])
        self.conv3_x = res_blocks[5]
        self.conv4_x = res_blocks[6]
        self.conv5_x = res_blocks[7]

        self.soa_layers = soa_layers
        if '4' in self.soa_layers:
            print("SOA_4:")
            self.soa4 = SOABlock(in_ch=last_feat_in // 2, k=4)
        if '5' in self.soa_layers:
            print("SOA_5:")
            self.soa5 = SOABlock(in_ch=last_feat_in, k=2)

####        if '4' in self.soa_layers:
####            print("SOA_4:")
####            self.soa4 = SOABlock(in_ch=last_feat_in // 2, k=8)
####        if '5' in self.soa_layers:
####            print("SOA_5:")
####            self.soa5 = SOABlock(in_ch=last_feat_in, k=8)


    def forward(self, x, mode='test'):
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2_x(x)
            x = self.conv3_x(x)
            x = self.conv4_x(x)

        # start SOA blocks
        if '4' in self.soa_layers:
            x, soa_m2 = self.soa4(x, mode == 'draw')
        
        x = self.conv5_x(x)
        if '5' in self.soa_layers:
            x, soa_m1 = self.soa5(x, mode == 'draw')

        if mode == 'draw':
            return x, soa_m2, soa_m1

        return x
