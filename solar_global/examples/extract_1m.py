import argparse
import fnmatch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

import torch
from torch.utils.model_zoo import load_url
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from solar_global.networks.imageretrievalnet import init_network, extract_vectors
from solar_global.datasets.testdataset import configdataset
from solar_global.utils.download import download_distractors
from solar_global.utils.evaluate import compute_map_and_print
from solar_global.utils.general import get_data_root, htime
from solar_global.utils.networks import load_network


# options
parser = argparse.ArgumentParser(description='Example Script for extracting and saving descriptors for R-1M disctractors')
parser.add_argument('--network', '-n', metavar='NETWORK', default='resnet101-solar-best.pth', 
                    help="network to be evaluated. ")
parser.add_argument('--image-size', '-imsize', dest='image_size', default=1024, type=int, metavar='N',
                    help="maximum size of longer image side used for testing (default: 1024)")
parser.add_argument('--multiscale', '-ms', metavar='MULTISCALE', default='[1, 2**(1/2), 1/2**(1/2)]',
                    help="use multiscale vectors for testing, " +
                    " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")
parser.add_argument('--soa', action='store_true',
                    help='use soa blocks')
parser.add_argument('--soa-layers', type=str, default='45',
                    help='config soa blocks for second-order attention')

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help="gpu id used for testing (default: '0')")


def main():
    args = parser.parse_args()

    # check if test dataset are downloaded
    # and download if they are not
    download_distractors(get_data_root())

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # loading network
    net = load_network(network_name=args.network)
    net.mode = 'test'

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

    # evaluate on test datasets
    dataset = 'revisitop1m' 
    start = time.time()

    print('>> {}: Extracting...'.format(dataset))

    # prepare config structure for the test dataset
    cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
    images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
    qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
    try:
        bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
    except:
        bbxs = None  # for holidaysmanrot and copydays

    # extract database and query vectors
    print('>> {}: database images...'.format(dataset))
    vecs = extract_vectors(net, images, args.image_size, transform, ms=ms, mode='test')
    torch.save(vecs, args.network + '_vecs_' + dataset + '.pt')

    print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))


if __name__ == '__main__':
    main()
