import os
import random
import pickle
import time

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from solar_global.datasets.datahelpers import default_loader, imresize, imcrop, cid2filename 
from solar_global.datasets.genericdataset import ImagesFromList
from solar_global.utils.general import get_data_root


class TuplesBatchedDataset(data.Dataset):
    """Data loader that loads training and validation tuples of
        Radenovic etal ECCV16: CNN image retrieval learns from BoW
    Args:
        name (string): dataset name: 'retrieval-sfm-120k'
        mode (string): 'train' or 'val' for training and validation parts of dataset
        imsize (int, Default: None): Defines the maximum size of longer image side
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional): A function to load an image given its path.
        nnum (int, Default:5): Number of negatives for a query image in a training tuple
        qsize (int, Default:1000): Number of query images, ie number of (q,p,n1,...nN) tuples, to be processed in one epoch
        poolsize (int, Default:10000): Pool size for negative images re-mining
     Attributes:
        images (list): List of full filenames for each image
    """

    def __init__(self, name, mode, imsize=None, nnum=5, qsize=2000, poolsize=20000, transform=None, loader=default_loader):

        if not (mode == 'train' or mode == 'val'):
            raise(RuntimeError("MODE should be either train or val, passed as string"))

        if name.startswith('retrieval-SfM'):
            # setting up paths
            data_root = get_data_root()
            db_root = os.path.join(data_root, 'train', name)
            ims_root = os.path.join(db_root, 'ims')

            # loading db
            db_fn = os.path.join(db_root, '{}.pkl'.format(name))
            with open(db_fn, 'rb') as f:
                db = pickle.load(f)[mode]

            # setting fullpath for images
            self.images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

        elif name.startswith('gl'):
            # setting up paths
            data_root = get_data_root()
            db_root = os.path.join(data_root, 'train', name)

            self.ims_root = os.path.join(db_root, 'jpg')

            # loading db
            db_fn = os.path.join(db_root, 'db_{}.pkl'.format(name))
            with open(db_fn, 'rb') as f:
                db = pickle.load(f)[mode]

            # setting fullpath for images
            self.images = [os.path.join(self.ims_root, db['cids'][i]+'.jpg') for i in range(len(db['cids']))]

        else:
            raise(RuntimeError("Unknown dataset name!"))

        # initializing tuples dataset
        self.name = name
        self.mode = mode
        self.imsize = imsize
        self.clusters = db['cluster']
        self.qpool = db['qidxs']
        self.ppool = db['pidxs']
        self.bbxs = db['bbxs']

        if mode == 'train':
            print('__'*50)
            print('Dataset:', name)
            print('__'*50)
        print('__'*50)
        print('Total number of', mode, 'samples: ', len(self.qpool))
        print('__'*50)

        ## If we want to keep only unique q-p pairs
        ## However, ordering of pairs will change, although that is not important
        # qpidxs = list(set([(self.qidxs[i], self.pidxs[i]) for i in range(len(self.qidxs))]))
        # self.qidxs = [qpidxs[i][0] for i in range(len(qpidxs))]
        # self.pidxs = [qpidxs[i][1] for i in range(len(qpidxs))]

        # size of training subset for an epoch
        self.nnum = nnum
        self.qsize = min(qsize, len(self.qpool))
        self.poolsize = min(poolsize, len(self.images))
        self.qidxs = None
        self.pidxs = None
        self.nidxs = None

        self.transform = transform
        self.loader = loader

        self.print_freq = 10

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            images tuple (q,p,n1,...,nN): Loaded train/val tuple at index of self.qidxs
        """
        if self.__len__() == 0:
            raise(RuntimeError("List qidxs is empty. Run ``dataset.create_epoch_tuples(net)`` method to create subset for train/val!"))

        output = []
        output_tensor = torch.empty(0, 3, self.imsize, self.imsize)
        # query image
        output.append(self.loader(self.images[self.qidxs[index]]))

        # positive image
        output.append(self.loader(self.images[self.pidxs[index]]))

        # negative images
        for i in range(len(self.nidxs[index])):
            output.append(self.loader(self.images[self.nidxs[index][i]]))

        if self.imsize is not None:
            crop_params = [transforms.RandomCrop.get_params(rgb, [min(rgb.size)]*2) for rgb in output]
            output = [imcrop(rgb, crop_params[i]) for i, rgb in enumerate(output)]
            output = [imresize(img, self.imsize) for img in output]

        if self.transform is not None:
            for i in range(len(output)):
                output_tensor = torch.cat([output_tensor, self.transform(output[i]).unsqueeze_(0)], dim=0)

        target = torch.Tensor([-1, 1] + [0]*len(self.nidxs[index]))

        return output_tensor, target

    def __len__(self):
        return self.qsize

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Name and mode: {} {}\n'.format(self.name, self.mode)
        fmt_str += '    Number of images: {}\n'.format(len(self.images))
        fmt_str += '    Number of training tuples: {}\n'.format(len(self.qpool))
        fmt_str += '    Number of negatives per tuple: {}\n'.format(self.nnum)
        fmt_str += '    Number of tuples processed in an epoch: {}\n'.format(self.qsize)
        fmt_str += '    Pool size for negative remining: {}\n'.format(self.poolsize)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def create_epoch_tuples(self, net):

        print('>> Creating tuples for an epoch of {}-{}...'.format(self.name, self.mode))
        print(">>>> used network: ")
        print(net.meta_repr())

        ## ------------------------
        ## SELECTING POSITIVE PAIRS
        ## ------------------------

        # draw qsize random queries for tuples
        idxs2qpool = torch.randperm(len(self.qpool))[:self.qsize]
        self.qidxs = [self.qpool[i] for i in idxs2qpool]
        self.pidxs = []
        for idx in idxs2qpool:
            self.pidxs += random.sample(self.ppool[idx], 1)
        ## ------------------------
        ## SELECTING NEGATIVE PAIRS
        ## ------------------------

        # if nnum = 0 create dummy nidxs
        # useful when only positives used for training
        if self.nnum == 0:
            self.nidxs = [[] for _ in range(len(self.qidxs))]
            return 0

        # draw poolsize random images for pool of negatives images
        idxs2images = torch.randperm(len(self.images))[:self.poolsize]

        # prepare network
        net.mode = 'test'
        net = nn.DataParallel(net)
        net.cuda()
        net.eval()

        batch_size = 50
        # no gradients computed, to reduce memory and increase speed
        with torch.no_grad():
            # QUERIES
            print('>> Extracting descriptors for query images...')
            # prepare query loader
            loader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=[self.images[i] for i in self.qidxs], imsize=self.imsize // 3, transform=self.transform),
                batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True
            )
            # extract query vectors
            qvecs = torch.zeros(net.module.meta['outputdim'], len(self.qidxs)).cuda()
            counter = 0
            with tqdm(total=len(self.qidxs)) as pbar:
                for i, _input in enumerate(loader):
                    batch_size_inner = _input.shape[0]
                    qvecs[:, i * batch_size_inner : ((i+1) * batch_size_inner)] = net(_input.cuda()).data.squeeze().permute(1, 0)
                    counter += batch_size_inner
                    if (counter) % self.print_freq < 1 or (counter) == len(self.qidxs):
                        print('\r>>>> {}/{} done...'.format(counter, len(self.qidxs)), end='')
                    time.sleep(0.01)
                    pbar.update(batch_size_inner)

            print('>> Extracting descriptors for negative pool...')
            # prepare negative pool data loader
            loader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=[self.images[i] for i in idxs2images], imsize=self.imsize // 3, transform=self.transform),
                batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True
            )
            # extract negative pool vectors
            poolvecs = torch.zeros(net.module.meta['outputdim'], len(idxs2images)).cuda()
            counter = 0
            with tqdm(total=len(idxs2images)) as pbar:
                for i, _input in enumerate(loader):
                    batch_size_inner = _input.shape[0]
                    poolvecs[:, i * batch_size_inner : ((i+1) * batch_size_inner)] = net(_input.cuda()).data.squeeze().permute(1, 0)
                    counter += batch_size_inner
                    if (counter) % self.print_freq < 1 or (counter) == len(idxs2images):
                        print('\r>>>> {}/{} done...'.format(counter, len(idxs2images)), end='')
                    time.sleep(0.01)
                    pbar.update(batch_size_inner)

            print('>> Searching for hard negatives...')
            # compute dot product scores and ranks on GPU
            scores = torch.mm(poolvecs.t(), qvecs)
            scores, ranks = torch.sort(scores, dim=0, descending=True)
            avg_ndist = torch.tensor(0).float().cuda()      # for statistics
            n_ndist = torch.tensor(0).float().cuda()  # for statistics
            # selection of negative examples
            self.nidxs = []
            for q in range(len(self.qidxs)):
                # do not use query cluster,
                # those images are potentially positive
                qcluster = self.clusters[self.qidxs[q]]
                clusters = [qcluster]
                nidxs = []
                r = 0
####                counter = 0
                while len(nidxs) < self.nnum:
                    potential = idxs2images[ranks[r, q]]
                    # take at most one image from the same cluster
                    if not self.clusters[potential] in clusters:
                        neg_dist = torch.pow(qvecs[:,q]-poolvecs[:,ranks[r, q]]+1e-6, 2).sum(dim=0).sqrt() 
####                        if neg_dist > 1:
                        nidxs.append(potential)
                        clusters.append(self.clusters[potential])
                        #avg_ndist += torch.pow(qvecs[:,q]-poolvecs[:,ranks[r, q]]+1e-6, 2).sum(dim=0).sqrt()
                        n_ndist += 1
                        avg_ndist += neg_dist
####                        counter += 1
                    r += 1
                self.nidxs.append(nidxs)
            print('>>>> Average negative l2-distance: {:.2f}'.format(avg_ndist/n_ndist))
            print('>>>> Done')

        del qvecs, poolvecs, _input
        torch.cuda.empty_cache()

        return (avg_ndist/n_ndist).item()  # return average negative l2-distance
