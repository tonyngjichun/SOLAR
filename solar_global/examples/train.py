import argparse
import fnmatch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import shutil
import time
import math
import pickle
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms
import torchvision.models as models

from solar_global.networks.imageretrievalnet import init_network, extract_vectors
from solar_global.layers.loss import ContrastiveLoss, TripletLoss, SOSLoss
from solar_global.datasets.datahelpers import collate_tuples, cid2filename, unnormalise
from solar_global.datasets.traindataset import TuplesBatchedDataset
from solar_global.datasets.testdataset import configdataset
from solar_global.utils.download import download_test
from solar_global.utils.whiten import whitenlearn, whitenapply
from solar_global.utils.evaluate import compute_map_and_print
from solar_global.utils.general import get_data_root, htime
from solar_global.utils.plots import *


warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore",  category=UserWarning)
warnings.filterwarnings("ignore",  category=FutureWarning)

training_dataset_names = ['retrieval-SfM-120k', 'gl18']
test_datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k', 'megadepth']
test_whiten_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
pool_names = ['mac', 'spoc', 'gem', 'gemmp']
loss_names = ['contrastive', 'triplet']
optimizer_names = ['sgd', 'adam']

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

# export directory, training and val datasets, test datasets
parser.add_argument('directory', metavar='EXPORT_DIR',
                    help='destination where trained network should be saved')
parser.add_argument('--training-dataset', '-d', metavar='DATASET', default='gl18', choices=training_dataset_names,
                    help='training dataset: ' + 
                        ' | '.join(training_dataset_names) +
                        ' (default: gl18)')
parser.add_argument('--base-path', '-bp', dest='base_path', default='./data/train/',
                    help='relative (or absolute) path to the training datasets.')
parser.add_argument('--no-val', dest='val', action='store_false',
                    help='do not run validation')
parser.add_argument('--test-datasets', '-td', metavar='DATASETS', default='roxford5k,rparis6k',
                    help='comma separated list of test datasets: ' + 
                        ' | '.join(test_datasets_names) + 
                        ' (default: roxford5k,rparis6k)')
parser.add_argument('--test-whiten', metavar='DATASET', default='', choices=test_whiten_names,
                    help='dataset used to learn whitening for testing: ' + 
                        ' | '.join(test_whiten_names) + 
                        ' (default: None)')
parser.add_argument('--test-freq', default=1, type=int, metavar='N', 
                    help='run test evaluation every N epochs (default: 1)')

# network architecture and initialization options
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet101', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet101)')
parser.add_argument('--pool', '-pl', metavar='POOL', default='gem', choices=pool_names,
                    help='pooling options: ' +
                        ' | '.join(pool_names) +
                        ' (default: gem)')
parser.add_argument('--p', type=float, default=3.,
                    help='power for pooling')
parser.add_argument('--local-whitening', '-lw', dest='local_whitening', action='store_true',
                    help='train model with learnable local whitening (linear layer) before the pooling')
parser.add_argument('--regional', '-r', dest='regional', action='store_true',
                    help='train model with regional pooling using fixed grid')
parser.add_argument('--whitening', '-w', dest='whitening', action='store_true',
                    help='train model with learnable whitening (linear layer) after the pooling')
parser.add_argument('--not-pretrained', dest='pretrained', action='store_false',
                    help='initialize model with random weights (default: pretrained on imagenet)')
parser.add_argument('--pretrained-type', dest='pretrained_type', type=str, default='gl18',
                    help='initialize model with custom pretrained weights (default: pretrained on imagenet)')
parser.add_argument('--loss', '-l', metavar='LOSS', default='contrastive',
                    choices=loss_names,
                    help='training loss options: ' +
                        ' | '.join(loss_names) +
                        ' (default: contrastive)')
parser.add_argument('--loss-margin', '-lm', metavar='LM', default=0.7, type=float,
                    help='loss margin: (default: 0.7)')
parser.add_argument('--lambda', '-lb', dest='_lambda', default=5, type=float,
                    help='loss term weight: (default: 5)')
parser.add_argument('--soa', dest='soa', action='store_true',
                    help='use non-local blocks for second-order attention (SOA)')
parser.add_argument('--soa-layers', dest='soa_layers', type=str, default='45',
                    help='config which layers of SOAs to include')
parser.add_argument('--unfreeze-last', '-ul', dest='unfreeze_last', action='store_true',
                    help='toggle whether to freeze weights of last layer')
parser.add_argument('--sos', dest='sos', action='store_true',
                    help='toggle whether to use second-order similarity loss')


# train/val options specific for image retrieval learning
parser.add_argument('--image-size', default=1024, type=int, metavar='N',
                    help='maximum size of longer image side used for training (default: 1024)')
parser.add_argument('--neg-num', '-nn', default=5, type=int, metavar='N',
                    help='number of negative image per train/val tuple (default: 5)')
parser.add_argument('--query-size', '-qs', default=2000, type=int, metavar='N',
                    help='number of queries randomly drawn per one train epoch (default: 2000)')
parser.add_argument('--pool-size', '-ps', default=20000, type=int, metavar='N',
                    help='size of the pool for hard negative mining (default: 20000)')
parser.add_argument('--val_query-size', '-vqs', default=400, type=int, metavar='N',
                    help='number of queries randomly drawn per one train epoch (default: 2000)')
parser.add_argument('--val_pool-size', '-vps', default=4000, type=int, metavar='N',
                    help='size of the pool for hard negative mining (default: 20000)')


# standard train/val options
parser.add_argument('--gpu-id', '-g', default='0', dest='gpu_id', metavar='N',
                    help='gpu id used for training (default: 0)')
parser.add_argument('--workers', '-j', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--batch-size', '-b', default=5, type=int, metavar='N', 
                    help='number of (q,p,n1,...,nN) tuples in a mini-batch (default: 5)')
parser.add_argument('--update-every', '-u', default=1, type=int, metavar='N',
                    help='update model weights every N batches, used to handle really large batches, ' + 
                        'batch_size effectively becomes update_every x batch_size (default: 1)')
parser.add_argument('--flatten-desc', dest='flatten_desc', action='store_true',
                    help='flatten descriptors to train in parallel, only applicable to small to moderate image/batch size')
parser.add_argument('--optimizer', '-o', metavar='OPTIMIZER', default='adam',
                    choices=optimizer_names,
                    help='optimizer options: ' +
                        ' | '.join(optimizer_names) +
                        ' (default: adam)')
parser.add_argument('--lr', '--learning-rate', default=1e-6, type=float,
                    metavar='LR', help='initial learning rate (default: 1e-6)')
parser.add_argument('-ld', '--lr-decay', dest='lr_decay', default=1e-2, type=float,
                    help='decay rate for learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='FILENAME',
                    help='name of the latest checkpoint (default: None)')


min_loss = float('inf')

def main():
    global args, min_loss, device
    args = parser.parse_args()

    # manually check if there are unknown test datasets
    for dataset in args.test_datasets.split(','):
        if dataset not in test_datasets_names:
            raise ValueError('Unsupported or unknown test dataset: {}!'.format(dataset))

    # check if test dataset are downloaded
    # and download if they are not
    download_test(get_data_root())

    # create export dir if it doesnt exist
    directory = "{}".format(args.training_dataset)
    directory += "_{}".format(args.arch)
    directory += "_{}".format(args.pool) + str(args.p)
    if args.local_whitening:
        directory += "_lwhiten"
    if args.regional:
        directory += "_r"
    if args.whitening:
        directory += "_whiten"
    if not args.pretrained:
        directory += "_notpretrained"
    if args.soa:
        directory += "_soa_"
        directory += args.soa_layers
        directory = os.path.join('second_order_attn', directory)
    if args.unfreeze_last:
        directory += "_unfreeze_last"
    if args.sos:
        directory += "_SOS_lambda{:.2f}".format(args._lambda)
    directory += "_{}".format(args.pretrained_type)
    directory += "_{}_m{:.2f}".format(args.loss, args.loss_margin)
    directory += "_{}_lr{:.1e}_lrd{:.1e}_wd{:.1e}".format(args.optimizer, args.lr, args.lr_decay, args.weight_decay)
    directory += "_nnum{}_qsize{}_psize{}".format(args.neg_num, args.query_size, args.pool_size)
    directory += "_bsize{}_uevery{}_imsize{}".format(args.batch_size, args.update_every, args.image_size)

    device = torch.device('cuda:'+ args.gpu_id)

    args.directory = os.path.join(args.directory, directory)
    print(">> Creating directory if it does not exist:\n>> '{}'".format(args.directory))
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    # set random seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    # initialize model
    if args.pretrained:
        print(">> Using pre-trained model '{}'".format(args.arch))
    else:
        print(">> Using model from scratch (random weights) '{}'".format(args.arch))
    model_params = {}
    model_params['architecture'] = args.arch
    model_params['pooling'] = args.pool
    model_params['p'] = args.p
    model_params['local_whitening'] = args.local_whitening
    model_params['regional'] = args.regional
    model_params['whitening'] = args.whitening
    # model_params['mean'] = ...  # will use default
    # model_params['std'] = ...  # will use default
    model_params['pretrained'] = args.pretrained
    model_params['pretrained_type'] = args.pretrained_type
    model_params['flatten_desc'] = args.flatten_desc
    model_params['soa'] = args.soa
    model_params['soa_layers'] = args.soa_layers

    model = init_network(model_params)
    
    # move network to gpu
    model.to(device)
    # define loss function (criterion) and optimizer
    if args.loss == 'contrastive':
        criterion = ContrastiveLoss(margin=args.loss_margin).to(device)
    elif args.loss == 'triplet':
        criterion = TripletLoss(margin=args.loss_margin).to(device)
    else:
        raise(RuntimeError("Loss {} not available!".format(args.loss)))

    if args.sos:
        criterionB = SOSLoss().to(device)
    else:
        criterionB = None

    # parameters split into features, pool, whitening 
    # IMPORTANT: no weight decay for pooling parameter p in GeM or regional-GeM
    parameters = []
    # add feature parameters
    if args.soa:
        for p in model.features.conv1.parameters():
            p.requires_grad = False
        for p in model.features.conv2_x.parameters():
            p.requires_grad = False
        for p in model.features.conv3_x.parameters():
            p.requires_grad = False
        for p in model.features.conv4_x.parameters():
            p.requires_grad = False
        if args.unfreeze_last:
            parameters.append({'params': model.features.conv5_x.parameters(), 'lr': args.lr * 0.0}) #, 'weight_decay': 0})
        if '4' in args.soa_layers:
            parameters.append({'params': model.features.soa4.parameters()})
        if '5' in args.soa_layers:
            parameters.append({'params': model.features.soa5.parameters()})
    else:
        parameters.append({'params': model.features.parameters()})
    # add local whitening if exists
    if model.lwhiten is not None:
        parameters.append({'params': model.lwhiten.parameters()})
    # add pooling parameters (or regional whitening which is part of the pooling layer!)
    if not args.regional:
        # global, only pooling parameter p weight decay should be 0
        if args.pool == 'gem':
            parameters.append({'params': model.pool.parameters(), 'lr': args.lr*100, 'weight_decay': 0})
        elif args.pool == 'gemmp':
            parameters.append({'params': model.pool.parameters(), 'lr': args.lr*100, 'weight_decay': 0})
    else:
        # regional, pooling parameter p weight decay should be 0, 
        # and we want to add regional whitening if it is there
        if args.pool == 'gem':
            parameters.append({'params': model.pool.rpool.parameters(), 'lr': args.lr*1, 'weight_decay': 0})
        elif args.pool == 'gemmp':
            parameters.append({'params': model.pool.rpool.parameters(), 'lr': args.lr*100, 'weight_decay': 0})
        if model.pool.whiten is not None:
            parameters.append({'params': model.pool.whiten.parameters()})
    # add final whitening if exists
    if model.whiten is not None:
        parameters.append({'params': model.whiten.parameters()})
    # define optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, args.lr, weight_decay=args.weight_decay)

    # define learning rate decay schedule
    # TODO: maybe pass as argument in future implementation?
    exp_decay = math.exp(-args.lr_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)

    # optionally resume from a checkpoint
    start_epoch = 0
    checkpoint_exists = False
    if args.resume: 
        args.resume = os.path.join(args.directory, args.resume)
        if os.path.isfile(args.resume):
            # load checkpoint weights and update model and optimizer
            print(">> Loading checkpoint:\n>> '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            min_loss = checkpoint['min_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            # important not to forget scheduler updating
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay, last_epoch=checkpoint['epoch']-1)
            checkpoint_exists = True
        else:
            print(">> No checkpoint found at '{}'".format(args.resume))

    # Data loading code
    #if args.training_dataset.startswith('megadepth'):
    #    model.meta['mean'] += sum(mean for mean in model.meta['mean']) / 3
    #    model.meta['std'] += sum(std for std in model.meta['std']) / 3

    normalize = transforms.Normalize(mean=model.meta['mean'], std=model.meta['std'])
    transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    if args.training_dataset.startswith('gl18'):
        train_dataset = TuplesBatchedDataset(
            name=args.training_dataset,
            mode='train',
            imsize=args.image_size,
            nnum=args.neg_num,
            qsize=args.query_size,
            poolsize=args.pool_size,
            transform=transform
        )
    else:
        train_dataset = TuplesDataset(
            name=args.training_dataset,
            mode='train',
            imsize=args.image_size,
            nnum=args.neg_num,
            qsize=args.query_size,
            poolsize=args.pool_size,
            transform=transform
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None,
        drop_last=True #, collate_fn=collate_tuples
    )
    if args.val:
        if args.training_dataset.startswith('gl18'):
            val_dataset = TuplesBatchedDataset(
                name=args.training_dataset,
                mode='val',
                imsize=args.image_size,
                nnum=args.neg_num,
                qsize=args.val_query_size,
                poolsize=args.val_pool_size,
                transform=transform
            )
        else:
            val_dataset = TuplesDataset(
                name=args.training_dataset,
                mode='val',
                imsize=args.image_size,
                nnum=args.neg_num,
                qsize=args.val_query_size,
                poolsize=args.val_pool_size,
                transform=transform
            )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        drop_last=True
        )

    # set up tensorboad
    summary = tb_setup(args.directory, checkpoint_exists)
    ################################################### evaluate the network before starting
    # this might not be necessary?
#   test(args.test_datasets, model, start_epoch, summary)

    for epoch in range(start_epoch, args.epochs):

        # set manual seeds per epoch

        random.seed(epoch)
        np.random.seed(epoch)
        torch.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)

        loss = train(train_loader, model, criterion, criterionB, optimizer, epoch, summary)

        # evaluate on test datasets every test_freq epochs
        with torch.no_grad():
            if args.val:
                loss = validate(val_loader, model, criterion, criterionB, epoch, summary)
            if (epoch + 1) % args.test_freq == 0:
                # evaluate on validation set
                test(args.test_datasets, model, epoch + 1, summary)

        # remember best loss and save checkpoint
        is_best = loss < min_loss
        min_loss = min(loss, min_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'meta': model.meta,
            'state_dict': model.state_dict(),
            'min_loss': min_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.directory)

        # train for one epoch on train set
        # adjust learning rate for each epoch
        scheduler.step()
        # # debug printing to check if everything ok
        # lr_feat = optimizer.param_groups[0]['lr']
        # lr_pool = optimizer.param_groups[1]['lr']
        # print('>> Features lr: {:.2e}; Pooling lr: {:.2e}'.format(lr_feat, lr_pool))


def tb_setup(save_dir, checkpoint_exists):
    # Setup for tensorboard
    tb_save_dir = os.path.join(
                    save_dir,
                    'summary',
                    )
    if not os.path.exists(tb_save_dir):
        os.makedirs(tb_save_dir)
    
    if not checkpoint_exists:
        trash_list = os.listdir(tb_save_dir)
        for entry in trash_list:
            filename = os.path.join(tb_save_dir, entry)
            if fnmatch.fnmatch(entry, '*tfevents*'):
                os.remove(filename)

    summary = SummaryWriter(log_dir=tb_save_dir)

    return summary

def train(train_loader, model, criterion, criterionB, optimizer, epoch, summary):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # create tuples for training
    with torch.no_grad():
        avg_neg_distance = train_loader.dataset.create_epoch_tuples(model)

    # switch to train mode
    model.mode = 'train'
    model.train()
    model.apply(set_batchnorm_eval)

    model = nn.DataParallel(model)

    # zero out gradients
    optimizer.zero_grad()

    end = time.time()
    for i, (_input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        _input = _input.to(device).float()

        nq = _input.shape[0] # number of training tuples
        ni = _input.shape[1] # number of images per tuple
        num_ch = _input.shape[2]
        h, w = _input.shape[-2], _input.shape[-1]
        # if flatten desc, only goes through network once instead of nq times
        _input = _input.view(-1, num_ch, h, w)

        if args.flatten_desc:
            ni = nq * ni
            nq = 1
            target = target.view(-1).unsqueeze_(0)

        h, w = _input.shape[-2], _input.shape[-1]   
        _input = _input.view(nq, ni, num_ch, h, w)
        
        for q in range(nq):
            # assign variable to this minibatch
            _input_q = _input[q]
            
            #output, attn_m2, attn_m1, fmap = model(_input_q)
            output = model(_input_q)
            try:
                p = model.module.pool.p
            except:
                p = torch.ones(1)

            output = output.permute(1, 0)
            loss_A = criterion(output, target[q].to(device))

            # second order regularisation
            if args.sos:
                loss_B = criterionB(output, target[q].to(device))
                loss = loss_A + args._lambda * loss_B
            else:
                loss = loss_A

            loss.backward()
            losses.update(loss.item() / ni)

        if (i + 1) % args.update_every == 0:
            # do one step for multiple batches
            # accumulated gradients are used
            optimizer.step()
            # zero out gradients so we can 
            # accumulate new ones over batches
            optimizer.zero_grad()
            #print('>> Train: [{0}][{1}/{2}]\t'
            #       'Weight update performed'.format(
            #        epoch+1, i+1, len(train_loader)))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or i == 0 or (i+1) == len(train_loader):
            print('\r>> Train: [{:3d}][{:3d}/{:3d}]\t'
                  'Time {batch_time.val:6.3f} ({batch_time.avg:6.3f})\t'
                  'Data {data_time.val:6.3f} ({data_time.avg:6.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch+1, i+1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses), end='')

            summary.add_scalar('train/loss_avg', losses.avg, global_step=epoch*args.query_size + i*args.batch_size)
            summary.add_scalar('train/loss_val', losses.val, global_step=epoch*args.query_size + i*args.batch_size)
            summary.add_scalar('train/1st-order_loss', loss_A.item(), global_step=epoch*args.query_size + i*args.batch_size)
            if args.sos:
                summary.add_scalar('train/2nd-order_loss', loss_B.item(), global_step=epoch*args.query_size + i*args.batch_size)
            summary.add_scalar('train/p', p.data, global_step=epoch*args.query_size + i*args.batch_size)

            step = epoch * len(train_loader) + i + 1
            plot_examples(unnormalise(_input_q), summary, 'train', step)
####            plot_self_attn(unnormalise(_input_q), attn_m2, attn_m1, summary, 'train', step)

    print('')
    model = model.module

    return losses.avg


def validate(val_loader, model, criterion, criterionB, epoch, summary):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # create tuples for validation
    avg_neg_distance = val_loader.dataset.create_epoch_tuples(model)

    # switch to evaluate mode
    model.mode = 'val'
    model.eval()
    model = nn.DataParallel(model)

    end = time.time()
    for i, (_input, target) in enumerate(val_loader):
        _input = _input.to(device)

        # _input: B * (2 + num_neg) * H * W
        nq = _input.shape[0] # number of training tuples
        ni = _input.shape[1] # number of images per tuple
        num_ch = _input.shape[2]
        h, w = _input.shape[-2], _input.shape[-1]

        _input = _input.view(-1, num_ch, h, w)

        output, fmap = model(_input)
        try:
            p = model.module.pool.p
        except:
            p = torch.ones(1)

        output = output.permute(1, 0)
        loss = criterion(output, target.view(-1).to(device))

        # second order regularisation
        if criterionB is not None:
            loss = loss + 5 * criterionB(output, target.view(-1).to(device))

        # record loss
        losses.update(loss.item()/nq, nq)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or i == 0 or (i+1) == len(val_loader):
            print('\r>> Val: [{:3d}][{:3d}/{:3d}]\t'
                  'Time {batch_time.val:6.3f} ({batch_time.avg:6.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch+1, i+1, len(val_loader), batch_time=batch_time, loss=losses), end='')

            summary.add_scalar('val/loss_avg', losses.avg, global_step=epoch*args.query_size + i*args.batch_size)
            summary.add_scalar('val/loss_val', losses.val, global_step=epoch*args.query_size + i*args.batch_size)

            summary.add_scalar('val/p', p.data, global_step=epoch*args.query_size + i*args.batch_size)

            step = epoch * len(val_loader) + i + 1
            #plot_attentions(unnormalise(_input), fmap, p.item(), summary, 'val', step)

    model = model.module

    return losses.avg

def test(datasets, net, epoch, summary):

    #ms = [1]
    ms = [1, 2**(1/2), 1/2**(1/2)]
    print('>> Evaluating network on test datasets...')

    # for testing we use image size of max 1024
    image_size = 1024

    # moving network to gpu and eval mode
    net.mode = 'test'
    net.to(device)
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

    # compute whitening
    if args.test_whiten:
        start = time.time()

        print('>> {}: Learning whitening...'.format(args.test_whiten))

        # loading db
        db_root = os.path.join(get_data_root(), 'train', args.test_whiten)
        ims_root = os.path.join(db_root, 'ims')
        db_fn = os.path.join(db_root, '{}-whiten.pkl'.format(args.test_whiten))
        with open(db_fn, 'rb') as f:
            db = pickle.load(f)
        images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

        # extract whitening vectors
        print('>> {}: Extracting...'.format(args.test_whiten))
        wvecs = extract_vectors(net, None, images, image_size, transform)   # implemented with torch.no_grad
        
        # learning whitening 
        print('>> {}: Learning...'.format(args.test_whiten))
        wvecs = wvecs.numpy()
        m, P = whitenlearn(wvecs, db['qidxs'], db['pidxs'])
        Lw = {'m': m, 'P': P}

        print('>> {}: elapsed time: {}'.format(args.test_whiten, htime(time.time()-start)))
    else:
        Lw = None

    # evaluate on test datasets
    datasets = args.test_datasets.split(',')
    for dataset in datasets: 
        start = time.time()

        print('>> {}: Extracting...'.format(dataset))

        # prepare config structure for the test dataset
        cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
        images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
        qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
        bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
        
        print('>> {}: database images...'.format(dataset))
        vecs = extract_vectors(net, images, image_size, transform, summary=summary, mode='test', ms=ms) # implemented with torch.no_grad
        
        print('>> {}: query images...'.format(dataset))
        qvecs = extract_vectors(net, qimages, image_size, transform, bbxs, summary=summary, mode='test', ms=ms)  # implemented with torch.no_grad

        print('>> {}: Evaluating...'.format(dataset))

        # convert to numpy
        vecs = vecs.numpy()
        qvecs = qvecs.numpy()

        # search, rank, and print
        scores = np.dot(vecs.T, qvecs)
        ranks = np.argsort(-scores, axis=0)
        compute_map_and_print(dataset, ranks, cfg['gnd'], summary=summary, epoch=epoch)

#       for protocol in ['hard']: #'easy', 'medium', 'hard']:
#           plot_ranks(qimages, images, ranks, cfg['gnd'], bbxs, summary, dataset, epoch, 20, protocol)
        if Lw is not None:
            # whiten the vectors
            vecs_lw  = whitenapply(vecs, Lw['m'], Lw['P'])
            qvecs_lw = whitenapply(qvecs, Lw['m'], Lw['P'])

            # search, rank, and print
            scores = np.dot(vecs_lw.T, qvecs_lw)
            ranks = np.argsort(-scores, axis=0)
            compute_map_and_print(dataset + ' + whiten', ranks, cfg['gnd'], summary=summary, epoch=epoch)

        print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))


def save_checkpoint(state, is_best, directory):
    filename = os.path.join(directory, 'model_epoch%d.pth.tar' % state['epoch'])
    torch.save(state, filename)
    if is_best:
        filename_best = os.path.join(directory, 'model_best.pth.tar')
        shutil.copyfile(filename, filename_best)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        # freeze running mean and std:
        # we do training one image at a time
        # so the statistics would not be per batch
        # hence we choose freezing (ie using imagenet statistics)
        m.eval()
        # # freeze parameters:
        # # in fact no need to freeze scale and bias
        # # they can be learned
        # # that is why next two lines are commented
        # for p in m.parameters():
            # p.requires_grad = False


if __name__ == '__main__':
    main()
