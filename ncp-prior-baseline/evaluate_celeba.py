import sys
import os

def set_gpu(gpu):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import argparse
import json
import random
import shutil
import copy
import logging
import datetime
import pickle
import itertools
import time
import math
from shutil import copyfile

import numpy as np

import torch
from torch import cuda
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

from nn import BaseDecoder, BaseEncoder
from torch.utils.tensorboard import SummaryWriter
from fid.fid_score import compute_statistics_of_generator, load_statistics, calculate_frechet_distance, save_statistics
from fid.inception import InceptionV3
import utils
import matplotlib.pyplot as plt
from train_celeba import Encoder, Decoder
from sklearn import mixture
from torch.utils import data
from dataset import ImageListDataset



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='training mode')
    parser.add_argument('--expdir', type=str, required=True, help='path to dataset')
    parser.add_argument('--dataset', default='celeba64',  help='SVHN|cifar10 | celeba64 | lsun | imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', default='./data/celeba/', help='path to dataset')
    parser.add_argument('--fid_dir', default='./fid_dir', help='path to fid')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--data_to_0_1', type=bool, default=False, help='Load data in [0, 1] range')
    
    # ------------ From RAE's config, config_id=20 --------------
    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--niter', type=int, default=70, help='no of epochs')
    parser.add_argument('--log_fid_with_smpls', type=int, default=1000, help='no of samples for FID calculation in intermediate epochs')
    parser.add_argument('--num_last_epoch_fid_samples', type=int, default=10000, help='no of smaples for FID in the last epoch')

    parser.add_argument('--exp_name', type=str, default="RAE-L2", help='name of exp')
    parser.add_argument('--recon_loss_type', type=str, default='l2', help='Type of reocn loss')
    parser.add_argument('--spec_norm_on_dec_only', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default='CELEBA_WAE_PAPER_MAN_EMB_SZIE', help='model name')
    parser.add_argument('--kernel_size', type=int, default=None, help='no of epochs')
    parser.add_argument('--num_filters', type=int, default=128, help='no of conv filters for encoder & decoder')
    parser.add_argument('--bottleneck_factor', type=int, default=64, help='latent embedding size')
    parser.add_argument('--gen_reg_type', type=str, default='l2', help='no of epochs')
    parser.add_argument('--gen_reg_weight', type=float, default=1e-7, help='no of epochs')
    parser.add_argument('--embedding_weight', type=float, default=1e-4, help='weight of embedding loss')
    parser.add_argument('--cycle_emd_loss_weight', type=bool, default=False, help='no of epochs')
    parser.add_argument('--include_batch_norm', type=bool, default=True, help='include batch norm in encoder and decoder')
    parser.add_argument('--n_components', type=int, default=10, help='no of componnets for Gaussian mixture for ex-post density estimation')
    # -----------------------------------------------------------
    parser.add_argument('--lrG', type=float, default=0.001, help='learning rate for G, default=0.0002')
    parser.add_argument('--lrI', type=float, default=0.001, help='learning rate for I, default=0.0002')
    parser.add_argument('--is_grad_clampG', type=bool, default=False, help='whether doing the gradient clamp for G')
    parser.add_argument('--max_normG', type=float, default=100, help='max norm allowed for G')
    parser.add_argument('--is_grad_clampI', type=bool, default=False, help='whether doing the gradient clamp for I')
    parser.add_argument('--max_normI', type=float, default=100, help='max norm allowed for I')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netI', default='', help="path to netI (to continue training)")


    parser.add_argument('--visIter', default=1, type=int, help='number of epochs we need to visualize')
    parser.add_argument('--plotIter', default=1, type=int, help='number of epochs we need to visualize')
    parser.add_argument('--evalIter', default=1, type=int,  help='number of epochs we need to evaluate')
    parser.add_argument('--saveIter', default=10, type=int, help='number of epochs we need to save the model')
    parser.add_argument('--diagIter', default=1, type=int, help='number of epochs we need to save the model')
    parser.add_argument('--n_printout', default=20, type=int, help='number of iters we need to print the stats')

    parser.add_argument('--outf', default='', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', default=42, type=int, help='42 is the answer to everything')
    parser.add_argument('--gpu', type=int, default=0, metavar='S', help='gpu id (default: 0)')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')

    opt = parser.parse_args()
    if torch.cuda.is_available() and not opt.cuda:
        opt.cuda = True

    return opt
    

def set_global_gpu_env(opt):
    torch.cuda.set_device(opt.gpu)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    print(" ")
    print("Using GPUs {}".format(opt.gpu))


def copy_source(file, output_dir):
    copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def setup_logging(output_dir):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger()
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)


def get_output_dir(exp_id, opt, fs_prefix='./'):
    t = str(opt.datetime) #datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if opt.mode == 'debug':
        output_dir = os.path.join(fs_prefix + 'output/' +  exp_id, 'debug_' + t)
    else:
        output_dir = os.path.join(fs_prefix + 'output/' + exp_id, t)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def set_seed(opt):

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    # np.random.seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
        torch.backends.cudnn.deterministic = True
        cudnn.benchmark = False



def infer(opt, test_loader, netI, netG, num_samples=10000, recon=True):
    input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize).cuda()
    to_range_0_1 = lambda x: (x + 1.) / 2.
    total = 0
    infered_zs = []
    recon_inputs = []
    for i, data in enumerate(test_loader):
        img = data
        img = img.cuda()
        batch_size = img.size(0)
        input.resize_as_(img).copy_(img)
        inputV = Variable(input)
        # inputV = to_range_0_1(inputV)

        with torch.no_grad():
            infer_z = netI(inputV)
            infered_zs.extend(infer_z.cpu())
            if recon:
                recon_input = netG(infer_z).cpu()
                recon_inputs.extend(recon_input)

        total += batch_size
        if total >= num_samples:
            break
    # infered_zs = torch.vstack(infered_zs)
    # recon_inputs = torch.vstack(recon_inputs)
    return infered_zs, recon_inputs



def get_fid(opt, sample_loader, test_loader, total_fid_samples):
    dims = 2048
    device = 'cuda'
    num_gpus = 1 
    num_sample_per_gpu = int(np.ceil(total_fid_samples / num_gpus))
    g = sample_loader

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], model_dir=opt.fid_dir).to(device)
    m, s = compute_statistics_of_generator(g, model, opt.batchSize, dims, device, max_samples=num_sample_per_gpu)

    # share m and s
    m = torch.from_numpy(m).cuda()
    s = torch.from_numpy(s).cuda()
    # take average across gpus
    opt.distributed=False  #! Warning: Need to handle this for large dataset
    utils.average_tensor(m, opt.distributed)
    utils.average_tensor(s, opt.distributed)

    # convert m, s
    m = m.cpu().numpy()
    s = s.cpu().numpy()

    # load precomputed m, s
    path = os.path.join(opt.fid_dir, opt.dataset + '.npz')

    if not os.path.exists(path):
        print(f'Computing fid stats of the {opt.dataset} data ...')
        m0, s0 = compute_statistics_of_generator(test_loader, model, opt.batchSize, dims, device, total_fid_samples)
        save_statistics(path, m0, s0)
        print('saved fid stats at %s' % path)
    else:
        m0, s0 = load_statistics(path)

    fid = calculate_frechet_distance(m0, s0, m, s)
    print(f'Note: Computed FID against {opt.dataset} .............')
    return fid

def get_mse(dataloader, netG, netI, imageSize, batchSize):
        input = torch.FloatTensor(batchSize, 3, imageSize, imageSize).cuda()

        total = 0
        batch_error = 0.0
        for i, data in enumerate(dataloader):
            img = data
            img = img.cuda()
            batch_size = img.size(0)
            input.resize_as_(img).copy_(img)
            inputV = Variable(input)
            # inputV = to_range_0_1(inputV)

            with torch.no_grad():
                infer_z = netI(inputV)
                recon_input = netG(infer_z).cpu()

            assert inputV.max() < 1.01 and inputV.min() > -1e-4
            assert recon_input.max() < 1.01 and recon_input.min() > -1e-4
            batch_error = batch_error + torch.sum((recon_input.data - inputV.cpu().data)**2)
            total = total + batch_size
          
        mse = batch_error.data.item() / total

        return mse



class GaussianMixtureSampler():
    '''
    train_data: numpy (N, dim)
    '''
    def __init__(self, opt, train_data, netG, n_components=10):
        self.opt = opt
        self.train_data = train_data
        self.n_components = n_components
        self.netG = netG

    def fit(self):
        print(f"Fitting {self.n_components} Mixture Gaussian ... ")
        gmm = mixture.GaussianMixture(n_components = self.n_components,
                                      covariance_type='full',
                                      max_iter=100,
                                      verbose=1,
                                      tol=1e-3)
        gmm.fit(self.train_data)                                                  
        self.gmm = gmm

    def sample(self, num_samples=10000):
        gen_samples = []
        gen_zs = []
        batch_size=100
        num_batches = num_samples // batch_size
        assert num_batches % batch_size == 0, f"num_samples should be integer multiple of {batch_size}"

        for i in range(num_batches):
            with torch.no_grad():
                z = torch.from_numpy(self.gmm.sample(batch_size)[0]).type(torch.float32).cuda()
                x = self.netG(z)
            gen_samples.extend(x.cpu())
            gen_zs.extend(z.cpu())

        return gen_samples, gen_zs

class NormalGaussianSampler():

    def __init__(self, opt, netG):
        self.opt = opt
        self.netG = netG

    def sample(self, num_samples=10000):
        gen_samples = []
        gen_zs = []
        batch_size=100
        num_batches = num_samples // batch_size
        assert num_batches % batch_size == 0, f"num_samples should be integer multiple of {batch_size}"

        for i in range(num_batches):
            with torch.no_grad():
                z = torch.randn(batch_size, opt.bottleneck_factor).cuda()
                x = self.netG(z)
            gen_samples.extend(x.cpu())
            gen_zs.extend(z.cpu())

        return gen_samples, gen_zs
    
def normalize_to_0_1(x):
    mx = torch.amax(x, dim=(1,2), keepdim=True)
    mn = torch.amin(x, dim=(1,2), keepdim=True)
    return (x- mn)/(mx - mn)


def main(opt):

    netI = Encoder(opt) 
    netG = Decoder(opt)
    assert opt.netI != ''
    assert opt.netG != ''

    ckpt = torch.load(opt.netI)
    netI.load_state_dict(ckpt['model_state'])
    if opt.cuda:
        netI.cuda()
    print(netI)

    ckpt = torch.load(opt.netG)
    netG.load_state_dict(ckpt['model_state'])
    if opt.cuda:
        netG.cuda()
    print(netG)

    netG.eval()
    netI.eval()

    from train_celeba import get_dataset
    train_dataloader, _ , test_dataloader = get_dataset(opt)
    NUM_GEN_SAMPLES = 10000

    outf_syn = opt.expdir + "/syn"
    out_f = open(opt.outf, 'w')
    

    #1. 
    # recon_fid
    # - InceptionV3 model for FID needs sampels in the range [0, 1]
    infered_test_zs, recon_test_inputs = infer(opt, test_dataloader, netI, netG, num_samples=NUM_GEN_SAMPLES)
    list_dataset = ImageListDataset(recon_test_inputs)
    recon_loader = torch.utils.data.DataLoader(list_dataset, batch_size=100, 
                                               shuffle=False, num_workers=int(opt.workers)
                                               )
    print("Computing FID of test reoconstruction samples ....")
    recon_fid = get_fid(opt, recon_loader, test_dataloader, len(infered_test_zs))
    recon_mse = get_mse(test_dataloader, netG, netI, opt.imageSize, 100)

    # 2.
    # sample fids
    # - Mixture trained on full train set. Sample num_samples points
    # - Calculate fid against val set
    infered_train_zs, _ = infer(opt, train_dataloader, netI, netG, num_samples=100000, recon=False)
    infered_train_zs = torch.vstack(infered_train_zs).numpy()
    # 2.1
    gmm10 =  GaussianMixtureSampler(opt, infered_train_zs, netG, n_components=10)
    gmm10.fit()
    gmm10_gen_samples, _ = gmm10.sample(num_samples=NUM_GEN_SAMPLES) #list
    gmm10_dataset = ImageListDataset(gmm10_gen_samples)
    list_loader = torch.utils.data.DataLoader(gmm10_dataset, batch_size=100,
                                               shuffle=False, num_workers=int(opt.workers))
    print("Computing FID of GMM10 samples against val set ....")
    gmm10_fid = get_fid(opt, list_loader, test_dataloader, len(gmm10_gen_samples))
    vutils.save_image(gmm10_gen_samples[:100], os.path.join(outf_syn, "gmm10_bestMSE_gen_samples.png"), normalize=True, nrow=10 )

    # 2.2
    normal_sampler = NormalGaussianSampler(opt, netG)
    normal_gen_samples, _ = normal_sampler.sample(num_samples=NUM_GEN_SAMPLES)
    normal_dataset = ImageListDataset(normal_gen_samples)
    list_loader = torch.utils.data.DataLoader(normal_dataset, batch_size=100,
                                                shuffle=False, num_workers=int(opt.workers))
    print("Computing FID of Normal prior generated samples against val set ....")
    normal_fid = get_fid(opt, list_loader, test_dataloader, len(normal_gen_samples))
    vutils.save_image(normal_gen_samples[:100], os.path.join(outf_syn, "normal_bestMSE_gen_samples.png"), normalize=True, nrow=10 )

    # 2.3 
    normal_gen_samples_ = [normalize_to_0_1(x) for x in normal_gen_samples]
    normal_dataset_ = ImageListDataset(normal_gen_samples_)
    list_loader = torch.utils.data.DataLoader(normal_dataset_, batch_size=100,
                                                shuffle=False, num_workers=int(opt.workers))
    print("Computing FID of Normal prior generated samples against val set ....")
    normal_fid_ = get_fid(opt, list_loader, test_dataloader, len(normal_gen_samples))
    vutils.save_image(normal_gen_samples_[:100], os.path.join(outf_syn, "normal_bestMSE_normalized_gen_samples.png"), normalize=True, nrow=10 )

    # Save the results
    out_f.write("Recon_FID:{}, Recon_MSE:{}, GMM10_sample_FID:{}, normal_sample_FID:{}, normal_normalized_sample_FID:{}".format(recon_fid, recon_mse, gmm10_fid, normal_fid, normal_fid_))
    out_f.close()
    print("Done!")




    

    



if __name__ == '__main__':
    opt = parse_args()
    set_global_gpu_env(opt)
    set_seed(opt)

    main(opt)

# FID: 
#   - imageList loader
#   - compute fid : confirm same pre-processing 
# mixture-sampler:
#   - gmm n-comp fit and sampler class 