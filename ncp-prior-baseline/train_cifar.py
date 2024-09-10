# RAE's encoder and decoder model

import os
import random

from shutil import copyfile

import datetime, time
import logging
import sys

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torch.utils.data

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse

from sklearn.metrics import roc_auc_score
from data import IgnoreLabelDataset, ConstantDataset, UniformDataset, DTDDataset

from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from fid.fid_score import compute_statistics_of_generator, load_statistics, calculate_frechet_distance, save_statistics
from fid.inception import InceptionV3
import utils
from snis import NIS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='training mode')
    parser.add_argument('--datetime', required=True, help='datetime')
    # parser.add_argument('--datetime', type=str, default='debug_'+datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), help='datetime')

    parser.add_argument('--fid_dataset', default='cifar10',  help='SVHN|cifar10 | celeba64 | lsun | imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', default='./data/cifar/', help='path to dataset')
    parser.add_argument('--fid_dir', default='./fid_dir', help='path to fid')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--data_to_0_1', type=bool, default=True, help='Load data in [0, 1] range')

    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--niter', type=int, default=100, help='no of epochs')
    parser.add_argument('--log_fid_with_smpls', type=int, default=2000, help='no of samples for FID calculation in intermediate epochs')
    parser.add_argument('--num_last_epoch_fid_samples', type=int, default=10000, help='no of smaples for FID in the last epoch')

    parser.add_argument('--exp_name', type=str, default="STD-VAE", help='name of exp')
    parser.add_argument('--recon_loss_type', type=str, default='l2', help='Type of reocn loss')
    parser.add_argument('--spec_norm_on_dec_only', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default='CIFAR_SEED', help='model name')
    parser.add_argument('--kernel_size', type=int, default=None, help='no of epochs')
    parser.add_argument('--num_filters', type=int, default=128, help='no of conv filters for encoder & decoder')
    parser.add_argument('--bottleneck_factor', type=int, default=128, help='latent embedding size')
    parser.add_argument('--gen_reg_type', type=str, default='l2', help='no of epochs')
    parser.add_argument('--gen_reg_weight', type=float, default=1e-7, help='no of epochs')
    parser.add_argument('--embedding_weight', type=float, default=0.007, help='weight of embedding loss')
    parser.add_argument('--cycle_emd_loss_weight', type=bool, default=False, help='no of epochs')
    parser.add_argument('--include_batch_norm', type=bool, default=True, help='include batch norm in encoder and decoder')
    parser.add_argument('--n_components', type=int, default=10, help='no of componnets for Gaussian mixture for ex-post density estimation')
    # -----------------------------------------------------------
    parser.add_argument('--nez', type=int, default=1, help='size of the output of ebm')
    parser.add_argument('--ndf', type=int, default=200)
    parser.add_argument('--lrE', type=float, default=0.001, help='learning rate for E, default=0.0002')
    parser.add_argument('--lrG', type=float, default=0.001, help='learning rate for G, default=0.0002')
    parser.add_argument('--lrI', type=float, default=0.001, help='learning rate for I, default=0.0002')
    parser.add_argument('--is_grad_clampE', type=bool, default=False, help='whether doing the gradient clamp for E')
    parser.add_argument('--is_grad_clampG', type=bool, default=False, help='whether doing the gradient clamp for G')
    parser.add_argument('--max_normG', type=float, default=100, help='max norm allowed for G')
    parser.add_argument('--is_grad_clampI', type=bool, default=False, help='whether doing the gradient clamp for I')
    parser.add_argument('--max_normI', type=float, default=100, help='max norm allowed for I')
    parser.add_argument('--max_normE', type=float, default=100, help='max norm allowed for E')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netI', default='', help="path to netI (to continue training)")
    parser.add_argument('--netE', default='', help="path to netE (to continue training)")

    parser.add_argument('--visIter', default=1, type=int, help='number of epochs we need to visualize')
    parser.add_argument('--plotIter', default=1, type=int, help='number of epochs we need to visualize')
    parser.add_argument('--fidIter', default=5, type=int,  help='number of epochs we need to evaluate')
    parser.add_argument('--saveIter', default=10, type=int, help='number of epochs we need to save the model')
    parser.add_argument('--diagIter', default=1, type=int, help='number of epochs we need to save the model')
    parser.add_argument('--n_printout', default=20, type=int, help='number of iters we need to print the stats')

    parser.add_argument('--outf', default='', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', default=42, type=int, help='42 is the answer to everything')
    parser.add_argument('--gpu', type=int, default=1, metavar='S', help='gpu id (default: 0)')
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


def unnormalize(img):
    return img / 2.0 + 0.5



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


def get_dataset(opt):
    from data import ImageMTDataset
    import PIL

    train_cache = os.path.join(opt.dataroot,'train','celeba64_train_full.pkl')
    val_cache = os.path.join(opt.dataroot,'val','celeba64_val_full.pkl')

    dataset = ImageMTDataset(os.path.join(opt.dataroot,'train/train/'), train_cache,  num_images=None,
                                normalize_to_0_1 = opt.data_to_0_1, 
                                 transforms=transforms.Compose([
                                     PIL.Image.fromarray,
                                     transforms.CenterCrop(140), # To be consistent with RAE's pre-processing
                                     transforms.Resize(opt.imageSize),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #TODO: Is this part of RAE's pre-processing?
                                 ]) )
    test_dataset = ImageMTDataset(os.path.join(opt.dataroot,'val/val/'), val_cache, num_images=None,
                                  normalize_to_0_1 = opt.data_to_0_1, 
                                 transforms=transforms.Compose([
                                     PIL.Image.fromarray,
                                     transforms.CenterCrop(140),
                                     transforms.Resize(opt.imageSize),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ]) )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=opt.batchSize, num_workers=int(opt.workers))
    return dataloader, dataset, test_dataloader

################################### AUROC ##############################
def get_cifar_dataset(opt):

    class CIFAR10NoLabels(datasets.CIFAR10):
        def __getitem__(self, index):
            img, _ = super(CIFAR10NoLabels, self).__getitem__(index)
            return img


    transform = transforms.Compose([
        transforms.Resize(opt.imageSize),  # Resize images to desired size
        transforms.ToTensor(),             # Convert PIL images to tensors
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
    ])

    # CIFAR-10 dataset
    train_dataset = CIFAR10NoLabels(root='./vae-baseline/data//mycifar10', train=True, download=True, transform=transform)
    test_dataset = CIFAR10NoLabels(root='./vae-baseline/data/mycifar10', train=False, download=True, transform=transform)

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                                  shuffle=True, num_workers=int(opt.workers))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                                 shuffle=False, num_workers=int(opt.workers))

    return train_dataloader, train_dataset, test_dataloader

def get_ood_dataset(opt, cifar_dataset):
    length = len(cifar_dataset)

    if opt.target_dataset == 'svhn':
        dataset = IgnoreLabelDataset(datasets.SVHN(root='data/svhn/', download=True, split='test',
                                transform=transforms.Compose([
                                    transforms.Resize(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])))
    elif opt.target_dataset == 'cifar10_train':

        dataset = IgnoreLabelDataset(datasets.CIFAR10(root='data/cifar/', train=True,
                                                      transform=transforms.Compose([
                                                          transforms.Resize(opt.imageSize),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                      ])))
    elif opt.target_dataset == 'random':


        dataset = UniformDataset(opt.imageSize, 3, length)
    elif opt.target_dataset == 'constant':


        dataset = ConstantDataset(opt.imageSize, 3, length)

    elif opt.target_dataset == 'texture':
        dataset = DTDDataset('data/dtd/images/',transform=transforms.Compose([
                                                          transforms.Resize(opt.imageSize),
                                                          transforms.CenterCrop(opt.imageSize),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                      ]))
    else:
        raise ValueError('no dataset')


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    return dataloader, dataset


##################################


def weights_init(m):
    """
    xavier initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_()
        # nn.init.xavier_normal_(m.weight)
        m.weight.data.normal_(0.0, 0.02)
        #m.weight.data.fill_(0.001)
    #elif classname.find('Linear') != -1:
        #xavier_uniform(m.weight)
    elif classname.find('BatchNorm') != -1:
        # m.weight.data.normal_(1.0, 0.02)
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)


class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.input_dim = (3, 32, 32)
        self.latent_dim = opt.bottleneck_factor
        self.n_channels = 3
        self.bn_mom = 0.01
        self.bn_eps = 1e-3

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Linear(opt.bottleneck_factor, 1024 * 8 * 8)))

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(1024, 512, 4, 2, padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(512, momentum=self.bn_mom, eps=self.bn_eps),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, padding=0, output_padding=1, bias=False),
                nn.BatchNorm2d(256, momentum=self.bn_mom, eps=self.bn_eps),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(256, self.n_channels, 4, 1, padding=2), nn.Sigmoid()
            )
        )  # Sigmoid at the end --> Transform the inputs to [0, 1]

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z): #: torch.Tensor, output_layer_levels: List[int] = None):

        max_depth = self.depth
        out = z

        for i in range(max_depth):
            out = self.layers[i](out)

            if i == 0:
                out = out.reshape(z.shape[0], 1024, 8, 8)
        return out
    
class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()

        self.input_dim = (3, 32, 32)
        self.latent_dim = opt.bottleneck_factor
        self.n_channels = 3

        layers = nn.ModuleList()
        self.bn_mom = 0.01
        self.bn_eps = 1e-3
        #! Kernels should be all (5,5)

        layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels, 128, 4, 2, padding=1),  # use kernel=5, pad=2?
                nn.BatchNorm2d(128, momentum=self.bn_mom, eps=self.bn_eps),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, padding=1), nn.BatchNorm2d(256, momentum=self.bn_mom, eps=self.bn_eps), 
                nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, padding=1), nn.BatchNorm2d(512, momentum=self.bn_mom, eps=self.bn_eps), 
                nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(512, 1024, 4, 2, padding=1), nn.BatchNorm2d(1024, momentum=self.bn_mom, eps=self.bn_eps), 
                nn.ReLU()
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(1024 * 2 * 2, opt.bottleneck_factor)
        self.log_sigma = nn.Linear(1024 * 2 * 2, opt.bottleneck_factor)
        self.tanh = nn.Tanh()

    # def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
    def forward(self, x):
        # output = ModelOutput() 

        # x = 2 * x - 1

        max_depth = self.depth

        out = x

        for i in range(max_depth):
            out = self.layers[i](out)
        mu = self.embedding(out.reshape(x.shape[0], -1))
        log_sigma = 5 * self.tanh(self.log_sigma(out.reshape(x.shape[0], -1)))
        # log_sigma = self.log_sigma(out.reshape(x.shape[0], -1))
        return mu, log_sigma


class _netE(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        apply_sn = lambda x: x #sn if args.e_sn else lambda x: x

        f = nn.LeakyReLU() #get_activation(args.e_activation, args)

        self.ebm = nn.Sequential(
            apply_sn(nn.Linear(args.bottleneck_factor, args.ndf)),
            f,

            apply_sn(nn.Linear(args.ndf, args.ndf)),
            f,

            apply_sn(nn.Linear(args.ndf, args.nez))
        )

    def forward(self, z):
        return self.ebm(z.squeeze()).view(-1, self.args.nez, 1, 1)


def compute_fid_from_datalaoder(sample_dataloader, ref_dataloader, 
                fid_dataset, fid_dir, 
                batchSize=100, 
                total_fid_samples=10000):

    '''
    model: decoder of vae
    dl: dataloader for the sampled images
    images should be in range [0, 1]
    '''
    dims = 2048
    device = 'cuda'
    num_gpus = 1 
    num_sample_per_gpu = int(np.ceil(total_fid_samples / num_gpus))
    g = sample_dataloader 
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], model_dir=fid_dir).to(device)
    m, s = compute_statistics_of_generator(g, model, batchSize, dims, device, max_samples=num_sample_per_gpu)
# share m and s
    m = torch.from_numpy(m).cuda()
    s = torch.from_numpy(s).cuda()
    # take average across gpus
    distributed=False  
    utils.average_tensor(m, distributed)
    utils.average_tensor(s, distributed)

    # convert m, s
    m = m.cpu().numpy()
    s = s.cpu().numpy()

    # load precomputed m, s
    path = os.path.join(fid_dir, fid_dataset + '.npz')

    if not os.path.exists(path):
        print('Computing fid stats of the train data ...')
        m0, s0 = compute_statistics_of_generator(ref_dataloader, model, batchSize, dims, device, total_fid_samples)
        save_statistics(path, m0, s0)
        print('saved fid stats at %s' % path)
    else:
        m0, s0 = load_statistics(path)

    fid = calculate_frechet_distance(m0, s0, m, s)
    return fid


def train(opt, output_dir):

    '''
    first define necessary functions
    '''

    # define energy and diagonal Normal log-probability
    def compute_energy(disc_score):
        # disc score: batch nez 1 1
        # return shape: [batch]
        if opt.energy_form == 'tanh':
            energy = F.tanh(-disc_score.squeeze())
        elif opt.energy_form == 'sigmoid':
            energy = F.sigmoid(disc_score.squeeze())
        elif opt.energy_form == 'identity':
            energy = disc_score.squeeze()
        elif opt.energy_form == 'softplus':
            # energy = F.softplus(-torch.sum(disc_score.squeeze(), dim=1))
            energy = F.softplus(disc_score.squeeze())
        return energy

    def diag_normal_NLL(z, z_mu, z_log_sigma):
        # z_log_sigma: log variance
        # sum over the dimension, but leave the batch dim unchanged
        # define the Negative Log Probablity of Normal which has diagonal cov
        # input:[batch nz, 1, 1] squeeze it to batch nz
        # return shape is [batch]
        nll = 0.5 * torch.sum(z_log_sigma.squeeze(), dim=1) + \
              0.5 * torch.sum((torch.mul(z - z_mu, z - z_mu) / (1e-6 + torch.exp(z_log_sigma))).squeeze(), dim=1)
        return nll.squeeze()

    def diag_standard_normal_NLL(z):
        # get the negative log-likelihood of standard normal distribution
        nll = 0.5 * torch.sum((torch.mul(z, z)).squeeze(), dim=1)
        return nll.squeeze()
    
    def loss_kl_divergence(mu, log_sigma):
        # kl = 0.5 * torch.sum(torch.exp(2*log_sigma) + torch.square(mu) - 1. - 2*log_sigma, axis=1) # This line has bug since, log_sigma is actually log_var
        kl = 0.5 * torch.sum(torch.exp(log_sigma) + torch.square(mu) - 1. - log_sigma, dim=1)
        return kl
    
    def mse_from_tf(y_true, y_pred):
        '''
        y_true, y_pred = H,C,H,W
        '''
        loss = torch.mean(torch.square(y_true - y_pred), dim=1) # mean over the channel
        return loss 

        #return kl_loss

    def reparametrize(mu, log_sigma, is_train=True):
        if is_train:
            # std = log_sigma.exp_() # This line has potential bugs cause it changes the log_sigma
            std = torch.exp(log_sigma.mul(0.5))
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def getGradNorm(net):
        pNorm = torch.sqrt(sum(torch.sum(p ** 2) for p in net.parameters()))
        gradNorm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in net.parameters()))
        return pNorm, gradNorm

    def plot_stats():
        p_i = 1
        p_n = len(stat_1)

        f = plt.figure(figsize=(20, p_n * 5))

        def plot(stats, stats_i):
            nonlocal p_i
            for j, (k, v) in enumerate(stats.items()):
                plt.subplot(p_n, 1, p_i)
                plt.plot(stats_i, v)
                plt.ylabel(k)
                p_i += 1

        plot(stat_1, stat_1_i)

        f.savefig(os.path.join(output_dir, 'stat.pdf'), bbox_inches='tight')
        plt.close(f)
    
    def save_model(opt_dict, epoch, idfier=None):
        for key in opt_dict:
            save_dict = {
                'epoch': epoch,
                'model_state': opt_dict[key][0].state_dict(),
                'optimizer_state': opt_dict[key][1].state_dict()
            }
            if idfier is not None:
                torch.save(save_dict, '%s/%s_%s.pth' % (output_dir, key, idfier))
            else:
                torch.save(save_dict, '%s/%s_epoch_%d.pth' % (output_dir, key, epoch))

    def eval_model(opt, model_type='bestmse'):
        
        assert model_type == 'bestmse' or model_type == 'bestfid', "Invalid model_type argument! Must be in ['bestmse', 'bestfid']"
        netG = Decoder(opt)
        if opt.cuda:
            netG.cuda()
        netG_path = '%s/%s_%s.pth' % (output_dir, 'netG', model_type)
        ckpt = torch.load(netG_path)
        netG.load_state_dict(ckpt['model_state'])  

        netI = Encoder(opt)
        if opt.cuda:
            netI.cuda()
        netI_path = '%s/%s_%s.pth' % (output_dir, 'netI', model_type)
        ckpt = torch.load(netI_path)
        netI.load_state_dict(ckpt['model_state'])  

        with torch.no_grad():
            mse_val = mse_score(test_dataloader, netG, netI, opt.imageSize, 100, outf_recon)
            fid_dict = get_fid_vae(netI, netG, opt, 10000, recon_fid=True)
        
        return mse_val, fid_dict['gen_fid'], fid_dict['recon_fid']



    def eval_flag():
        netG.eval()
        netI.eval()
        # netE.eval()

    def train_flag():
        netG.train()
        netI.train()
        # netE.train()

    def create_generator_vae(netG, batch_size, num_total_samples):
        '''
        netG: generator/decoder of vae
        return
        ------
        output_img: Of the same shape as input to inference model of vae
        '''
        num_iters = int(np.ceil(num_total_samples / batch_size))
        assert num_total_samples % batch_size == 0, "num_total_samples must be an integer multiple of batch_size."
        for i in range(num_iters):
            with torch.no_grad():
                output_img = netG(new_noise())
            yield output_img.float()
    
    def create_recon_generator(opt, netI, netG, img_loader, batch_size, num_total_samples):
        num_iters = int(np.ceil(num_total_samples / batch_size))
        assert num_total_samples % batch_size == 0, "num_total_samples must be an integer multiple of batch_size."
        total = 0

        input = torch.FloatTensor(batch_size, 3, opt.imageSize, opt.imageSize).cuda()
        for i, data in enumerate(img_loader):
            img = data
            img = img.cuda()
            batch_size = img.size(0)
            total += batch_size
            if total >= num_total_samples:
                break
            input.resize_as_(img).copy_(img)
            inputV = Variable(input)
            with torch.no_grad():
                mu, _ = netI(inputV)
                out = netG(mu)
            yield out



    def get_fid_vae(netI, netG, opt, total_fid_samples, recon_fid=False):
        '''
        model: decoder of vae
        '''
        fid_dict = {'gen_fid':None, 'recon_fid': None}
        sample_generator = create_generator_vae(netG, 100, total_fid_samples)
        gen_fid = compute_fid_from_datalaoder(sample_generator, test_dataloader, opt.fid_dataset, opt.fid_dir, batchSize=100, total_fid_samples=total_fid_samples)
        fid_dict['gen_fid'] = gen_fid

        if recon_fid:
            recon_generator = create_recon_generator(opt, netI, netG, test_dataloader, 100, total_fid_samples)
            recon_fid =  compute_fid_from_datalaoder(recon_generator, test_dataloader, opt.fid_dataset, opt.fid_dir, batchSize=100, total_fid_samples=total_fid_samples)
            fid_dict['recon_fid'] = recon_fid
        return fid_dict

    def mse_score(dataloader,netG, netI, imageSize, batchSize, saveFolder):

        input = torch.FloatTensor(batchSize, 3, imageSize, imageSize).cuda()

        total = 0
        batch_error = 0.0
        for i, data in enumerate(dataloader, 0):
            img = data
            img = img.cuda()
            batch_size = img.size(0)
            input.resize_as_(img).copy_(img)
            inputV = Variable(input)

            with torch.no_grad():
                infer_z_mu_input, _ = netI(inputV)
                recon_input = netG(infer_z_mu_input).cpu()

            batch_error = batch_error + torch.sum((recon_input.data - inputV.cpu().data)**2)
            total = total + batch_size

            if i % 10 ==0:
                # get the grid representation for first batch (easy to examine)
                vutils.save_image(inputV.data, os.path.join(saveFolder, "step_{}_input_test.png".format(i)),
                                normalize=True, nrow=10)
                vutils.save_image(recon_input.data, os.path.join(saveFolder, "step_{}_recon_test.png".format(i)),
                                normalize=True, nrow=10)

                
        mse = batch_error.data.item() / total

        return mse



    '''
    setup auxiliaries
    '''
    output_subdirs = output_dir + opt.outf
    try:
        os.makedirs(output_subdirs)
    except OSError:
        pass

    outf_recon = output_subdirs + '/recon'
    outf_syn = output_subdirs + '/syn'
    outf_err = output_subdirs + '/curve'
    try:
        os.makedirs(outf_recon)
        os.makedirs(outf_syn)
        os.makedirs(outf_err)
    except OSError:
        pass
    ## open file for later use
    out_f = open("%s/results.txt" % output_dir, 'w')

    # Tensorbaord
    tb_path = "runs/"+ "".join(output_dir.split("output")[1:])
    tb_writer = SummaryWriter(tb_path)

    ## get constants
    nz = int(opt.bottleneck_factor)
    nc = 3

    dataloader, dataset_full, test_dataloader = get_cifar_dataset(opt)

    input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz).normal_()  # for visualize

    mse_loss = nn.MSELoss(size_average=False)
    bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

    new_noise = lambda: noise.resize_(opt.batchSize, nz).normal_()
    num_samples = 10000

    '''
    create networks
    '''
    netG = Decoder(opt)
    # netG.apply(weights_init)
    if opt.cuda:
        netG.cuda()
    
    optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lrG, weight_decay=0, betas=(0.9, 0.999))
    if opt.netG != '':
        ckpt = torch.load(opt.netG)
        netG.load_state_dict(ckpt['model_state'])
        optimizerG.load_state_dict(ckpt['optimizer_state'])
    print(netG)

    netI = Encoder(opt)
    # netI.apply(weights_init)
    if opt.cuda:
        netI.cuda()
    optimizerI = torch.optim.Adam(netI.parameters(), lr=opt.lrI, weight_decay=0, betas=(0.9, 0.999))
    if opt.netI != '':
        ckpt = torch.load(opt.netI)
        netI.load_state_dict(ckpt['model_state'])
        optimizerI.load_state_dict(ckpt['optimizer_state'])

    print(netI)

    # critic model
    netE = _netE(opt)
    # netE.apply(weights_init)
    if opt.cuda:
        netE.cuda()
    optimizerE = torch.optim.Adam(netE.parameters(), lr=opt.lrE, weight_decay=0, betas=(0.9, 0.999))

    if opt.netE != '':
        ckpt = torch.load(opt.netE)
        netE.load_state_dict(ckpt['model_state'])
        optimizerE.load_state_dict(ckpt['optimizer_state'])
    print(netE)

    if opt.netE != '':
        start_epoch = torch.load(opt.netE)['epoch'] + 1
    else:
        start_epoch = 0

    if opt.cuda:
        input = input.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
        mse_loss.cuda()
        bce_loss.cuda()

    fixed_noiseV = Variable(fixed_noise)

    lrE_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerE, factor=0.5, patience=5)


    '''
    define stats
    ----------
    stats_headings: keys for stats_values
    stats_values: batch wise statistics. Initilaizes every epoch
    stat_1: epoch wise statistics
    '''
    # TODO merge with code below, print to file and plot pdf
    stats_headings = [ #['epoch',          '{:>14}',  '{:>14d}'],
                      ['err(E)',         '{:>14}',  '{:>14.3f}'],
                      ['norm(grad(E))',  '{:>14}',  '{:>14.3f}'],
                      ['norm(weight(E))',  '{:>14}',  '{:>14.3f}'],
                    ]

    # stat_1 stores values in every diagIter and plots them in stat.pdf 
    stat_1_i = []

    stat_1 = {k[0]:[] for k in stats_headings}
    stat_1['lrE'] = []


    fid = 0.0
    best_fid = 1e5
    best_recon_fid = 1e5
    fid_dict={'gen_fid':1e5, 'recon_fid':1e5}
    inception = 0.0
    inception_std = 0.0
    mse_val = 0.0
    best_err = 1e5
    auroc = 0


    for epoch in range(start_epoch, opt.niter):

        epoch_start_time = time.time()

        stats_values = {k[0]: 0 for k in stats_headings}
        stats_values['epoch'] = epoch

        num_batch = len(dataloader.dataset) / opt.batchSize
        global_step = 0
        train_flag()

        for i, data in enumerate(dataloader, 0):
            """
          
            """
            real_cpu = data
            batch_size = real_cpu.size(0)
            if opt.cuda:
                real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            inputV = Variable(input)

            z_input_mu, z_input_log_sigma = netI(inputV)
            z_pos = reparametrize(z_input_mu, z_input_log_sigma)
            z_neg = torch.randn(batch_size, nz)
            if opt.cuda: 
                z_neg = z_neg.cuda()

            netI.zero_grad()
            netG.zero_grad()
            netE.zero_grad()

            # latent classifier
            ones, zeros = torch.ones(batch_size,), torch.zeros(batch_size,)
            labels = torch.cat((ones, zeros)).type(torch.float) #.type(torch.long) # BCE target needs float
            if opt.cuda:
                labels = labels.cuda()
            inputs = torch.cat((z_pos, z_neg), dim=0)
            idxs = torch.randperm(inputs.shape[0])
            #shuffle
            inputs, labels = inputs[idxs], labels[idxs]
            outputs = netE(inputs).squeeze() # 2B, 1

            errE = bce_loss(outputs, labels) / (2 * batch_size)
            
            errE.backward()
            
            if opt.is_grad_clampE:
                torch.nn.utils.clip_grad_norm_(netE.parameters(), opt.max_normE)
            optimizerE.step()
            netEpNorm, netEgradNorm = getGradNorm(netE)

            if i % opt.n_printout == 0:
                logging.info('[%3d/%3d][%3d/%3d] errE: %6.2f, param norm(E): %6.2f, grad norm(E): %6.2f'
                    % (epoch+1, opt.niter, i+1, len(dataloader),
                        errE.data.item(),
                        netEpNorm.data.item(), netEgradNorm.data.item(), 
                        ))
                

            stats_values['err(E)'] += errE.data.item() / num_batch
            # stats_values['err(E)'] += 0
            
            stats_values['norm(grad(E))'] += netEgradNorm.data.item() / num_batch
            stats_values['norm(weight(E))'] += netEpNorm.data.item() / num_batch

                
            global_step += 1

            #? Add breakpoint for debugging
            # break
        
        # end of datalaoder

        eval_flag()

        epoch_err = stats_values['err(E)']    
        if epoch_err < best_err:
            best_err = epoch_err
            best_epoch = epoch + 1
            opt_dict = {'netE': (netE, optimizerE)}
            save_model(opt_dict, epoch+1, idfier='bestbce')            

        epoch_end_time = time.time()
        print("Epoch {} took {} secs...".format(epoch+1, epoch_end_time - epoch_start_time))

        lrE_schedule.step(stats_values['err(E)'])
        last_lrE = lrE_schedule._last_lr[0]

                # diagnostics
        if (epoch+1) % opt.diagIter == 0 or epoch == 0 or epoch + 1==opt.niter:
            stat_1_i.append(epoch+1)
            stat_1['err(E)'].append(stats_values['err(E)'])
            stat_1['norm(grad(E))'].append(stats_values['norm(grad(E))']) #netIgradNorm.data.item())
            stat_1['norm(weight(E))'].append(stats_values['norm(weight(E))']) #netIpNorm.data.item())

            stat_1['lrE'].append(last_lrE)


            plot_stats()

        torch.cuda.empty_cache()
        # end of epoch
    
    # best_mse, fid_gen_at_fid, fid_recon_at_fid = eval_model(opt, model_type='bestfid')
    res_str3 = "Best BCE Model: Epoch={}, Loss={:.4f}".format(best_epoch, best_err)
    out_f.write(res_str3+"\n")
    out_f.flush()


def evaluate(opt, output_dir):

    def image_generator_from_latent_loader(netG, loader, batchSize, num_total_samples):
        num_iters = int(np.ceil(num_total_samples / batchSize))
        assert num_total_samples % batchSize == 0, "num_total_samples must be an integer multiple of batch_size."
        total = 0
        for i, data in enumerate(loader):
            zs = data
            zs = zs.cuda()
            batch_size = zs.shape[0]
            total += batch_size
            if total >= num_total_samples:
                break 
            with torch.no_grad():
                out = netG(zs)
            if i < 3:
                vutils.save_image(out.data, os.path.join(output_dir,'syn', "ncp_gen_samples{}.png".format(i)),
                                  normalize=True, nrow=10)
            yield out

    netG = Decoder(opt)
    # netG.apply(weights_init)
    if opt.cuda:
        netG.cuda()
    # if opt.netG != '':
    print("Loading netG from {} .....".format(opt.netG))
    ckpt = torch.load(opt.netG)
    netG.load_state_dict(ckpt['model_state'])
    print(netG)

    K = 5000
    NUM_FID_SAMPLES = opt.num_last_epoch_fid_samples

    netE = _netE(opt)
    # netE.apply(weights_init)
    if opt.cuda:
        netE.cuda()
    print("Loading netE from {} .....".format(opt.netE))
    ckpt = torch.load(opt.netE)
    netE.load_state_dict(ckpt['model_state'])
    print(netE)

    snis = NIS(K, [opt.bottleneck_factor], 
                netE,
                energy_hidden_sizes=None,
                proposal=None,
                data_mean=None,
                reparameterize_proposal_samples=False,
                dtype=torch.float,
                )
    sampled_zs = []
    for i in range(NUM_FID_SAMPLES): 
        sampled_zs.append(snis.sample(num_samples=1).detach().cpu()) # num_samples=1 is hard coded inside SNIS
    print("sampled {} latent vectors from NCP prior".format(len(sampled_zs)))
    # make dataset and loader
    from data import ImageListDataset
    genset = ImageListDataset(sampled_zs)
    genloader = torch.utils.data.DataLoader(genset, batch_size=100,
                                            shuffle=False, num_workers=int(opt.workers))
    img_generator = image_generator_from_latent_loader(netG, genloader, 100, NUM_FID_SAMPLES)
    
    fid = compute_fid_from_datalaoder(img_generator, 
                                        None, opt.fid_dataset, 
                                        opt.fid_dir, 
                                        batchSize=100, 
                                        total_fid_samples=NUM_FID_SAMPLES)
    
    out_f = open("%s/eval_results.txt" % output_dir, 'w')
    res_str = "Eval Results: FID = {:.3f} with n_proposals = {}".format(fid, K)
    out_f.write(res_str+"\n")
    out_f.flush()


def main():
    opt = parse_args()
    set_global_gpu_env(opt)
    set_seed(opt)
##
    if opt.mode == 'eval':
        # if running with eval mode, take output dir from model path
        output_dir = "/".join(opt.netE.split("/")[:-1])
        evaluate(opt, output_dir)

    else:
        data_exp = os.path.splitext(os.path.basename(__file__))[0]
        exp_id = os.path.join(data_exp, opt.exp_name, opt.model_name)
        output_dir = get_output_dir(exp_id, opt)

    # if opt.mode == 'train':
        copy_source(__file__, output_dir)
        setup_logging(output_dir)
        logging.info(opt)

        train(opt, output_dir)
        opt.netE = '%s/%s_%s.pth' % (output_dir, 'netE', 'bestbce')
        opt.mode='eval'
        evaluate(opt, output_dir)
    
    
        

if __name__ == '__main__':
    main()