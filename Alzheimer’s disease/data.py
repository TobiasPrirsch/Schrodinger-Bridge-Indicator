import numpy as np

import torch
import torch.nn as nn
import torch.distributions as td
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import random

from prefetch_generator import BackgroundGenerator
# from scipy.stats import gaussian_kde

import util
# from ipdb import set_trace as debug


def build_boundary_distribution(opt):
    print(util.magenta("build boundary distribution..."))

    opt.data_dim = get_data_dim(opt.problem_name)   #[2]
    prior = build_data_sampler(opt, opt.final, 'mild')
    pdata = build_data_sampler(opt, opt.init, 'gauss')

    return pdata, prior


def get_data_dim(problem_name):
    return {
        'gmm':          [2],
        'checkerboard': [2],
        'morris':       [2],
        'moon-to-spiral': [2],
        'mnist':       [1,32,32],
        'celebA32':    [3,32,32],
        'celebA64':    [3,64,64],
        'cifar10':     [3,32,32],
    }.get(problem_name)


def build_prior_sampler(opt, batch_size):
    if opt.problem_name == 'moon-to-spiral':
        # 'moon-to-spiral' uses Moon as prior distribution
        return Moon(batch_size)

    # image+VESDE -> use (sigma_max)^2; otherwise use 1.
    cov_coef = opt.sigma_max**2 if (util.is_image_dataset(opt) and not util.use_vp_sde(opt)) else 1.
    prior = td.MultivariateNormal(torch.zeros(opt.data_dim), cov_coef*torch.eye(opt.data_dim[-1]) * 100)

    return PriorSampler(prior, batch_size, opt.device)

def build_data_sampler(opt, batch_size, name):

    if util.is_toy_dataset(opt):
        return {
            'gmm': MixMultiVariateNormal,
            'checkerboard': Sne,
            'moon-to-spiral': Morris_Lecar,
        }.get(opt.problem_name)(batch_size, name)

    elif util.is_image_dataset(opt):
        dataset_generator = {
            'mnist':      generate_mnist_dataset,
            'celebA32':   generate_celebA_dataset,
            'celebA64':   generate_celebA_dataset,
            'cifar10':    generate_cifar10_dataset,
        }.get(opt.problem_name)
        dataset = dataset_generator(opt)
        return DataSampler(dataset, batch_size, opt.device)

    else:
        raise RuntimeError()


class MixMultiVariateNormal:
    def __init__(self, batch_size, num=8, radius=12, sigmas=None):

        # build mu's and sigma's
        arc = 2*np.pi/num
        xs = [np.cos(arc*idx)*radius for idx in range(num)]
        ys = [np.sin(arc*idx)*radius for idx in range(num)]
        mus = [torch.Tensor([x, y]) for x, y in zip(xs, ys)]
        dim = len(mus[0])
        sigmas = [torch.eye(dim) * 0.01 for _ in range(num)] if sigmas is None else sigmas

        if batch_size%num!=0:
            raise ValueError('batch size must be devided by number of gaussian')
        self.num = num
        self.batch_size = batch_size
        self.dists = [
            td.multivariate_normal.MultivariateNormal(mu, sigma) for mu, sigma in zip(mus, sigmas)
        ]

    def log_prob(self, x):
        # assume equally-weighted
        densities = [torch.exp(dist.log_prob(x))+1e-41 for dist in self.dists]

        return torch.log(sum(densities)/len(self.dists))

    def sample(self):
        ind_sample = self.batch_size/self.num
        samples = [dist.sample([int(ind_sample)]) for dist in self.dists]
        samples = torch.cat(samples, dim=0)
        return samples


class GaussianKDE2D(nn.Module):
    def __init__(self, bandwidth=0.5):
        super(GaussianKDE2D, self).__init__()
        self.bandwidth = bandwidth

    def forward(self, x, data):
        # 计算 KDE
        diffs = x.unsqueeze(1) - data.unsqueeze(0)
        kernel_scores = torch.exp(-0.5 * (diffs.norm(dim=-1) / self.bandwidth)**2) / (self.bandwidth * (2 * torch.tensor(torch.pi)).sqrt())
        kde_estimate = kernel_scores.mean(dim=1)

        return kde_estimate

# 高斯核函数
def gaussian_kernel(x, h):

    return (1 / torch.sqrt(2 * torch.tensor(np.pi))) * torch.exp(-0.5 * (x / h)**2)


# 二维核密度估计函数
def kde_2d(xt, data, h):
    xt_x = xt[:, 0]
    xt_y = xt[:, 1]
    data_x = data[:, 0]
    data_y = data[:, 1]
    num = xt.shape[0]
    kde_values = torch.zeros_like(xt_x)

    for i in range(num):
        kde_values[i] = torch.sum(gaussian_kernel((xt_x - data_x[i]) / h, h) * gaussian_kernel((xt_y - data_y[i]) / h, h))

    return kde_values / (num * h**2)


class Morris_Lecar:

    def __init__(self, batch_size, name):
        self.batch_size = batch_size
        self.name = name
        self.data = np.load('./results/data_steady.npy')

        self.kde_model = GaussianKDE2D()
        x_mean = self.data[:, 0].mean()
        y_mean = self.data[:, 1].mean()

        mu = torch.tensor([x_mean, y_mean])
        sigma = torch.eye(2) * 0.0005

        self.dist = td.multivariate_normal.MultivariateNormal(mu, sigma)

    def log_prob(self, x):

        densities = torch.exp(self.dist.log_prob(x)) + 1e-41

        return torch.log(sum(densities))

    def sample(self):
        n = self.batch_size
        if self.name == 'steady':
            data = np.load('./results/data_steady.npy')
        elif self.name == 'cycle':
            data = np.load('./results/data_cycle.npy')
        data = torch.Tensor(data)

        sample = data[0:n, :]
        return sample


class Sne:

    def __init__(self, batch_size, name):
        self.batch_size = batch_size
        self.name = name
        self.data = np.load('./results/experiments/mildx.npy')
        # self.data = np.load('./results/experiments/mild_part.npy')

        # self.kde_model = GaussianKDE2D()
        # # print(kde_model)
        self.data = torch.tensor(self.data)

        xs = [i for i in self.data[:, 0]]
        ys = [i for i in self.data[:, 1]]

        mus = [torch.Tensor([x, y]) for x, y in zip(xs, ys)]
        sigmas = [torch.eye(2) * 0.1 for _ in range(len(self.data))]

        self.dists = [
            td.multivariate_normal.MultivariateNormal(mu, sigma) for mu, sigma in zip(mus, sigmas)
        ]

    def log_prob(self, x):

        densities = [torch.exp(dist.log_prob(x)) + 1e-41 for dist in self.dists]

        return torch.log(sum(densities) / len(self.dists))

    def sample(self):
        n = self.batch_size
        if self.name == 'non':
            data = np.load('./results/experiments/nonx.npy')
        if self.name == 'gauss':
            data = np.load('./results/experiments/gau3000_unique_right.npy')
        elif self.name == 'mild':
            data = np.load('./results/experiments/gau3000_target_right.npy')

        data = torch.Tensor(data)

        sample = data[0:n, :]
        return sample


class CheckerBoard:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sample(self):
        n = self.batch_size
        n_points = 3*n
        n_classes = 2
        freq = 5
        x = np.random.uniform(-(freq//2)*np.pi, (freq//2)*np.pi, size=(n_points, n_classes))
        mask = np.logical_or(np.logical_and(np.sin(x[:,0]) > 0.0, np.sin(x[:,1]) > 0.0), \
        np.logical_and(np.sin(x[:, 0]) < 0.0, np.sin(x[:,1]) < 0.0))
        y = np.eye(n_classes)[1*mask]
        x0=x[:,0]*y[:,0]
        x1=x[:,1]*y[:,0]
        sample=np.concatenate([x0[...,None],x1[...,None]],axis=-1)
        sqr=np.sum(np.square(sample),axis=-1)
        idxs=np.where(sqr==0)
        sample=np.delete(sample,idxs,axis=0)
        # res=res+np.random.randn(*res.shape)*1
        sample=torch.Tensor(sample)
        sample=sample[0:n,:]
        # print(sample.shape)
        # print(ss)
        return sample


class Spiral:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sample(self):
        n = self.batch_size
        theta = np.sqrt(np.random.rand(n))*3*np.pi-0.5*np.pi # np.linspace(0,2*pi,100)

        r_a = theta + np.pi
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        x_a = data_a + 0.25*np.random.randn(n, 2)
        samples = np.append(x_a, np.zeros((n, 1)), axis=1)
        samples = samples[:, 0:2]
        return torch.Tensor(samples)


class Moon:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sample(self):
        n = self.batch_size
        x = np.linspace(0, np.pi, n // 2)
        u = np.stack([np.cos(x) + .5, -np.sin(x) + .2], axis=1) * 10.
        u += 0.5*np.random.normal(size=u.shape)
        v = np.stack([np.cos(x) - .5, np.sin(x) - .2], axis=1) * 10.
        v += 0.5*np.random.normal(size=v.shape)
        x = np.concatenate([u, v], axis=0)
        return torch.Tensor(x)


class DataSampler:  # a dump data sampler
    def __init__(self, dataset, batch_size, device):
        self.num_sample = len(dataset)
        # dataset.to(device)
        self.dataloader = setup_loader(dataset, batch_size)
        self.batch_size = batch_size
        self.device = device

    def sample(self):
        data = next(self.dataloader)
        return data[0].to(self.device)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子


class PriorSampler: # a dump prior sampler to align with DataSampler
    def __init__(self, prior, batch_size, device):
        self.prior = prior
        self.batch_size = batch_size
        self.device = device

    def log_prob(self, x):
        return self.prior.log_prob(x)

    def sample(self):

        return self.prior.sample([self.batch_size]).to(self.device)


def setup_loader(dataset, batch_size): 
    train_loader = DataLoaderX(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    # train_loader = DataLoaderX(dataset, batch_size=batch_size,shuffle=True,num_workers=4, pin_memory=True)
    print("number of samples: {}".format(len(dataset)))
    # train_loader.to(device)
    # https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/image_datasets.py#L52-L53
    # https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/train_util.py#L166
    # return train_loader
    while True:
        yield from train_loader


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def generate_celebA_dataset(opt,load_train=True):
    if opt.problem_name=='celebA32': #Our own data preprocessing
        transforms_list=[
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    elif opt.problem_name=='celebA64':
        transforms_list=[ #Normal Data preprocessing
            transforms.Resize([64,64]), #DSB type resizing
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    else:
        raise RuntimeError()

    if util.use_vp_sde(opt):
        transforms_list+=[transforms.Lambda(lambda t: (t * 2) - 1),]

    return datasets.ImageFolder(
        root='data/celebA/img_align_celeba/',
        transform=transforms.Compose(transforms_list)
    )


def generate_mnist_dataset(opt,load_train=True):
    transforms_list=[
        transforms.Pad(2,fill=0), #left and right 2+2=4 padding
        transforms.ToTensor(),
    ]
    if util.use_vp_sde(opt):
        transforms_list+=[transforms.Lambda(lambda t: (t * 2) - 1),]

    return datasets.MNIST(
        'data',
        train= not opt.compute_NLL,
        download=load_train,
        transform=transforms.Compose(transforms_list)
    )


def generate_cifar10_dataset(opt, load_train=True):
    transforms_list=[
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(), #Convert to [0,1]
    ]
    if util.use_vp_sde(opt):
        transforms_list+=[transforms.Lambda(lambda t: (t * 2) - 1),]

    return datasets.CIFAR10(
        'data',
        train=not opt.compute_NLL,
        download=load_train,
        transform=transforms.Compose(transforms_list)
    )
