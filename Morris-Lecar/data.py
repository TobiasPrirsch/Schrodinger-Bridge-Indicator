import numpy as np

import torch
import torch.nn as nn
import torch.distributions as td
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

import util


def build_boundary_distribution(opt):
    print(util.magenta("build boundary distribution..."))

    opt.data_dim = get_data_dim(opt.problem_name)   #[2]
    prior = build_data_sampler(opt, opt.samp_bs, 'cycle')
    pdata= build_prior_sampler(opt, opt.samp_bs)

    return pdata, prior


def get_data_dim(problem_name):
    return {
        'gmm':          [2],
        'checkerboard': [2],
        'morris':       [2],

    }.get(problem_name)


def build_prior_sampler(opt, batch_size):

    data = np.load('./results/data_steady.npy')

    x_mean = data[:, 0].mean()
    y_mean = data[:, 1].mean()

    mu = torch.tensor([x_mean, y_mean]).to(torch.float32)
    sigma = torch.eye(2) * torch.tensor([1e-4, 1e-8])

    prior = td.MultivariateNormal(mu, sigma)

    return PriorSampler(prior, batch_size, opt.device)


def build_data_sampler(opt, batch_size, name):

    if util.is_toy_dataset(opt):
        return {
            'gmm': MixMultiVariateNormal,
            'checkerboard': Morris_Lecar,

        }.get(opt.problem_name)(batch_size, name)


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


class Morris_Lecar:

    def __init__(self, batch_size, name):
        self.batch_size = batch_size
        self.name = name
        self.data = np.load('./results/data_cycle.npy')

        xs = [i for i in self.data[:, 0]]
        ys = [i for i in self.data[:, 1]]


        mus = [torch.Tensor([x, y]) for x, y in zip(xs, ys)]
        sigmas = [torch.eye(2) * torch.tensor([1e-2, 1e-4]) for _ in range(len(self.data))]

        self.dists = [
            td.multivariate_normal.MultivariateNormal(mu, sigma) for mu, sigma in zip(mus, sigmas)
        ]

    def log_prob(self, x):

        densities = [torch.exp(dist.log_prob(x)) + 1e-41 for dist in self.dists]

        return torch.log(sum(densities) / len(self.dists))

    def sample(self):
        n = self.batch_size
        if self.name == 'steady':
            data = np.load('./results/data_steady.npy')
        elif self.name == 'cycle':
            # data = np.load('./results/data_cycle_30_300_0.5.npy')
            data = np.load('./results/data_cycle.npy')
        data = torch.Tensor(data)

        sample = data[0:n, :]
        return sample


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


