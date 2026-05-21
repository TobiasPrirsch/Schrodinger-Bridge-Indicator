
import torch
import util
import numpy as np
from ipdb import set_trace as debug
import torch.nn as nn


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)


def sample_e(opt, x):
    return {
        'gaussian': sample_gaussian_like,
        'rademacher': sample_rademacher_like,
    }.get(opt.noise_type)(x)


def compute_div_gz(opt, dyn, ts, xs, policy, return_zs=False):
    # print(xs.shape)
    # print(ts.shape)
    # print('***********')
    zs = policy(xs, ts)

    g_ts = dyn.g(ts)
    g_ts = g_ts[:, None, None, None] if util.is_image_dataset(opt) else g_ts[:, None]
    gzs = g_ts*zs

    e = sample_e(opt, xs)
    e_dzdx = torch.autograd.grad(gzs, xs, e, create_graph=True)[0]
    div_gz = e_dzdx * e
    # approx_div_gz = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)

    return [div_gz, zs] if return_zs else div_gz


def compute_sb_nll_alternate_train(opt, dyn, ts, xs, zs_impt, policy_opt, return_z=False):
    """ Implementation of Eq (18,19) in our main paper.
    """
    assert opt.train_method == 'alternate'
    assert xs.requires_grad
    assert not zs_impt.requires_grad

    batch_x = opt.train_bs_x
    batch_t = opt.train_bs_t

    with torch.enable_grad():
        div_gz, zs = compute_div_gz(opt, dyn, ts, xs, policy_opt, return_zs=True)
        loss = zs*(0.5*zs + zs_impt) + div_gz
        # print(torch.sum(div_gz), torch.sum(zs*(0.5*zs + zs_impt)))
        loss = torch.sum(loss * dyn.dt) / batch_x / batch_t  # sum over x_dim and T, mean over batch
    return loss, zs if return_z else loss


def kde_estimate(x, bandwidth=0.5):
    # 使用高斯核函数进行核密度估计

    x1 = x.unsqueeze(1)  # 将输入张量变形为 (n, 1, d)
    x2 = x.unsqueeze(0)  # 创建一个矩阵，每一行都是原始数据点 (1, n, d)
    diff = x1 - x2
    kde = torch.exp(-0.5 * (diff / bandwidth)**2).prod(dim=-1) / (bandwidth * torch.sqrt(2 * torch.tensor(3.141592653589793)))*2
    return kde.mean(dim=(0, 1))  # 对所有数据点的估计值取平均得到密度估计


def kl_divergence(p, q):
    # 计算两个分布之间的KL散度
    # print(p.shape)
    # print(p)
    # print(q)
    # print(torch.log(p))
    # print(torch.log(q))
    # print((torch.log(p) - torch.log(q)))
    # print(ss)
    # print(torch.log(p) - torch.log(q))
    # print(p * (torch.log(p) - torch.log(q)))
    return (p * (torch.log(p) - torch.log(q))).sum()


def total_variation_distance(p, q):
    return 0.5 * torch.abs(p - q).sum()


def kolmogorov_smirnov_distance(p, q):
    return torch.max(torch.abs(torch.cumsum(p, dim=0) - torch.cumsum(q, dim=0)))


def loss_joint_train(dyn, x_term_f):

    loss = torch.nn.functional.softmax(- dyn.q.log_prob(x_term_f)).mean()


    return loss


def compute_sb_nll_joint_train(opt, batch_x, dyn, ts, xs_f, zs_f, x_term_f, policy_b, policy_f, target):
    """ Implementation of Eq (16) in our main paper.
    """
    assert opt.train_method == 'joint'
    assert policy_b.direction == 'backward'
    assert xs_f.requires_grad and zs_f.requires_grad and x_term_f.requires_grad

    div_gz_b, zs_b = compute_div_gz(opt, dyn, ts, xs_f, policy_b, return_zs=True)
    # div_gz_f, _ = compute_div_gz(opt, dyn, ts, xs_f, policy_f, return_zs=True)
    # print(ts)
    # print(ss)

    z_f = zs_f.reshape(-1, 100, 2)
    z_b = zs_b.reshape(-1, 100, 2)
    d_b = div_gz_b.reshape(-1, 100, 2)
    # d_f = div_gz_f.reshape(-1, 100, 2)

    cost = 0.5*(z_f + z_b)**2 + d_b
    # cost1 = 0.5*(z_f + z_b)**2 + d_f

    cost = torch.sum(cost * dyn.dt, axis=[0, -1]) / batch_x
    # cost1 = torch.sum(cost1 * dyn.dt, axis=[0, -1]) / batch_x
    
    loss = 0.5*(zs_f + zs_b)**2 + div_gz_b
    # print(z_f)
    # print(z_b)
    # print(div_gz_b)
    # print(loss.shape)
    # print(torch.sum(cost))
    loss1 = torch.sum(loss*dyn.dt) / batch_x
    pdist = nn.PairwiseDistance(p=2)
    loss2 = pdist(x_term_f, target).mean()
    # print(loss2)c
    # print(ss)
    # loss2 = - dyn.q.log_prob(x_term_f).mean()
    # print(cost)
    # print(loss1.item(), loss2.item())
    # print(ss)
    loss = loss1 + loss2 
    print(loss1.item(), loss2.item())
    # print(loss2.item())

    return loss, cost, z_f, z_b, div_gz_b