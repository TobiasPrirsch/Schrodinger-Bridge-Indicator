
import torch
import util



def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)


def sample_e(opt, x):
    return {
        'gaussian': sample_gaussian_like,
        'rademacher': sample_rademacher_like,
    }.get(opt.noise_type)(x)

def m_inf(v):
    V1 = -1.2
    V2 = 18
    temp = (v - V1) / V2

    return 0.5 * (1 + torch.tanh(temp))


def w_inf(v):
    V3 = 2
    V4 = 30
    temp = (v - V3) / V4

    return 0.5 * (1 + torch.tanh(temp))


def tau_w(v):
    V3 = 2
    V4 = 30
    temp = (v - V3) / (2 * V4)
    return torch.cosh(temp) ** -1

def f(x):
    C = 20
    phi = 0.04
    I = 92
    V_ca = 120
    V_k = -84
    V_l = -60
    g_ca = 4.4
    g_k = 8
    g_l = 2
    v = x[:, 0]
    w = x[:, 1]
    drift_v = (- g_ca * m_inf(v) * (v - V_ca) - g_k * w * (v - V_k) - g_l * (v - V_l) + I) / C
    drift_w = phi * (w_inf(v) - w) / tau_w(v)

    drift = torch.cat([drift_v.reshape(-1,1), drift_w.reshape(-1,1)], dim=1)
    
    return drift 

def compute_div_gz(opt, dyn, ts, xs, policy, return_zs=False):

    zs = policy(xs, ts)
    g_ts = dyn.g(ts) 
    g_ts = g_ts[:, None, None, None] if util.is_image_dataset(opt) else g_ts[:, None]
    gzs = g_ts * zs * torch.tensor([5e-1, 5e-3])
    drift= f(xs)

    sign = 1. if policy.direction=='forward' else -1.
    
    e = sample_e(opt, xs)
    e_dzdx = torch.autograd.grad(gzs + sign * drift, xs, e, create_graph=True)[0]
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

        loss = torch.sum(loss * dyn.dt) / batch_x / batch_t  # sum over x_dim and T, mean over batch
    return loss, zs if return_z else loss

def compute_sb_nll_joint_train(opt, batch_x, dyn, ts_f, xs_f, zs_f, x_term_f, policy_b):
    """ Implementation of Eq (16) in our main paper.
    """
    assert opt.train_method == 'joint'
    assert policy_b.direction == 'backward'
    assert xs_f.requires_grad and zs_f.requires_grad and x_term_f.requires_grad

    div_gz_f, zs_f_hat = compute_div_gz(opt, dyn, ts_f, xs_f, policy_b, return_zs=True)

    z_f = zs_f.reshape(-1, 100, 2)

    lossf_run = 0.5*(zs_f + zs_f_hat)**2 + div_gz_f
    lossf1 = torch.sum(lossf_run*dyn.dt) / batch_x
    lossf2 = - dyn.q.log_prob(x_term_f).mean()
    loss = lossf1 + lossf2

    return loss, z_f
