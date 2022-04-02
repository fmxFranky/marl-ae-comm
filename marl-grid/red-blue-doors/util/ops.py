from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
import torch.nn.functional as F


def to_state_var(x, use_gpu=True, dtype=np.float32):
    if type(x) is dict:
        # multi-agent
        return {k: _to_state_var(v, use_gpu, dtype) for k, v in x.items()}
    else:
        # single-agent
        return _to_state_var(x, use_gpu, dtype)


def _to_state_var(x, use_gpu=True, dtype=np.float32):
    if isinstance(x, dict):
        # observation_style=='dict'
        return dict_to_state_var(x, use_gpu, dtype)
    else:
        # observation_style=='img'
        return img_to_state_var(x, use_gpu, dtype)


def dict_to_state_var(x, use_gpu=True, dtype=np.float32):
    for k, v in x.items():
        v = np.array(v, dtype=dtype)
        var = torch.from_numpy(v)
        if k == "pov":
            var = var.permute(2, 0, 1).unsqueeze(0)
        x[k] = var.cuda() if use_gpu else var
    return x


def img_to_state_var(x, use_gpu=True, dtype=np.float32):
    x = np.array(x, dtype=dtype)
    var = torch.from_numpy(x)
    var = var.permute(2, 0, 1).unsqueeze(0)
    return var.cuda() if use_gpu else var


def to_torch(x, use_gpu=True, dtype=np.float32):
    if isinstance(x, list):
        x = x[0]
    if torch.is_tensor(x):
        return x.cuda() if use_gpu is not None else x
    x = np.array(x, dtype=dtype)
    var = torch.from_numpy(x)
    return var.cuda() if use_gpu else var


def to_numpy(x):
    if isinstance(x, int) or isinstance(x, float):
        return x
    if isinstance(x, (list, np.ndarray)):
        return np.array([to_numpy(_x) for _x in x])
    return x.detach().cpu().numpy()


def norm_col_init(weights, std=1.0):
    """
    Normalized column initializer
    """
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x ** 2).sum(1, keepdim=True))
    return x

def multi_head_attention(q, k, v, mask=None):
    # q shape = (B, n_heads, n, key_dim)   : n can be either 1 or N
    # k,v shape = (B, n_heads, N, key_dim)
    # mask.shape = (B, group, N)

    B, n_heads, n, key_dim = q.shape

    # score.shape = (B, n_heads, n, N)
    score = torch.matmul(q, k.transpose(2, 3)) / np.sqrt(q.size(-1))

    if mask is not None:
        score += mask[:, None, :, :].expand_as(score)

    shp = [q.size(0), q.size(-2), q.size(1) * q.size(-1)]
    attn = torch.matmul(F.softmax(score, dim=3), v).transpose(1, 2)
    return attn.reshape(*shp)


def make_heads(qkv, n_heads):
    shp = (qkv.size(0), qkv.size(1), n_heads, -1)
    return qkv.reshape(*shp).transpose(1, 2)