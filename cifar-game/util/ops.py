from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch


def to_state_var(x, use_gpu=True, dtype=np.float32):
    if type(x) is dict:
        # multi-agent
        return {k: _to_state_var(v, use_gpu, dtype) for k, v in x.items()}
    elif type(x) is list:
        return [_mnist_to_state_var(v, use_gpu, dtype) for v in x]
    else:
        # single-agent
        return _to_state_var(x, use_gpu, dtype)


def _mnist_to_state_var(x, use_gpu=True, dtype=np.float32):
    for k, v in x.items():
        if type(v) is int or type(v) is float:
            v = np.array([v], dtype=dtype)
            v = torch.from_numpy(v)
        elif type(v) is np.ndarray:
            v = torch.from_numpy(v).float()
        elif type(v) is list:
            v = torch.FloatTensor(v)
        x[k] = v.cuda() if use_gpu else v
    return x


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
