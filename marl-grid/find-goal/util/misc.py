from __future__ import absolute_import, division, print_function, unicode_literals

import datetime
import random
import warnings

import numpy as np
import torch
from envs.grid_world_environment import get_env_name


def check_config(cfg):
    assert cfg.env_cfg.comm_len > 0
    assert not (cfg.env_cfg.num_adversaries > 0 and cfg.env_cfg.num_blind_agents > 0)

    if (
        cfg.env_cfg.observe_position
        or cfg.env_cfg.observe_done
        or cfg.env_cfg.observe_self_position
    ):
        if cfg.env_cfg.observation_style != "dict":
            cprint("AUTO: correcting observation_style to _dict_", "r")
            cfg.env_cfg.observation_style = "dict"

    assert cfg.env_cfg.num_blind_agents <= cfg.env_cfg.num_agents
    assert cfg.env_cfg.num_adversaries <= cfg.env_cfg.num_agents
    if cfg.env_cfg.active_after_done and cfg.mask:
        raise ValueError("active_after_done and mask cannot both be True")

    if cfg.env_cfg.observe_position and cfg.env_cfg.observe_self_position:
        raise ValueError(
            "observe_position and observe_self_position cannot " "both be True"
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_config(cfg):
    cfg.env_cfg.env_name = get_env_name(cfg.env_cfg)
    if cfg.seed is None:
        cfg.env_cfg.seed = cfg.seed = random.randint(0, 2 ** 32 - 1)
        print("AUTO: setting seed to {}".format(cfg.seed))

    # automatically generate exp name based on configs
    curr_time = str(datetime.datetime.now())[:16].replace(" ", "_")

    id_args = [
        ["algo", cfg.algo],
        ["seed", cfg.env_cfg.seed],
        ["lr", cfg.lr],
        ["tmax", cfg.tmax],
        ["workers", cfg.num_workers],
        ["ms", cfg.env_cfg.max_steps],
        ["ae_type", cfg.ae_type],
    ]

    if cfg.comm_vf:
        id_args += [["commvf", "True"]]

    if cfg.ae_pg:
        id_args += [["ae_pg", cfg.ae_pg]]

    if cfg.ae_agent_done:
        id_args += [["agentdone", cfg.ae_agent_done]]

    if cfg.img_feat_dim != 64:
        id_args += [["imgdim", cfg.img_feat_dim]]

    cfg_id = "_".join([f"{n}-{v}" for n, v in id_args])

    if cfg.id:
        cfg_id = "{}_{}".format(cfg.id, cfg_id)

    if eval:
        cfg_id += "_eval"

    exp_name = "{}/a3c_{}_{}".format(cfg.env_cfg.env_name, cfg_id, curr_time)

    cfg.exp_name = exp_name


class bcolors:
    HEADER = "\033[95m"
    b = blue = OKBLUE = "\033[94m"
    g = green = OKGREEN = "\033[92m"
    y = yellow = WARNING = "\033[93m"
    r = red = FAIL = "\033[91m"
    c = cyan = "\033[36m"
    lb = lightblue = "\033[94m"
    p = pink = "\033[95m"
    o = orange = "\033[33m"
    p = pink = "\033[95m"
    lc = lightcyan = "\033[96m"
    end = ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def cprint(print_str, color=None, float_num=False, return_str=False):
    if float_num is not False:
        # make it colorful
        cmap = [31, 32, 33, 34, 35, 36, 37, 91, 92, 93, 94, 95, 96, 97]
        cmap_idx = int(float_num * (len(cmap) - 1))  # floor
        c = "\033[{}m".format(cmap[cmap_idx])
    else:
        if not hasattr(bcolors, color):
            warnings.warn("Unknown color {}".format(color))
            if return_str:
                return print_str
            print(print_str)
        else:
            c = getattr(bcolors, color)
    e = getattr(bcolors, "end")
    c_str = "{}{}{}".format(c, print_str, e)
    if return_str:
        return c_str
    print(c_str)
    return


def check_done(done):
    if type(done) is bool:
        return done
    elif type(done) is dict:
        return done["__all__"]
    else:
        raise ValueError(f"unknown done signal {type(done)}")
