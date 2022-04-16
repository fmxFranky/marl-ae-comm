from __future__ import absolute_import, division, print_function, unicode_literals

import random

import numpy as np
import torch
import torch.nn.functional as F


def to_torch(x, use_gpu=True, dtype=np.float32):
    x = np.array(x, dtype=dtype)
    var = torch.from_numpy(x)
    return var.cuda() if use_gpu is not None else var


def kld_loss(mu, var):
    bsz = mu.size()[0]
    mu, var = mu.contiguous(), var.contiguous()
    mu, var = mu.view(bsz, -1), var.view(bsz, -1)
    return torch.mean(0.5 * (mu ** 2 + torch.exp(var) - var - 1), dim=1)


def reparamterize(mu_var, use_gpu=False):
    vae_dim = int(mu_var.size()[1] / 2)
    mu, var = mu_var[:, :vae_dim], mu_var[:, vae_dim:]
    eps = to_torch(torch.randn(mu.size()), use_gpu=use_gpu)
    z = mu + eps * torch.exp(var / 2)  # var -> std
    return z, mu, var


def discrete_policy_gradient_loss(
    policy_logit, action, advantage, gae=False, value_weight=0.5, entropy_weight=0.01
):
    policy = F.softmax(policy_logit, dim=-1)[0]
    log_policy = F.log_softmax(policy_logit, dim=-1)[0]
    log_policy_action = log_policy[action]

    if gae is not False and gae is not None:
        policy_loss = -log_policy_action * gae[0].detach()
    else:
        policy_loss = -log_policy_action * advantage[0]

    value_loss = advantage ** 2
    entropy = -(policy * log_policy).sum()

    loss = policy_loss + value_weight * value_loss - entropy_weight * entropy

    return loss, policy_loss, value_loss, entropy


def policy_gradient_loss(
    policy_logit,
    action,
    advantage,
    gae=False,
    value_weight=0.5,
    entropy_weight=0.01,
    action_space=None,
):
    if action_space is None:
        # default to Discrete
        loss, policy_loss, value_loss, entropy = discrete_policy_gradient_loss(
            policy_logit, action, advantage, gae, value_weight, entropy_weight
        )
        return loss, (policy_loss, value_loss, entropy)

    elif action_space.__class__.__name__ == "MultiDiscrete":
        policy_logit = torch.split(policy_logit, action_space.nvec.tolist(), dim=-1)

        loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        entropy = 0.0
        for i, logit in enumerate(policy_logit):
            l, pl, vl, ent = discrete_policy_gradient_loss(
                logit, action[i], advantage, gae, value_weight, entropy_weight
            )
            loss += l
            policy_loss += pl
            value_loss += vl
            entropy += ent
        loss /= len(action)
        policy_loss /= len(action)
        value_loss /= len(action)
        entropy /= len(action)

        return loss, (policy_loss, value_loss, entropy)

    elif action_space.__class__.__name__ == "Box":
        value_loss = advantage ** 2
        return value_loss, (0.0, value_loss, 0.0)

    else:
        raise NotImplementedError


def mlm_loss(input_processor, attetnion_net, obses):
    assert hasattr(input_processor, "encode_obs")
    # obses.shape = [mlm_bsz, mlm_length, num_agents, ...]

    mlm_agents = attetnion_net.mlm_agents
    mlm_ratio = attetnion_net.mlm_ratio
    mlm_bsz, mlm_length, num_agents, *img_shape = list(obses.shape)

    # build label
    with torch.no_grad():
        aug_obses = attetnion_net.transformation(obses.flatten(0, 2))
    aug_obses = aug_obses.view(mlm_bsz, mlm_length, num_agents, *img_shape)
    if mlm_length > 1 and mlm_agents > 1:
        aug_obses = aug_obses.transpose(1, 2).flatten(0, 2)
        shp = (mlm_bsz, mlm_length * num_agents)
    elif mlm_length == 1 and mlm_agents > 1:
        aug_obses = aug_obses.flatten(0, 2)
        shp = (mlm_bsz, mlm_agents)
    elif mlm_length > 1 and mlm_agents == 1:
        aug_obses = aug_obses.transpose(1, 2).flatten(0, 2)
        shp = (mlm_bsz * num_agents, mlm_length)
    else:
        raise ValueError("mlm_length * mlm_agents must be > 1")
    seq_label_feat = input_processor.encode_obs(aug_obses)
    seq_label_feat = seq_label_feat.view(*shp, -1)
    feat_dim = seq_label_feat.shape[-1]

    # build mlm outputs
    aug_obses = obses
    if mlm_length > 1:
        aug_obses = aug_obses.transpose(1, 2)
        masked = torch.zeros((mlm_bsz, num_agents, mlm_length), dtype=torch.bool)
        for i in range(mlm_bsz):
            for row in range(num_agents):
                for col in range(mlm_length):
                    if random.random() < mlm_ratio:
                        aug_obses[i, row, col] = 0
                        masked[i, row, col] = True
        if mlm_agents == 1:
            aug_obses = aug_obses.flatten(0, 1)
            masked = masked.flatten(0, 1)
        else:
            aug_obses = aug_obses.flatten(1, 2)
            masked = masked.flatten(1, 2)
    elif mlm_length == 1 and mlm_agents > 1:
        aug_obses = aug_obses.flatten(1, 2)
        masked = torch.zeros((mlm_bsz, num_agents), dtype=torch.bool)
        for j in range(mlm_bsz):
            masked_agent_id = np.random.choice(num_agents)
            for i in range(num_agents):
                if i == masked_agent_id:
                    aug_obses[j, i] = 0
                    masked[j, i] = True
    shp = aug_obses.shape
    with torch.no_grad():
        aug_obses = attetnion_net.transformation(aug_obses.flatten(0, 1))
    seq_feat = input_processor.encode_obs(aug_obses)
    seq_feat = seq_feat.view(*shp[:2], feat_dim)
    seq_feat = attetnion_net(seq_feat)

    # mse loss
    loss = F.mse_loss(seq_feat[masked], seq_label_feat[masked])
    return loss
