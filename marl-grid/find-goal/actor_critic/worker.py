from __future__ import absolute_import, division, print_function, unicode_literals

from collections import deque

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import RandomCrop
from loss import jpr_loss, policy_gradient_loss
from util import ops
from util.decorator import within_cuda_device
from util.misc import check_done


class Worker(mp.Process):
    """
    A3C worker. Each worker is responsible for collecting data from the
    environment and updating the master network by supplying the gradients.
    The worker re-synchronizes the weight at ever iteration.

    Args:
        master: master network instance.
        net: network with same architecture as the master network
        env: environment
        worker_id: worker id. used for tracking and debugging.
        gpu_id: the cuda gpu device id used to initialize variables. the
            `within_cuda_device` decorator uses this.
        t_max: maximum number of steps to take before applying gradient update.
            Default: `20`
        use_gae: uses generalized advantage estimation.
            Default: `True`
        gamma: hyperparameter for the reward decay.
            Default: `0.99`
        tau: gae hyperparameter.
    """

    def __init__(
        self,
        master,
        net,
        attention_net,
        env,
        worker_id,
        gpu_id=0,
        t_max=20,
        use_gae=True,
        gamma=0.99,
        tau=1.0,
        num_acts=1,
        anneal_comm_rew=False,
        add_auxiliary_loss=False,
        aux_task="jpr",
        aux_loss_k=5.0,
        aux_agents=1,
        aux_rb_size=int(1e5),
        aux_length=8,
        aux_bsz=64,
        aux_pred_only_latest_latent=False,
        log_queue=None,
        **kwargs,
    ):
        super().__init__()

        self.worker_id = worker_id
        self.net = net
        self.attention_net = attention_net
        self.env = env
        self.master = master
        self.t_max = t_max
        self.use_gae = use_gae
        self.gamma = gamma
        self.tau = tau
        self.gpu_id = gpu_id
        self.reward_log = deque(maxlen=5)  # track last 5 finished rewards
        self.pfmt = (
            "policy loss: {} value loss: {} entropy loss: {} reward: {} aux_loss: {}"
        )
        self.agents = [f"agent_{i}" for i in range(self.env.num_agents)]
        self.num_acts = num_acts
        self.anneal_comm_rew = anneal_comm_rew
        self.add_auxiliary_loss = add_auxiliary_loss
        self.aux_task = aux_task
        self.aux_loss_k = aux_loss_k
        self.aux_agents = aux_agents
        self.aux_rb_size = aux_rb_size
        self.aux_length = aux_length
        self.aux_bsz = aux_bsz
        self.aux_pred_only_latest_latent = aux_pred_only_latest_latent
        assert aux_task in ["mlm", "jpr"]
        assert aux_agents == 1 or aux_agents == self.env.num_agents
        self.log_queue = log_queue
        self.env_action_size = env.action_space[0].n
        self.obs_keys = list(env.observation_space.spaces.keys())

    @within_cuda_device
    def get_trajectory(self, hidden_state, state_var, done, weight_iter):
        """
        extracts a trajectory using the current policy.

        The first three return values (traj, val, tval) have `num_acts` length.

        Args:
            hidden_state: last hidden state observed
            state_var: last state observed
            done: boolean value to determine whether the env should be reset
        Returns:
            trajectory: (pi, a, v, r) trajectory [state is not tracked]
            values: reversed trajectory values .. used for GAE.
            target_value: the last time-step value
            done: updated indicator
        """
        # mask first (environment) actions after an agent is done
        env_mask_idx = [None for _ in range(len(self.agents))]

        trajectory = [[] for _ in range(self.num_acts)]

        while not check_done(done) and len(trajectory[0]) < self.t_max:
            plogit, value, hidden_state = self.net(
                self.transform(state_var), hidden_state, env_mask_idx=env_mask_idx
            )
            action, _, _ = self.net.take_action(plogit)
            state, reward, done, info = self.env.step(action)
            state_var = ops.to_state_var(state)
            act_info = None

            if self.num_acts == 1:
                trajectory[0].append((plogit, action, value, reward, act_info))
            else:
                for i in range(self.num_acts):
                    # use separate reward on env and comm policies
                    rew = info["rew_by_act"][i]

                    if i > 0:
                        if self.anneal_comm_rew:
                            for k, v in rew.items():
                                rew[k] *= weight_iter / self.master.max_iteration

                    action_by_id = {k: v[i] for k, v in action.items()}
                    trajectory[i].append(
                        (plogit[i], action_by_id, value[i], rew, act_info)
                    )

            # mask unavailable env actions after individual done
            for agent_id, a in enumerate(self.agents):
                if info[a]["done"] and env_mask_idx[agent_id] is None:
                    env_mask_idx[agent_id] = [0, 1, 2, 3]

            if self.add_auxiliary_loss:
                for key in self.obs_keys:
                    data = [
                        state_var[f"agent_{i}"][key] for i in range(self.env.num_agents)
                    ]
                    data = (
                        torch.cat(data, dim=0)
                        if key == "pov"
                        else torch.stack(data, dim=0)
                    )
                    np.copyto(
                        self.buffer[key][self.aux_rb_idx], data.cpu().numpy(),
                    )
                acts = [action[f"agent_{i}"][0] for i in range(self.env.num_agents)]
                np.copyto(
                    self.buffer["act"][self.aux_rb_idx], np.array(acts),
                )
                self.aux_rb_idx = (self.aux_rb_idx + 1) % self.aux_rb_size
                self.aux_rb_full = self.aux_rb_full or self.aux_rb_idx == 0

        # end condition
        if check_done(done):
            target_value = [{k: 0 for k in self.agents} for _ in range(self.num_acts)]
        else:
            with torch.no_grad():
                target_value = self.net(
                    self.transform(state_var), hidden_state, env_mask_idx=env_mask_idx
                )[1]
                if self.num_acts == 1:
                    target_value = [target_value]

        #  compute Loss: accumulate rewards and compute gradient
        values = [{k: None for k in self.agents} for _ in range(self.num_acts)]
        if self.use_gae:
            for k in self.agents:
                for aid in range(self.num_acts):
                    values[aid][k] = [x[k] for x in list(zip(*trajectory[aid]))[2]]
                    values[aid][k].append(ops.to_torch([target_value[aid][k]]))
                    values[aid][k].reverse()

        return trajectory, values, target_value, done

    @torch.no_grad()
    @within_cuda_device
    def transform(self, state_var):
        if not hasattr(self, "transform_op"):
            self.transform_op = nn.Sequential(
                nn.ReplicationPad2d(2),
                RandomCrop(state_var["agent_0"]["pov"].shape[-2:]),
            )
        if self.attention_net:
            for agent in self.agents:
                state_var[agent]["pov"] = self.transform_op(state_var[agent]["pov"])
        return state_var

    @within_cuda_device
    def run(self):
        self.master.init_tensorboard()
        done = True
        reward_log = 0.0

        if self.add_auxiliary_loss:
            dummy_obs = ops.to_state_var(self.env.reset())["agent_0"]
            self.buffer = {}
            for key in self.obs_keys:
                shp = (
                    dummy_obs[key].shape if key == "pov" else dummy_obs[key][None].shape
                )
                shp = [self.aux_rb_size, shp[0] * self.env.num_agents] + list(shp[1:])
                self.buffer[key] = np.empty(shp, dtype=np.float32)
            self.buffer["act"] = np.empty(
                [self.aux_rb_size, self.env.num_agents], dtype=np.long
            )
            self.aux_rb_idx = 0
            self.aux_rb_full = False

        while not self.master.is_done():
            # synchronize network parameters
            weight_iter = self.master.copy_weights(self.net, self.attention_net)
            self.net.zero_grad()
            self.attention_net.zero_grad()

            # reset environment if new episode
            if check_done(done):
                state = self.env.reset()
                state_var = ops.to_state_var(state)
                hidden_state = None

                if self.net.is_recurrent:
                    hidden_state = self.net.init_hidden()

                done = False

                self.reward_log.append(reward_log)
                reward_log = 0.0

            # extract trajectory
            trajectory, values, target_value, done = self.get_trajectory(
                hidden_state, state_var, done, weight_iter
            )

            all_pls = [[] for _ in range(self.num_acts)]
            all_vls = [[] for _ in range(self.num_acts)]
            all_els = [[] for _ in range(self.num_acts)]

            # compute loss for each action
            loss = 0
            for aid in range(self.num_acts):
                traj = trajectory[aid]
                val = values[aid]
                tar_val = target_value[aid]

                # compute loss - computed backward
                traj.reverse()

                for agent in self.agents:
                    gae = torch.zeros(1, 1).cuda()
                    t_value = tar_val[agent]

                    pls, vls, els = [], [], []
                    for i, (pi_logit, action, value, reward, act_info) in enumerate(
                        traj
                    ):

                        # clip reward (optional)
                        if False:
                            reward = float(np.clip(reward, -1.0, 1.0))

                        # Agent A3C Loss
                        t_value = reward[agent] + self.gamma * t_value
                        advantage = t_value - value[agent]

                        if self.use_gae:
                            # Generalized advantage estimation (GAE)
                            delta_t = (
                                reward[agent]
                                + self.gamma * val[agent][i].data
                                - val[agent][i + 1].data
                            )
                            gae = gae * self.gamma * self.tau + delta_t
                        else:
                            gae = False

                        if aid == 1:
                            tl, (pl, vl, el) = policy_gradient_loss(
                                pi_logit[agent],
                                action[agent],
                                advantage,
                                gae=gae,
                                action_space=self.net.comm_action_space,
                            )
                        else:
                            tl, (pl, vl, el) = policy_gradient_loss(
                                pi_logit[agent], action[agent], advantage, gae=gae
                            )

                        pls.append(ops.to_numpy(pl))
                        vls.append(ops.to_numpy(vl))
                        els.append(ops.to_numpy(el))

                        reward_log += reward[agent]
                        loss += tl

                    all_pls[aid].append(np.mean(pls))
                    all_vls[aid].append(np.mean(vls))
                    all_els[aid].append(np.mean(els))

                    # compute backward locally
                    loss.backward(retain_graph=True)

            self.master.apply_gradients(self.net, loss)

            # auxilary task
            if self.add_auxiliary_loss:
                idxs = np.random.randint(
                    0,
                    self.aux_rb_size - self.aux_length - 1
                    if self.aux_rb_full
                    else self.aux_rb_idx - self.aux_length - 1,
                    size=self.aux_bsz,
                ).reshape(-1, 1)
                step = np.arange(self.aux_length + 1).reshape(1, -1)
                idxs = idxs + step
                temporal_samples = [{key: {} for key in self.agents}] * (
                    self.aux_length + 1
                )
                for i in range(self.aux_length + 1):
                    for j in range(self.env.num_agents):
                        for key in self.obs_keys:
                            temporal_samples[i][self.agents[j]][key] = torch.from_numpy(
                                self.buffer[key][idxs][:, i, j]
                            ).cuda()
                        temporal_samples[i][self.agents[j]]["act_onehot"] = (
                            F.one_hot(
                                torch.from_numpy(self.buffer["act"][idxs][:, i, j]),
                                num_classes=self.env_action_size,
                            )
                            .cuda()
                            .float()
                        )

                aux_loss = self.aux_loss_k * jpr_loss(
                    self.attention_net,
                    temporal_samples,
                    pred_only_latest_latent=self.aux_pred_only_latest_latent,
                )
                aux_loss.backward()
                self.master.apply_aux_gradients(self.attention_net)

            # log training info to tensorboard
            if self.worker_id == 0:
                log_dict = {}
                for act_id, act in enumerate(["env", "comm"][: self.num_acts]):
                    for agent_id, agent in enumerate(self.agents):
                        log_dict[f"{act}_policy_loss/{agent}"] = all_pls[act_id][
                            agent_id
                        ]
                        log_dict[f"{act}_value_loss/{agent}"] = all_vls[act_id][
                            agent_id
                        ]
                        log_dict[f"{act}_entropy/{agent}"] = all_els[act_id][agent_id]
                    log_dict[f"policy_loss/{act}"] = np.mean(all_pls[act_id])
                    log_dict[f"value_loss/{act}"] = np.mean(all_vls[act_id])
                    log_dict[f"entropy/{act}"] = np.mean(all_els[act_id])
                if self.add_auxiliary_loss:
                    log_dict["aux_loss"] = aux_loss.item()
                for k, v in log_dict.items():
                    self.master.writer.add_scalar(k, v, weight_iter)
                if self.log_queue:
                    log_dict["train_weight_iter"] = weight_iter
                    self.log_queue.put(log_dict)

            # all_pls, all_vls, all_els shape == (num_acts, num_agents)
            progress_str = self.pfmt.format(
                np.around(np.mean(all_pls, axis=-1), decimals=5),
                np.around(np.mean(all_vls, axis=-1), decimals=5),
                np.around(np.mean(all_els, axis=-1), decimals=5),
                np.around(np.mean(self.reward_log), decimals=2),
                np.around(aux_loss.item(), decimals=5)
                if self.add_auxiliary_loss
                else np.nan,
            )

            self.master.increment(progress_str)
            self.master.momentum_update()

        if self.log_queue:
            self.log_queue.put(f"worker {self.worker_id} is done.")
        else:
            print(f"worker {self.worker_id} is done.")

        return
