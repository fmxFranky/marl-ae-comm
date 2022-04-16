from __future__ import absolute_import, division, print_function, unicode_literals

from collections import deque

import numpy as np
import torch
import torch.multiprocessing as mp
from loss import mlm_loss, policy_gradient_loss
from util import ops
from util.decorator import within_cuda_device
from util.misc import check_done


class WorkerAE(mp.Process):
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
        gamma=0.99,
        tau=1.0,
        ae_loss_k=1.0,
        mlm_encoded=False,
        mlm_rb_size=int(1e5),
        mlm_bsz=64,
        mlm_length=10,
        mlm_loss_k=0.1,
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
        self.gamma = gamma
        self.tau = tau
        self.gpu_id = gpu_id
        self.reward_log = deque(maxlen=5)  # track last 5 finished rewards
        self.pfmt = "policy loss: {} value loss: {} entropy loss: {} ae loss: {} reward: {} mlm_loss: {}"
        self.agents = [f"agent_{i}" for i in range(self.env.num_agents)]
        self.num_acts = 1
        self.ae_loss_k = ae_loss_k
        self.mlm_encoded = mlm_encoded
        self.mlm_rb_size = mlm_rb_size
        self.mlm_bsz = mlm_bsz
        self.mlm_length = mlm_length
        self.mlm_loss_k = mlm_loss_k
        self.log_queue = log_queue

    @within_cuda_device
    def get_trajectory(self, hidden_state, state_var, done):
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
        trajectory = [[] for _ in range(self.num_acts)]

        while not check_done(done) and len(trajectory[0]) < self.t_max:
            if self.mlm_encoded:
                # get the encoded representation of the current state
                np.copyto(
                    self.obs_buffer[self.mlm_rb_idx],
                    torch.cat(
                        [
                            state_var[f"agent_{i}"]["pov"]
                            for i in range(self.env.num_agents)
                        ],
                        dim=0,
                    )
                    .cpu()
                    .numpy(),
                )
                self.mlm_rb_idx = (self.mlm_rb_idx + 1) % self.mlm_rb_size
                self.mlm_rb_full = self.mlm_rb_full or self.mlm_rb_idx == 0

            plogit, value, hidden_state, comm_out, comm_ae_loss = self.net(
                state_var, hidden_state
            )
            action, _, _, all_actions = self.net.take_action(plogit, comm_out)
            state, reward, done, info = self.env.step(all_actions)
            state_var = ops.to_state_var(state)

            # assert self.num_acts == 1:
            trajectory[0].append((plogit, action, value, reward, comm_ae_loss))

        # end condition
        if check_done(done):
            target_value = [{k: 0 for k in self.agents} for _ in range(self.num_acts)]
        else:
            with torch.no_grad():
                target_value = self.net(state_var, hidden_state)[1]
                if self.num_acts == 1:
                    target_value = [target_value]

        #  compute Loss: accumulate rewards and compute gradient
        values = [{k: None for k in self.agents} for _ in range(self.num_acts)]

        # GAE
        for k in self.agents:
            for aid in range(self.num_acts):
                values[aid][k] = [x[k] for x in list(zip(*trajectory[aid]))[2]]
                values[aid][k].append(ops.to_torch([target_value[aid][k]]))
                values[aid][k].reverse()

        return trajectory, values, target_value, done

    @within_cuda_device
    def run(self):
        self.master.init_tensorboard()
        done = True
        reward_log = 0.0

        if self.mlm_encoded:
            shp = list(ops.to_state_var(self.env.reset())["agent_0"]["pov"].shape)
            shp[0] *= self.env.num_agents
            self.obs_buffer = np.empty([self.mlm_rb_size, *shp], dtype=np.float32)
            self.mlm_rb_idx = 0
            self.mlm_rb_full = False

        while not self.master.is_done():
            # synchronize network parameters
            weight_iter = self.master.copy_weights(self.net)
            self.net.zero_grad()

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
                hidden_state, state_var, done
            )

            all_pls = [[] for _ in range(self.num_acts)]
            all_vls = [[] for _ in range(self.num_acts)]
            all_els = [[] for _ in range(self.num_acts)]

            comm_ae_losses = []

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
                    for i, (pi_logit, action, value, reward, comm_ae_loss) in enumerate(
                        traj
                    ):
                        comm_ae_losses.append(comm_ae_loss.item())

                        # Agent A3C Loss
                        t_value = reward[agent] + self.gamma * t_value
                        advantage = t_value - value[agent]

                        # Generalized advantage estimation (GAE)
                        delta_t = (
                            reward[agent]
                            + self.gamma * val[agent][i].data
                            - val[agent][i + 1].data
                        )
                        gae = gae * self.gamma * self.tau + delta_t

                        tl, (pl, vl, el) = policy_gradient_loss(
                            pi_logit[agent], action[agent], advantage, gae=gae
                        )

                        pls.append(ops.to_numpy(pl))
                        vls.append(ops.to_numpy(vl))
                        els.append(ops.to_numpy(el))

                        reward_log += reward[agent]
                        loss += tl + comm_ae_loss * self.ae_loss_k

                    all_pls[aid].append(np.mean(pls))
                    all_vls[aid].append(np.mean(vls))
                    all_els[aid].append(np.mean(els))

            # auxilary task
            if self.mlm_encoded:
                idxs = np.random.randint(
                    0,
                    self.mlm_rb_size - self.mlm_length
                    if self.mlm_rb_full
                    else self.mlm_rb_idx - self.mlm_length,
                    size=self.mlm_bsz,
                ).reshape(-1, 1)
                step = np.arange(self.mlm_length).reshape(1, -1)
                idxs = idxs + step
                obses = torch.from_numpy(self.obs_buffer[idxs]).cuda()
                al = mlm_loss(self.net.input_processor, self.attention_net, obses)
                loss += self.mlm_loss_k * al

            # accumulate gradient locally
            loss.backward()

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
                log_dict["ae_loss"] = np.mean(comm_ae_losses)
                if self.mlm_encoded:
                    log_dict["mlm_loss"] = al.item()
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
                np.around(np.mean(comm_ae_losses), decimals=5),
                np.around(np.mean(self.reward_log), decimals=2),
                np.around(al.item(), decimals=5) if self.mlm_encoded else np.nan,
            )

            self.master.apply_gradients(self.net)
            self.master.increment(progress_str)

        print(f"worker {self.worker_id} is done.")
        return
