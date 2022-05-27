from __future__ import absolute_import, division, print_function, unicode_literals

import time

import torch
import util.ops as ops
from torch.utils.tensorboard import SummaryWriter


def set_requires_grad(modules, value):
    for m in modules:
        for p in m.parameters():
            p.requires_grad = value


class Master(object):
    """ A master network. Think of it as a container that holds weight and the
    optimizer for the workers

    Args
        net: a neural network A3C model
        opt: shared optimizer
        gpu_id: gpu device id
    """

    def __init__(
        self,
        net,
        attention_net,
        opt,
        aux_opt,
        global_iter,
        global_done,
        master_lock,
        writer_dir,
        momentum_update_freq=2,
        momentum_tau=0.05,
        max_iteration=100,
        log_queue=None,
    ):
        self.lock = master_lock
        self.iter = global_iter
        self.done = global_done
        self.max_iteration = max_iteration
        self.momentum_update_freq = momentum_update_freq
        self.momentum_tau = momentum_tau
        self.net = net
        self.attention_net = attention_net
        self.opt = opt
        self.aux_opt = aux_opt
        if attention_net is not None:
            self.attention_net.input_processor = self.net.input_processor
            self.attention_net.share_memory()
        self.net.share_memory()
        self.writer_dir = writer_dir
        self.log_queue = log_queue
        self.start_time = time.time()

    def init_tensorboard(self):
        """ initializes tensorboard by the first worker """
        with self.lock:
            if not hasattr(self, "writer"):
                self.writer = SummaryWriter(self.writer_dir)
        return

    def copy_weights(self, net, attention_net, with_lock=False):
        """ copy weight from master """

        if with_lock:
            with self.lock:
                for p, mp in zip(net.parameters(), self.net.parameters()):
                    p.data.copy_(mp.data)
                if attention_net is not None:
                    for p, mp in zip(
                        attention_net.parameters(), self.attention_net.parameters()
                    ):
                        p.data.copy_(mp.data)
            return self.iter.value
        else:
            for p, mp in zip(net.parameters(), self.net.parameters()):
                p.data.copy_(mp.data)
            if attention_net is not None:
                for p, mp in zip(
                    attention_net.parameters(), self.attention_net.parameters()
                ):
                    p.data.copy_(mp.data)
            return self.iter.value

    def _apply_aux_gradients(self, attention_net):
        # backward prop and clip gradients
        self.aux_opt.zero_grad()
        torch.nn.utils.clip_grad_norm_(attention_net.parameters(), 10.0)
        for p, mp in zip(attention_net.parameters(), self.attention_net.parameters()):
            if p.grad is not None:
                mp.grad = p.grad.cpu()
        self.aux_opt.step()

    def apply_aux_gradients(self, attention_net, with_lock=False):
        """ apply gradient to the master network """
        if with_lock:
            with self.lock:
                self._apply_aux_gradients(attention_net)
        else:
            self._apply_aux_gradients(attention_net)
        return

    def _apply_gradients(self, net):
        # backward prop and clip gradients
        self.opt.zero_grad()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 40.0)
        for p, mp in zip(net.parameters(), self.net.parameters()):
            if p.grad is not None:
                mp.grad = p.grad.cpu()
        self.opt.step()

    def apply_gradients(self, net, with_lock=False):
        """ apply gradient to the master network """
        if with_lock:
            with self.lock:
                self._apply_gradients(net)
        else:
            self._apply_gradients(net)
        return

    def momentum_update(self, with_lock=False):
        if self.attention_net is None:
            return
        with self.iter.get_lock():
            if self.iter.value % self.momentum_update_freq != 0:
                return
        if with_lock:
            with self.lock:
                ops.soft_update_params(
                    self.attention_net.input_processor,
                    self.attention_net.target_input_processor,
                    self.momentum_tau,
                )
                ops.soft_update_params(
                    self.attention_net.projector,
                    self.attention_net.target_projector,
                    self.momentum_tau,
                )
        else:
            ops.soft_update_params(
                self.attention_net.input_processor,
                self.attention_net.target_input_processor,
                self.momentum_tau,
            )
            ops.soft_update_params(
                self.attention_net.projector,
                self.attention_net.target_projector,
                self.momentum_tau,
            )
        return

    def increment(self, progress_str=None):
        with self.iter.get_lock():
            self.iter.value += 1

            if self.iter.value % 100 == 0:
                if progress_str is not None:
                    s = f"[{self.iter.value}/{self.max_iteration}] {progress_str}"
                    t = f"time pass: {((time.time()-self.start_time)/60):.2f}mins"
                    if self.log_queue:
                        self.log_queue.put(f"{s} {t}")
                    else:
                        print(f"{s} {t}")

                else:
                    s = f"[{self.iter.value}/{self.max_iteration}] workers are working hard."
                    if self.log_queue:
                        self.log_queue.put(s)
                    else:
                        print(s)

            if self.iter.value > self.max_iteration:
                self.done.value = 1
        return

    def is_done(self):
        return self.done.value

    def save_ckpt(self, weight_iter, save_path):
        ckpt = {
            "net": self.net.state_dict(),
            "opt": self.opt.state_dict(),
            "iter": weight_iter,
        }
        if self.attention_net:
            ckpt["attention_net"] = self.attention_net.state_dict()
            ckpt["aux_opt"] = self.aux_opt.state_dict()
        torch.save(
            ckpt, save_path,
        )
