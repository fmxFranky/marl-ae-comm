from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import os
import os.path as osp

import hydra
import torch
import torch.multiprocessing as mp
from actor_critic import Evaluator, Master, Worker, WorkerAE
from envs.environments import make_environment
from model import (
    AENetwork,
    AttentionModule,
    HardSharedNetwork,
    JointPredReprModule,
    RichSharedNetwork,
)
from omegaconf import DictConfig, OmegaConf
from util.misc import check_config, set_config, set_seed_everywhere
from util.shared_opt import SharedAdam
from util.wandb import WandbLoggingProcess


@hydra.main(config_name="config", config_path=".")
def main(cfg: DictConfig):
    check_config(cfg)
    set_config(cfg)
    set_seed_everywhere(cfg.seed)

    if cfg.run_dir is None:
        save_dir = os.getcwd()
    elif os.path.isabs(cfg.run_dir):
        save_dir = cfg.run_dir
    else:
        save_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.run_dir)
    save_dir_fmt = osp.join(save_dir, cfg.exp_name + "/{}")
    print(">> {}".format(cfg.exp_name))

    # (1) create environment
    env = make_environment(cfg.env_cfg)

    if "ae" in cfg.algo:
        net = AENetwork(
            obs_space=env.observation_space,
            act_space=env.action_space,
            num_agents=cfg.env_cfg.num_agents,
            comm_len=cfg.env_cfg.comm_len,
            discrete_comm=cfg.env_cfg.discrete_comm,
            ae_pg=cfg.ae_pg,
            ae_type=cfg.ae_type,
            img_feat_dim=cfg.img_feat_dim,
        )
    else:
        if cfg.env_cfg.observation_style == "dict" and cfg.env_cfg.comm_len <= 0:
            net = HardSharedNetwork(
                obs_space=env.observation_space,
                action_size=env.action_space.n,
                num_agents=cfg.env_cfg.num_agents,
                num_blind_agents=cfg.env_cfg.num_blind_agents,
                share_critic=cfg.share_critic,
                layer_norm=cfg.layer_norm,
            )
        elif cfg.env_cfg.observation_style == "dict":
            net = RichSharedNetwork(
                obs_space=env.observation_space,
                act_space=env.action_space,
                num_agents=cfg.env_cfg.num_agents,
                comm_size=2,
                comm_len=cfg.env_cfg.comm_len,
                discrete_comm=cfg.env_cfg.discrete_comm,
                num_blind_agents=cfg.env_cfg.num_blind_agents,
                share_critic=cfg.share_critic,
                layer_norm=cfg.layer_norm,
                comm_rnn=cfg.comm_rnn,
            )
        else:
            raise ValueError(
                "Observation style {} not supported".format(
                    cfg.env_cfg.observation_style
                )
            )

    # (2) create master network.
    # hogwild-style update will be applied to the master weight.
    master_lock = mp.Lock()
    net.share_memory()
    params = list(net.parameters())
    if cfg.add_auxiliary_loss:
        attention_net = JointPredReprModule(
            input_processor=net.input_processor,
            feat_dim=net.comm_ae.preprocessor.feat_dim if "ae" in cfg.algo else 288,
            num_agents=cfg.env_cfg.num_agents,
            obs_space=env.observation_space,
            act_space=env.action_space,
            jpr_bsz=cfg.aux_bsz,
            jpr_length=cfg.aux_length,
            jpr_agents=cfg.aux_agents,
        )
        attention_net.share_memory()
        aux_opt = SharedAdam(attention_net.parameters(), lr=cfg.aux_lr)

    opt = SharedAdam(net.parameters(), lr=cfg.lr)

    if cfg.resume_path:
        ckpt = torch.load(cfg.resume_path)
        global_iter = mp.Value("i", ckpt["iter"])
        net.load_state_dict(ckpt["net"])
        opt.load_state_dict(ckpt["opt"])
        if cfg.add_auxiliary_loss:
            attention_net.load_state_dict(ckpt["attention_net"])
            aux_opt.load_state_dict(ckpt["aux_opt"])
        print(">>>>> Loaded ckpt from iter", ckpt["iter"])
    else:
        global_iter = mp.Value("i", 0)
    global_done = mp.Value("i", 0)

    master = Master(
        net,
        attention_net if cfg.add_auxiliary_loss else None,
        opt,
        aux_opt if cfg.add_auxiliary_loss else None,
        global_iter,
        global_done,
        master_lock,
        momentum_update_freq=cfg.momentum_update_freq,
        momentum_tau=cfg.momentum_tau,
        writer_dir=save_dir_fmt.format("tb"),
        max_iteration=cfg.train_iter,
    )

    if cfg.use_wandb:
        wandb_logger = WandbLoggingProcess(
            master,
            save_dir_fmt=save_dir_fmt.format(""),
            log_queue=mp.Queue(),
            name=cfg.exp_name,
            project=cfg.wandb_project_name,
            dir=save_dir_fmt.format(""),
            config=OmegaConf.to_object(cfg),
            notes=cfg.get("wandb_notes", None),
        )
        wandb_logger.start()

    # (3) create workers
    WorkerInstance = WorkerAE if "ae" in cfg.algo else Worker
    num_acts = 1
    if cfg.env_cfg.comm_len > 0 and "ae" not in cfg.algo:
        num_acts = 2
    workers = []
    for worker_id in range(cfg.num_workers):
        gpu_id = cfg.gpu[worker_id % len(cfg.gpu)]
        print(f"(worker {worker_id}) initializing on gpu {gpu_id}")

        with torch.cuda.device(gpu_id):
            workers += [
                WorkerInstance(
                    master,
                    copy.deepcopy(net).cuda(),
                    copy.deepcopy(attention_net).cuda()
                    if cfg.add_auxiliary_loss is True
                    else None,
                    copy.deepcopy(env),
                    worker_id=worker_id,
                    gpu_id=gpu_id,
                    num_acts=num_acts,
                    anneal_comm_rew=cfg.anneal_comm_rew,
                    ae_loss_k=cfg.ae_loss_k,
                    add_auxiliary_loss=cfg.add_auxiliary_loss,
                    aux_rb_size=cfg.aux_rb_size,
                    aux_bsz=cfg.aux_bsz,
                    aux_length=cfg.aux_length,
                    aux_loss_k=cfg.aux_loss_k,
                    log_queue=wandb_logger.log_queue if cfg.use_wandb else None,
                ),
            ]

    # (4) create a separate process to dump latest result (optional)
    eval_gpu_id = cfg.gpu[-1]

    with torch.cuda.device(eval_gpu_id):
        evaluator = Evaluator(
            master,
            copy.deepcopy(net).cuda(),
            copy.deepcopy(env),
            save_dir_fmt=save_dir_fmt,
            gpu_id=eval_gpu_id,
            sleep_duration=10,
            video_save_freq=cfg.video_save_freq,
            ckpt_save_freq=cfg.ckpt_save_freq,
            num_eval_episodes=cfg.num_eval_episodes,
            net_type="ae" if "ae" in cfg.algo else "",
            log_queue=wandb_logger.log_queue if cfg.use_wandb else None,
        )
        workers.append(evaluator)

    # (5) start training

    # > start the processes
    [w.start() for w in workers]

    # > join when done
    [w.join() for w in workers]

    master.save_ckpt(
        cfg.train_iter, osp.join(save_dir_fmt.format("ckpt"), "latest.pth")
    )

    if cfg.use_wandb:
        wandb_logger.log_queue.put(None)
        wandb_logger.join()


if __name__ == "__main__":
    # (0) cfg and steps to make this work.
    # Disable the python spawned processes from using multiple threads.
    mp.set_start_method("spawn", force=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
