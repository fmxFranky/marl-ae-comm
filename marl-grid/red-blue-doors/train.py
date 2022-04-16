from __future__ import absolute_import, division, print_function, unicode_literals

import os
import os.path as osp

import hydra
import torch
import torch.multiprocessing as mp
from actor_critic import Evaluator, Master, Worker, WorkerAE
from envs.environments import make_environment
from model import AENetwork, AttentionModule, HardSharedNetwork, RichSharedNetwork
from omegaconf import DictConfig, OmegaConf
from util.misc import check_config, set_config, set_seed_everywhere
from util.shared_opt import SharedAdam
from util.wandb import WandbLoggingProcess


@hydra.main(config_name="config", config_path=".")
def main(cfg: DictConfig):
    check_config(cfg)
    set_config(cfg)
    set_seed_everywhere(cfg.seed)

    save_dir_fmt = osp.join(f"./{cfg.run_dir}", cfg.exp_name + "/{}")
    print(">> {}".format(cfg.exp_name))

    # (1) create environment
    create_env = lambda: make_environment(cfg.env_cfg)
    env = create_env()

    if "ae" in cfg.algo:
        create_net = lambda: AENetwork(
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
            create_net = lambda: HardSharedNetwork(
                obs_space=env.observation_space,
                action_size=env.action_space.n,
                num_agents=cfg.env_cfg.num_agents,
                num_blind_agents=cfg.env_cfg.num_blind_agents,
                share_critic=cfg.share_critic,
                layer_norm=cfg.layer_norm,
                observe_door=cfg.env_cfg.observe_door,
            )
        elif cfg.env_cfg.observation_style == "dict":
            create_net = lambda: RichSharedNetwork(
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

    create_attention_net = lambda: AttentionModule(
        feat_dim=cfg.img_feat_dim if "ae" in cfg.algo else 288,
        num_agents=cfg.env_cfg.num_agents,
        obs_space=env.observation_space,
        mlm_bsz=cfg.mlm_bsz,
        mlm_length=cfg.mlm_length,
        mlm_ratio=cfg.mlm_ratio,
        mlm_agents=cfg.mlm_agents,
    )

    # (2) create master network.
    # hogwild-style update will be applied to the master weight.
    master_lock = mp.Lock()
    net = create_net()
    net.share_memory()
    params = list(net.parameters())
    if cfg.mlm_encoded:
        attention_net = create_attention_net()
        attention_net.share_memory()
        params += list(attention_net.parameters())
    opt = SharedAdam(params, lr=cfg.lr)

    if cfg.resume_path:
        ckpt = torch.load(cfg.resume_path)
        global_iter = mp.Value("i", ckpt["iter"])
        net.load_state_dict(ckpt["net"])
        opt.load_state_dict(ckpt["opt"])
        print(">>>>> Loaded ckpt from iter", ckpt["iter"])
    else:
        global_iter = mp.Value("i", 0)
    global_done = mp.Value("i", 0)

    master = Master(
        net,
        attention_net if cfg.mlm_encoded else None,
        opt,
        global_iter,
        global_done,
        master_lock,
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
            dir=osp.join(f"./{cfg.run_dir}", cfg.exp_name),
            config=OmegaConf.to_object(cfg),
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
                    create_net().cuda(),
                    create_attention_net().cuda() if cfg.mlm_encoded is True else None,
                    create_env(),
                    worker_id=worker_id,
                    gpu_id=gpu_id,
                    num_acts=num_acts,
                    anneal_comm_rew=cfg.anneal_comm_rew,
                    ae_loss_k=cfg.ae_loss_k,
                    mlm_encoded=cfg.mlm_encoded,
                    mlm_rb_size=cfg.mlm_rb_size,
                    mlm_bsz=cfg.mlm_bsz,
                    mlm_length=cfg.mlm_length,
                    mlm_loss_k=cfg.mlm_loss_k,
                    log_queue=wandb_logger.log_queue if cfg.use_wandb else None,
                ),
            ]

    # (4) create a separate process to dump latest result (optional)
    eval_gpu_id = cfg.gpu[-1]

    with torch.cuda.device(eval_gpu_id):
        evaluator = Evaluator(
            master,
            create_net().cuda(),
            create_env(),
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
    # (0) args and steps to make this work.
    # Disable the python spawned processes from using multiple threads.
    mp.set_start_method("spawn", force=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
