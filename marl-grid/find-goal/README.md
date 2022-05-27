# ae-comm: FindGoal
This repository contains implementation of models used for `FindGoal` environment
in NeurIPS 2021 submission ***"Learning to Ground Multi-Agent Communication with Autoencoders"***.

## Requirements
Prerequisite: Python 3.6 or higher

To install MARL grid environment, follow instructions in `../env/README.md`.

To install other requirements:
```setup
pip install gym==0.23.0 easydict pillow scikit-video scikit-image wandb moviepy matplotlib opencv-python tensorboard hydra-core kornia
```

## Training
To train the models in the paper, run the following commands:
```train
# no-comm (baseline)
python train.py num_workers=4 gpu=[0]

# rl-comm (baseline)
python train.py num_workers=4 env_cfg.comm_len=10 gpu=[0]

# ae-rl-comm (baseline)
python train.py -num_workers=4 env_cfg.comm_len=10 ae_type='fc' gpu=[0]

# ae-comm (baseline)
python train.py -num_workers=4 env_cfg.comm_len=10 ae_type='' gpu=[0]

# ae-comm with auxilary task
python train.py -num_workers=4 env_cfg.comm_len=10 ae_type='' add_auxiliary_loss=true gpu=[0]
```

Videos, tensorboard logs, and checkpoints generated during training are saved in `./runs/${env}/${exp_name}` by default.

## Code layout

| Code          | Detail |
| :-------------: |:-------------:|
| actor_critic/master.py | A3C master weight and optimizer |
| actor_critic/worker.py | asynchronous worker for no-comm / rl-comm agent|
| actor_critic/worker_ae.py | asynchronous worker for ae-comm / ae-rl-comm agent|
| actor_critic/evaluator.py | separate worker to compute and log results |
| :-------------: |:-------------:|
| env/environment.py | entry point for all environments | 
| env/grid_world_environment.py | marl grid world environment wrapper |
| env/wrapper.py | additional wrappers for input preprocessing and output postprocessing |
| :-------------: |:-------------:|
| model/ae.py | base for ae-comm agent input & output | 
| model/rich.py | base for no-comm / rl-comm agent input & output | 

## License
All content in this repository is licensed under the MIT license.