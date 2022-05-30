pip install -e ./marl-grid/env
pip install gym==0.23.0
wandb login 31ce01e4120061694da54a54ab0dafbee1262420
python marl-grid/find-goal/train.py gpu='[0,1,2,3,4,5,6,7]' num_workers=32 \
    use_wandb=true wandb_project_name="marl-jpr-comm" wandb_notes="eval 100 episodes"