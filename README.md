# CassieGymRL: PPO is All You Need
## 1 What we have

- Cassie OpenAI Gym environment*[1]
    - Gait cycle-based reward function design for gait locomotion
- Scalable 2-layer LSTM network of policy function
- Ray-based multi-process sample on CPU
- PPO loss-based agent upgrade*[2] on GPU
- An Intuitive Tensorboard-based training process demonstration

The framework of CassieGymRL: PPO is All You Need is shown below:
![The overall framework of our method.](./readme-images/framework.svg)


## 2 How to use

2.1 Installation

1) Environment,

- `Ubuntu20.04`
- `Python 3.9`
- `Pytorch 2.4.0`
- `MUJOCO2.0.0`

Note:
- If encounter `ERROR: Could not open activation key file (null)`, it is a mujoco error, try to add `export MUJOCO_KEY_PATH=/path/to/.mujoco/mjkey.txt` to the last line of `~/.bashrc`, then `source ~/.bashrc`[3].


2) Install `gym=0.21`,

```
$conda install -c conda-forge gym=0.21.0
```

3) Install other dependencies,

```
pip install -r requirements.txt
```


2.2 Usage

First, run the `train.py` to train the robot agent model, e.g.,

```
$ python train.py --num_steps 2000 --entropy_coeff 0.0005 --max_traj_len 100 \
--lr_a 1e-3 --lr_c 1e-3 --hidden_width 256 --num_procs 2 --minibatch_size 100 --critic_loss_scale 2 --seed 3737
```

Second, open another terminal, run the following command to check the training process,

```
$ tensorboard --logdir=/home/.../This_Project_Dir/trained_models/ppo

# Note: Copying the full path would be recommended
```

![The tensorboard.](./readme-images/tensorboard.png)

Third, run `eval.py` directly, and a video will show out with `Total reward` displaying in the terminal.

E.g.,

```
python eval.py --exp_id 0524-16-20-23-s42

# Note:
# 1. If output "Segmentation fault (core dumped)", it is normal, try to run the command again.
# 2. The exp_id is set defaultly as your latest one.
```



## 3 One more thing

The authors would like to express heartfelt thanks to Prof. Guillaume Adrien Sartoretti@NUS, for his invaluable guidance. The author also thanks his teammates Duhy and Guozc for their fundamental works.



## Reference
*[1] Adapted from https://github.com/osudrl/cassie-mujoco-sim, https://github.com/lbermillo/cassie-curriculum-learning/ and https://github.com/XHN-1/Cassie_mujoco_RL.*

*[2] Adapted from https://github.com/Lizhi-sjtu/DRL-code-pytorch.*

*[3] https://github.com/osudrl/apex/issues/25.*

---

Appendice A: Gait Cycle Reward Design (in Gym)

![Appendice A.](./readme-images/gait_cycle_design.svg)
