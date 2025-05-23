# CassieGymRL: PPO is All You Need
## 1 What we have

- Cassie OpenAI Gym environment*[1]
    - Gait cycle-based reward function design for gait locomotion
- Scalable 2-layer LSTM network of policy function
- Ray-based multi-process sample on CPU
- PPO loss-based agent upgrade*[2] on GPU
- An Intuitive Tensorboard-based training process demonstration

![The overall framework of our method.](./readme-images/framework.svg)


## 2 How to use

2.1 Installation

1) Environment

You must install the following environment:

`Ubuntu20.04`, `Python 3.9`, `Pytorch 2.40`, `MUJOCO2.0.0`


2) Install gym=0.21

```
$conda install -c conda-forge gym=0.21.0
```


2.2 Usage

First, then activate the conda environment, run the `train.py` to train the robot agent model.

```
$ python train.py
```

Second, run the following command to check the training process.

```
$ tensorboard --logdir=./trained_models/ppo--------------(copying the full path would be recommended)
```


Third, run `eval.py` directly, and a video will show out.

```
python eval.py model_path=----------------------(set you model path)
```



## 3 One more thing

The authors would like to express heartfelt thanks to Prof. Guillaume Adrien Sartoretti@NUS, for his invaluable guidance. The author also would like to thank so many companies are hiring RL engineering. Now I am still open-hiring, please contact maij@u.nus.edu if you are willing to offer an opportunity.



## Reference
*[1] Adapted from https://github.com/osudrl/cassie-mujoco-sim, https://github.com/lbermillo/cassie-curriculum-learning/ and https://github.com/XHN-1/Cassie_mujoco_RL.*
*[2] Adapted from https://github.com/Lizhi-sjtu/DRL-code-pytorch*


---

Appendice A: Gait Cycle Reward Design (in Gym)

![The overall framework of our method.](./readme-images/gait_cycle_design.svg)
