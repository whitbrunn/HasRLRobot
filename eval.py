import torch
import gym
import gym_pkg  # 确保 gym_pkg 已正确安装并包含所需的环境
import numpy as np
import time
from torch.distributions import Normal
import os
import argparse


def get_dist(model, s, hidden1=None, hidden2=None):
    orig_shape = s.shape

    # 自动添加 batch 维 即可
    if s.dim() == 1:
        # Actually will not get in here.
        # print(f"Here is actor, {orig_shape}")
        s = s.unsqueeze(0).unsqueeze(0)
    elif s.dim() == 2:    
        s = s.unsqueeze(0)
    elif s.dim() == 3:            
        pass
    
    mean, std, (h1, h2) = model(s, hidden1, hidden2)

    if len(orig_shape) == 1:
        # print(f"Here is actor, {orig_shape}")
        mean = mean.squeeze(0).squeeze(0)
        std = std.squeeze(0).squeeze(0)
    elif len(orig_shape) == 2:
        mean = mean.squeeze(0)
        std = std.squeeze(0)

    return Normal(mean, std),(h1,h2)


def choose_action(model, max_action,  s):
    s = torch.tensor(s, dtype=torch.float)
    # print(f"\nHere is choose_action, {s.shape}.\n")
    
    with torch.no_grad():
        dist,_ = get_dist(model, s)
        a = dist.sample()  # Sample the action according to the probability distribution
        # print(f"Here is choose action, dist shape{dist.batch_shape}")
        a = torch.clamp(a, -max_action, max_action)  # [-max,max]
        a_logprob = dist.log_prob(a)  # The log probability density of the action
        # print(f"Here is choose_action, a_logprob:{a_logprob}")
        # print(f"Here is choose_action, a:{a.cpu().numpy().flatten()}")
        # print(f"Here is choose_action, a_logprob:{a_logprob.cpu().numpy().flatten()}")
    return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()


# 加载训练好的模型
def load_model(model_path):
    model = torch.load(model_path)
    model.eval()  # 设置模型为评估模式，以确保在测试过程中没有 dropout 或 batchnorm 之类的行为
    return model

# 运行环境并进行可视化
def visualize_policy(env, model, max_action=1, render=True):
    state= env.reset()  # 初始化环境
    done = False
    total_reward = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    while not done:
        if render:
            env.render()  # 可视化环境
            time.sleep(0.01)  # 控制可视化帧率

        # 状态转换为张量并传入模型
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device) 
        with torch.no_grad():
            action, _ = choose_action(model, max_action, state_tensor) # 获取模型动作，并将其移动到 CPU

        # 执行动作
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state

    print(f"Total reward: {total_reward}")

def eval(args):
    
    MODEL_PATH = os.path.join(args.exp_path, args.exp_id, "actor.pt") # 训练好的模型文件路径

    # 创建环境
    env = gym.make("me5418-Cassie-v0")

    # 加载训练好的 PPO 模型
    model = load_model(MODEL_PATH)

    # 开始可视化
    visualize_policy(env, model)

    # 关闭环境
    env.close()



if __name__ == "__main__":

    exp_path_d = os.path.abspath("./trained_models/ppo/me5418-Cassie-v0")
    exp_id_d = os.listdir(exp_path_d)[-1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=str, default=exp_id_d)
    parser.add_argument("--exp_path", type=str, default=exp_path_d)
    args = parser.parse_args()
    eval(args)

    