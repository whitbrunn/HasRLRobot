import torch
import gym
import gym_pkg  # 确保 gym_pkg 已正确安装并包含所需的环境
import numpy as np
import time

# 加载训练好的模型
def load_model(model_path):
    model = torch.load(model_path)
    model.eval()  # 设置模型为评估模式，以确保在测试过程中没有 dropout 或 batchnorm 之类的行为
    return model

# 运行环境并进行可视化
def visualize_policy(env, model, render=True):
    state= env.reset()  # 初始化环境
    done = False
    total_reward = 0.0

    while not done:
        if render:
            env.render()  # 可视化环境
            time.sleep(0.01)  # 控制可视化帧率

        # 状态转换为张量并传入模型
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # 确保在 CPU 上运行
        with torch.no_grad():
            action = model(state_tensor).numpy()[0]  # 获取模型动作，并将其移动到 CPU

        # 执行动作
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state

    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    # Gym 环境名和模型路径
    MODEL_PATH = '/home/jy/fun_project/Cassie_RL/cassie_ppo/trained_models/ppo/me5418-Cassie-v0/0517-10-15-26-s42/actor.pt'# 训练好的模型文件路径

    # 创建环境
    env = gym.make("me5418-Cassie-v0")

    # 加载训练好的 PPO 模型
    model = load_model(MODEL_PATH)

    # 开始可视化
    visualize_policy(env, model)

    # 关闭环境
    env.close()
