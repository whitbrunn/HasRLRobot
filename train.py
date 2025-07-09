import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import argparse
import os
from time import time, sleep
from datetime import datetime
from ppo import PPO_Agent, PPOBuffer, buffer_merge
from utils import env_factory, WrapEnv, create_logger

import ray


def set_seed(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # 如果使用了 GPU，建议加上这行：
    if device == torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_save(save_path, agent):
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    
    torch.save(agent.actor, os.path.join(save_path, "actor.pt"))
    torch.save(agent.critic, os.path.join(save_path, "critic.pt"))


@ray.remote(num_gpus=1)# 并行化采样
@torch.no_grad()# 禁止梯度计算，加快速率
def sample(agent, gamma, lam, env_fn, min_steps, max_traj_len, device, term_thresh=0):
    # import torch
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)# 限制 PyTorch 只使用一个线程

    env = WrapEnv(env_fn)

    memory = PPOBuffer(gamma, lam)# 初始化环境和存储器 memory

    num_steps = 0
    while num_steps < min_steps:# 进入一个循环，直到采样的步数达到 min_steps
        state = env.reset()# 每次采样新轨迹时，环境都会被重置为初始状态，并转换成张量

        done = False
        value = 0
        traj_len = 0

        
        hidden1, hidden2 = agent.actor.init_hidden_state()
        # print("Here is sample, actor init")

        if hasattr(agent.critic, 'init_hidden_state'): # 检查价值网络
            agent.critic.init_hidden_state()
            print("Here is sample, critic init")

        # 在轨迹结束前，生成动作，计算状态的价值，进行环境交互，并将结果存储到 memory
        while not done and traj_len < max_traj_len:
            action, a_logprob , (hidden1, hidden2) = agent.choose_action(state, hidden1, hidden2) 
            # print(f"Here is sample, a_logprob:{a_logprob}")
            # deterministic=False：表示使用随机策略，允许探索行为。如果设置为 True，策略将输出确定性动作，通常用于评估阶段
            with torch.no_grad():    
                value = agent.critic(torch.from_numpy(state).float().to(device))
                value = value.cpu().numpy()

            state_, reward, done, _ = env.step(action, term_thresh=term_thresh)
            # env.step() 执行一部动作，并返回下一个状态，奖励 是否结束 附加信息
            
            memory.store(state.squeeze(0), action, [a_logprob.sum()], reward, value.squeeze(0)) # 环境返回的下一个状态转换为 PyTorch 张量
            state = state_
            traj_len += 1
            num_steps += 1
        with torch.no_grad():    
            value = agent.critic(torch.from_numpy(state).float().to(device))
            value = value.cpu().numpy()
        # print(f"\nHere is sample, {value.shape}.\n")
        # print(f"Here is sample, last value={(not done) * value.squeeze(0)}")
        memory.finish_path(last_val=(not done) * value.squeeze(0))

    return memory



def sample_parallel(worker, w_args, workers, total_steps, result):

    ready_ids, _ = ray.wait(workers, num_returns=1)

    # update result
    result.append(ray.get(ready_ids[0])) # ray.get(ready_ids[0]) 获取已经完成任务的工作者的结果，并将其添加到 result


    workers.remove(ready_ids[0])# 从工作者列表 workers 中移除已经完成任务的工作者 (ready_ids[0])

    total_steps += len(result[-1]) 
    workers.append(worker.remote(*w_args)) 

    return total_steps, result


def test_sample_parallel(agent, gamma,lam, env_fn, test_steps, n_proc, max_traj_len, device, curr_thresh):
    worker = sample    
    w_args = (
    agent, gamma, lam, env_fn, test_steps // n_proc, max_traj_len, device, curr_thresh)

    workers = [worker.remote(*w_args) for _ in range(n_proc)]

    result = []
    total_steps = 0

    while total_steps < test_steps:
        total_steps, result = sample_parallel(worker, w_args, workers, total_steps,result)
    batch = buffer_merge(result,gamma,lam)

    return batch

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gamma = args.gamma
    lam = args.lam # GAE的参数
    minibatch_size = args.minibatch_size # 每次更新策略时数据集被拆分为若干个小批量，每次用一个小批量的数据进行参数更新
    num_steps = args.num_steps
    max_traj_len = args.max_traj_len
    n_proc = args.num_procs

    highest_reward = -1
    
    logger = None
    if args.debugger == False:
        logger = create_logger(args)

    env_fn = env_factory(args.env_name)
    args.state_dim = env_fn().observation_space.shape[0]
    args.action_dim = env_fn().action_space.shape[0]
    args.max_action = float(env_fn().action_space.high[0])

    # 限制 PyTorch 在并行计算中只使用一个线程，以避免与 Ray 的多进程并行出现冲突
    os.environ['OMP_NUM_THREADS'] = '1'
    if not ray.is_initialized():
        if args.redis_address is not None:
            ray.init(num_cpus=args.num_procs, num_gpus=args.num_gpus,redis_address=args.redis_address)
        else:
            ray.init(num_cpus=args.num_procs, num_gpus=args.num_gpus)


    set_seed(args.seed, device)


    agent = PPO_Agent(args,device)

    
    curr_thresh = 0
    start_itr = 0
    for itr in range(args.n_itr):
        print("================= Iteration {} =====================".format(itr))

        sample_start = time()
        
        if curr_thresh < 0.35:
            curr_thresh = .1 * 1.0006 ** (itr - start_itr)
        
        worker = sample
        w_args = (
        agent, gamma, lam, env_fn, num_steps // n_proc, max_traj_len, device, curr_thresh)

        workers = [worker.remote(*w_args) for _ in range(n_proc)]

        result = []
        total_steps = 0

        while total_steps < num_steps:
            total_steps, result = sample_parallel(worker, w_args, workers, total_steps,result)

            if args.debugger == True:
                sleep(2)
        batch = buffer_merge(result, gamma,lam)

        samp_time = time() - sample_start

        num_trajs = len(batch.traj_idx)-1
        # print(f"Here is train, batch traj indx: {batch.traj_idx}")
        print(f"Here is train, batch traj nums: {num_trajs}")
        # print(f"Here is train, batch reward nums: {len(batch.rewards)}")
        print(f"Here is train, batch return nums: {len(batch.returns)}")

        observations, actions, alogps, returns, advantages = map(
            lambda x: torch.tensor(x, dtype=torch.float32, device=device),
            batch.get(args.adv_scale)  
        )
        
        optimizer_start = time()

        random_indices = SubsetRandomSampler(range(num_trajs)) # 首先使用 SubsetRandomSampler 从轨迹

        if minibatch_size > len(random_indices):
            minibatch_size = num_trajs
        sampler = BatchSampler(random_indices, minibatch_size, drop_last=True)
        
        for indices in sampler:
            obs_batch = [observations[batch.traj_idx[i]:batch.traj_idx[i + 1]] for i in indices]
            action_batch = [actions[batch.traj_idx[i]:batch.traj_idx[i + 1]] for i in indices]
            alogp_batch = [alogps[batch.traj_idx[i]:batch.traj_idx[i + 1]] for i in indices]
            return_batch = [returns[batch.traj_idx[i]:batch.traj_idx[i + 1]] for i in indices]
            advantage_batch = [advantages[batch.traj_idx[i]:batch.traj_idx[i + 1]] for i in indices]
            

            obs_batch = pad_sequence(obs_batch, batch_first=True)
            action_batch = pad_sequence(action_batch, batch_first=True)
            alogp_batch = pad_sequence(alogp_batch, batch_first=True)
            return_batch = pad_sequence(return_batch, batch_first=True)
            advantage_batch = pad_sequence(advantage_batch, batch_first=True)
            
            actor_loss, entropy_mean,critic_loss, ratio_mean = agent.update(obs_batch, 
                                                                            action_batch,
                                                                            alogp_batch, 
                                                                            return_batch, 
                                                                            advantage_batch,
                                                                            itr)
            # print(f"Here is train, entropy{entropy_mean}")
        opt_time = time() - optimizer_start


        if logger is not None:
            evaluate_start = time()
            test_steps = num_steps //2
            test_batch = test_sample_parallel(agent, gamma,lam, env_fn, test_steps, n_proc, max_traj_len, device, curr_thresh)
            eval_time = time() - evaluate_start
            
            avg_eval_reward = np.mean(test_batch.ep_returns)
            avg_batch_reward = np.mean(batch.ep_returns)
            avg_ep_len = np.mean(batch.ep_lens)

            logger.add_scalar("Train/Actor Loss", actor_loss, itr)
            logger.add_scalar("Train/Entopy(Mean)", entropy_mean, itr)
            logger.add_scalar("Train/Critic Loss", critic_loss, itr)
            logger.add_scalar("Train/Ratio(Mean)", ratio_mean, itr)

            logger.add_scalar("Rewards/Test", avg_eval_reward, itr)
            logger.add_scalar("Rewards/Train", avg_batch_reward, itr)

            logger.add_scalar("T/Sample Times", samp_time, itr)
            logger.add_scalar("T/Optimize Times", opt_time, itr)
            logger.add_scalar("T/Test Times", eval_time, itr)

            if highest_reward < avg_eval_reward:
                highest_reward = avg_eval_reward
                model_save(logger.dir, agent) # 如果当前的评估奖励 (avg_eval_reward) 大于之前记录的最高奖励avg_eval_reward
                # 则更新最高奖励并保存模型的参数

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝ 1 General & Experiment Settings ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    parser.add_argument("--run_name", type=str, default=None,
                        help="Unique identifier for this training run (used in logs and checkpoints)")
    parser.add_argument("--previous", type=str, default=None,
                        help="Path to a previous run to resume or fine-tune from")
    parser.add_argument("--logdir", type=str, default="./trained_models/ppo/",
                        help="Root directory for tensorboard logs and saved models")
    parser.add_argument("--seed", type=int, default=37,
                        help="Global random seed for reproducibility")
    parser.add_argument("--debugger", action="store_true",
                        help="Enable verbose debugging mode")

    # ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝ 2 Environment & Resource Settings ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to use")
    parser.add_argument("--redis_address", type=str, default=None,
                        help="Redis server address for distributed rollout (if applicable)")
    parser.add_argument("--env_name", type=str, default="me5418-Cassie-v0",
                        help="Gym-compatible environment ID")

    # ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝ 3 Network Architecture & Init ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    parser.add_argument("--hidden_width", type=int, default=128,
                        help="Number of neurons per hidden layer")
    parser.add_argument("--use_tanh", type=bool, default=True,
                        help="Use tanh as activation function; otherwise ReLU")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True,
                        help="Apply orthogonal weight initialization (PPO Trick 8)")
    parser.add_argument("--set_adam_eps", type=bool, default=True,
                        help="Explicitly set Adam's ε to improve numerical stability")

    # ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝ 4 Learning‑Rate & Optimizer ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    parser.add_argument("--lr_a", type=float, default=3e-4,
                        help="Actor network learning rate")
    parser.add_argument("--lr_c", type=float, default=3e-4,
                        help="Critic network learning rate")
    parser.add_argument("--use_lr_decay", type=bool, default=True,
                        help="Enable linear learning-rate decay across iterations")

    # ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝ 5 Training Schedule & Sampling ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    parser.add_argument("--n_itr", type=int, default=10000,
                        help="Total number of training iterations")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Optimization epochs per PPO update")
    parser.add_argument("--minibatch_size", type=int, default=50,
                        help="Minibatch size for each gradient step")
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Environment steps collected per iteration per worker")
    parser.add_argument("--num_procs", type=int, default=2,
                        help="Number of parallel environment workers")
    parser.add_argument("--max_grad_norm", type=float, default=0.05,
                        help="Gradient-clipping threshold (L2 norm)")
    parser.add_argument("--max_traj_len", type=int, default=20,
                        help="Maximum horizon length of one trajectory")
    parser.add_argument("--adv_scale", type=float, default=1.0,
                        help="Scalar to rescale advantages before policy update")

    # ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝ 6 Core RL / PPO Hyper‑parameters ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for future rewards")
    parser.add_argument("--lam", type=float, default=0.95,
                        help="GAE lambda parameter")
    parser.add_argument("--use_gae", type=bool, default=False,
                        help="Enable Generalized Advantage Estimation (GAE)")
    parser.add_argument("--clip", type=float, default=0.3,
                        help="PPO clipping parameter for policy ratio")
    parser.add_argument("--entropy_coeff", type=float, default=0.005,
                        help="Entropy regularization coefficient")
    parser.add_argument("--critic_loss_scale", type=float, default=1.0,
                        help="Weight applied to critic (value) loss in total loss")

    # ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝ 7 Optional Features ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    parser.add_argument("--learn_gains", action="store_true", default=False,
                        help="Learn feedback gains instead of using fixed values")

    args = parser.parse_args()

    print()
    print("Synchronous Distributed PPO Configuration:")
    print(" ┌──── Experiment Settings")
    print(" ├ seed:                 {}".format(args.seed))
    print(" ├ debugger:             {}".format(args.debugger))

    print(" ├──── Parallelism & Environment")
    print(" ├ num procs:            {}".format(args.num_procs))

    print(" ├──── Learning Rates & Optimizer")
    print(" ├ lr_actor:             {}".format(args.lr_a))
    print(" ├ lr_critic:            {}".format(args.lr_c))
    print(" ├ use_lr_decay:         {}".format(args.use_lr_decay))

    print(" ├──── PPO Core Settings")
    print(" ├ gamma:                {}".format(args.gamma))
    print(" ├ lambda:               {}".format(args.lam))
    print(" ├ entropy coeff:        {}".format(args.entropy_coeff))
    print(" ├ clip ratio:           {}".format(args.clip))

    print(" ├──── Training Schedule")
    print(" ├ steps per iter:       {}".format(args.num_steps))
    print(" ├ minibatch size:       {}".format(args.minibatch_size))
    print(" ├ epochs per update:    {}".format(args.epochs))
    print(" ├ max trajectory len:   {}".format(args.max_traj_len))

    print(" └──── Network Config")
    print("   use orthogonal init:  {}".format(args.use_orthogonal_init))
    print()
    train(args)


