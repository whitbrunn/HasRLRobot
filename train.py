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

        if hasattr(agent.actor, 'init_hidden_state'): # 检查策略网络是否有 init_hidden_state 方法，从而
            # 每次让隐藏状态都重置到初始状态
            agent.actor.init_hidden_state()

        if hasattr(agent.critic, 'init_hidden_state'): # 检查价值网络
            agent.critic.init_hidden_state()

        # 在轨迹结束前，生成动作，计算状态的价值，进行环境交互，并将结果存储到 memory
        while not done and traj_len < max_traj_len:
            action, a_logprob = agent.choose_action(state) 
            # print(f"Here is sample, a_logprob:{a_logprob}")
            # deterministic=False：表示使用随机策略，允许探索行为。如果设置为 True，策略将输出确定性动作，通常用于评估阶段
            # anneal=anneal：可以是一个控制探索与利用权衡的参数，用于调整动作的随机性，通常用于逐步减少探索
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

        observations, actions, alogps, returns, values = map(
            lambda x: torch.tensor(x, dtype=torch.float32, device=device),
            batch.get()
        )
        
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        
        
        optimizer_start = time()
        

        random_indices = SubsetRandomSampler(range(num_trajs)) # 首先使用 SubsetRandomSampler 从轨迹

        if minibatch_size > len(random_indices):
            minibatch_size = num_trajs
        hidden1, hidden2 = agent.actor.init_hidden_state(minibatch_size)
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
            
            hidden1, hidden2 = agent.update(obs_batch, action_batch,alogp_batch, return_batch, advantage_batch, hidden1, hidden2)

        opt_time = time() - optimizer_start


        if logger is not None:
            evaluate_start = time()
            test_steps = num_steps //2
            test_batch = test_sample_parallel(agent, gamma,lam, env_fn, test_steps, n_proc, max_traj_len, device, curr_thresh)
            eval_time = time() - evaluate_start
            
            avg_eval_reward = np.mean(test_batch.ep_returns)
            avg_batch_reward = np.mean(batch.ep_returns)
            avg_ep_len = np.mean(batch.ep_lens)


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
    parser.add_argument("--simrate", default=20, type=int, help="simrate of environment")################simrate##############3
    parser.add_argument("--run_name", default=None)  # run name
    parser.add_argument('--num_gpus', type=int, default=1, help='GPU')#GPU数量
    parser.add_argument("--learn_gains", default=False, action='store_true', dest='learn_gains')####是否学习增益#######
    parser.add_argument("--previous", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="./trained_models/ppo/")  # Where to log diagnostics to
    parser.add_argument("--seed", default=37, type=int)  # 设置Gym随机种子
    parser.add_argument("--history", default=0, type=int)  # number of previous states
    parser.add_argument("--redis_address", type=str, default=None)  # redis
    parser.add_argument("--env_name", default="me5418-Cassie-v0")
    # PPO algo args
    parser.add_argument("--input_norm_steps", type=int, default=50)###########输入归一化的步数################
    parser.add_argument("--n_itr", type=int, default=1000, help="Number of iterations of the learning algorithm")#############迭代轮数###############
    parser.add_argument("--hidden_width", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--use_lr_decay", type=bool, default=False)
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=bool, default=True, help="use Adam eps")### Adam 优化器的 epsilon 值########
    parser.add_argument("--lam", type=float, default=0.95, help="GAE")#####广义优势估计####
    parser.add_argument("--gamma", type=float, default=0.99, help="MDP")
    parser.add_argument("--learn_stddev", default=False, action='store_true', help="learn std_dev or keep it fixed")
    parser.add_argument("--anneal", default=1.0, action='store_true', help="anneal rate for stddev")##退火###
    parser.add_argument("--std_dev", type=int, default=-1.5, help="exponent of exploration std_dev")
    parser.add_argument("--entropy_coeff", type=float, default=0.01, help="Coefficient for entropy regularization")
    parser.add_argument("--clip", type=float, default=0.3,help="Clipping parameter for PPO surrogate loss")
    parser.add_argument("--minibatch_size", type=int, default=50, help="Batch size for PPO updates")###############PPO 更新时的minibatch大小############
    parser.add_argument("--epochs", type=int, default=10, help="Number of optimization epochs per PPO update")  ############epoch################
    parser.add_argument("--num_steps", type=int, default=1000,help="Number of sampled ")##每次梯度估计采样步数###
    parser.add_argument("--use_gae", type=bool, default=True,help="GAE")
    parser.add_argument("--num_procs", type=int, default=2, help="Number of threads to train on")###################并行化采样的步数#############
    parser.add_argument("--max_grad_norm", type=float, default=0.05, help="Value to clip gradients at.")#梯度裁剪的最大值#
    parser.add_argument("--max_traj_len", type=int, default=20, help="Max traj horizon")#最大轨迹长度#
    parser.add_argument("--bounded", type=bool, default=False)
    parser.add_argument("--debugger", type=bool, default=False)
    args = parser.parse_args()

    print()
    print("Synchronous Distributed Proximal Policy Optimization:")
    print(" ├ seed:           {}".format(args.seed))
    print(" ├ num procs:      {}".format(args.num_procs))
    print(" ├ lr_a:           {}".format(args.lr_a))
    print(" ├ lr_c:           {}".format(args.lr_c))
    print(" ├ lam:            {}".format(args.lam))
    print(" ├ gamma:          {}".format(args.gamma))
    print(" ├ learn stddev:   {}".format(args.learn_stddev))
    print(" ├ entropy coeff:  {}".format(args.entropy_coeff))
    print(" ├ clip:           {}".format(args.clip))
    print(" ├ minibatch size: {}".format(args.minibatch_size))
    print(" ├ epochs:         {}".format(args.epochs))
    print(" ├ num steps:      {}".format(args.num_steps))
    print(" ├ max traj len:   {}".format(args.max_traj_len))
    print(" └ debugger:       {}".format(args.debugger))
    print()
    train(args)


