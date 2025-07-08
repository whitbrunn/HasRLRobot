import numpy as np

class PPOBuffer:
    def __init__(self, gamma=0.99, lam=0.95, use_gae=False):
        self.states = []
        self.actions = []
        self.alogs = []        
        self.rewards = []
        self.states_ = []
        self.values = []
        self.returns = []


        self.ep_returns = []  
        self.ep_lens = []

        self.gamma, self.lam = gamma, lam 

        self.ptr = 0  
        self.traj_idx = [0]  # 存储每个轨迹的起始位置，初始为 [0]

    def __len__(self):
        return len(self.states)  # 返回 states 列表的长度，即缓冲区中存储的条目数

    def storage_size(self):
        return len(self.states)

    def store(self, state, action, a_logprob, reward, value):
        # print(f"Here is store, a:{action}")
        # print(f"Here is store, alogb:{a_logprob}")
        self.states += [state]  # 将状态展平
        self.actions += [action]# 展平
        self.alogs += [a_logprob]
        self.rewards += [reward]# 奖励一般是标量
        self.values += [value]
        

        self.ptr += 1 # 每次存储后指针 ptr 自增 1

    def finish_path(self, last_val):  # 用于标记一个轨迹的结束，并计算对应的回报
        self.traj_idx += [self.ptr]  # 这个列表存储next轨迹开始的索引
        
        rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1]] # 提取从上一个轨迹终止位置到当前终止位置的所有奖励，self.traj_idx[-2] 表示上一个轨迹的终止位置（也是当前轨迹的起始位置），self.traj_idx[-1] 表示当前轨迹的终止位置
        returns = [] # returns：用来存储当前轨迹计算出来的回报。回报表示未来所有奖励的折现和

        R = float(last_val)
        for reward in reversed(rewards):
            R = self.gamma * R + reward
            returns.insert(0, R)

        self.returns += returns
        
        self.ep_returns += [np.sum(rewards)]# 记录轨迹奖励总和与轨迹长度
        self.ep_lens += [len(rewards)]

    def get(self, adv_scale):
        

        returns = np.array(self.returns) 
        values = np.array(self.values)
        advantages = returns - values
        advantages = adv_scale*(advantages - advantages.mean()) / (advantages.std() + 1e-5)

        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.alogs),
            returns,
            advantages
        )


def buffer_merge(buffers, gamma, lam):#合并多个 Buffer 实例
    merged = PPOBuffer(gamma, lam)
    for buf in buffers: # buffers 是多个 Buffer 的列表
        offset = len(merged)

        merged.states += buf.states
        merged.actions += buf.actions
        merged.rewards += buf.rewards
        merged.values += buf.values
        merged.returns += buf.returns
        merged.alogs += buf.alogs
        merged.ep_returns += buf.ep_returns
        merged.ep_lens += buf.ep_lens

        merged.traj_idx += [ offset + i for i in buf.traj_idx[1:]] 
        # merged.traj_idx 应该记录个buf在merge中的起始索引，而不是中止索引！
        merged.ptr += buf.ptr # 将每个 Buffer 中的 states 和 actions 添加到合并后的
        # merged 中，确保所有轨迹数据被合并为一个缓冲区
    return merged