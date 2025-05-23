import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Beta, Normal


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Beta(nn.Module):
    def __init__(self, args):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.alpha_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.beta_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean


class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = self.max_action * torch.tanh(self.mean_layer(s))  # [-1,1]->[-max_action,max_action]
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist

class Actor_LSTM(nn.Module):
    def __init__(self, args, device, dist_type="Gaussian"):
        super(Actor_LSTM, self).__init__()
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.hidden_width = args.hidden_width
        self.max_action = args.max_action
        self.dist_type = dist_type
        self.device = device

        self.lstm1 = nn.LSTM(input_size=self.state_dim,
                             hidden_size=self.hidden_width,
                             batch_first=True)

        self.lstm2 = nn.LSTM(input_size=self.hidden_width,
                             hidden_size=self.hidden_width,
                             batch_first=True)

        if dist_type == "Beta":
            self.alpha_layer = nn.Linear(self.hidden_width, self.action_dim)
            self.beta_layer = nn.Linear(self.hidden_width, self.action_dim)
        else:
            self.mean_layer = nn.Linear(self.hidden_width, self.action_dim)
            self.log_std = nn.Parameter(torch.zeros(1, self.action_dim))

        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

    def init_hidden_state(self, batch_size=1):
        h0 = torch.zeros(1, batch_size, self.hidden_width).to(self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_width).to(self.device)
        return (h0.clone(), c0.clone()), (h0.clone(), c0.clone())

    def forward(self, s, hidden1=None, hidden2=None):
        """
        s: (state_dim,) or (batch_size, seq_len, state_dim)
        hidden1, hidden2: optional hidden states (h, c) tuple
        """
        if hidden1 is None or hidden2 is None:
            batch_size = s.shape[0]
            hidden1, hidden2 = self.init_hidden_state(batch_size)

        out1, h1 = self.lstm1(s, hidden1)  # [B, T, H]
        out1 = self.activate_func(out1)
        out2, h2 = self.lstm2(out1, hidden2)  # [B, T, H]
        out2 = self.activate_func(out2)

        
        mean = self.max_action * torch.tanh(self.mean_layer(out2))
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return mean, std, ((h1[0].detach(),h1[1].detach()), (h2[0].detach(),h2[1].detach()))

    def get_dist(self, s, hidden1=None, hidden2=None):
        orig_shape = s.shape

        # 自动添加 batch 维 即可
        if s.dim() == 1:
            # Actually will not get in here.
            print(f"Here is actor, {orig_shape}")
            s = s.unsqueeze(0).unsqueeze(0)
        elif s.dim() == 2:    
            s = s.unsqueeze(0)
        elif s.dim() == 3:            
            pass
        
        mean, std, (h1, h2) = self.forward(s, hidden1, hidden2)

        if len(orig_shape) == 1:
            print(f"Here is actor, {orig_shape}")
        elif len(orig_shape) == 2:
            mean = mean.squeeze(0)
            std = std.squeeze(0)

        return Normal(mean, std),(h1,h2)


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        orig_shape = s.shape

        if s.dim() == 1:
            print(f"Here is critic, {orig_shape}")
            s = s.unsqueeze(0).unsqueeze(0)
        elif s.dim() == 2:
            s = s.unsqueeze(0)
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)

        if len(orig_shape) == 1:
            return v_s.squeeze(0).squeeze(0)
        elif len(orig_shape) == 2:
            return v_s.squeeze(0)
        else:
            return v_s



class PPO_Agent():
    def __init__(self, args, device):
        self.policy_dist = "Gaussian"
        self.max_action = args.max_action
        
        self.max_train_steps = args.num_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.epsilon = args.clip  # PPO clip parameter
        self.K_epochs = args.epochs  # PPO parameter
        self.entropy_coef = args.entropy_coeff  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        
        self.use_lr_decay = args.use_lr_decay
        self.device = device

        self.actor = Actor_LSTM(args, self.device, self.policy_dist).to(self.device)
        self.critic = Critic(args).to(self.device)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float, device=self.device), 0)
        if self.policy_dist == "Beta":
            a = self.actor.mean(s).detach().numpy().flatten() 
        else:
            a = self.actor(s).detach().numpy().flatten()
        return a

    def choose_action(self, s):
        s = torch.tensor(s, dtype=torch.float, device=self.device)
        # print(f"\nHere is choose_action, {s.shape}.\n")
        
        with torch.no_grad():
            dist,_ = self.actor.get_dist(s)
            a = dist.sample()  # Sample the action according to the probability distribution
            a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
            a_logprob = dist.log_prob(a)  # The log probability density of the action
            # print(f"Here is choose_action, a_logprob:{a_logprob}")
            # print(f"Here is choose_action, a:{a.cpu().numpy().flatten()}")
            # print(f"Here is choose_action, a_logprob:{a_logprob.cpu().numpy().flatten()}")
        return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()

    def update(self, obs_batch, action_batch,alogp_batch, return_batch, adv_batch, hidden1=None, hidden2=None):
        # print(f"\nHere is update, {obs_batch.shape}.\n")
        # print(f"\nHere is update, {action_batch.shape}.\n")
        # print(f"\nHere is update, {alogp_batch.shape}.\n")
        # print(f"\nHere is update, {return_batch.shape}.\n")
        # print(f"\nHere is update, {adv_batch.shape}.\n")

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            dist_now, (h1,h2) = self.actor.get_dist(obs_batch, hidden1, hidden2)
            values = self.critic(obs_batch)

            dist_entropy = dist_now.entropy()
            a_logprob_now = dist_now.log_prob(action_batch)
            # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
            ratios = torch.exp(a_logprob_now.sum(-1, keepdim=True) - alogp_batch)  # shape(mini_batch_sizeXseq_lenX 1)

            surr1 = ratios * adv_batch  # Only calculate the gradient of 'a_logprob_now' in ratios
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv_batch
            actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
            # Update actor
            self.optimizer_actor.zero_grad()
            actor_loss.mean().backward()

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer_actor.step()

            critic_loss = F.mse_loss(return_batch, values)
            # Update critic
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer_critic.step()
        return h1, h2

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now