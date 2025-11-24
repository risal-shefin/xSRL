import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # Add this import
from torch.distributions import Categorical
from torch.utils.data.sampler import *


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Actor_RNN(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size*N, actor_input_dim),prob.shape=(mini_batch_size*N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        prob = torch.softmax(self.fc2(self.rnn_hidden), dim=-1)
        return prob


class Critic_RNN(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size*N, critic_input_dim), value.shape=(mini_batch_size*N, 1)
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value


class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size, episode_limit, N, actor_input_dim), prob.shape(mini_batch_size, episode_limit, N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        prob = torch.softmax(self.fc3(x), dim=-1)
        return prob


class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size, episode_limit, N, critic_input_dim), value.shape=(mini_batch_size, episode_limit, N, 1)
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value


class MAPPO:
    def __init__(self, args):
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.rnn_hidden_dim = args.rnn_hidden_dim

        self.use_central_q = getattr(args, "use_central_q", False)
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_rnn = args.use_rnn
        self.add_agent_id = args.add_agent_id
        self.use_value_clip = args.use_value_clip

        # get the input dimension of actor and critic
        self.actor_input_dim = args.obs_dim
        self.critic_input_dim = args.state_dim
        if self.add_agent_id:
            print("------add agent id------")
            self.actor_input_dim += args.N
            self.critic_input_dim += args.N

        if self.use_rnn:
            print("------use rnn------")
            self.actor = Actor_RNN(args, self.actor_input_dim)
            self.critic = Critic_RNN(args, self.critic_input_dim)
            if self.use_central_q:
                q_input_dim = self.critic_input_dim + self.N * self.action_dim
                self.central_q = Critic_RNN(args, q_input_dim)
        else:
            self.actor = Actor_MLP(args, self.actor_input_dim)
            self.critic = Critic_MLP(args, self.critic_input_dim)
            if self.use_central_q:
                q_input_dim = self.critic_input_dim + self.N * self.action_dim
                self.central_q = Critic_MLP(args, q_input_dim)

        if not self.use_central_q:
            self.central_q = None

        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.use_central_q:
            self.ac_parameters += list(self.central_q.parameters())
        if self.set_adam_eps:
            print("------set adam eps------")
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr)

    @staticmethod
    def _apply_action_mask(probs, mask):
        if mask is None:
            return probs
        probs = probs * mask
        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs = probs / (probs_sum + 1e-8)
        return probs

    def choose_action(self, obs_n, evaluate, action_masks=None):
        with torch.no_grad():
            actor_inputs = []
            # Fix: Convert list of numpy arrays to a single numpy array before creating tensor
            if isinstance(obs_n, list) and isinstance(obs_n[0], np.ndarray):
                obs_n = np.array(obs_n)
            obs_n = torch.tensor(obs_n, dtype=torch.float32)  # obs_n.shape=(N，obs_dim)
            actor_inputs.append(obs_n)
            if self.add_agent_id:
                """
                    Add an one-hot vector to represent the agent_id
                    For example, if N=3
                    [obs of agent_1]+[1,0,0]
                    [obs of agent_2]+[0,1,0]
                    [obs of agent_3]+[0,0,1]
                    So, we need to concatenate a N*N unit matrix(torch.eye(N))
                """
                actor_inputs.append(torch.eye(self.N))

            actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # actor_input.shape=(N, actor_input_dim)

            # Reset the RNN hidden state if using RNN
            if self.use_rnn:
                batch_size = actor_inputs.size(0)  # Should match N
                self.actor.rnn_hidden = torch.zeros(batch_size, self.rnn_hidden_dim,
                                                   device=actor_inputs.device)

            prob = self.actor(actor_inputs)  # prob.shape=(N,action_dim)
            if action_masks is not None:
                mask_list = []
                for m in action_masks:
                    if m is None:
                        mask_list.append(torch.ones(self.action_dim, device=prob.device))
                    else:
                        mask_list.append(torch.tensor(m, dtype=torch.float32, device=prob.device))
                mask = torch.stack(mask_list, dim=0)
                prob = self._apply_action_mask(prob, mask)

            if evaluate:
                a_n = prob.argmax(dim=-1)
                return a_n.numpy(), None
            else:
                dist = Categorical(probs=prob)
                a_n = dist.sample()
                a_logprob_n = dist.log_prob(a_n)
                return a_n.numpy(), a_logprob_n.numpy()

    def get_value(self, s):
        with torch.no_grad():
            critic_inputs = []
            # Because each agent has the same global state, we need to repeat the global state 'N' times.
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)  # (state_dim,)-->(N,state_dim)
            critic_inputs.append(s)
            if self.add_agent_id:  # Add an one-hot vector to represent the agent_id
                critic_inputs.append(torch.eye(self.N))
            critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_input.shape=(N, critic_input_dim)

            # Reset the RNN hidden state if using RNN
            if self.use_rnn:
                batch_size = critic_inputs.size(0)  # Should be N
                self.critic.rnn_hidden = torch.zeros(batch_size, self.rnn_hidden_dim,
                                                    device=critic_inputs.device)

            v_n = self.critic(critic_inputs)  # v_n.shape(N,1)
            return v_n.numpy().flatten()

    def get_central_q(self, s, a_n):
        if not self.use_central_q:
            raise RuntimeError("Central Q network is disabled. Enable it by setting use_central_q to True.")

        with torch.no_grad():
            critic_inputs = []
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)
            critic_inputs.append(s_tensor)
            if self.add_agent_id:
                critic_inputs.append(torch.eye(self.N))
            critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)

            actions = torch.tensor(a_n, dtype=torch.long)
            action_one_hot = F.one_hot(actions, num_classes=self.action_dim).float()
            joint_action = action_one_hot.reshape(1, self.N * self.action_dim).repeat(self.N, 1)
            q_inputs = torch.cat([critic_inputs, joint_action], dim=-1)

            if self.use_rnn:
                batch_size = q_inputs.size(0)
                self.central_q.rnn_hidden = torch.zeros(batch_size, self.rnn_hidden_dim, device=q_inputs.device)

            q_values = self.central_q(q_inputs)
            return q_values.numpy().flatten()

    def train(self, replay_buffer, total_steps):
        batch = replay_buffer.get_training_data()  # get training data

        # mask out padded transitions using recorded episode lengths
        lengths = batch['lengths']  # shape: (batch_size,)
        device = lengths.device
        valid_mask = (torch.arange(self.episode_limit, device=device).unsqueeze(0)
                      < lengths.unsqueeze(1)).unsqueeze(-1).float()

        # Calculate the advantage using GAE
        adv = []
        gae = 0
        with torch.no_grad():  # adv and td_target have no gradient
            deltas = batch['r_n'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['done_n']) - batch['v_n'][:, :-1]  # deltas.shape=(batch_size,episode_limit,N)
            for t in reversed(range(self.episode_limit)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)  # adv.shape(batch_size,episode_limit,N)
            v_target = adv + batch['v_n'][:, :-1]  # v_target.shape(batch_size,episode_limit,N)
            if self.use_adv_norm:
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        """
            Get actor_inputs and critic_inputs
            actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
            critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        """
        actor_inputs, critic_inputs, q_inputs = self.get_inputs(batch)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                """
                    get probs_now and values_now
                    probs_now.shape=(mini_batch_size, episode_limit, N, action_dim)
                    values_now.shape=(mini_batch_size, episode_limit, N)
                """
                if self.use_rnn:
                    # If use RNN, we need to reset the rnn_hidden of the actor and critic.
                    self.actor.rnn_hidden = None
                    self.critic.rnn_hidden = None
                    if self.use_central_q:
                        self.central_q.rnn_hidden = None
                    probs_now, values_now = [], []
                    q_values_now = [] if self.use_central_q else None
                    for t in range(self.episode_limit):
                        prob = self.actor(actor_inputs[index, t].reshape(self.mini_batch_size * self.N, -1))  # prob.shape=(mini_batch_size*N, action_dim)
                        mask = batch['action_mask_n'][index, t].reshape(self.mini_batch_size * self.N, -1)
                        prob = self._apply_action_mask(prob, mask)
                        probs_now.append(prob.reshape(self.mini_batch_size, self.N, -1))  # prob.shape=(mini_batch_size,N,action_dim）
                        v = self.critic(critic_inputs[index, t].reshape(self.mini_batch_size * self.N, -1))  # v.shape=(mini_batch_size*N,1)
                        values_now.append(v.reshape(self.mini_batch_size, self.N))  # v.shape=(mini_batch_size,N)
                        if self.use_central_q:
                            q = self.central_q(q_inputs[index, t].reshape(self.mini_batch_size * self.N, -1))
                            q_values_now.append(q.reshape(self.mini_batch_size, self.N))
                    probs_now = torch.stack(probs_now, dim=1)
                    values_now = torch.stack(values_now, dim=1)
                    if self.use_central_q:
                        q_values_now = torch.stack(q_values_now, dim=1)
                else:
                    probs_now = self.actor(actor_inputs[index])
                    probs_now = self._apply_action_mask(probs_now, batch['action_mask_n'][index])
                    values_now = self.critic(critic_inputs[index]).squeeze(-1)
                    if self.use_central_q:
                        q_values_now = self.central_q(q_inputs[index]).squeeze(-1)

                dist_now = Categorical(probs_now)
                dist_entropy = dist_now.entropy()  # dist_entropy.shape=(mini_batch_size, episode_limit, N)
                a_logprob_n_now = dist_now.log_prob(batch['a_n'][index])  # a_logprob_n_now.shape=(mini_batch_size, episode_limit, N)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_n_now - batch['a_logprob_n'][index].detach())  # ratios.shape=(mini_batch_size, episode_limit, N)
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                valid_mask_batch = valid_mask[index]
                actor_loss = (-torch.min(surr1, surr2) - self.entropy_coef * dist_entropy)
                actor_loss = (actor_loss * valid_mask_batch).sum() / valid_mask_batch.sum()

                if self.use_value_clip:
                    values_old = batch["v_n"][index, :-1].detach()
                    values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - v_target[index]
                    values_error_original = values_now - v_target[index]
                    critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss = (values_now - v_target[index]) ** 2
                critic_loss = (critic_loss * valid_mask_batch).sum() / valid_mask_batch.sum()

                self.ac_optimizer.zero_grad()
                ac_loss = actor_loss + critic_loss
                if self.use_central_q:
                    if self.use_value_clip:
                        q_values_old = batch["q_n"][index].detach()
                        q_error_clip = torch.clamp(q_values_now - q_values_old, -self.epsilon, self.epsilon) + q_values_old - v_target[index]
                        q_error_original = q_values_now - v_target[index]
                        q_loss = torch.max(q_error_clip ** 2, q_error_original ** 2)
                    else:
                        q_loss = (q_values_now - v_target[index]) ** 2
                    q_loss = (q_loss * valid_mask_batch).sum() / valid_mask_batch.sum()
                    ac_loss = ac_loss + q_loss
                ac_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now

    def get_inputs(self, batch):
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch['obs_n'])
        critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1))
        if self.add_agent_id:
            # agent_id_one_hot.shape=(mini_batch_size, max_episode_len, N, N)
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.episode_limit, 1, 1)
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)

        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # actor_inputs.shape=(batch_size, episode_limit, N, actor_input_dim)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_inputs.shape=(batch_size, episode_limit, N, critic_input_dim)

        q_inputs = None
        if self.use_central_q:
            action_one_hot = F.one_hot(batch['a_n'], num_classes=self.action_dim).float()
            joint_action = action_one_hot.reshape(self.batch_size, self.episode_limit, 1, self.N * self.action_dim)
            joint_action = joint_action.repeat(1, 1, self.N, 1)
            q_inputs = torch.cat([critic_inputs, joint_action], dim=-1)

        return actor_inputs, critic_inputs, q_inputs

    def save_model(self, env_name, number, seed, total_steps):
        torch.save(self.actor.state_dict(), "./model/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, int(total_steps / 1000)))
        torch.save(self.critic.state_dict(), "./model/MAPPO_critic_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, int(total_steps / 1000)))
        if self.use_central_q:
            torch.save(self.central_q.state_dict(), "./model/MAPPO_central_q_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, int(total_steps / 1000)))

    def load_model(self, env_name, number, seed, step):
        self.actor.load_state_dict(torch.load("./model/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, step)))
        self.critic.load_state_dict(torch.load("./model/MAPPO_critic_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, step)))
        if self.use_central_q:
            self.central_q.load_state_dict(torch.load("./model/MAPPO_central_q_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, step)))

    def load_model_from_directory(self, path):
        # self.actor.load_state_dict(torch.load(path))
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        if self.use_central_q:
            self.central_q.load_state_dict(checkpoint["central_q_state_dict"])

    def select_action(self, obs, agent_id, evaluate=False, action_mask=None, return_dist=False):
        """
        Select an action for the specified agent based on the observation
        
        Args:
            obs: The observation for the agent
            agent_id: The ID of the agent
            evaluate: Whether to use deterministic policy (for evaluation) or stochastic policy (for training)
            
        Returns:
            action: The selected action
        """
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            obs = torch.tensor(obs, dtype=torch.float).unsqueeze(0)  # shape: (1, obs_dim)
            
            # Add agent ID if needed
            if self.add_agent_id:
                agent_id_one_hot = torch.zeros(1, self.N)
                agent_id_one_hot[0, agent_id] = 1.0
                actor_input = torch.cat([obs, agent_id_one_hot], dim=-1)
            else:
                actor_input = obs
            
            # Reset RNN hidden state if using RNN
            if self.use_rnn:
                batch_size = actor_input.size(0)  # Should be 1
                self.actor.rnn_hidden = torch.zeros(batch_size, self.rnn_hidden_dim, 
                                                   device=actor_input.device)
            
            # Get action probabilities
            action_probs = self.actor(actor_input)  # shape: (1, action_dim)
            if action_mask is not None:
                mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=action_probs.device)
                action_probs = self._apply_action_mask(action_probs, mask_tensor)
            dist = torch.distributions.Categorical(action_probs)
            if evaluate:  # Use deterministic policy for evaluation
                action = torch.argmax(action_probs, dim=-1).item()  # Select the action with highest probability
            else:  # Use stochastic policy for training
                # For discrete action spaces
                action = dist.sample().item()
        
        if return_dist:
            return action, dist
        return action
    
    def compute_log_prob(self, obs, agent_id, action, action_mask=None):
        # Convert to tensor and add batch dimension
        obs = obs.unsqueeze(0)  # shape: (1, obs_dim)
        
        # Add agent ID if needed
        if self.add_agent_id:
            agent_id_one_hot = torch.zeros(1, self.N)
            agent_id_one_hot[0, agent_id] = 1.0
            actor_input = torch.cat([obs, agent_id_one_hot], dim=-1)
        else:
            actor_input = obs
        
        # Reset RNN hidden state if using RNN
        if self.use_rnn:
            batch_size = actor_input.size(0)  # Should be 1
            self.actor.rnn_hidden = torch.zeros(batch_size, self.rnn_hidden_dim, 
                                            device=actor_input.device)
        
        # Get action probabilities
        action_probs = self.actor(actor_input)  # shape: (1, action_dim)
        if action_mask is not None:
            mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=action_probs.device)
            action_probs = self._apply_action_mask(action_probs, mask_tensor)
        dist = torch.distributions.Categorical(action_probs)
        return dist.log_prob(action)
    
    # gradient enabled version of get_value
    def compute_value(self, s_tensor):
        critic_inputs = []
        # Because each agent has the same global state, we need to repeat the global state 'N' times.
        s = s_tensor.unsqueeze(0).repeat(self.N, 1)  # (state_dim,)-->(N,state_dim)
        critic_inputs.append(s)
        if self.add_agent_id:  # Add an one-hot vector to represent the agent_id
            critic_inputs.append(torch.eye(self.N))
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_input.shape=(N, critic_input_dim)
        
        # Reset the RNN hidden state if using RNN
        if self.use_rnn:
            batch_size = critic_inputs.size(0)  # Should be N
            self.critic.rnn_hidden = torch.zeros(batch_size, self.rnn_hidden_dim, 
                                                device=critic_inputs.device)
        
        v_n = self.critic(critic_inputs)  # v_n.shape(N,1)
        return v_n
    
    # gradient enabled version of select_action
    def compute_action(self, obs, agent_id, evaluate=False, action_mask=None, return_dist=False):
        """
        Select an action for the specified agent based on the observation
        
        Args:
            obs: The observation for the agent
            agent_id: The ID of the agent
            evaluate: Whether to use deterministic policy (for evaluation) or stochastic policy (for training)
            
        Returns:
            action: The selected action
        """
        
        # Add agent ID if needed
        if self.add_agent_id:
            agent_id_one_hot = torch.zeros(1, self.N)
            agent_id_one_hot[0, agent_id] = 1.0
            actor_input = torch.cat([obs, agent_id_one_hot], dim=-1)
        else:
            actor_input = obs
        
        # Reset RNN hidden state if using RNN
        if self.use_rnn:
            batch_size = actor_input.size(0)  # Should be 1
            self.actor.rnn_hidden = torch.zeros(batch_size, self.rnn_hidden_dim, 
                                                device=actor_input.device)
        
        # Get action probabilities
        action_probs = self.actor(actor_input)  # shape: (1, action_dim)
        if action_mask is not None:
            mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=action_probs.device)
            action_probs = self._apply_action_mask(action_probs, mask_tensor)
        dist = torch.distributions.Categorical(action_probs)
        if evaluate:  # Use deterministic policy for evaluation
            action = torch.argmax(action_probs, dim=-1) # Select the action with highest probability
        else:  # Use stochastic policy for training
            # For discrete action spaces
            action = dist.sample()
    
        if return_dist:
            return action, dist
        return action
