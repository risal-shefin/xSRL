import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.batch_size = args.batch_size
        self.action_dim = args.action_dim
        self.use_central_q = getattr(args, "use_central_q", False)
        self.episode_num = 0
        self.buffer = None
        self.reset_buffer()
        # create a buffer (dictionary)

    def reset_buffer(self):
        self.buffer = {
            'obs_n': np.zeros([self.batch_size, self.episode_limit, self.N, self.obs_dim], dtype=np.float32),
            's': np.zeros([self.batch_size, self.episode_limit, self.state_dim], dtype=np.float32),
            'v_n': np.zeros([self.batch_size, self.episode_limit + 1, self.N], dtype=np.float32),
            'a_n': np.zeros([self.batch_size, self.episode_limit, self.N], dtype=np.float32),
            'a_logprob_n': np.zeros([self.batch_size, self.episode_limit, self.N], dtype=np.float32),
            'r_n': np.zeros([self.batch_size, self.episode_limit, self.N], dtype=np.float32),
            # done mask default to 1 so GAE stops for unfilled steps
            'done_n': np.ones([self.batch_size, self.episode_limit, self.N], dtype=np.float32),
            # action mask default to all ones meaning all actions available
            'action_mask_n': np.ones([self.batch_size, self.episode_limit, self.N, self.action_dim], dtype=np.float32),
            # track actual episode lengths for masking padded steps
            'lengths': np.zeros([self.batch_size], dtype=np.int32)
        }
        if self.use_central_q:
            self.buffer['q_n'] = np.zeros([self.batch_size, self.episode_limit, self.N], dtype=np.float32)
        self.episode_num = 0

    def store_transition(self, episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n, action_mask_n, q_n=None):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.buffer['a_n'][self.episode_num][episode_step] = a_n
        self.buffer['a_logprob_n'][self.episode_num][episode_step] = a_logprob_n
        self.buffer['r_n'][self.episode_num][episode_step] = r_n
        self.buffer['done_n'][self.episode_num][episode_step] = done_n
        self.buffer['action_mask_n'][self.episode_num][episode_step] = action_mask_n
        if self.use_central_q and q_n is not None:
            self.buffer['q_n'][self.episode_num][episode_step] = q_n

    def store_last_value(self, episode_step, v_n):
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.buffer['lengths'][self.episode_num] = episode_step
        self.episode_num += 1

    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            if key == 'a_n' or key == 'lengths':
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32)
        return batch
