import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
# from pytz import timezone
from torch.optim import Adam
# from dotmap import DotMap
import cv2
from AdvEx_RL.network import GaussianPolicy, QNetwork, DeterministicPolicy, QNetworkConstraint, StochasticPolicy, grad_false
from AdvEx_RL.utils import *

class SAC(object):
    def __init__(self,
                 observation_space,
                 action_space,
                 args,
                 logdir,
                 adv_agent=False,
                 env = None
                 ):
          self.learning_steps = 0
          #---------------------------------------------
          self.policy_type = args.policy    #Gaussian   Deterministic #else Stochastic
          self.gamma = args.gamma
          self.tau = args.tau
          self.alpha = args.alpha
          self.env_name = args.env_name
          
          self.target_update_interval = args.target_update_interval
        
          self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
          self.logdir = logdir
          #===========================SAC CRITIC================================
          self.critic = QNetwork(observation_space,
                                 action_space,
                                 args.hidden_size).to(device=self.device)
          self.critic_target = QNetwork(observation_space,
                                        action_space,
                                        args.hidden_size).to(device=self.device)
          self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

          #===========Initialize Target ========================================
          hard_update(self.critic_target, self.critic)
          #=====================================================================
          #==========================SAC ENTROPY================================
          self.target_entropy = -torch.prod(torch.Tensor(action_space).to(self.device)).item()
          self.log_alpha = torch.zeros(1,requires_grad=True, device=self.device)
          self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
          #=====================================================================
          #============================SAC POLICY ACTOR=========================
          if self.policy_type == "Gaussian":
              self.policy = GaussianPolicy( observation_space,
                                            action_space,
                                            args.hidden_size).to(self.device)
              self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
         
          elif self.policy_type == "Deterministic":
              self.policy = DeterministicPolicy(observation_space,
                                                action_space,
                                                args.hidden_size
                                                ).to(self.device)
              self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
          
          else:
              self.policy = StochasticPolicy( observation_space,
                                              action_space,
                                              args.hidden_size).to(self.device)
              self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
          self.adv = adv_agent
          if not env==None:
              self.env_action_space = env.action_space
          #=====================================================================

          #=======================##=======================
    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval is False:
          action, _, _,_, entropy = self.policy.sample(state)
        else:
            _, _, action,_,entropy = self.policy.sample(state)
        action = action.detach().cpu().numpy()
        # print(action)
        # print(action.dtype)
        return action[0], entropy

    #+++++++++FOR STRATEGIC ATTACK ++++++++++++++++++
    def get_qval_diff(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        state_batch = state.repeat(1000, 1)
        sampled_actions = torch.FloatTensor(
                np.array([self.env_action_space.sample()
                          for _ in range(1000)])).to(self.device)
        
        with torch.no_grad():
            q1, q2 = self.critic(state_batch, sampled_actions)
        qval_vec = torch.max(q1, q2)
        min_qval = torch.min(qval_vec)
        # min_index = 
        max_qval = torch.max(qval_vec)
        qval_diff = max_qval-min_qval

        return qval_diff.item()
    #++++++++++++++++++++++++++++++++++++
    
    def get_batch_tensor(self, memory, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        return state_batch, action_batch, next_state_batch, reward_batch, mask_batch

    def get_shield_value(self, state, action):
        with torch.no_grad():
            q1, q2 = self.critic(state, action)
        return torch.max(q1, q2)

    def update_agent_critic(self, memory, batch_size):
        state_batch, action_batch, next_state_batch, reward_batch, mask_batch = self.get_batch_tensor(memory, batch_size)
        with torch.no_grad():
          next_state_action, next_state_log_pi, _,_ = self.policy.sample(next_state_batch)
          qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
          
          if self.adv:
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            # next_q_value = reward_batch +  self.gamma * (min_qf_next_target)
          else:
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
             #   if self.adv_agent:
        next_q_value = reward_batch + mask_batch *self.gamma * (min_qf_next_target)
        #   else:
        #     next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        
        qf1, qf2 = self.critic(state_batch, action_batch)

        #==========Critic Loss =====================
        qf1_loss = F.mse_loss(qf1, next_q_value) 
        qf2_loss = F.mse_loss(qf2, next_q_value)
        #==========Loss Optim=======================
        self.critic_optim.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.critic_optim.step()
        return qf1_loss.item(), qf2_loss.item()

    #*********************POLICY & ENTROPY UPDATE ******************************
    def update_agent_policy_and_entropy(self, memory, batch_size):
        state_batch, action_batch, next_state_batch, reward_batch, mask_batch = self.get_batch_tensor(memory, batch_size)
        
        pi, log_pi, _,_ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        #==========POLICY LOSS ============================
        if self.adv:
            policy_loss = (- min_qf_pi).mean() 
        else:
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() 
        #==========POLICY Loss Optim=======================
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        #==========UPDATE ENTROPY==========================
        alpha_loss = -(self.log_alpha *(log_pi + self.target_entropy).detach()).mean()
        
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()
        alpha_tlogs = self.alpha.clone()

        return policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    #************************************************************************************

    def update_parameters(self, memory, batch_size, nu=None, safety_critic=None):
        self.learning_steps+=1
        qloss1, qloss2 = self.update_agent_critic(memory, batch_size)
        policy_loss, entropy_loss, entropy_loss_tlog = self.update_agent_policy_and_entropy(memory, batch_size)
        #===========UPDATE TARGET OCCASIONALLY===================
        if self.learning_steps % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        critic_loss = (qloss1+qloss2)/2
        return critic_loss, policy_loss, entropy_loss, entropy_loss_tlog
    
    def save_best_model(self, reward):
        # tz = timezone('EST')
        time = datetime.now().strftime("%b-%d-%Y")
        path = self.logdir
        model_dir = os.path.join(self.logdir,'Agent_model','Best_Agent_Model','DateTime_{}_reward_{}'.format( time, reward))
        critic_path = os.path.join(model_dir,'critic')
        policy_path = os.path.join(model_dir,'policy')
        if not os.path.exists(critic_path):
            os.makedirs(critic_path)
        if not os.path.exists(policy_path):
            os.makedirs(policy_path)
        critic_path = os.path.join(critic_path, 'critic_net.pth')
        policy_path = os.path.join(policy_path, 'policy_net.pth')
        self.critic.save(critic_path)
        self.policy.save(policy_path)

    def save_model(self, n, reward):
        # tz = timezone('EST')
        time = datetime.now().strftime("%b-%d-%Y")
        path = self.logdir
        model_dir = os.path.join(self.logdir,'Agent_model','Models_checkpoints','{}_Agent_Model_{}'.format( n, reward))
        critic_path = os.path.join(model_dir,'critic')
        policy_path = os.path.join(model_dir,'policy')
        if not os.path.exists(critic_path):
            os.makedirs(critic_path)
        if not os.path.exists(policy_path):
            os.makedirs(policy_path)
        critic_path = os.path.join(critic_path, 'critic_net.pth')
        policy_path = os.path.join(policy_path, 'policy_net.pth')
        self.critic.save(critic_path)
        self.policy.save(policy_path)

    def load_best_model(self, path):
        model_dir = path
        critic_path = os.path.join(model_dir,'critic')
        policy_path = os.path.join(model_dir,'policy')
  
        critic_path = os.path.join(critic_path, 'critic_net.pth')
        policy_path = os.path.join(policy_path, 'policy_net.pth')
        self.critic.load(critic_path)
        self.policy.load(policy_path)
        grad_false(self.critic)
        grad_false(self.policy)








