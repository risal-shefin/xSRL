from datetime import datetime
import gym
import os
import os.path as osp
import pickle
import numpy as np
import itertools
import torch
import cv2
import copy
from torch import nn, optim
from AdvEx_RL.sac import SAC
from AdvEx_RL.memory import ReplayMemory, ConstraintReplayMemory
import copy
from matplotlib import pyplot 
import matplotlib.pyplot as plt

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Adv_Experiment():
#******************************************************************************************************************
    def __init__(self, env, cfg, test_env):
      #**************************************************************
        self.agent_observation_space = env.observation_space.shape[0]
        self.agent_action_space = env.action_space.shape[0]
      #**************************************************************
        self.cfg = cfg
        self.env = env
        self.test_env = test_env
    
        # tz = timezone('EST')
        time = datetime.now().strftime("%b-%d-%Y|%H:%M|%p")
        self.logdir = os.path.join(self.cfg.logdir, '{}_SAC_{}_{}'.format(time,
                                                                          self.cfg.env_name, 
                                                                          self.cfg.policy,
                                                         ))
        if not os.path.exists(self.logdir):
              os.makedirs(self.logdir)
        # print("LOGDIR: ", self.logdir)
        pickle.dump(self.cfg,open(os.path.join(self.logdir, "args.pkl"), "wb"))
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.env.seed(self.cfg.seed)
        # agent_dir = os.path.join(self.logdir, "victim_agent")
        adversary_dir = os.path.join(self.logdir, "adversary_agent")
        self.adversary_agent = SAC(self.agent_observation_space,
                    self.agent_action_space,
                    self.cfg,
                    adversary_dir
                    )
        self.adv_critic_memory = ReplayMemory(self.cfg.replay_size, self.cfg.seed)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        self.no_of_training_episode = self.cfg.num_eps
        self.total_numsteps = 0
        self.num_viols = 0
        self.num_successes = 0

        self.best_adv_reward = -np.inf
        self.best_adv_step_count = 99999

        self.best_safe_step = -np.inf
        self.best_safty_step_count = 0
        
        self.lowest_recovery_loss = 10000
        self.lowest_adv_critic_loss = 10000
        self.episode_count = 0
        self.safety_look_ahead = 100
        self.adv_reward_val= 0.0
#******************************************************************************************************************
#******************************************************************************************************************
#******************************************************************************************************************   
    
    def torchify(self, x): 
        return torch.FloatTensor(x).to(TORCH_DEVICE).unsqueeze(0)
    
    def agent_training(self):
        adv_critic_loss_vec =[]
        adv_policy_loss_vec =[]
        adv_reward_vec = []
        critic_val_vector = []
        
       
        training_start_flag = False
        # print(self.no_of_training_episode)
        for n in range(self.no_of_training_episode):
            adv_reward = self.run_adv_agent_episode(Eval=False, iter_no = n)
            print(f'Episode: {n}  '
                  f'Adversary Reward: {adv_reward:<3.2f}  '
                  )
            
            if (len(self.adv_critic_memory) > self.cfg.epoch*self.cfg.batch_size) and (n>=self.cfg.train_start):
                if not training_start_flag:
                  print("######Training Started######")   
                  training_start_flag = True

                adv_critic_loss_sum = 0
                adv_policy_loss_sum = 0
                for i in range(self.cfg.epoch):
                        #----------------------------------            
                        adv_critic_loss, adv_policy_loss, _ , _ =self.adversary_agent.update_parameters(self.adv_critic_memory, min(self.cfg.batch_size, len(self.adv_critic_memory)))
                        adv_critic_loss_sum+=adv_critic_loss
                        adv_policy_loss_sum+=adv_policy_loss
                        #----------------------------------
                avg_adv_crtitic_loss = adv_critic_loss_sum/self.cfg.epoch
                avg_adv_policy_loss = adv_policy_loss_sum/self.cfg.epoch

                adv_critic_loss_vec.append(avg_adv_crtitic_loss)
                adv_policy_loss_vec.append(avg_adv_policy_loss)


            if (len(self.adv_critic_memory) > self.cfg.epoch*self.cfg.batch_size) and (n%10==0):
                ad_reward, vec = self.run_adv_eval_episode()
                adv_reward_vec.append(ad_reward)
                critic_val_vector.extend(vec)
                if self.best_adv_reward<=ad_reward:
                  self.best_adv_reward = ad_reward
                  self.adversary_agent.save_best_model(self.best_adv_reward)
                
                if n%500==0:
                    self.adversary_agent.save_model(n, ad_reward)
                    

                print("-"*10,"Evaluation","-"*10)
                print(f'Episode: {n}  '
                      f'Adv Agent Reward: {ad_reward:<4}  '
                      )
                print("-"*(20+len(" Evaluation ")))
                
                plot_dir= os.path.join(self.logdir, 'Plots')
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                
                adv_reward_path = os.path.join(plot_dir, 'adversary_reward_plot')
                self.plot_reward(adv_reward_vec, adv_reward_path)   
                    
                adv_agent_critic_loss_path = os.path.join(plot_dir, 'adv_agent_critic_loss_plot')
                adv_agent_policy_loss_path = os.path.join(plot_dir, 'adv_agent_policy_loss_plot')
                critic_shield_value_data_path = os.path.join(plot_dir, 'critic_shield_value_for_unsafe_plot')
                self.plot(adv_critic_loss_vec, adv_agent_critic_loss_path)
                self.plot(adv_policy_loss_vec, adv_agent_policy_loss_path)
                self.plot_histogram(critic_val_vector,critic_shield_value_data_path)    
                
                critic_val_path = os.path.join(plot_dir, 'adv_critic_value.npy')
                with open(critic_val_path, 'wb') as f:
                      np.save(f, np.array(critic_val_vector))

    def run_adv_agent_episode(self, Eval=False, iter_no=None):
        episode_reward = 0
        total_adv_reward = 0
        epi_adv_reward = 0
        episode_steps = 0
        done = False
        state = self.env.reset()
        self.episode_count+=1
        while not done:
            episode_steps += 1
            action = self.adversary_agent.select_action(state, eval=Eval)
            next_state, _, done, info = self.env.step(action)
            adv_r = float(info['adv_reward'])
            epi_adv_reward += adv_r
            self.total_numsteps += 1
            mask = int(not done)
            done = done or episode_steps == self.env._max_episode_steps
            # done = done 
            #**********************************************************
            self.adv_critic_memory.push(state, action, adv_r, next_state, mask)
            #**********************************************************
            state = next_state
            if done:
                break
        return adv_r


    def run_adv_eval_episode(self):
        critic_val_vec=[]
        indx = 0
        capacity = 3
        
        episode_reward = 0
        total_adv_reward = 0
        # unsafe_state = 0
        episode_steps = 0
        done = False
        state = self.test_env.reset()
        self.episode_count+=1
        
        while not done:
            episode_steps += 1
            action = self.adversary_agent.select_action(state, eval=True)
            next_state, reward, done, info = self.test_env.step(action)
            adv_r = float(info['adv_reward'])
            total_adv_reward += adv_r
            critic_val = self.adversary_agent.get_shield_value(self.torchify(state),self.torchify(action))
            critic_val = critic_val.detach().cpu().numpy()[0]
            if len(critic_val_vec) < capacity:
                critic_val_vec.append(None)
                critic_val_vec[indx]=(critic_val)
                indx = (indx+1)%capacity

            done = done or episode_steps == self.env._max_episode_steps
            state = next_state
            if done:
                break
        return total_adv_reward, critic_val_vec


    def plot(self, data, path):
        plt.figure()
        plt.plot(range(len(data)), data)
        # plt.xlabel('Episodes:{}'.format(self._episodes))
        plt.xlabel('No of iter:')
        plt.ylabel('Loss:')
        plt.savefig(path+'.png', format='png')
        plt.close()

    def plot_reward(self, data, path):
        plt.figure()
        plt.plot(range(len(data)), data)
        # plt.xlabel('Episodes:{}'.format(self._episodes))
        plt.xlabel('No of iter:')
        plt.ylabel('Reward:')
        plt.savefig(path+'.png', format='png')
        plt.close()

    def plot_histogram(self, data, path):
        plt.figure()
        plt.hist(data, density=False, bins=20)
        plt.ylabel('Frequency')
        plt.xlabel('Critic Value');
        plt.savefig(path+'.png', format='png')
        plt.close()


            






            
        
        
