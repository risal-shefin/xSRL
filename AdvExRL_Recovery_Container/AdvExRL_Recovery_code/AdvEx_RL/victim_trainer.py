from datetime import datetime
# from pytz import timezone
import gym
import os
import os.path as osp
import pickle
import numpy as np
import itertools
import torch
import cv2
from torch import nn, optim
from AdvEx_RL.sac import SAC
from AdvEx_RL.memory import ReplayMemory
import copy
from matplotlib import pyplot 
import matplotlib.pyplot as plt

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



class Victim_Experiment():
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
        self.logdir = os.path.join(self.cfg.logdir, '{}_SAC_{}'.format(time,
                                                                          self.cfg.env_name, 
                                  ))
        if not os.path.exists(self.logdir):
              os.makedirs(self.logdir)
        print("LOGDIR: ", self.logdir)
        pickle.dump(self.cfg,open(os.path.join(self.logdir, "args.pkl"), "wb"))
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.env.seed(self.cfg.seed)
        # agent_dir = os.path.join(self.logdir, "victim_agent")
        victim_dir = os.path.join(self.logdir, "victim_agent")
        self.victim_agent = SAC(self.agent_observation_space,
                    self.agent_action_space,
                    self.cfg,
                    victim_dir
                    )
        self.victim_critic_memory = ReplayMemory(self.cfg.replay_size, self.cfg.seed)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        self.no_of_training_episode = self.cfg.num_eps
        self.total_numsteps = 0
        self.num_viols = 0
        self.num_successes = 0

        self.best_vic_reward = -np.inf
        self.best_vic_step_count = 99999

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
        critic_loss_vec =[]
        policy_loss_vec =[]
        victim_reward_vec = []
        critic_val_vector = []
        
       
        training_start_flag = False
        # print(self.no_of_training_episode)
        for n in range(self.no_of_training_episode):
            reward = self.run_agent_episode(Eval=False, iter_no = n)
            print(f'Episode: {n}  '
                  f'Adversary Reward: {reward:<3.2f}  '
                  )
            
            if (len(self.victim_critic_memory) > self.cfg.epoch*self.cfg.batch_size) and (n>=self.cfg.train_start):
                if not training_start_flag:
                  print("######Training Started######")   
                  training_start_flag = True

                critic_loss_sum = 0
                policy_loss_sum = 0
                for i in range(self.cfg.epoch):
                        #----------------------------------            
                        c_loss, p_loss, _ , _ =self.victim_agent.update_parameters(self.victim_critic_memory, min(self.cfg.batch_size, len(self.victim_critic_memory)))
                        critic_loss_sum+=c_loss
                        policy_loss_sum+=p_loss
                        #----------------------------------
                avg_crtitic_loss = critic_loss_sum/self.cfg.epoch
                avg_policy_loss = policy_loss_sum/self.cfg.epoch

                critic_loss_vec.append(avg_crtitic_loss)
                policy_loss_vec.append(avg_policy_loss)


            if (len(self.victim_critic_memory) > self.cfg.epoch*self.cfg.batch_size) and (n%10==0):
                eval_reward = self.run_eval_episode()
                victim_reward_vec.append(eval_reward)
                if self.best_vic_reward<=eval_reward:
                  self.best_vic_reward = eval_reward
                  self.victim_agent.save_best_model(self.best_vic_reward)
                
                if n%500==0:
                    self.victim_agent.save_model(n, eval_reward)
                    

                print("-"*10,"Evaluation","-"*10)
                print(f'Episode: {n}  '
                      f'Adv Agent Reward: {eval_reward:<4}  '
                      )
                print("-"*(20+len(" Evaluation ")))
                
                plot_dir= os.path.join(self.logdir, 'Plots')
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                
                adv_reward_path = os.path.join(plot_dir, 'adversary_reward_plot')
                self.plot_reward(victim_reward_vec, adv_reward_path)   
                    
                agent_critic_loss_path = os.path.join(plot_dir, 'adv_agent_critic_loss_plot')
                agent_policy_loss_path = os.path.join(plot_dir, 'adv_agent_policy_loss_plot')
               
                self.plot(critic_loss_vec, agent_critic_loss_path)
                self.plot(policy_loss_vec, agent_policy_loss_path)
                
            

    def run_agent_episode(self, Eval=False, iter_no=None):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = self.env.reset()
        self.episode_count+=1
        while not done:
            episode_steps += 1
            action = self.victim_agent.select_action(state, eval=Eval)
            next_state, reward, done, info = self.env.step(action)
            episode_reward += reward
            self.total_numsteps += 1
            mask = int(not done)
            done = done or episode_steps == self.env._max_episode_steps
            # done = done 
            #**********************************************************
            self.victim_critic_memory.push(state, action, reward, next_state, mask)
            #**********************************************************
            state = next_state
            if done:
                break
        return reward


    def run_eval_episode(self):
        total_adv_reward = 0
        # unsafe_state = 0
        episode_steps = 0
        done = False
        state = self.test_env.reset()
        self.episode_count+=1
        
        while not done:
            episode_steps += 1
            action = self.victim_agent.select_action(state, eval=True)
            next_state, reward, done, info = self.test_env.step(action)
            reward += reward
    
            done = done or episode_steps == self.env._max_episode_steps
            state = next_state
            if done:
                break
        return reward

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


            






            
        
        
