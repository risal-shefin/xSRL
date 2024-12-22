from datetime import datetime
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
from AdvEx_RL.safety_agent import Safety_Agent
from AdvEx_RL.memory import ReplayMemory, ConstraintReplayMemory
import copy
from matplotlib import pyplot 
import matplotlib.pyplot as plt
from AdvEx_RL_config.victim_config import get_victim_args
from AdvEx_RL_config.adversary_config import get_adv_args
import os

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def torchify(x): 
  return torch.FloatTensor(x).to(TORCH_DEVICE)

class Safety_trainer():
#******************************************************************************************************************
    def __init__(self, env, cfg, test_env):
        self.cfg = cfg
        self.victim_cfg = get_victim_args(cfg.env_name)
        self.adversary_cfg = get_adv_args(cfg.env_name)
      #**************************************************************
        self.agent_observation_space = env.observation_space.shape[0]
        self.agent_action_space = env.action_space.shape[0]
        current_path = os.getcwd()
        self.expert_agent_path = current_path + self.victim_cfg.saved_model_path
        self.adv_agent_path = current_path+ self.adversary_cfg.saved_model_path
      #**************************************************************
        self.env = env
        self.test_env = test_env
        # tz = timezone('EST')
        time = datetime.now().strftime("%b-%d-%Y|%H:%M|%p")
        self.logdir = os.path.join(self.cfg.logdir, '{}_SafetyAgent_{}'.format(time,
                                                                          self.cfg.env_name 
                                                                       ))

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        print("Created LOGDIR: ", self.logdir)
        pickle.dump(self.victim_cfg, open(os.path.join(self.logdir, "victim_agent_args.pkl"), "wb"))
        pickle.dump(self.adversary_cfg, open(os.path.join(self.logdir, "adv_agent_args.pkl"), "wb"))
        pickle.dump(self.cfg, open(os.path.join(self.logdir, "safety_args.pkl"), "wb"))
        #-------------------------------------------------------------------
        # if self.cfg.env_name == "nav2":  
        self.expert_agent = SAC(self.agent_observation_space,
                                self.agent_action_space,
                                self.victim_cfg,
                                self.logdir
                                )
        self.expert_agent.load_best_model(self.expert_agent_path)
        #-------------------------------------------------------------------
        self.adv_agent = SAC(self.agent_observation_space,
                                self.agent_action_space,
                                self.adversary_cfg,
                                self.logdir,
                                env = env,
                            )
        self.adv_agent.load_best_model(self.adv_agent_path)
        #-------------------------------------------------------------------
        self.safety_agent = Safety_Agent(observation_space = self.agent_observation_space, 
                                             action_space= self.agent_action_space,
                                             args=self.cfg,
                                             logdir=self.logdir,
                                             env = self.env,
                                             adv_agent=self.adv_agent
                                             #********************
                                            )
        #------------------------------------------------------------------
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.env.seed(self.cfg.seed)
        self.test_env.seed(self.cfg.seed)
        #++++++++++++++++++++++++++++++++++++++++++
        self.safety_constrained_memory = ConstraintReplayMemory(10000, self.cfg.seed)
        #++++++++++++++++++++++++++++++++++++++++++
        self.no_of_training_episode = self.cfg.num_eps
        self.total_numsteps = 0
        self.num_viols = 0
        self.num_successes = 0

        self.best_safety_ratio = -999999
        self.worst_safety_ratio = 999999
        self.best_step_count = 0
        self.episode_count = 0
        self.rec_training_epoch = 2
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.shield_theshold = self.adversary_cfg.shield_threshold #0.93 #.98
        self.beta = self.cfg.beta
        self.eta =  self.cfg.eta
#******************************************************************************************************************
        
    def agent_training(self):
        safety_pol_loss_vec = []
        safety_policy_safety = []

        task_count = []
        safety_count = []
        
        task_safety_reward = []
        only_tsk_reward_vec = []
        Flag_train_start = False
        # print(self.no_of_training_episode)
        for n in range(self.no_of_training_episode):
            safety, safety_ratio = self.run_safety_train_episode()
            print(f'Episode: {n}  '
                  f'Safety: {safety:<6.2f}  '
                  f'Safety Ratio: {safety_ratio:<6.2f}  '
                  )
            batch_size = 256
            if (len(self.safety_constrained_memory) > self.rec_training_epoch*self.cfg.batch_size) and (n>=self.cfg.train_start):
                if (len(self.safety_constrained_memory) > self.rec_training_epoch*self.cfg.batch_size) and not Flag_train_start:
                    print("Training Started")   
                    Flag_train_start = True
                safety_pol_loss = 0  
                for i in range(self.rec_training_epoch):
                    
                    safety_policy_loss = self.safety_agent.update_parameters(self.safety_constrained_memory, batch_size)
                    safety_pol_loss+=safety_policy_loss
                    
                safety_pol_loss = safety_pol_loss/self.rec_training_epoch
                
                safety_pol_loss_vec.append(safety_pol_loss)
                # rec_critic_loss_vec.append(rec_crtc_loss)

            if (n>0) and (n%20==0):
                eval_safety, eval_safety_ratio  = self.run_safety_eval_episode()
                safety_policy_safety.append(eval_safety_ratio)
                 #------------------------------------------------
                if self.best_safety_ratio<=eval_safety_ratio:
                    self.best_safety_ratio = eval_safety_ratio
                    self.safety_agent.save_best_safety_model(self.best_safety_ratio)
                if (n%10)==0:
                    self.safety_agent.save_best_safety_model(eval_safety_ratio, interval=n)
               
                print("-"*10,"Evaluation","-"*10)
                print(f'Episode: {n}  '
                      f'Safety: {eval_safety:<6.2f}  '
                      f'Safety Ratio: {eval_safety_ratio:<6.2f}  '
                )
                print("-"*(20+len(" Evaluation ")))
                #----------------Plot Loss data-----------------------
                plot_dir= os.path.join(self.logdir, 'Plots')
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                #----------------Plot data-----------------------
                safety_pl_plot_dir= os.path.join(plot_dir, 'safety_policy_loss_plot')
                self.plot(safety_pol_loss_vec, safety_pl_plot_dir, loss=True)

                safety_plot_dir= os.path.join(plot_dir, 'safety_plot')
                self.plot(safety_policy_safety, safety_plot_dir, safety=True)

            if n%20==0:
                reward, task_cnt, sfty_count, only_tsk_reward = self.run_task_rec_episode()
                task_safety_reward.append(reward)
                task_count.append(task_cnt)
                safety_count.append(sfty_count)
                only_tsk_reward_vec.append(only_tsk_reward)
                plot_dir= os.path.join(self.logdir, 'Plots')
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                task_rec_plot_dir= os.path.join(plot_dir, 'task_rec_reward_plot')
                self.plot_task_safety_reward(task_safety_reward, task_count, safety_count, only_tsk_reward_vec, task_rec_plot_dir)
                #------------------------------------------------
    
    def run_safety_train_episode(self):
        #****************************************
        safety, safety_ratio = self.rollout_trajectory()
        return safety, safety_ratio
        #****************************************
        
        #****************************************
    def select_random_action(self, env):
        action = np.random.uniform(env.action_space.low.min(), 
                                   env.action_space.high.max(),
                                   env.action_space.shape).astype(np.float32)
        return action

  #====================================================================
    def rollout_trajectory(self, rec_iter=1):
        safety = 0
        safety_ratio = 0
        episode_steps = 0
        state = self.env.reset()
        done = False
        while not done:
            episode_steps+=1
            tmp_env = copy.deepcopy(self.env)
            for _ in range(rec_iter):
                _ ,sf, sfr = self.safety_trajectory(tmp_env, state)
                safety+=sf
                safety_ratio+=sfr
            safety=safety/rec_iter
            safety_ratio=safety_ratio/rec_iter
            if np.random.rand()<self.beta:
                action = self.expert_agent.select_action(state, eval=True)
            else:
                if np.random.rand()>self.eta:
                    action = self.safety_agent.adv_critic.select_action(state, eval=True)
                else:
                    action = self.select_random_action(self.env)

            next_state, reward, done, _ = self.env.step(action)
            # episode_reward+=reward
            state = next_state
            done =done or episode_steps==self.env._max_episode_steps
            if done:
                break
        return safety/episode_steps, safety_ratio/episode_steps
        
  #==============================================================
    def safety_trajectory(self, env, init_state):
        # env.reversed_reward=True #Just to increase one step at absorbing state
        episode_safety_ratio = 0
        episode_steps = 0
        if self.cfg.env_name =="maze":
            env.set_state_to_target(init_state)
            state = init_state
        else:
            # env.extended_abs = True
            env.set_curr_state(init_state)
            state = init_state
        safety = 0
        done = False
        unsafe_count = 0
        episode_reward=0
        while not done:
            episode_steps+=1
            action = self.safety_agent.select_action(state, eval=False)
            next_state, reward, done, info = env.step(action)
            penalty = info['adv_reward']
            unsafe_count+=penalty
            episode_reward+=reward
            mask = float(not done)
            self.safety_constrained_memory.push(state, action, reward, penalty, next_state, mask)
            state = next_state
            done =done or episode_steps==env._max_episode_steps
            if done:
                if unsafe_count>0:
                    safety=0
                else:
                    safety=1
                episode_safety_ratio=episode_steps/env._max_episode_steps
                break

        return episode_reward, safety, episode_safety_ratio
#==============================================================
    def run_safety_eval_episode(self):
        episode_safety_ratio = 0
        episode_steps = 0
        state = self.test_env.reset()
        safety = 0
        done = False
        unsafe_count = 0
        episode_reward = 0
        while not done:
            episode_steps+=1
            action = self.safety_agent.select_action(state, eval=True)
            next_state, reward, done, info = self.test_env.step(action)
            penalty = info['adv_reward']
            unsafe_count+=penalty
            episode_reward+=reward
            state = next_state
            done =done or episode_steps==self.test_env._max_episode_steps
            if done:
                if unsafe_count>0:
                    safety=0
                else:
                    safety=1
                episode_safety_ratio=episode_steps/self.test_env._max_episode_steps
                break
        return safety, episode_safety_ratio
  #====================================================================
  #====================================================================
    def torchify(self, x): 
        return torch.FloatTensor(x).to(self.device).unsqueeze(0)

    def run_task_rec_episode(self):
        safety_cnt = 0
        tsk_cnt = 0
        epi_reward = 0

        done = False
        state = self.test_env.reset()
        copy_env = copy.deepcopy(self.test_env)
        init_state = state.copy()
        epi_task_reward = self.run_task_only_episode(copy_env, init_state)
        epi_step_count=0
        while not done:
            epi_step_count+=1
            action_tsk = self.expert_agent.select_action(state, eval=True)
            shield_val_tsk = self.safety_agent.adv_critic.get_shield_value(self.torchify(state), self.torchify(action_tsk))
            if shield_val_tsk>self.shield_theshold:   
                action = self.safety_agent.select_action(state, eval=True)
                safety_cnt+=1
            else:
                action = action_tsk
                tsk_cnt+=1
            n_state, reward, done, info = self.test_env.step(action)
            epi_reward += reward
            state = n_state
            done = done or epi_step_count==self.test_env._max_episode_steps
            if done:
                break
        return epi_reward, tsk_cnt, safety_cnt, epi_task_reward
  #====================================================================
    def run_task_only_episode(self, env, init_state):
        done =False
        epi_reward = 0
        if self.cfg.env_name =="Maze":
            env.set_state_to_target(init_state)
            state = init_state
        else:
            env.state = init_state
            state = init_state
        epi_step_count=0
        while not done:
            epi_step_count+=1
            action = self.expert_agent.select_action(state, eval=True)
            nxt_state, reward, done, _ = env.step(action)
            epi_reward+=reward
            state = nxt_state
            done = done or epi_step_count==self.test_env._max_episode_steps
            if done:
                break
        return epi_reward
  #====================================================================
            
    def plot(self, data1, path, loss=False, safety=False):
        plt.figure()
        plt.plot(range(len(data1)), data1)
        if loss:
            plt.xlabel('No of iter')
            plt.ylabel('Loss:')
        if safety:
            plt.xlabel('No of iter:')
            plt.ylabel('Safety:')
        plt.savefig(path+'.png', format='png')
        plt.close()

    def plot_task_safety_reward(self, reward, task_cnt, sfty_cnt, tsk_reward, path):
        plt.figure()

        plt.plot(range(len(reward)), reward, "-g", linewidth=3.0 ,label="Task+Shield+Safety reward")
        plt.plot(range(len(tsk_reward)), tsk_reward, ":c", linewidth=2.0 ,label="Task policy reward",)
        plt.plot(range(len(task_cnt)), task_cnt, "--r",label="task policy count")
        plt.plot(range(len(sfty_cnt)), sfty_cnt, "--b",label="safety policy count")
        plt.legend(loc="best")
        
        plt.savefig(path+'.png', format='png')
        plt.close()




            






            
        
        
