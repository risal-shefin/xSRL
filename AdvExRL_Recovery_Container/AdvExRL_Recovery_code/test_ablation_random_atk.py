from AdvEx_RL.sac import SAC
import argparse
import torch
import numpy as np
import copy
from tqdm import tqdm
from datetime import datetime
import os
from matplotlib import pyplot 
import pickle
import matplotlib.pyplot as plt
from AdvEx_RL_config.victim_config import get_victim_args
from AdvEx_RL_config.adversary_config import get_adv_args
from AdvEx_RL_config.safety_config import get_safety_args
from AdvEx_RL.safety_agent import Safety_Agent
from RecoveryRL.recRL_comparison_exp_random_atk import *
from plot_scripts.ablation_plot_all_new_functions import *
import warnings
warnings.filterwarnings("ignore")

#====================================================================
def torchify(x):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    return torch.FloatTensor(x).to(device).unsqueeze(0)
#====================================================================
def run_eval_episode(env,  
                    expert_agent, 
                    safety_agent=None, 
                    shield_threshold = 0.0, 
                    atk_rate=0.20, 
                    epsilon=0.20
                   
                    ):
    rec_cnt = 0
    tsk_cnt = 0
    done =False
    epi_reward = 0
    epi_step_count=0
    state = env.reset()
    adv_reward = 0
    safety = 0
    unsafe_cnt = 0
    info_vec = []
    while not done:
        epi_step_count+=1

        action_tsk = expert_agent.select_action(state, eval=True)
        #******************************************************************************
        if np.random.rand()<atk_rate:
            perturbation = np.random.uniform(-epsilon, epsilon)
            action_tsk = action_tsk+perturbation
            action_tsk = np.clip(action_tsk, env.action_space.low.min(), env.action_space.high.max())
      
      #******************************************************************************   
        if not safety_agent==None:
            shield_val_tsk = safety_agent.get_shield_value(torchify(state), torchify(action_tsk))
            
            if shield_val_tsk>=shield_threshold:
                action = safety_agent.select_ablation_action(state)
                rec_cnt+=1
            else:
                action = action_tsk 
                tsk_cnt+=1     
        else:
            action = action_tsk 
            tsk_cnt+=1    

        nxt_state, reward, done, info = env.step(action)
        info_vec.append(info)
        
        epi_reward+=reward

        adv_reward +=info['adv_reward']
        if info['adv_reward']>0:
            unsafe_cnt+=1
        state = nxt_state
        done = done or (epi_step_count==env._max_episode_steps)
        if done:
            # if not safety_agent==None:
            #     print(shield_val_tsk)
            if adv_reward<0:
                safety=1
            else:
                safety = epi_step_count/env._max_episode_steps
            break
    return safety, tsk_cnt, rec_cnt, epi_reward, adv_reward, epi_step_count, info_vec
#====================================================================
#====================================================================
   
            
def run(env_name=None, eval_epi_no=100):
  epsilon_list = [0.0, 0.25, 0.50, 0.75, 1]
  eval_epi_no = eval_epi_no

  if env_name == "maze":
      from env.maze import MazeNavigation
      env = MazeNavigation()
  elif env_name == 'nav1':
      from env.navigation1 import Navigation1
      env = Navigation1()
  elif env_name == 'nav2':
      from env.navigation2 import Navigation2
      env = Navigation2()
  
  agent_cfg =  get_victim_args(env_name)
  safety_cfg = get_safety_args(env_name)
  adv_cfg = get_adv_args(env_name)
  
  current_path = os.getcwd()
  expdata_path = current_path+agent_cfg.exp_data_dir
  shield_threshold = adv_cfg.shield_threshold
  
  expert_agent_path = current_path + agent_cfg.saved_model_path
  safety_policy_path = current_path + safety_cfg.saved_model_path
  shield_path = current_path + adv_cfg.saved_model_path

  agent_observation_space = env.observation_space.shape[0]
  agent_action_space = env.action_space.shape[0]
  logdir = ' '
  #====================================================================
  expert_agent = SAC(agent_observation_space,
                   agent_action_space,
                   agent_cfg,
                   logdir,
                   env=env
                  )
  task_algorithm = "SAC"
  expert_agent.load_best_model(expert_agent_path)
  #====================================================================
  adv_agent = SAC(agent_observation_space,
                   agent_action_space,
                   adv_cfg,
                   logdir,
                   env=env
                  )
  adv_agent.load_best_model(shield_path)
  #====================================================================
  safety_agent = Safety_Agent(observation_space = agent_observation_space, 
                                action_space= agent_action_space,
                                args=safety_cfg,
                                logdir=logdir,
                                env = env,
                                adv_agent=adv_agent
                                )
  safety_agent.load_safety_model(safety_policy_path)
  AdvEx_RL_data_path = os.path.join(expdata_path, env_name, 'AdvEx_RL','shield_threshold_{}'.format(shield_threshold))
  #====================================================================
  for eps in epsilon_list:
        #-------------------------------------------------------------------
        epi_task_only_safety_vec = []
        epi_task_rec_safety_vec = []

        epi_only_task_reward_vec = []
        epi_tsk_only_adv_reward_vec = []
        tsk_test_info_vec = []
        epi_task_rec_reward_vec = []
        epi_task_rec_tsk_cnt = []
        epi_task_rec_rec_cnt = []

        epi_rec_reward_vec = []
        epi_rec_advreward_vec = []

        task_stp_cnt = []
        tskrec_epi_stp_cnt = []
        tsk_rec_test_info_vec = []
        #-------------------------------------------------------------------
        atk_rate = eps
        epsilon = eps
        logdir = os.path.join(AdvEx_RL_data_path ,'atk_rate_{}_eps_{}'.format(atk_rate, epsilon))
        if not os.path.exists(logdir):
                os.makedirs(logdir)
        #**********************************************************************************************************
        for i in tqdm(range(eval_epi_no)):
              tsk_safety, tsk_count , _ , tsk_reward, tsk_adv_reward, epi_step, tsk_test_info = run_eval_episode(env, expert_agent, atk_rate=atk_rate, epsilon=epsilon)
              epi_task_only_safety_vec.append(tsk_safety)
              task_stp_cnt.append(epi_step)
              epi_only_task_reward_vec.append(tsk_reward)
              epi_tsk_only_adv_reward_vec.append(tsk_adv_reward)
              tsk_test_info_vec.append(tsk_test_info)

              rectask_safety, t_cnt, r_cnt, e_reward, adv_r, epi_stp, tsk_rec_test_info = run_eval_episode(env, expert_agent, safety_agent, shield_threshold=shield_threshold, atk_rate=atk_rate, epsilon=epsilon)
              epi_task_rec_safety_vec.append(rectask_safety)
              epi_task_rec_tsk_cnt.append(t_cnt)
              epi_task_rec_rec_cnt.append(r_cnt)
              epi_rec_reward_vec.append(e_reward)
              epi_rec_advreward_vec.append(adv_r)
              tskrec_epi_stp_cnt.append(epi_stp)
              tsk_rec_test_info_vec.append(tsk_rec_test_info)

        Data ={'shield_threshold': shield_threshold,
              'attack Rate':atk_rate,
              'epsilon': epsilon,

              'task_only_safety': epi_task_only_safety_vec,
              'task_only_reward': epi_only_task_reward_vec,
              'task_only_adv_reward':epi_tsk_only_adv_reward_vec,
              'task_only_epi_step':task_stp_cnt,
              
              'epi_task_rec_safety': epi_task_rec_safety_vec,
              'epi_task_rec_reward': epi_rec_reward_vec,
              'epi_task_rec_tsk_cnt': epi_task_rec_tsk_cnt,
              'epi_task_rec_rec_cnt': epi_task_rec_rec_cnt,
              'epi_task_rec_adv_reward': epi_rec_advreward_vec,
              'epi_task_rec_epi_step': tskrec_epi_stp_cnt
              }

        Info_data = {'tsk_info':tsk_test_info_vec,
                    'tsk_rec_info':tsk_rec_test_info_vec
                    }
        data_file = os.path.join(logdir,'')
        
        with open(data_file+'Exp_data.pkl', 'wb') as f:
            pickle.dump(Data, f)
      
        with open(data_file+'Info_for_plotting_data.pkl', 'wb') as f2:
            pickle.dump(Info_data, f2)
        # **********************************************************************************************************
        # **********************************************************************************************************
        # *****************************************CALL RECOVERY RL COMPARISON************************************
        if env_name=="maze": 
            RecRL_model_path = current_path+'/RecoveryRL/RecoveryRL_Model/Maze'
        elif env_name=="nav1":
            RecRL_model_path = current_path+'/RecoveryRL/RecoveryRL_Model/Navigation1'
        elif env_name=="nav2":
            RecRL_model_path = current_path+'/RecoveryRL/RecoveryRL_Model/Navigations2'
        RecRLexp_data_path = os.path.join(expdata_path, env_name, 'RecRL')
        RecRLexp_data_dir_sub_folders = os.path.join(RecRLexp_data_path,'Atk_rate{}_eps{}'.format(atk_rate, epsilon))
        if not os.path.exists(RecRLexp_data_dir_sub_folders):
             os.makedirs(RecRLexp_data_dir_sub_folders)
        recRL_experiment_data = run_comparison(env=env, env_model_path= RecRL_model_path, atk_rt=atk_rate, eps=epsilon, eval_episode= eval_epi_no)
        recRL_data_path = os.path.join(RecRLexp_data_dir_sub_folders,'saved_exp_data.pkl')
        with open(recRL_data_path, 'wb') as f:
            pickle.dump(recRL_experiment_data, f)
        #**********************************************************************************************************
        #**********************************************************************************************************
  plot_path = os.path.join(expdata_path, env_name, 'Plots')
  atk_name = 'random'
  our_data = env_safety_data_our_model(AdvEx_RL_data_path, env_name)
  recRL_data = get_all_Recovery_RL_data(RecRLexp_data_path,env_name)
  draw_safety_plot(recRL_data, our_data, env_name, atk_name, plot_path)
  draw_success_rate_plot(recRL_data,our_data, env_name, atk_name, plot_path)
  draw_all_safety_with_policy_ratio_plot(recRL_data, our_data, env_name, atk_name, plot_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configure-env', default='none', help='Set test environment to setup all configuration')
    parser.add_argument('--exp-data-dir', default='/Experimental_Data/', help='Set experiment data location')
    arg = parser.parse_args()
    name = arg.configure_env
    test_epi_no = 100
    run(env_name=name, eval_epi_no=test_epi_no)