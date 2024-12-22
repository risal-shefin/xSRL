from AdvEx_RL.sac import SAC
from AdvEx_RL.safety_agent import Safety_Agent
from AdvEx_RL_config.victim_config import get_victim_args
from AdvEx_RL_config.adversary_config import get_adv_args
from AdvEx_RL_config.safety_config import get_safety_args
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
from matplotlib import pyplot 
import pickle
import matplotlib.pyplot as plt
import os
from RecoveryRL.recRL_comparison_exp_aaa_atk import *
from plot_scripts.plot_all_new_functions import *
from plot_scripts.plot_all_new_functions import *
import warnings
warnings.filterwarnings("ignore")
import random as random
import matplotlib.pyplot as plt
import numpy as np

#====================================================================
def torchify(x):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    return torch.FloatTensor(x).to(device).unsqueeze(0)
#====================================================================
def run_eval_episode(env, 
                    expert_agent, 
                    safety_agent=None, 
                    use_safety=False,
                    shield_threshold = 0.0, 
                    atk_rate=0.20, 
                    epsilon=0.20,
                    aaa_agent = None,
                    aaa_atk = True,
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
        if np.random.rand()<atk_rate and aaa_atk:
            action_tsk = aaa_agent.select_action(state)
        #******************************************************************************   
        if use_safety:
            shield_val_tsk = safety_agent.get_shield_value(torchify(state), torchify(action_tsk))
            if shield_val_tsk>=shield_threshold:
                action = safety_agent.select_action(state, eval=True)
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
            if adv_reward<0:
                safety=1
            else:
                safety = epi_step_count/env._max_episode_steps
            break
          
    return safety, tsk_cnt, rec_cnt, epi_reward, adv_reward, epi_step_count, info_vec
#====================================================================
#====================================================================
   
            
def run(env_name=None, eval_epi_no=100):
  eval_epi_no = eval_epi_no
  shield_threshold = np.arange(0.0, 2.0, 0.1)
#   shield_threshold = [.10, .20, .30, 0.40, .50, .60, .70, .80, .90, 1, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00]
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
  AdvEx_RL_plot_path = os.path.join(expdata_path, env_name, 'AdvEx_RL','different_shield_threshold')
  Safety_data_vec = []
  success_safety_vec = []
  #====================================================================
  for threshold in shield_threshold:
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
        atk_rate = 0.50
        epsilon = 0.50
        logdir = os.path.join(AdvEx_RL_plot_path,'Threshold_{}'.format(threshold))
        if not os.path.exists(logdir):
                os.makedirs(logdir)
        #**********************************************************************************************************
        for i in tqdm(range(eval_epi_no)):
              tsk_safety, tsk_count , _ , tsk_reward, tsk_adv_reward, epi_step, tsk_test_info = run_eval_episode(env, expert_agent,safety_agent, use_safety=False, atk_rate=atk_rate, epsilon=epsilon, aaa_agent=adv_agent, aaa_atk=True)
              epi_task_only_safety_vec.append(tsk_safety)
              task_stp_cnt.append(epi_step)
              epi_only_task_reward_vec.append(tsk_reward)
              epi_tsk_only_adv_reward_vec.append(tsk_adv_reward)
              tsk_test_info_vec.append(tsk_test_info)

              rectask_safety, t_cnt, r_cnt, e_reward, adv_r, epi_stp, tsk_rec_test_info = run_eval_episode(env, expert_agent, safety_agent, use_safety=True, shield_threshold=threshold, atk_rate=atk_rate, epsilon=epsilon, aaa_agent=adv_agent, aaa_atk=True)
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
        plot_file_path = data_file+'Info_for_plotting_data.pkl'
        with open(plot_file_path, 'wb') as f2:
            pickle.dump(Info_data, f2)
        
        Data = get_our_safety_data(plot_file_path, env_name)
        Safety_data_vec.append(Data['rec_sfty_percentage'])
        success_safety_vec.append(Data['rec_success_rate'])
        # **********************************************************************************************************
        # **********************************************************************************************************
        # *****************************************CALL RECOVERY RL COMPARISON************************************
  colors = []
  for i in range(0, len(Safety_data_vec)+1):
     colors.append(tuple(np.random.choice(range(0, 2), size=3)))
  fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
  plt.rcParams.update({'font.size': 25})

#   mean, lb, ub = get_stats(np.array(Safety_data_vec))
  ax.fill_between(shield_threshold, Safety_data_vec, 0, color='skyblue', 
                            alpha=.25, label='')
  ax.plot(shield_threshold, Safety_data_vec, color='skyblue', linewidth='3')
#   print(Safety_data_vec)
  ax.set_xticks(shield_threshold)
  ax.set_yticks([0,25,50,75,100])
  ax.tick_params(axis='x', labelsize=20)
  ax.tick_params(axis='y', labelsize=20)
  ax.set_xlabel('Safety threshold value', fontsize=30)
  ax.set_ylabel('Safety (%)', fontsize=30)
  plt.savefig(AdvEx_RL_plot_path+'/safety_wrt_different_threshold.png',dpi=300,format='png',bbox_inches='tight')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configure-env', default='none', help='Set test environment to setup all configuration')
    parser.add_argument('--exp-data-dir', default='/Experimental_Data/', help='Set experiment data location')
    arg = parser.parse_args()
    name = arg.configure_env
    test_epi_no = 50
    run(env_name=name, eval_epi_no=test_epi_no)