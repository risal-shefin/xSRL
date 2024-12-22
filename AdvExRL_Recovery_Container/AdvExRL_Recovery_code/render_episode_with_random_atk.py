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
from RecoveryRL.render_RecRL_random_atk import *
from plot_scripts.plot_all_new_functions import *
import moviepy.editor as mpy
import warnings
warnings.filterwarnings("ignore")

#====================================================================
def torchify(x):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    return torch.FloatTensor(x).to(device).unsqueeze(0)


def npy_to_gif(im_list, filepath, fps=4):
    if not os.path.exists(filepath):
                os.makedirs(filepath)
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filepath + '/rendered_fig.gif')
#====================================================================
def render_run_eval_episode(env,  
                     env_name,
                     expert_agent, 
                     safety_agent=None, 
                     shield_threshold = 0.0, 
                     atk_rate=0.20, 
                     epsilon=0.20,
                     render_function=None,
                     sample = 0
                    ):
    done = False
    epi_step_count = 0
    state = env.reset()
    if env_name=='maze':
        im_list = [env._get_obs(images=True)]
    else:
        im_list = [render_function(state).squeeze()]
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
                action = safety_agent.select_action(state, eval=True)
            else:
                action = action_tsk 
        else:
            action = action_tsk   
        nxt_state, _, done, _ = env.step(action)
        if not sample==0:
            if epi_step_count%sample==0:
                # im_list.append(render(next_state).squeeze())
                if env_name=='maze':
                    im_list.append(env._get_obs(images=True))
                else:
                    im_list.append(render_function(nxt_state).squeeze())
        else:
            if env_name=='maze':
                    im_list.append(env._get_obs(images=True))
            else:
                    im_list.append(render_function(nxt_state).squeeze())
        state = nxt_state
        done = done or (epi_step_count==env._max_episode_steps)
        if done:
            if not env_name=="maze":
                 render_function([0,0],end=True)
            break
    return im_list
#====================================================================
#====================================================================
   
            
def run(env_name=None, eval_epi_no=100):
  epsilon_list = [0.25, 0.50, 0.75, 1]
  
  eval_epi_no = eval_epi_no

  render = None
  if env_name == "maze":
      from env.maze import MazeNavigation
      env = MazeNavigation()
      sample_rate = 0
  elif env_name == 'nav1':
      from env.navigation1 import Navigation1, render
      env = Navigation1()
      sample_rate = 5
  elif env_name == 'nav2':
      from env.navigation2 import Navigation2, render
      env = Navigation2()
      sample_rate = 5
  render_function = render
  
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
        atk_rate = eps
        epsilon = eps
        logdir = os.path.join(AdvEx_RL_data_path ,'atk_rate_{}_eps_{}'.format(atk_rate, epsilon))
        if not os.path.exists(logdir):
                os.makedirs(logdir)
        #**********************************************************************************************************
        for i in tqdm(range(eval_epi_no)):
              im_list_Adv_Ex_RL = render_run_eval_episode(env, env_name, expert_agent, safety_agent, shield_threshold=shield_threshold, atk_rate=atk_rate, epsilon=epsilon, render_function=render_function, sample=5)
              gif_path = logdir+'/Adv_Ex_RL/'
              npy_to_gif(im_list_Adv_Ex_RL, gif_path)
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
        recRL_experiment_data = run_comparison(env=env, env_name=env_name, rend_fun=render_function,  env_model_path= RecRL_model_path, s_rate= sample_rate, atk_rt=atk_rate, eps=epsilon, eval_episode= eval_epi_no)
        for algo in recRL_experiment_data['algos']:
            gif_path = RecRLexp_data_dir_sub_folders+'/{}/'.format(algo)
            img_vec = recRL_experiment_data['algos'][algo]['render_vec']['safety_render']
            npy_to_gif(img_vec, gif_path)
        #**********************************************************************************************************

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configure-env', default='none', help='Set test environment to setup all configuration')
    parser.add_argument('--exp-data-dir', default='/Experimental_Data/', help='Set experiment data location')
    arg = parser.parse_args()
    name = arg.configure_env
    test_epi_no = 1
    run(env_name=name, eval_epi_no=test_epi_no)