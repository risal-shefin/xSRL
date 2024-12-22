from AdvEx_RL.sac import SAC
import torch
import argparse
from AdvEx_RL_config.victim_config import get_victim_args
from AdvEx_RL_config.adversary_config import get_adv_args
from AdvEx_RL_config.safety_config import get_safety_args
from matplotlib import pyplot 
from AdvEx_RL.safety_agent import Safety_Agent
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def plot_histogram(data, path):
      plt.figure()
      plt.hist(data, 
              20,
              density=True,
              histtype='bar',
              facecolor='b',
              alpha=0.5
              )
      # plt.hist(data, density=False, bins=20)
      plt.ylabel('Frequency')
      plt.xlabel('Critic Value')
      plt.savefig(path+'/critc_val_hist.png', format='png')
      plt.close()


def torchify(x): 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return torch.FloatTensor(x).to(device).unsqueeze(0)

def run_adv_rollout(env, sh_agent, agent=None, cap = 5):
  critic_vec = []
  indx = 0
  capacity = cap

  done = False
  epi_step = 0
  epi_reward = 0
  state = env.reset()
  adv_reward_total = 0
  while not done:
    epi_step+=1
    # action = agent.select_action(state, eval=True)
    if agent==None:
      action = np.random.uniform(env.action_space.low.min(), env.action_space.high.max(), env.action_space.shape).astype(np.float32)
    else:
      # if np.random.rand()<0.2:
      #   action = np.random.uniform(env.action_space.low.min(), env.action_space.high.max(), env.action_space.shape).astype(np.float32)
      # else:
        action = agent.select_action(state, eval=True)
    n_state, reward, done , info = env.step(action)
    #------------------------------------------------
    #------------------------------------------------
    critic_val = sh_agent.get_shield_value(torchify(state),torchify(action))
    critic_val = critic_val.cpu().detach().numpy()[0]
    if len(critic_vec) < capacity:
       critic_vec.append(None)
    critic_vec[indx]=(critic_val)
    indx = (indx+1)%capacity
    #------------------------------------------------
    #------------------------------------------------
    adv_reward_total+=info['adv_reward']
    done = done or epi_step==env._max_episode_steps or (info['adv_reward'])>0
    state = n_state
    if done:
      break
  if adv_reward_total>0:
      return adv_reward_total, critic_vec
  else:
      return adv_reward_total, critic_vec
    

def run(env_name):
  if env_name == 'maze':
        from env.maze import MazeNavigation
        env = MazeNavigation()
        test_env = MazeNavigation()
  elif env_name == 'nav1':
        from env.navigation1 import Navigation1
        env = Navigation1()
        test_env = Navigation1()
  elif env_name == 'nav2':
        from env.navigation2 import Navigation2
        env = Navigation2()
        test_env = Navigation2()

  agent_cfg =  get_victim_args(env_name)
  safety_cfg = get_safety_args(env_name)
  adv_cfg = get_adv_args(env_name)
  
  current_path = os.getcwd()
  
  expert_agent_path = current_path + agent_cfg.saved_model_path
  safety_policy_path = current_path + safety_cfg.saved_model_path
  adversary_path = current_path + adv_cfg.saved_model_path
  
  observation_space = env.observation_space.shape[0]
  action_space = env.action_space.shape[0]
  logdir = ''
  expert_agent = SAC(observation_space,
                   action_space,
                   agent_cfg,
                   agent_cfg.logdir,
                  )
  expert_agent.load_best_model(expert_agent_path)
  
  adv_agent = SAC(observation_space,
                   action_space,
                   adv_cfg,
                   adv_cfg.logdir,
                  )
  adv_agent.load_best_model(adversary_path)
  
  safety_agent = Safety_Agent(observation_space = observation_space, 
                              action_space= action_space,
                              args=safety_cfg,
                              logdir=safety_cfg.logdir,
                              env = env,
                              adv_agent=adv_agent
                              #********************
                               )
  safety_agent.load_safety_model(safety_policy_path)
  adv_critic_val_vec = []

  for i in tqdm(range(1000)):
    if np.random.rand()<0.1:
      _, critic_val = run_adv_rollout(test_env, safety_agent, agent=expert_agent)
    else:
      _, critic_val = run_adv_rollout(test_env, safety_agent, agent=adv_agent)
    
    if not critic_val==None:
      adv_critic_val_vec.extend(critic_val)
  #-----------------------------------------------------------------------------
  #-----------------------------------------------------------------------------
  critic_val_path = current_path+'/Collected_Shield_Values/'
  critic_val_path = os.path.join(critic_val_path,'critic_value', env_name)
  if not os.path.exists(critic_val_path):
              os.makedirs(critic_val_path)

  critic_val_file = os.path.join(critic_val_path,'shield_critic_val')
  with open(critic_val_file, 'wb') as f:
          np.save(f, np.array(adv_critic_val_vec))
  #-----------------------------------------------------------------------------
  #-----------------------------------------------------------------------------
  plot_histogram(adv_critic_val_vec, critic_val_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configure-env', default='none', help='Set test environment to setup all configuration')
    arg = parser.parse_args()
    name = arg.configure_env
    run(env_name=name)