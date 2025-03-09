from AdvEx_RL.sac import SAC
from AdvEx_RL.safety_agent import Safety_Agent
from AdvEx_RL_config.victim_config import get_victim_args
from AdvEx_RL_config.adversary_config import get_adv_args
from AdvEx_RL_config.safety_config import get_safety_args
from RecoveryRL.recoverRL_args import recRL_get_args
import argparse
from common.DictionarySummaryModel import DictionarySummaryModel
from common.enums import PolicyEnum, SummaryMethodEnum
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
import warnings
warnings.filterwarnings("ignore")
from math import *
import random
from data import Data
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

discrete_action = { 0: [1,0], # east
                    1: [sqrt(2),sqrt(2)], # north east
                    2: [0,1], # north
                    3: [-sqrt(2),sqrt(2)], # north west
                    4: [-1,0], # west
                    5: [-sqrt(2),-sqrt(2)], # south west
                    6: [0,-1], # south
                    7: [sqrt(2),-sqrt(2)]} # south east

def eu_dist(a,b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return sqrt(dotproduct(v, v))

def compute_angle(v1, v2):
    cosine_of_angle = dotproduct(v1, v2) / (length(v1) * length(v2))
    cosine_of_angle = max(-1.0, min(1.0, cosine_of_angle))  # Clamp value to the valid range
    angle = acos(cosine_of_angle) / pi * 180
    if v2[1] < 0:
        angle = 360 - angle
    return angle


def binning_action(a):
    # action should be in terms of a = [ax, ay]
    # where ax, ay in range -1 to 1

    """
    min_dist = [-1, 1]
    for k,v in discrete_action.items():
        if eu_dist(v, a) < min_dist[1]:
            min_dist[0] = k
            min_dist[1] = eu_dist(v,a)
    return min_dist[0]
    """
    angle = compute_angle([1,0], a)
    if (angle >= 0 and angle <= 20) or (angle > 335 and angle <= 360):
        return 0
    elif angle > 20 and angle <= 70:
        return 1
    elif angle > 70 and angle <= 110:
        return 2
    elif angle > 110 and angle <= 160:
        return 3
    elif angle > 160 and angle <= 200:
        return 4
    elif angle > 200 and angle <= 250:
        return 5
    elif angle > 250 and angle <= 290:
        return 6
    else:
        return 7

def round_state(s):
    rounded_s = []
    for ele in s:
        ele = round(ele , 2)
        rounded_s.append(ele)
    return rounded_s
    
#====================================================================
def torchify(x):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    return torch.FloatTensor(x).to(device).unsqueeze(0)

def left_risk(state):
    if abs(state[1]) <= 15 and state[0] >= -40 and state[0] <= -30:
        return 1
    else:
        return 0
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
                    recovery_rl_exp = None,
                    ctf_action_method = None
                    ):
    rec_cnt = 0
    tsk_cnt = 0

    done =False
    epi_reward = 0
    epi_step_count=0
    state = env.reset()
    print(f'start state {state}')
    adv_reward = 0
    safety = 0
    unsafe_cnt = 0
    info_vec = []

    if recovery_rl_exp is not None:
        # Need to set it false before each episode
        recovery_rl_exp.ctf_action_taken = False

    # ADD
    # actions holds the binning_action and env_actions holds the actual actions applied on the environment
    episode_data = {'states': [], 'actions': [], 'entropy': [], 'dones': [], 'rewards': [], 'env_actions': [], 'policies': [], 'task_critic_vals': [], 'safety_critc_vals': []}
    all_state = set()
    is_print = True
    if is_print:
        print(f'state is {state}')
        print(f'rounded state is {round_state(state)}')
    all_state.add(tuple(round_state(state)))
    # ADD
    print('Start of the main loop')
    print(f'attack {aaa_atk}')
    print(f'safety {use_safety}')
    while not done:
        epi_step_count+=1
        action_tsk, entropy = expert_agent.select_action(state, eval=True)
        used_policy = PolicyEnum.TaskPolicy
        task_critic_val = 0.0
        safety_critic_val = 0.0
        #******************************************************************************   
        if np.random.rand() < atk_rate and aaa_atk:
        #if left_risk(state) and aaa_atk and np.random.rand() < atk_rate:
            action_tsk, entropy = aaa_agent.select_action(state)
            print("Triggered Attack")
        #****************************************************************************** 
        if use_safety:
            if recovery_rl_exp is None: # AdvExRL
                print(f'shield threshold is {shield_threshold}')
                shield_val_tsk = safety_agent.get_shield_value(torchify(state), torchify(action_tsk))
                if shield_val_tsk>=shield_threshold:
                    action, entropy = safety_agent.select_action(state, eval=True) 
                    rec_cnt+=1
                    used_policy = PolicyEnum.SafetyPolicy
                else:
                    action = action_tsk
                    tsk_cnt+=1

                safety_critic_val = safety_agent.get_shield_value(torchify(state), torchify(action)).item()
                task_q1, task_q2 = expert_agent.critic(torchify(state), torchify(action))
                task_critic_val = task_q1.item()
            else:
                agent_action, rec_action, recovery_selected, entropy = recovery_rl_exp.get_action(env, state, atk_rate, epsilon, aaa_atk, True, ctf_action_method)

                # agent_action can be from an expert agent or from an attacking agent. It depends on the attack rate.
                # recovery_selected is true if a recovery policy is used to determine the action.
                if recovery_selected:
                    action = rec_action
                    rec_cnt += 1
                    used_policy = PolicyEnum.SafetyPolicy
                else:
                    action = agent_action
                    tsk_cnt += 1
                
                safety_critic_val = recovery_rl_exp.agent.safety_critic.get_value(torchify(state), torchify(action)).item()
                task_q1, task_q2 = recovery_rl_exp.agent.critic(torchify(state), torchify(action))
                task_critic_val = task_q1.item()
        else:
            action = action_tsk
            tsk_cnt+=1
            task_q1, task_q2 = expert_agent.critic(torchify(state), torchify(action))
            safety_critic_val = safety_agent.get_shield_value(torchify(state), torchify(action)).item()
            task_critic_val = task_q1.item()


        # ADD
        entropy = entropy.detach().cpu().numpy().reshape(-1)[0]
        # ADD_end


        action_d = binning_action(action)
        if is_print:
            print('\n\n\n\-------------')
            print(f'timestep #{epi_step_count}')
            print(f'angle is {compute_angle([1,0], action)}')
            print(f'action is {action}')
            print(f'action is {action_d}')
            print(f'entropy is {entropy}')

        # ADD
        # episode_data['states'].append(state)
        episode_data['states'].append(tuple(round_state(state)))
        episode_data['actions'].append(action_d)
        episode_data['entropy'].append(entropy)
        episode_data['env_actions'].append(action)
        episode_data['policies'].append(used_policy)
        episode_data['task_critic_vals'].append(task_critic_val)
        episode_data['safety_critc_vals'].append(safety_critic_val)
        # ADD_end

        nxt_state, reward, done, info = env.step(action)
        if is_print:
            print(f'state is {state}')
            print(f'rounded state is {round_state(state)}')
            print(f'next state is {nxt_state}')
            print(f'rounded next_state is {round_state(nxt_state)}')
            print(f'reward is {reward}')
            print(f'done is {done}')
            print(f'success is {info["success"]}')
            print("Task Critic Val: {}, Safety Critic Val: {}".format(task_critic_val, safety_critic_val))
        all_state.add(tuple(round_state(state)))
        info_vec.append(info)
        epi_reward+=reward

        # adv_reward +=info['adv_reward']
        # if info['adv_reward']>0:
        #     unsafe_cnt+=1
        state = nxt_state
        done = done or (epi_step_count==env._max_episode_steps)
        # ADD
        episode_data['dones'].append(int(done))
        episode_data['rewards'].append(reward)
        # ADD_end
        if done:
            # if adv_reward<0:
            #     safety=1
            # else:
            #     safety = epi_step_count/env._max_episode_steps
            break
          
    # return safety, tsk_cnt, rec_cnt, epi_reward, adv_reward, epi_step_count, info_vec
    # ADD
    print("\n --- Action Count:", tsk_cnt, "Recovery Activated Action Count:", rec_cnt)
    return episode_data, all_state
    # ADD_end
#====================================================================
#====================================================================
   
            
def run(env_name, args):
  eval_epi_no = args.num_episodes if hasattr(args, 'num_episodes') else 100
  base_atk_rt = args.attack_rate if hasattr(args, 'attack_rate') else 0.0
  recovery_rl_algo_name = args.recovery_rl if hasattr(args, 'recovery_rl') else ''
  user_test = args.user_test if hasattr(args, 'user_test') else False
  use_safety = args.use_safety if hasattr(args, 'use_safety') else False
  numpy_seed = args.numpy_seed if hasattr(args, 'numpy_seed') else 0
  ctf_action_method = args.ctf_action_method if hasattr(args, 'ctf_action_method') else None
  # eval_epi_no = 10  # used to be 100, now change it to 10
  epsilon_list = [0.0, 0.25, 0.50, 0.75, 1]
  atk_rate_idx = 2 # choose idx to set the attack rate from the epsilon_list & assign attack rate values.
  current_path = os.path.join(os.getcwd(), "AdvExRL_Recovery_Container", "AdvExRL_Recovery_code")
  NUMPY_SEED = numpy_seed
  extra_dicts_list = list[DictionarySummaryModel]()
  if env_name == "maze":
      from env.maze import MazeNavigation
      env = MazeNavigation()
      env.seed(1234)
      RecRL_model_path = current_path+'/RecoveryRL/RecoveryRL_Model/Maze'
  elif env_name == 'nav1':
      from env.navigation1 import Navigation1
      env = Navigation1()
      env.seed(31415)
      RecRL_model_path = current_path+'/RecoveryRL/RecoveryRL_Model/Navigation1'
  elif env_name == 'nav2':
      from env.navigation2 import Navigation2
      env = Navigation2()
    #   env.seed(27156)
      env.action_space.seed(NUMPY_SEED)
      np.random.seed(NUMPY_SEED)
      torch.manual_seed(NUMPY_SEED)
      random.seed(NUMPY_SEED)
    #   env.seed(123456)
    #   env.action_space.seed(123456)
    #   np.random.seed(123456)
    #   torch.manual_seed(123456)
    #   random.seed(123456)
      RecRL_model_path = current_path+'/RecoveryRL/RecoveryRL_Model/Navigations2'

  agent_cfg =  get_victim_args(env_name)
  safety_cfg = get_safety_args(env_name)
  adv_cfg = get_adv_args(env_name)
  recoveryRL_cfg = recRL_get_args()
#   print(adv_cfg)
  # current_path = os.getcwd()
  
  expdata_path = current_path+agent_cfg.exp_data_dir
  #shield_threshold = adv_cfg.shield_threshold
  
  
  expert_agent_path = current_path + agent_cfg.saved_model_path
  safety_policy_path = current_path + safety_cfg.saved_model_path
  adv_path = current_path + adv_cfg.saved_model_path

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
#   #====================================================================
  adv_agent = SAC(agent_observation_space,
                   agent_action_space,
                   adv_cfg,
                   logdir,
                   env=env
                  )
  adv_agent.load_best_model(adv_path)
#   #====================================================================
  safety_agent = Safety_Agent(observation_space = agent_observation_space, 
                                action_space= agent_action_space,
                                args=safety_cfg,
                                logdir=logdir,
                                env = env,
                                adv_agent=adv_agent
                                )
  safety_agent.load_safety_model(safety_policy_path)
  #=====================================================================
  # recoveryRL_agent = Recovery_SAC(observation_space = agent_observation_space,
  #                           action_space = agent_action_space,
  #                           args = recoveryRL_cfg,
  #                           env= env
  #                           )
  #recoveryRL_agent.load_models(expert_agent_path, RecRL_model_path)

  recovery_rl_exp = None
  if use_safety and recovery_rl_algo_name != '':
      algo_model_path_map = get_model_directories(RecRL_model_path)
      recovery_rl_args = set_algo_configuration(recovery_rl_algo_name, recoveryRL_cfg)

      expert_path =  algo_model_path_map['algos'][recovery_rl_algo_name]['agent']
      recovery_path = algo_model_path_map['algos'][recovery_rl_algo_name]['recovery_agent']

      recovery_rl_exp = Comparative_Experiment(env=env, 
                                               exp_cfg=recovery_rl_args,
                                                expert_path=expert_path, 
                                                recovery_path=recovery_path,
                                                adv_path=adv_path,
                                                adv_cfg=adv_cfg)
  

  print(f'state space {agent_observation_space}')
  print(f'action space {agent_action_space}')
  # ADD

  highlights_data = []
  num_feats = 2
  action_dim = 8
  model = None
  # ADD_end

  """
  state range [(-60 -> 10), (-30, 30)]
  start state = [-50, 0]
  goal state = [0,0]
  action = [x_velocity, y_velocity]
  
  """
  episode_all_state = set()
  for i in tqdm(range(eval_epi_no)):
    print('#######################')
    print(f'episode #{i}')

    episode_data, all_state = run_eval_episode(env, expert_agent, safety_agent, use_safety, aaa_atk=True, aaa_agent=adv_agent, atk_rate=base_atk_rt,
                                shield_threshold=0.2, recovery_rl_exp=recovery_rl_exp, ctf_action_method=ctf_action_method)

    # ADD
    highlights_data.append(episode_data)
    episode_all_state = episode_all_state.union(all_state)
    # ADD_end

#   print('finally')
#   print(episode_all_state)
#   print(len(episode_all_state))
  print(" ---- NUMPY SEED VALUE: ", NUMPY_SEED)
  from Approximators.failure_search import Approximator, Trainer, Approx_Buffer
  from Approximators.risk_estimation import EstimateRisk, Rollout
  from Approximators.train_risk_estimation import estimate_agent_capability
  F_trainer = Trainer(Approximator(tuple([agent_observation_space+1])), Approx_Buffer())

  if recovery_rl_algo_name == '' or not use_safety: # Not recovery rl. We need to do tasks with SAC.
    experiment_data = estimate_agent_capability(env, expert_agent, adv_agent, safety_agent, F_trainer, model, episode_all_state, 
                                base_atk_rt, user_test, use_safety)
    fail_dic = experiment_data[0]
    ts_dic = experiment_data[1]
    reward_dic = experiment_data[2]
    fail_dic_user = experiment_data[3]
    ts_dic_user = experiment_data[4]
    reward_dic_user = experiment_data[5]

    safety_critic_dict = {}
    agent_critic_dict = {}
    state_action_list = gen_state_action_list(highlights_data)

    for state, action in state_action_list:
        safety_critic_val = safety_agent.get_shield_value(torchify(state), torchify(action))
        task_q1, task_q2 = expert_agent.critic(torchify(state), torchify(action))
        agent_task_critic_val = task_q1

        safety_critic_dict[tuple(round_state(state))] = safety_critic_val.item()
        agent_critic_dict[tuple(round_state(state))] = agent_task_critic_val.item()

    extra_dicts_list.append(DictionarySummaryModel("Safety Critic Average (by AdvExRL Model)", safety_critic_dict, SummaryMethodEnum.Average))
    extra_dicts_list.append(DictionarySummaryModel("Safety Critic Max (by AdvExRL Model)", safety_critic_dict, SummaryMethodEnum.Max))
    extra_dicts_list.append(DictionarySummaryModel("Agent Task Critic Average", agent_critic_dict, SummaryMethodEnum.Average))

    # print("Fail Dic:", fail_dic, type(fail_dic))
    # print("Ts Dic:", ts_dic, type(ts_dic))
    return highlights_data, num_feats, action_dim, expert_agent, fail_dic, ts_dic, reward_dic, fail_dic_user, ts_dic_user, reward_dic_user, extra_dicts_list
  
  else:
    print("----------RECOVERY RL COMPARISON----------")
    # RECOVERY RL COMPARISON

    algo_name = recovery_rl_algo_name
    
    if env_name=="maze": 
        RecRL_model_path = current_path+'/RecoveryRL/RecoveryRL_Model/Maze'
    elif env_name=="nav1":
        RecRL_model_path = current_path+'/RecoveryRL/RecoveryRL_Model/Navigation1'
    elif env_name=="nav2":
        RecRL_model_path = current_path+'/RecoveryRL/RecoveryRL_Model/Navigations2'
    RecRLexp_data_path = os.path.join(expdata_path, env_name, 'RecRL')

    #atk_rate = epsilon = epsilon_list[atk_rate_idx]
    atk_rate = epsilon = base_atk_rt
    RecRLexp_data_dir_sub_folders = os.path.join(RecRLexp_data_path,'Atk_rate{}_eps{}'.format(atk_rate, epsilon))
    if not os.path.exists(RecRLexp_data_dir_sub_folders):
        os.makedirs(RecRLexp_data_dir_sub_folders)

    # recRL_experiment_data = run_comparison(env_name=env_name, env_model_path= RecRL_model_path, atk_rt=atk_rate, eps=epsilon, aaa_agent_path=shield_path, aaa_cfg=adv_cfg, eval_episode= eval_epi_no)
    
    # If user_test is true, we will generate the failure dictionaries using 'Recovery RL' and 'SAC without safety'
    # so that we'll get the transitions of failure probabilities from 'SAC without safety' to 'Recovery RL'.
    # As a result, user_test=False is being passed here since we will make a separate call below for further computations.
    recRL_experiment_data = run_comparison(env=env, env_model_path=RecRL_model_path, atk_rt=atk_rate, eps=epsilon, 
                                            aaa_agent_path=adv_path, aaa_cfg=adv_cfg, eval_episode=eval_epi_no, 
                                            all_states=episode_all_state, algo_name=algo_name, user_test=False, ctf_method=ctf_action_method)
    
    recRL_data_path = os.path.join(RecRLexp_data_dir_sub_folders,'saved_exp_data.pkl')

    # recovery_fail_dic = recRL_experiment_data['algos'][algo_name]['result'][0]
    # recovery_ts_dic = recRL_experiment_data['algos'][algo_name]['result'][1]
    # recovery_fail_user_dic = {}
    # recovery_ts_user_dic = {}
    # print("Recovery Fail Dic: ", recovery_fail_dic, type(recovery_fail_dic))
    # print("Recovery ts Dic: ", recovery_ts_dic, type(recovery_ts_dic))

    # if user_test:
    #     recovery_fail_user_dic = recRL_experiment_data['algos'][algo_name]['result'][2]
    #     recovery_ts_user_dic = recRL_experiment_data['algos'][algo_name]['result'][3]

    #     print("Recovery Fail User Dic: ", recovery_fail_user_dic, type(recovery_fail_user_dic))
    #     print("Recovery ts User Dic: ", recovery_ts_user_dic, type(recovery_ts_user_dic))

    # Safety and task critic values after rollout
    # extra_dicts_list.append(DictionarySummaryModel("Safety Critic Average", recRL_experiment_data['algos'][algo_name]['result'][6]["safety_critic_dic"], SummaryMethodEnum.Average))
    # extra_dicts_list.append(DictionarySummaryModel("Agent Task Critic Average", recRL_experiment_data['algos'][algo_name]['result'][6]["agent_critic_dic"], SummaryMethodEnum.Average)) # agent critic dic
    # extra_dicts_list.append(DictionarySummaryModel("Safety Critic Max", recRL_experiment_data['algos'][algo_name]['result'][6]["safety_critic_max_dic"], SummaryMethodEnum.Max))

    # Please see the comment before run_comparison()
    if user_test:
        # Pass your desired attack rate for which you are computing data for "SAC without safety".
        experiment_data = estimate_agent_capability(env, expert_agent, adv_agent, safety_agent, F_trainer, 
            model, episode_all_state, atk_rate=base_atk_rt, user_test=False, use_safety=False)
        recovery_fail_dic = experiment_data[0]
        recovery_ts_dic = experiment_data[1]
        recovery_reward_dic = experiment_data[2]

        recovery_fail_dic_user = recRL_experiment_data['algos'][algo_name]['result'][0]
        recovery_ts_dic_user = recRL_experiment_data['algos'][algo_name]['result'][1]
        recovery_reward_dic_user = recRL_experiment_data['algos'][algo_name]['result'][2]

        experiment_data_0_atk = estimate_agent_capability(env, expert_agent, adv_agent, safety_agent, F_trainer, 
            model, episode_all_state, atk_rate=0, user_test=False, use_safety=False)
        extra_dicts_list.append(DictionarySummaryModel("Fail Dic", experiment_data_0_atk[0], SummaryMethodEnum.Average)) # fail dic
        extra_dicts_list.append(DictionarySummaryModel("Ts Dic", experiment_data_0_atk[1], SummaryMethodEnum.Average)) # ts dic
        extra_dicts_list.append(DictionarySummaryModel("Reward Dic", experiment_data_0_atk[2], SummaryMethodEnum.Average)) # reward dic
    else:
        recovery_fail_dic = recRL_experiment_data['algos'][algo_name]['result'][0]
        recovery_ts_dic = recRL_experiment_data['algos'][algo_name]['result'][1]
        recovery_reward_dic = recRL_experiment_data['algos'][algo_name]['result'][2]
        recovery_fail_dic_user = {}
        recovery_ts_dic_user = {}
        recovery_reward_dic_user = {}


    # Safety and task critic values of the states explored in highlights data only.
    safety_critic_dict = {}
    agent_critic_dict = {}
    state_action_list = gen_state_action_list(highlights_data)

    for state, action in state_action_list:
        safety_critic_val = recovery_rl_exp.agent.safety_critic.get_value(torchify(state), torchify(action))
        task_q1, task_q2 = recovery_rl_exp.agent.critic(torchify(state), torchify(action))
        agent_task_critic_val = task_q1

        safety_critic_dict[tuple(round_state(state))] = safety_critic_val.item()
        agent_critic_dict[tuple(round_state(state))] = agent_task_critic_val.item()

    extra_dicts_list.append(DictionarySummaryModel("Safety Critic Average", safety_critic_dict, SummaryMethodEnum.Average))
    extra_dicts_list.append(DictionarySummaryModel("Safety Critic Max", safety_critic_dict, SummaryMethodEnum.Max))
    extra_dicts_list.append(DictionarySummaryModel("Safety Critic Min", safety_critic_dict, SummaryMethodEnum.Min))
    extra_dicts_list.append(DictionarySummaryModel("Agent Task Critic Average", agent_critic_dict, SummaryMethodEnum.Average))
    extra_dicts_list.append(DictionarySummaryModel("Agent Task Critic Max", agent_critic_dict, SummaryMethodEnum.Max))
    extra_dicts_list.append(DictionarySummaryModel("Agent Task Critic Min", agent_critic_dict, SummaryMethodEnum.Min))

        
    # print("Recovery Fail Dic: ", recovery_fail_dic, type(recovery_fail_dic))
    # print("Recovery ts Dic: ", recovery_ts_dic, type(recovery_ts_dic))
    # print("Recovery Reward Dic: ", recovery_reward_dic, type(recovery_reward_dic))
    # print("Recovery Fail User Dic: ", recovery_fail_dic_user, type(recovery_fail_dic_user))
    # print("Recovery ts User Dic: ", recovery_ts_dic_user, type(recovery_ts_dic_user))
    # print("Recovery Reward User Dic: ", recovery_reward_dic_user, type(recovery_reward_dic_user))
    
    return highlights_data, num_feats, action_dim, expert_agent, recovery_fail_dic, recovery_ts_dic, recovery_reward_dic, recovery_fail_dic_user, recovery_ts_dic_user, recovery_reward_dic_user, extra_dicts_list








#   AdvEx_RL_data_path = os.path.join(expdata_path, env_name, 'AdvEx_RL','shield_threshold_{}'.format(shield_threshold))
  #====================================================================
#   for eps in epsilon_list:
#         epi_task_only_safety_vec = []
#         epi_task_rec_safety_vec = []
#         epi_only_task_reward_vec = []
#         epi_tsk_only_adv_reward_vec = []
#         tsk_test_info_vec = []
#         epi_task_rec_reward_vec = []
#         epi_task_rec_tsk_cnt = []
#         epi_task_rec_rec_cnt = []
#         epi_rec_reward_vec = []
#         epi_rec_advreward_vec = []
#         task_stp_cnt = []
#         tskrec_epi_stp_cnt = []
#         tsk_rec_test_info_vec = []
#         #-------------------------------------------------------------------
#         atk_rate = eps
#         epsilon = eps
#         logdir = os.path.join(AdvEx_RL_data_path,'atk_rate_{}_eps_{}'.format(atk_rate, epsilon))
#         if not os.path.exists(logdir):
#                 os.makedirs(logdir)
#         #**********************************************************************************************************
#         for i in tqdm(range(eval_epi_no)):
#               tsk_safety, tsk_count , _ , tsk_reward, tsk_adv_reward, epi_step, tsk_test_info = run_eval_episode(env, expert_agent,safety_agent, use_safety=False, atk_rate=atk_rate, epsilon=epsilon, aaa_agent=adv_agent, aaa_atk=True)
#               epi_task_only_safety_vec.append(tsk_safety)
#               task_stp_cnt.append(epi_step)
#               epi_only_task_reward_vec.append(tsk_reward)
#               epi_tsk_only_adv_reward_vec.append(tsk_adv_reward)
#               tsk_test_info_vec.append(tsk_test_info)

#               rectask_safety, t_cnt, r_cnt, e_reward, adv_r, epi_stp, tsk_rec_test_info = run_eval_episode(env, expert_agent, safety_agent, use_safety=True, shield_threshold=shield_threshold, atk_rate=atk_rate, epsilon=epsilon, aaa_agent=adv_agent, aaa_atk=True)
#               epi_task_rec_safety_vec.append(rectask_safety)
#               epi_task_rec_tsk_cnt.append(t_cnt)
#               epi_task_rec_rec_cnt.append(r_cnt)
#               epi_rec_reward_vec.append(e_reward)
#               epi_rec_advreward_vec.append(adv_r)
#               tskrec_epi_stp_cnt.append(epi_stp)
#               tsk_rec_test_info_vec.append(tsk_rec_test_info)

#         Data ={'shield_threshold': shield_threshold,
#               'attack Rate':atk_rate,
#               'epsilon': epsilon,

#               'task_only_safety': epi_task_only_safety_vec,
#               'task_only_reward': epi_only_task_reward_vec,
#               'task_only_adv_reward':epi_tsk_only_adv_reward_vec,
#               'task_only_epi_step':task_stp_cnt,
              
#               'epi_task_rec_safety': epi_task_rec_safety_vec,
#               'epi_task_rec_reward': epi_rec_reward_vec,
#               'epi_task_rec_tsk_cnt': epi_task_rec_tsk_cnt,
#               'epi_task_rec_rec_cnt': epi_task_rec_rec_cnt,
#               'epi_task_rec_adv_reward': epi_rec_advreward_vec,
#               'epi_task_rec_epi_step': tskrec_epi_stp_cnt
#               }

#         Info_data = {'tsk_info':tsk_test_info_vec,
#                     'tsk_rec_info':tsk_rec_test_info_vec
#                     }
                    
#         data_file = os.path.join(logdir,'')
        
#         with open(data_file+'Exp_data.pkl', 'wb') as f:
#             pickle.dump(Data, f)
      
#         with open(data_file+'Info_for_plotting_data.pkl', 'wb') as f2:
#             pickle.dump(Info_data, f2)
#         # **********************************************************************************************************
#         # **********************************************************************************************************
#         # *****************************************CALL RECOVERY RL COMPARISON************************************
#         if env_name=="maze": 
#             RecRL_model_path = current_path+'/RecoveryRL/RecoveryRL_Model/Maze'
#         elif env_name=="nav1":
#             RecRL_model_path = current_path+'/RecoveryRL/RecoveryRL_Model/Navigation1'
#         elif env_name=="nav2":
#             RecRL_model_path = current_path+'/RecoveryRL/RecoveryRL_Model/Navigations2'
#         RecRLexp_data_path = os.path.join(expdata_path, env_name, 'RecRL')
#         RecRLexp_data_dir_sub_folders = os.path.join(RecRLexp_data_path,'Atk_rate{}_eps{}'.format(atk_rate, epsilon))
#         if not os.path.exists(RecRLexp_data_dir_sub_folders):
#             os.makedirs(RecRLexp_data_dir_sub_folders)
#         # recRL_experiment_data = run_comparison(env_name=env_name, env_model_path= RecRL_model_path, atk_rt=atk_rate, eps=epsilon, aaa_agent_path=shield_path, aaa_cfg=adv_cfg, eval_episode= eval_epi_no)
#         recRL_experiment_data = run_comparison(env=env, env_model_path= RecRL_model_path, atk_rt=atk_rate, eps=epsilon, aaa_agent_path=adv_path, aaa_cfg=adv_cfg, eval_episode= eval_epi_no)
#         recRL_data_path = os.path.join(RecRLexp_data_dir_sub_folders,'saved_exp_data.pkl')
#         with open(recRL_data_path, 'wb') as f:
#             pickle.dump(recRL_experiment_data, f)
#         #**********************************************************************************************************
#         #**********************************************************************************************************
  
#   plot_path = os.path.join(expdata_path, env_name, 'Plots')
#   atk_name = 'aaa'
#   our_data = env_safety_data_our_model(AdvEx_RL_data_path, env_name)
#   recRL_data = get_all_Recovery_RL_data(RecRLexp_data_path,env_name)
#   draw_safety_plot(recRL_data, our_data, env_name, atk_name, plot_path)
#   draw_success_rate_plot(recRL_data,our_data, env_name, atk_name, plot_path)
#   draw_all_safety_with_policy_ratio_plot(recRL_data, our_data, env_name, atk_name, plot_path)


def gen_state_action_list(highlights_data):
    state_action_list = list()
    for episode_data in highlights_data:
        for i, state in enumerate(episode_data['states']):
            action = episode_data['env_actions'][i]
            state_action_list.append([state, action])

    return state_action_list


# Normalized Root Mean Squared Error
def NRMSE(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred) / (max(y_true) - min(y_true))

def calculate_fidelity_nav2(args, all_clusters, dataset: Data, num_episodes=10):
    env_name = args.env
    current_path = '/content/drive/MyDrive/XRL/AdvExRL_Submission/AdvExRL_code'

    if env_name == "maze":
        from env.maze import MazeNavigation
        env = MazeNavigation()
        RecRL_model_path = current_path+'/RecoveryRL/RecoveryRL_Model/Maze'
    elif env_name == 'nav1':
        from env.navigation1 import Navigation1
        env = Navigation1()
        RecRL_model_path = current_path+'/RecoveryRL/RecoveryRL_Model/Navigation1'
    elif env_name == 'nav2':
        from env.navigation2 import Navigation2
        env = Navigation2()
        RecRL_model_path = current_path+'/RecoveryRL/RecoveryRL_Model/Navigations2'

    base_atk_rt = args.attack_rate if hasattr(args, 'attack_rate') else 0.0
    recovery_rl_algo_name = args.recovery_rl if hasattr(args, 'recovery_rl') else ''
    use_safety = args.use_safety if hasattr(args, 'use_safety') else False
    numpy_seed = args.numpy_seed if hasattr(args, 'numpy_seed') else 0

    NUMPY_SEED = numpy_seed

    env.action_space.seed(NUMPY_SEED)
    np.random.seed(NUMPY_SEED)
    torch.manual_seed(NUMPY_SEED)
    random.seed(NUMPY_SEED)

    all_actions = dataset.actions
    all_policies = dataset.policies
    all_task_critic_vals = dataset.task_critic_vals
    all_safety_critic_vals = dataset.safety_critic_vals

    action_matches = list()
    q_safety_true = list()
    q_safety_pred = list()
    q_task_true = list()
    q_task_pred = list()
    policy_matches = list()    

    def load_trained_agents():
        agent_cfg =  get_victim_args(env_name)
        safety_cfg = get_safety_args(env_name)
        adv_cfg = get_adv_args(env_name)
        recoveryRL_cfg = recRL_get_args()
        #   print(adv_cfg)
        # current_path = os.getcwd()
        
        expdata_path = current_path+agent_cfg.exp_data_dir
        #shield_threshold = adv_cfg.shield_threshold
        
        
        expert_agent_path = current_path + agent_cfg.saved_model_path
        safety_policy_path = current_path + safety_cfg.saved_model_path
        adv_path = current_path + adv_cfg.saved_model_path

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
        #   #====================================================================
        adv_agent = SAC(agent_observation_space,
                        agent_action_space,
                        adv_cfg,
                        logdir,
                        env=env
                        )
        adv_agent.load_best_model(adv_path)
        #   #====================================================================
        safety_agent = Safety_Agent(observation_space = agent_observation_space, 
                                        action_space= agent_action_space,
                                        args=safety_cfg,
                                        logdir=logdir,
                                        env = env,
                                        adv_agent=adv_agent
                                        )
        safety_agent.load_safety_model(safety_policy_path)
        #=====================================================================
        # recoveryRL_agent = Recovery_SAC(observation_space = agent_observation_space,
        #                           action_space = agent_action_space,
        #                           args = recoveryRL_cfg,
        #                           env= env
        #                           )
        #recoveryRL_agent.load_models(expert_agent_path, RecRL_model_path)

        recovery_rl_exp = None
        if use_safety and recovery_rl_algo_name != '':
            algo_model_path_map = get_model_directories(RecRL_model_path)
            recovery_rl_args = set_algo_configuration(recovery_rl_algo_name, recoveryRL_cfg)

            expert_path =  algo_model_path_map['algos'][recovery_rl_algo_name]['agent']
            recovery_path = algo_model_path_map['algos'][recovery_rl_algo_name]['recovery_agent']

            recovery_rl_exp = Comparative_Experiment(env=env, 
                                                    exp_cfg=recovery_rl_args,
                                                        expert_path=expert_path, 
                                                        recovery_path=recovery_path,
                                                        adv_path=adv_path,
                                                        adv_cfg=adv_cfg)
            
        return expert_agent, adv_agent, safety_agent, recovery_rl_exp
    

    def get_cluster_action(clusters, num_feats=2, num_actions=8):
        if clusters == []:
            action = np.random.randint(0, num_actions)
            return action
        
        taken_actions = np.zeros(num_actions)
        for cluster in clusters:
            ids = cluster.getInstanceIds()
            actions = all_actions[ids]
            # print(actions)
            for i in range(len(actions)):
                taken_actions[actions[i]] += 1
        
        policy = taken_actions / np.sum(taken_actions)

        action = np.random.choice(np.arange(num_actions), p=policy)
        return action
    

    def get_cluster_q_safety(clusters):
        if clusters == []:
            return 0
        
        q_safety = -inf
        for cluster in clusters:
            ids = cluster.getInstanceIds()
            q_safety_vals = all_safety_critic_vals[ids]
            for i in range(len(q_safety_vals)):
                q_safety = max(q_safety, q_safety_vals[i])
        
        return q_safety
    
    def get_cluster_q_task(clusters):
        if clusters == []:
            return 0
        
        q_task_critic = 0
        total_count = 0
        for cluster in clusters:
            ids = cluster.getInstanceIds()
            q_task_critic_vals = all_task_critic_vals[ids]
            for i in range(len(q_task_critic_vals)):
                q_task_critic += q_task_critic_vals[i]
                total_count += 1
        
        return q_task_critic / total_count
    
    def get_cluster_policy_choice(clusters):
        if clusters == []:
            return PolicyEnum.TaskPolicy
        
        safety_policy_count = 0
        task_policy_count = 0
        for cluster in clusters:
            ids = cluster.getInstanceIds()
            policy_choices = all_policies[ids]
            for i in range(len(policy_choices)):
                if policy_choices[i] == PolicyEnum.SafetyPolicy:
                    safety_policy_count += 1
                else:
                    task_policy_count += 1
        
        return PolicyEnum.SafetyPolicy if (safety_policy_count >= task_policy_count) else PolicyEnum.TaskPolicy

    
    def find_clusters(obs, clusters, num_feats=2):
        obs = np.reshape(obs, [-1])
        valid_clusters = []
        for cluster in clusters:
            in_cluster = True
            for i in range(num_feats):
                bounds = cluster.get_bounds(i)
                if obs[i] < bounds[1] or obs[i] > bounds[0]: #if feat not in cluster
                    in_cluster = False
            
            if in_cluster:
                valid_clusters.append(cluster)
        
        return valid_clusters
    

    def perform_episode():
        done = False
        state = env.reset()
        num_steps = 0

        while not done:
            num_steps+=1
            action_tsk, entropy = expert_agent.select_action(state, eval=True)
            task_critic_val = 0.0
            safety_critic_val = 0.0
            used_policy = PolicyEnum.TaskPolicy
            #******************************************************************************   
            if np.random.rand() < base_atk_rt: # aaa attack
            #if left_risk(state) and aaa_atk and np.random.rand() < atk_rate:
                action_tsk, entropy = adv_agent.select_action(state)
            #****************************************************************************** 
            if use_safety:
                if recovery_rl_exp is None: # AdvExRL
                    shield_threshold = 0.2
                    shield_val_tsk = safety_agent.get_shield_value(torchify(state), torchify(action_tsk))
                    if shield_val_tsk>=shield_threshold:
                        action, entropy = safety_agent.select_action(state, eval=True)
                        used_policy = PolicyEnum.SafetyPolicy
                    else:
                        action = action_tsk

                    safety_critic_val = safety_agent.get_shield_value(torchify(state), torchify(action)).item()
                    task_q1, task_q2 = expert_agent.critic(torchify(state), torchify(action))
                    task_critic_val = task_q1.item()
                else:
                    agent_action, rec_action, recovery_selected, _ = recovery_rl_exp.get_action(env, state, base_atk_rt)

                    # agent_action can be from an expert agent or from an attacking agent. It depends on the attack rate.
                    # recovery_selected is true if a recovery policy is used to determine the action.
                    if recovery_selected:
                        action = rec_action
                        used_policy = PolicyEnum.SafetyPolicy
                    else:
                        action = agent_action
                    
                    safety_critic_val = recovery_rl_exp.agent.safety_critic.get_value(torchify(state), torchify(action)).item()
                    task_q1, task_q2 = recovery_rl_exp.agent.critic(torchify(state), torchify(action))
                    task_critic_val = task_q1.item()
            else:
                action = action_tsk
                task_q1, task_q2 = expert_agent.critic(torchify(state), torchify(action))
                safety_critic_val = safety_agent.get_shield_value(torchify(state), torchify(action)).item()
                task_critic_val = task_q1.item()

            action_d = binning_action(action)

            ######## Fidelity Score Calculations #####
            cls = find_clusters(state, all_clusters)

            abstract_action = get_cluster_action(cls)
            action_matches.append(int(abstract_action==action_d))

            abstract_q_safety = get_cluster_q_safety(cls)
            q_safety_true.append(safety_critic_val)
            q_safety_pred.append(abstract_q_safety)

            abstract_q_task = get_cluster_q_task(cls)
            q_task_true.append(task_critic_val)
            q_task_pred.append(abstract_q_task)

            abstract_policy_choice = get_cluster_policy_choice(cls)
            policy_matches.append(int(abstract_policy_choice == used_policy))
            ######## Fidelity Score Calculations #####
            
            nxt_state, reward, done, info = env.step(action)
            state = nxt_state
            done = done or (num_steps==env._max_episode_steps)
    

    expert_agent, adv_agent, safety_agent, recovery_rl_exp = load_trained_agents()
    
    for _ in range(num_episodes):
        perform_episode()

    fidelity = {
        'action_correct_ratio': sum(action_matches) / len(action_matches),
        'q_task_MAE': mean_absolute_error(q_task_true, q_task_pred),
        'q_task_MSE': mean_squared_error(q_task_true, q_task_pred),
        'q_task_RMSE': root_mean_squared_error(q_task_true, q_task_pred),
        'q_task_NRMSE': NRMSE(q_task_true, q_task_pred),
        'q_task_MAPE': mean_absolute_percentage_error(q_task_true, q_task_pred),
        'q_safety_MAE': mean_absolute_error(q_safety_true, q_safety_pred),
        'q_safety_MSE': mean_squared_error(q_safety_true, q_safety_pred),
        'q_safety_RMSE': root_mean_squared_error(q_safety_true, q_safety_pred),
        'q_safety_NRMSE': NRMSE(q_safety_true, q_safety_pred),
        'q_safety_MAPE': mean_absolute_percentage_error(q_safety_true, q_safety_pred),
        'policy_correct_ratio': sum(policy_matches) / len(policy_matches)
    }
    return fidelity
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configure-env', default='none', help='Set test environment to setup all configuration')
    parser.add_argument('--exp-data-dir', default='/Experimental_Data/', help='Set experiment data location')
    arg = parser.parse_args()
    name = arg.configure_env
    test_epi_no = 100
    run(env_name=name, eval_epi_no=test_epi_no)
