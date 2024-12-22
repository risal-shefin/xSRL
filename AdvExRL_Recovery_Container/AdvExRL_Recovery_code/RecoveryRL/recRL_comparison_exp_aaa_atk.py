import pickle
from constants import CONSTANT_VALUES
from common.enums import CtfActionMethodEnum
import torch
from RecoveryRL.recoverRL_args import recRL_get_args
import os
import sys
import numpy as np
import datetime
from RecoveryRL.recoveryRL_agent import Recovery_SAC
import copy
from AdvEx_RL.sac import SAC
from tqdm import tqdm
import math



TORCH_DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


def torchify(x):
    return torch.FloatTensor(x).to(TORCH_DEVICE)
    
def left_risk(state):
    if abs(state[1]) <= 15 and state[0] >= -40 and state[0] <= -30:
        return 1
    else:
        return 0

class Comparative_Experiment():
        def __init__(self, env, exp_cfg, expert_path, recovery_path, adv_path, adv_cfg=None):
            self.exp_cfg = exp_cfg
            if adv_cfg==None:
                self.adv_cfg = exp_cfg
            else:
                self.adv_cfg = adv_cfg
            #***********************************
            self.env = env
            self.agent_observation_space = env.observation_space.shape[0]
            self.agent_action_space = env.action_space.shape[0]
            # SAFETY NET FLAG SET. 
            # self.exp_cfg.use_recovery = False
            #***********************************
            #***********************************
            self.expert_path = expert_path
            self.adv_path = adv_path
            self.recovery_path = recovery_path
            #***********************************
            #***********************************
            self.agent = Recovery_SAC(self.agent_observation_space,
                                      self.agent_action_space ,
                                      self.exp_cfg,
                                      env=self.env
                                     )
            self.agent.load_models(self.expert_path, self.recovery_path)
            #***********************************
            logdir = ''
            self.aaa_agent = SAC(self.agent_observation_space,
                                    self.agent_action_space,
                                    self.adv_cfg,
                                    logdir,
                                    env=self.env
                                    )
            self.aaa_agent.load_best_model(self.adv_path)

            # Track the state of counterfactual action taken
            self.ctf_action_taken = False
            

        def get_action(self, env, state, atk_rate, epsilon=0.2, aaa=True, entropy=False, ctf_method=None):
                def recovery_thresh(state, action):
                    if not self.exp_cfg.use_recovery:
                        return False
                    critic_val = self.agent.safety_critic.get_value(
                        torchify(state).unsqueeze(0),
                        torchify(action).unsqueeze(0))
                    if critic_val > self.exp_cfg.eps_safe:
                        return True
                    return False

                entropy_val = torch.zeros(1)

                # Risky Counterfactual Action Selection
                if ctf_method is not None and not self.ctf_action_taken:
                    risky_action = self.get_riskiest_action(state, list(CONSTANT_VALUES.NAV2_DISCRETE_ACTIONS.values()))
                    if risky_action is not None:
                        self.ctf_action_taken = True
                        return risky_action, risky_action, False, entropy_val # TODO - Shielding Mechanism
                elif ctf_method == CtfActionMethodEnum.RiskyAlways and self.ctf_action_taken:
                    # Reaching here means that we took ctf action earlier and will take normal action now.
                    # Setting the flag false enables us to take ctf action again in the future.
                    self.ctf_action_taken = False

                # Sample action from policy
                if entropy:
                  action, entropy_val = self.agent.select_action(state, eval=True, ret_entropy=entropy)
                else:
                  action = self.agent.select_action(state, eval=True)

                #******************************************************************************
                if np.random.rand()<atk_rate and aaa:
                    if entropy:
                      action, entropy_val = self.aaa_agent.select_action(state, eval=True)
                    else:
                      action, _ = self.aaa_agent.select_action(state, eval=True)
                #******************************************************************************          
                if recovery_thresh(state, action) and self.exp_cfg.use_recovery:
                    recovery = True
                    real_action = self.agent.safety_critic.select_action(state)
                else:
                    recovery = False
                    real_action = np.copy(action)

                return action, real_action, recovery, entropy_val

        def run_test_experiment(self, n_episode, atk_rate, epsilon, ):
            
            taskonly_safety_vec = []
            taskonly_tsk_cnt_vec = []
            taskonly_reward_vec = []
            taskonly_adv_reward_vec = []
            taskonly_info_vec = []


            rectsk_safety_vec = []
            rectsk_tsk_cnt_vec = []
            rectsk_rec_cnt_vec = []
            rectsk_reward_vec = []
            rectsk_adv_reward_vec = []
            rectsk_info_vec = []


            for i in tqdm(range(n_episode)):
                env1 = copy.deepcopy(self.env)
                env2 = copy.deepcopy(self.env)
                
                rec_epi_rec_safety, rec_epi_tsk_cnt, rec_epi_rec_cnt, rec_epi_reward, rec_epi_adv_reward, rectsk_info = self.test_roll_out(env2, atk_rate, epsilon, aaa_atk= True)
                rectsk_safety_vec.append(rec_epi_rec_safety)
                rectsk_tsk_cnt_vec.append(rec_epi_tsk_cnt)
                rectsk_rec_cnt_vec.append(rec_epi_rec_cnt)
                rectsk_reward_vec. append(rec_epi_reward)
                rectsk_adv_reward_vec.append(rec_epi_adv_reward)
                rectsk_info_vec.append(rectsk_info)
            
            task_agent = { 'reward': taskonly_reward_vec,
                           'safety': taskonly_safety_vec,
                           'adv_reward': taskonly_adv_reward_vec,
                           'tsk_cnt': taskonly_tsk_cnt_vec,
                           'info': taskonly_info_vec
                        }
            task_rec_agent = {'reward': rectsk_reward_vec,
                              'safety': rectsk_safety_vec,
                              'adv_reward': rectsk_adv_reward_vec,
                              'tsk_cnt': rectsk_tsk_cnt_vec,
                              'rec_cnt': rectsk_rec_cnt_vec,
                              'info':rectsk_info_vec
                             }

            Data = { 'task_agent':task_agent,
                     'task_rec_agent': task_rec_agent
                    }
            return Data

        def test_roll_out(self, 
                          env, 
                          atk_rate=0.00, 
                          epsilon=0.00,
                          task_only = False,
                          aaa_atk = False,
                          ):
            done = False
            reward = 0
            epi_step = 0
            epi_reward = 0
            adv_reward = 0
            safety = 0

            rec_cnt = 0
            tsk_cnt = 0
            info_vec = []
            state = self.env.reset()
            while not done:
                epi_step+=1
                action, final_action, recovery_selected = self.get_action(env, state, atk_rate=atk_rate, epsilon= epsilon, aaa=aaa_atk)
                if (not task_only) and recovery_selected:
                    rec_cnt+=1
                else:
                    tsk_cnt+=1

                if task_only:
                    next_state, reward, done, info = self.env.step(action)  # Step
                else:
                    next_state, reward, done, info = self.env.step(final_action)  # Step
                #### add a print statement here ####
                info_vec.append(info)
                epi_reward+=reward
                adv_reward+=info['adv_reward']
                done = done or epi_step == env._max_episode_steps or (float(info['adv_reward'])>0)
                state = next_state
                if done:
                    if adv_reward<0:
                        safety=1
                    else:
                        safety = epi_step/env._max_episode_steps
                    break
            return safety, tsk_cnt, rec_cnt, epi_reward, adv_reward, info_vec

        def get_failure_data(self, n_episode, all_states, atk_rate=0.5, epsilon=0.0, user_test=False, ctf_method=None):
          # Given a finite number of states in all_states,
          # returns the failure probabilities for each x belongs to all_states.
          data = dict()
          fail_dic = {}
          ts_dic = {}
          reward_dic = {}
          fail_dic_user = {}
          ts_dic_user = {}
          reward_dic_user = {}
          safety_critic_dic = {}
          agent_critic_dic = {}
          safety_critic_max_dic = {}
          half = False # indicate if rollout half of the state, so starting perturbed action
          total_state_count = len(all_states)

          for i, x in enumerate(all_states):
            if user_test and i >= total_state_count/2:
                half = True

            failed_average = 0
            ts_average = 0
            reward_average = 0
            safety_critic_average = 0
            agent_critic_average = 0
            safety_critic_max = 0
            for j in range(n_episode):
              failed, ts, ret_data = self.rollout(x, atk_rate=atk_rate, epsilon=epsilon, task_only=(user_test and not half), ctf_method=ctf_method)
              failed_average += failed
              ts_average += ts
              reward_average += ret_data["total_reward"]
              safety_critic_average += ret_data["safety_critic_average"]
              agent_critic_average += ret_data["agent_critic_average"]
              safety_critic_max = ret_data["safety_critic_max"]

            failed_average /= n_episode
            ts_average /= n_episode
            reward_average /= n_episode
            safety_critic_average /= n_episode
            agent_critic_average /= n_episode

            # should be tuple(x) for new gridworld
            # should be x for old gridworld
            if not half:
              fail_dic[tuple(x)] = failed_average
              ts_dic[tuple(x)] = ts_average
              reward_dic[tuple(x)] = reward_average
              #print("Recovery RL State: {}, Failure Probability: {}, Expected Timestep to Finish {}".format(x, failed_average, ts_average))
            else:
              fail_dic_user[tuple(x)] = failed_average
              ts_dic_user[tuple(x)] = ts_average
              reward_dic_user[tuple(x)] = reward_average
              #print("I am here")
              #print("Recovery RL State: {}, perturbed Failure Probability: {}, Expected Timestep to Finish {}, Per TS Expected Attack {} and Expected Recovery Selected {}".format(x, failed_average, ts_average, atk_per_ts, safety_per_ts))

            safety_critic_dic[tuple(x)] = safety_critic_average
            agent_critic_dic[tuple(x)] = agent_critic_average
            safety_critic_max_dic[tuple(x)] = safety_critic_max

          addtional_dicts = {
            "safety_critic_dic": safety_critic_dic,
            "agent_critic_dic": agent_critic_dic,
            "safety_critic_max_dic": safety_critic_max_dic
          }
          return fail_dic, ts_dic, reward_dic, fail_dic_user, ts_dic_user, reward_dic_user, addtional_dicts

        def rollout(self,
                    init_state,
                    atk_rate=0.5, 
                    epsilon=0.0,
                    task_only = False,
                    aaa_atk = True,
                    ctf_method = None
                    ):
          done = False
          ts = 0

          state = list(init_state)
          self.env.reset_state(state)
          adv_reward = 0
          total_reward = 0
          total_agent_critic = 0
          total_safety_critic = 0
          max_safety_critic = 0
          self.ctf_action_taken = False

          while not done:
            ts+=1
            action, final_action, recovery_selected, entropy_val = self.get_action(self.env, state, atk_rate=atk_rate, 
                                                                            epsilon=epsilon, aaa=aaa_atk, entropy=True, ctf_method=ctf_method)
            
            # task_only = True means we will only use the agent action without an additional recovery policy.
            if task_only:
              next_state, reward, done, info = self.env.step(action)  # Step
              selected_action = action
            else:
              next_state, reward, done, info = self.env.step(final_action)  # Step
              selected_action = final_action

            adv_reward += float(info['adv_reward'])
            done = done or ts >= self.env._max_episode_steps or (float(info['adv_reward'])>0)
            #print(next_state)
            state = next_state
            total_reward += reward

            safety_critic_val = self.agent.safety_critic.get_value(torchify(state).unsqueeze(0), torchify(selected_action).unsqueeze(0)).item()
            total_safety_critic += safety_critic_val
            max_safety_critic = max(max_safety_critic, safety_critic_val)
            agent_critic, _ = self.agent.critic(torchify(state).unsqueeze(0), torchify(selected_action).unsqueeze(0))
            total_agent_critic += agent_critic.item()

          #print("done: ", done, " info: ", info, " ts: ", ts, " max_episode_steps: ", self.env._max_episode_steps)
          #print(f'start at {start_state} and return 0, {ts}')
            
          additional_data = {
            "total_reward": total_reward,
            "safety_critic_average": total_safety_critic / ts,
            "agent_critic_average": total_agent_critic / ts,
            "safety_critic_max": max_safety_critic
          }

        #   """
        #   @ if the agent reaches the target or avoids being hit by an obstacle, 0%
        #   @ else 100%
        #   """
        #   if info["success"] or adv_reward <= 0:
        #     return (0, ts)
        #   else:
        #     return (1, ts)
          
          """
          @ if agent reach 100 steps 100%
          @ if agent finish in 100 steps, 0%
          @ if agent go into red area, 100%
          """
          if ts == 100 and not info["success"]:
              return 1, ts, additional_data   # fail
          elif info["success"]:
              return 0, ts, additional_data
          else:
              return 1, ts, additional_data
        

        def get_riskiest_action(self, state, actions):
            riskiest_action = None
            max_critic_val = -math.inf

            for action in actions:
                critic_val = self.agent.safety_critic.get_value(
                    torchify(state).unsqueeze(0),
                    torchify(action).unsqueeze(0))
                
                # not risky action
                if critic_val <= self.exp_cfg.eps_safe:
                    continue

                if critic_val > max_critic_val:
                    max_critic_val = critic_val
                    riskiest_action = action

            return riskiest_action


def set_algo_configuration(algo, args): # 7 baselines
    if algo=='RRL_MF':
        args.cuda = True
        args.seed = 1234
        args.use_recovery=True
        args.MF_recovery= True
        args.Q_sampling_recovery = False
        args.gamma_safe = 0.8
        args.eps_safe = 0.2

    elif algo=='unconstrained':
        args.use_recovery=False
        args.MF_recovery= False
        args.cuda = True
        args.seed = 1234
        
    elif algo=='RSPO':
        args.cuda = True
        args.seed = 1234
        args.use_recovery=False
        args.MF_recovery= False
        args.DGD_constraints= True
        args.gamma_safe = 0.8
        args.eps_safe = 0.2

    elif algo=='RCPO':
        args.cuda = True
        args.use_recovery=False
        args.MF_recovery= False
        args.seed = 1234
        args.gamma_safe = 0.8
        args.eps_safe = 0.2
        args.RCPO = True

    elif algo=='RP':
        args.cuda = True
        args.use_recovery=False
        args.MF_recovery= False
        args.seed = 1234
        args.constraint_reward_penalty = 1000

    elif algo=='SQRL':
        args.cuda = True
        args.seed = 1234
        args.use_constraint_sampling= True
        args.use_recovery=False
        args.MF_recovery= False
        args.DGD_constraints= True
        args.gamma_safe = 0.8
        args.eps_safe = 0.2

    elif algo=='LR':
        args.cuda = True
        args.use_recovery=False
        args.MF_recovery= False
        args.seed = 1234
        args.DGD_constraints= True
        args.gamma_safe = 0.8
        args.eps_safe = 0.2
    return args

def get_model_directories(logdir):
    algo = []
    experiment_map = {}
    experiment_map['algos'] = {}
    best_agent_path = None
    
    print("---\n get_model_directories(): recRL_comparison_exp_aa_atk.py ----------")
    for fname in os.listdir(logdir):
      best_agent = -999999999
      algo.append(fname)
      agent_path = os.path.join(logdir, fname, 'sac_agent', 'best') 
    #   print(agent_path)
      for model in os.listdir(agent_path):
          name = model.split('reward_')[-1]
          if best_agent<=float(name):
            best_agent = float(name)
            best_agent_path = os.path.join(agent_path, model) 
      
      recovery_path = os.path.join(logdir, fname, 'safety_critic_recovery')
      for rec_model in os.listdir(recovery_path):
          recovery_agent_path = os.path.join(recovery_path, rec_model)

    #   if 'SQRL' in recovery_agent_path:
    #       recovery_agent_path = os.path.join(recovery_path, 'Mar-21-2024')
    #   if 'RRL_MF' in recovery_agent_path:
    #       recovery_agent_path = os.path.join(recovery_path, 'Apr-25-2024')

      result = None
      agents_path = {'agent': best_agent_path,
                    'recovery_agent': recovery_agent_path ,
                    'result': result
                    } 
      experiment_map['algos'][fname] = agents_path
      print(fname, agent_path, agents_path)

    return experiment_map 
    
def run_comparison(env, env_model_path, atk_rt=0.5, eps=0.0, aaa_agent_path=None, aaa_cfg=None, 
                   eval_episode = 100, all_states=None, algo_name="RRL_MF", user_test=False, ctf_method=None):
    args = recRL_get_args()
    env = env
    eval_episode = eval_episode
    algo_model_path_map = get_model_directories(env_model_path)
    # loop through the algos
    # algo_names = ["RRL_MF", "SQRL"]
    for algo in algo_model_path_map['algos']:
      if algo.strip().lower() == algo_name.strip().lower():
        args = set_algo_configuration(algo, args)
        expert_path =  algo_model_path_map['algos'][algo]['agent']
        recovery_path = algo_model_path_map['algos'][algo]['recovery_agent']
        print("-----\n run_comparison() -----\n", expert_path, recovery_path)
        evaluate_algo = Comparative_Experiment(env, args, expert_path, recovery_path, aaa_agent_path, aaa_cfg)
        #print(" >>>> Use Recovery, EPS SAFE, EXP CFG: ", evaluate_algo.exp_cfg.use_recovery, evaluate_algo.exp_cfg.eps_safe, evaluate_algo.exp_cfg)
        #exp_data = evaluate_algo.run_test_experiment(eval_episode, atk_rate=atk_rate, epsilon= epsilon)
        exp_data = evaluate_algo.get_failure_data(eval_episode, all_states, atk_rt, eps, user_test, ctf_method)
       
        algo_model_path_map['algos'][algo]['result'] = exp_data

        print("all_states count: ", len(all_states), ", Number of episodes: ", eval_episode)
    return algo_model_path_map
