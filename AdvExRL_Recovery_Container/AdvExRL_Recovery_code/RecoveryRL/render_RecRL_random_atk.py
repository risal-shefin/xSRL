import pickle
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
TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def torchify(x):
    return torch.FloatTensor(x).to(TORCH_DEVICE)

def npy_to_gif(im_list, filepath, fps=4):
    if not os.path.exists(filepath):
                os.makedirs(filepath)
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filepath +'_rendered_fig.gif')


class Comparative_Experiment():
        def __init__(self, env, exp_cfg, expert_path, recovery_path, env_name, render_function, sample=0):
            self.exp_cfg = exp_cfg
            #***********************************
            self.env = env
            self.env_name = env_name
            self.sample = sample
            self.env_render_function = render_function
            self.agent_observation_space = env.observation_space.shape[0]
            self.agent_action_space = env.action_space.shape[0]
            #***********************************
            #***********************************
            self.expert_path = expert_path
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
            #***********************************

        def get_action(self, env, state, atk_rate, epsilon):
                def recovery_thresh(state, action):
                    if not self.exp_cfg.use_recovery:
                        return False

                    critic_val = self.agent.safety_critic.get_value(
                        torchify(state).unsqueeze(0),
                        torchify(action).unsqueeze(0))

                    if critic_val > self.exp_cfg.eps_safe:
                        return True
                    return False

                action = self.agent.select_action(state, eval=True)  # Sample action from policy
                #******************************************************************************
                if np.random.rand()<atk_rate:
                    perturbation = np.random.uniform(-epsilon, epsilon)
                    action = action+perturbation
                    action = np.clip(action, env.action_space.low.min(), env.action_space.high.max())
                #******************************************************************************   
                if recovery_thresh(state, action) and self.exp_cfg.use_recovery:
                    recovery = True
                    real_action = self.agent.safety_critic.select_action(state)
                else:
                    recovery = False
                    real_action = np.copy(action)
                return action, real_action, recovery

        def run_test_experiment(self, n_episode, atk_rate, epsilon):
            for i in tqdm(range(n_episode)):
                env1 = copy.deepcopy(self.env)
                env2 = copy.deepcopy(self.env)
                
                tsk_render_img_vec = self.test_roll_out(env1, atk_rate, epsilon, task_only=True)
                safety_render_img_vec = self.test_roll_out(env2, atk_rate, epsilon)

            Data = { 'without_safety_render':tsk_render_img_vec,
                     'safety_render': safety_render_img_vec
                    }
            return Data

        def test_roll_out(self, 
                          env, 
                          atk_rate=0.00, 
                          epsilon=0.00,
                          task_only = False,
                          ):
            done = False
            epi_step=0
            rec_cnt =0
            tsk_cnt = 0
            state = env.reset()
            if self.env_name=='maze':
                im_list = [env._get_obs(images=True)]
            else:
                im_list = [self.env_render_function(state).squeeze()]
            while not done:
                epi_step+=1
                action, final_action, recovery_selected = self.get_action(env, state, atk_rate=atk_rate, epsilon= epsilon)
                if (not task_only) and recovery_selected:
                    rec_cnt+=1
                else:
                    tsk_cnt+=1
                next_state, reward, done, info = env.step(final_action)  # Step
                if not self.sample==0:
                    if epi_step%self.sample==0:
                        if self.env_name=='maze':
                            im_list.append(env._get_obs(images=True))
                        else:
                            im_list.append(self.env_render_function(next_state).squeeze())
                else:
                    if self.env_name=='maze':
                            im_list.append(env._get_obs(images=True))
                    else:
                            im_list.append(self.env_render_function(next_state).squeeze())
                
                done = done or epi_step == env._max_episode_steps or (float(info['adv_reward'])>0)
                state = next_state
                if done:
                    if not self.env_name=="maze":
                        self.env_render_function([0,0],end=True)
                    break
            return im_list

def set_algo_configuration(algo, args):
    if algo=='RRL_MF':
        args.cuda = True
        args.seed = 1234
        args.use_recovery=True
        args.MF_recovery= True
        args.Q_sampling_recovery = False
        args.gamma_safe = 0.8
        args.eps_safe = 0.3

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
        args.eps_safe = 0.3

    elif algo=='RCPO':
        args.cuda = True
        args.use_recovery=False
        args.MF_recovery= False
        args.seed = 1234
        args.gamma_safe = 0.8
        args.eps_safe = 0.3
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
        args.eps_safe = 0.3

    elif algo=='LR':
        args.cuda = True
        args.use_recovery=False
        args.MF_recovery= False
        args.seed = 1234
        args.DGD_constraints= True
        args.gamma_safe = 0.8
        args.eps_safe = 0.3
    return args

def get_model_directories(logdir):
    algo = []
    experiment_map = {}
    experiment_map['algos'] = {}
    best_agent_path = None
    
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
      image_vec = None
      agents_path = {'agent': best_agent_path,
                    'recovery_agent': recovery_agent_path ,
                    'render_vec': image_vec
                    } 
      experiment_map['algos'][fname] = agents_path
      
    return experiment_map 
    
def run_comparison(env, env_name, rend_fun, env_model_path, s_rate=0, atk_rt=0.0, eps=0.0, eval_episode=100):
    args = recRL_get_args()
    env = env
    env_name = env_name
    render_func = rend_fun
    sample_rate = s_rate
    atk_rate = atk_rt
    epsilon = eps
    eval_episode = eval_episode

    algo_model_path_map = get_model_directories(env_model_path)
    for algo in algo_model_path_map['algos']:
       args = set_algo_configuration(algo, args)
       expert_path =  algo_model_path_map['algos'][algo]['agent']
       recovery_path = algo_model_path_map['algos'][algo]['recovery_agent']
       evaluate_algo = Comparative_Experiment(env, args, expert_path, recovery_path, env_name=env_name, render_function=render_func,sample=sample_rate)
       exp_data = evaluate_algo.run_test_experiment(eval_episode, atk_rate=atk_rate, epsilon= epsilon)
       algo_model_path_map['algos'][algo]['render_vec'] = exp_data
    return algo_model_path_map

