import sys
from common.enums import CtfActionMethodEnum
import torch
from torch.autograd import Variable
import numpy as np
import random
import ray
from CAPS import explain
from topin_baseline import gen_apg
from config import argparser
import sys
import pickle
 
# adding Folder_2 to the system path
sys.path.insert(0, './AdvExRL_Recovery_Container/AdvExRL_Recovery_code')

from data import Data
from abstract import APG
from zahavy_baseline import explain_zahavy
from translation import MazePredicates, Nav2Predicates
from AdvExRL_Recovery_Container.AdvExRL_Recovery_code.test_nav_maze import run as run_nav_maze
from AdvExRL_Recovery_Container.AdvExRL_Recovery_code.test_nav_maze import calculate_fidelity_nav2


# it is called from inside "elif args.env == 'nav2' or args.env == 'maze':" section
def set_nav_maze_args(args):
    args.num_episodes = 1
    args.attack_rate = 0.0 # [0,1] range
    ## To use recovery_rl, you must set the args.use_safety to True.
    ## If args.use_safety=True and no args.recovery_rl is set, AdvExRL's safety agent will be used.
    # args.use_safety = True
    # args.recovery_rl = 'RRL_MF'
    # args.recovery_rl = 'SQRL'
    args.user_test_nav2 = False # Experimental. Set True to get failure probability and timestep transition. May need fixes.
    # args.numpy_seed = np.random.randint((1<<31)-1)
    # args.numpy_seed = 1433448179
    # args.numpy_seed = 123456
    #args.ctf_action_method = CtfActionMethodEnum.RiskyAlways
    # args.calc_fidelity = True

if __name__ == '__main__':

    args = argparser()
    model_path = args.path

    # Due to some argparse conflicts existing in this project, command line arguments produce errors for nav2.
    # So, we need to set the arguments manually here if we don't want to use the default values.
    args.env = 'nav2'
    # args.env = 'maze'

    # if args.env != 'safety_grid':
    #   assert model_path != ''
    fidelity_fn = None
    reward_dic = None
    reward_dic_user = None
    extra_dicts_list = None

    if args.env == 'nav2' or args.env == 'maze': # we'll follow similar workflows for both nav2 and maze
        def torchify(x, device):
            return torch.FloatTensor(x).to(device).unsqueeze(0)
        
        set_nav_maze_args(args)
        
        user_test = args.user_test_nav2 # True to get RE and TS transition
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # path = '/content/drive/MyDrive/XRL/AdvExRL_Recovery_Container/AdvExRL_Recovery_code/Experimental_Data/random/render'
        print('start running')
        # data, num_feats, num_actions, fail_dic, ts_dic, agent = run_nav2('nav2', path)

        exp_data = run_nav_maze(args.env, args)
        data = exp_data[0]
        num_feats = exp_data[1]
        num_actions = exp_data[2] 
        agent = exp_data[3] 
        fail_dic = exp_data[4]
        ts_dic = exp_data[5]
        reward_dic = exp_data[6]
        fail_dic_user = exp_data[7]
        ts_dic_user = exp_data[8] 
        reward_dic_user = exp_data[9]
        extra_dicts_list = exp_data[10]

        print('finish running')

        if args.calc_fidelity:
            fidelity_fn = calculate_fidelity_nav2

        print('start clustering')
        def value_fn(state):
          action, _ = agent.select_action(state, eval=True)
          # state = torch.FloatTensor(state).to(device).unsqueeze(0)
          with torch.no_grad():
            # print(f'debug --- state {state}')
            # print(f'debug --- action {action}')
            q1, q2 = agent.critic(torchify(state, device), torchify(action, device))
          value = torch.max(q1,q2)
          value = value.detach().cpu().numpy().reshape(-1)[0]
          return value
        
        dataset = Data(data, value_fn)
        #print(dataset)
        print('value function is ok :)')

        if args.env == 'nav2':
            translator = Nav2Predicates(num_feats=num_feats)
        elif args.env == 'maze':
            translator = MazePredicates(num_feats=num_feats)
        #fail_dic_user = []
        #ts_dic_user = []
    else:
        raise ValueError('Enter valid environment')

    if args.zahavy_baseline:
        abstract_baseline = APG(num_actions, value_fn, translator)
        explain_zahavy(args, dataset, translator, abstract_baseline, num_actions, fidelity_fn, model_path, mode=args.alg)
    elif args.topin_baseline:
        info = {'states': dataset.states, 'actions': dataset.actions, 'next_states': dataset.next_states, 'dones': dataset.dones, 'entropies': dataset.entropies}
        abstract_baseline = APG(num_actions, value_fn, translator, info=info)
        gen_apg(abstract_baseline, model_path, fidelity_fn, mode=args.alg)
    else: # what is running at the end
        abstract_baseline = APG(num_actions, value_fn, translator)
        print("run fail user: ", fail_dic_user)
        explain(args, dataset, model_path, translator, num_feats, num_actions, fidelity_fn, abstract_baseline, mode=args.alg, 
                fail=fail_dic, ts=ts_dic, fail_user=fail_dic_user, ts_user=ts_dic_user, user_test=user_test, reward=reward_dic, reward_user=reward_dic_user, extra_dicts=extra_dicts_list)
        
