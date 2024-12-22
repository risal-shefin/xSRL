from AdvEx_RL_config.adversary_config import get_adv_args
from AdvEx_RL.adv_trainer import Adv_Experiment
import torch
import warnings
warnings.filterwarnings('ignore')
import os
import argparse

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
    adv_cfg = get_adv_args(env_name) 
    current_path = os.getcwd()
    adv_cfg.logdir = current_path+'/AdvEx_RL_Trained_Models_New/Adversary/'
    experiment = Adv_Experiment(env, adv_cfg, test_env)
    experiment.agent_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configure-env', default='none', help='Set test environment to setup all configuration')
    arg = parser.parse_args()
    name = arg.configure_env
    run(env_name=name)

