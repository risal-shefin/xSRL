from AdvEx_RL_config.safety_config import get_safety_args
from AdvEx_RL.safety_trainer import Safety_trainer
import torch
import os
import argparse
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
    safety_cfg = get_safety_args(env_name)  

    current_path = os.getcwd()
    safety_cfg.logdir = current_path+'/AdvEx_RL_Trained_Models_New/Safety_policy/'
    
    exp1 = Safety_trainer(env, safety_cfg, test_env)
    exp1.agent_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configure-env', default='none', help='Set test environment to setup all configuration')
    arg = parser.parse_args()
    name = arg.configure_env
    run(env_name=name)
