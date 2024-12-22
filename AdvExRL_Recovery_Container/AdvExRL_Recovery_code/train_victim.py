from AdvEx_RL_config.victim_config import get_victim_args
from AdvEx_RL.victim_trainer import Victim_Experiment
import warnings
import os
warnings.filterwarnings('ignore')
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
    victim_cfg = get_victim_args(env_name) 
    victim_cfg.env_name = env_name
    current_path = os.getcwd()
    victim_cfg.logdir = current_path+'/AdvEx_RL_Trained_Models_New/Victim/'
    experiment = Victim_Experiment(env, victim_cfg, test_env)
    experiment.agent_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configure-env', default='none', help='Set test environment to setup all configuration')
    arg = parser.parse_args()
    name = arg.configure_env
    run(env_name=name)
