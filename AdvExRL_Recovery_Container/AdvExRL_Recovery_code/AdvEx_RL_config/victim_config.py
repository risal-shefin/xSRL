class BaseVictimConfig:
    """Base configuration class for Victim RL arguments."""

    def __init__(self):
        # Global Parameters
        self.configure_env = 'none'  # ''
        self.exp_data_dir = '/Experimental_Data/'  # 'Set experiment data location'
        self.env_change = 1.0  # 'multiplier for variation of env dynamics'
        self.device = ''  # 'run on CUDA (default: False)'
        self.logdir = 'runs'  # 'exterior log directory'
        self.logdir_suffix = ''  # 'log directory suffix'
        self.epoch = 1  # 'model updates per simulator step (default: 1)'
        self.seed = 123456  # 'random seed (default: 123456)'
        self.train_start = 10  # 'No of episode to start training'
        self.num_steps = 1000000  # 'maximum number of steps (default: 1000000)'
        self.num_eps = 1000000  # 'maximum number of episodes (default: 1000000)'
        self.model_path = 'None'  # 'Loaded model dir'

        # SAC Parameters
        self.hidden_size = 512  # 'hidden size (default: 256)'
        self.gamma = 0.99  # 'discount factor for reward (default: 0.99)'
        self.tau = 0.005  # 'target smoothing coefficient(τ) (default: 0.005)'
        self.alpha = 0.25  # 'Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)'
        self.lr = 0.0001  # 'learning rate (default: 0.0003)'
        self.batch_size = 64  # 'batch size (default: 256)'
        self.policy = 'Gaussian'  # 'Policy Type: Gaussian | Deterministic (default: Gaussian)'
        self.target_update_interval = 1  # 'Value target update per no. of updates per step (default: 1)'
        self.replay_size = 1000000  # 'size of replay buffer (default: 1000000)'
        self.env_change_rate = 1.0  # 'multiplier for variation of env dynamics'


class MazeVictimConfig(BaseVictimConfig):
    """Configuration for the Maze environment."""

    def __init__(self):
        super().__init__()
        self.env_name = 'maze'  # 'Gym environment (default: maze)'
        self.saved_model_path = '/AdvEx_RL_Trained_Models/Victim/Jul-28-2022_03_41_AM_SAC_maze_Gaussian/victim_agent/Agent_model/Best_Agent_Model/DateTime_Jul-28-2022_reward_-0.0'  # 'exterior log directory'


class Nav1VictimConfig(BaseVictimConfig):
    """Configuration for the Nav1 environment."""

    def __init__(self):
        super().__init__()
        self.env_name = 'nav1'  # 'Gym environment (default: maze)'
        self.saved_model_path = '/AdvEx_RL_Trained_Models/Victim/Jul-29-2022_04_30_AM_SAC_nav1_Gaussian/victim_agent/Agent_model/Best_Agent_Model/DateTime_Jul-29-2022_reward_-6.207633963897395'  # 'exterior log directory'
        self.hidden_size = 128  # 'hidden size (default: 256)'


class Nav2VictimConfig(BaseVictimConfig):
    """Configuration for the Nav2 environment."""

    def __init__(self):
        super().__init__()
        self.env_name = 'nav2'  # 'Gym environment (default: maze)'
        self.saved_model_path = '/AdvEx_RL_Trained_Models/Victim/Jul-27-2022_21_56_PM_SAC_nav2_Gaussian/victim_agent/Agent_model/Best_Agent_Model/DateTime_Jul-27-2022_reward_-6.407141558867541'  # 'exterior log directory'


def get_victim_args(env_name):
    if env_name =='maze':
        return MazeVictimConfig()
    elif env_name =='nav1':
        return Nav1VictimConfig()
    elif env_name =='nav2':
        return Nav2VictimConfig()
    
    