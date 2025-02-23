class BaseAdversaryConfig:
    """Base configuration class for Adversary RL arguments."""

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
        self.model_path = 'runs'  # 'exterior log directory'

        # SAC Parameters
        self.hidden_size = 512  # 'hidden size (default: 256)'
        self.gamma = 0.99  # 'discount factor for reward (default: 0.99)'
        self.tau = 0.005  # 'target smoothing coefficient(τ) (default: 0.005)'
        self.alpha = 0.20  # 'Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)'
        self.lr = 0.0003  # 'learning rate (default: 0.0003)'
        self.batch_size = 64  # 'batch size (default: 256)'
        self.policy = 'Gaussian'  # 'Policy Type: Gaussian | Deterministic (default: Gaussian)'
        self.target_update_interval = 1  # 'Value target update per no. of updates per step (default: 1)'
        self.replay_size = 1000000  # 'size of replay buffer (default: 1000000)'


class MazeAdversaryConfig(BaseAdversaryConfig):
    """Configuration for the Maze environment."""

    def __init__(self):
        super().__init__()
        self.env_name = 'maze'  # 'Gym environment (default: maze)'
        # parser.add_argument('--saved_model_path', default='/AdvEx_RL_Trained_Models/Adversary/Jul-28-2022_02_55_AM_SAC_maze_Gaussian/adversary_agent/Agent_model/Best_Agent_Model/DateTime_Jul-28-2022_reward_1.0', help='exterior log directory')
        self.saved_model_path = '/AdvEx_RL_Trained_Models/Adversary/Nov-07-2024_16_35_PM_SAC_maze_Gaussian/adversary_agent/Agent_model/Best_Agent_Model/DateTime_Nov-08-2024_reward_1.0'  # 'exterior log directory'
        self.shield_threshold = 0.90  # 'Shield threshold value default .97'


class Nav1AdversaryConfig(BaseAdversaryConfig):
    """Configuration for the Nav1 environment."""

    def __init__(self):
        super().__init__()
        self.env_name = 'nav1'  # 'Gym environment (default: maze)'
        self.saved_model_path = '/AdvEx_RL_Trained_Models/Adversary/Jul-28-2022_21_11_PM_SAC_nav1_Gaussian/adversary_agent/Agent_model/Best_Agent_Model/DateTime_Jul-28-2022_reward_1.0'  # 'exterior log directory'
        self.shield_threshold = 0.80  # 'Shield threshold value'


class Nav2AdversaryConfig(BaseAdversaryConfig):
    """Configuration for the Nav2 environment."""

    def __init__(self):
        super().__init__()
        self.env_name = 'nav2'  # 'Gym environment (default: maze)'
        self.saved_model_path = '/AdvEx_RL_Trained_Models/Adversary/Jul-27-2022_21_38_PM_SAC_nav2_Gaussian/adversary_agent/Agent_model/Best_Agent_Model/DateTime_Jul-27-2022_reward_1.0'  # 'exterior log directory'
        self.shield_threshold = 0.50  # 'Shield threshold value'


def get_adv_args(env_name):
    if env_name =='maze':
        return MazeAdversaryConfig()
    elif env_name =='nav1':
        return Nav1AdversaryConfig()
    elif env_name =='nav2':
        return Nav2AdversaryConfig()
    
    