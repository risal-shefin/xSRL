class BaseSafetyConfig:
    """Base configuration class for Safety RL arguments."""

    def __init__(self):
        # Global Parameters
        self.configure_env = 'none'  # ''
        self.env_change = 1.0  # 'multiplier for variation of env dynamics'
        self.exp_data_dir = '/Experimental_Data/'  # 'Set experiment data location'
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
        self.hidden_size = 256  # 'hidden size (default: 256)'
        self.gamma = 0.99  # 'discount factor for reward (default: 0.99)'
        self.tau = 0.005  # 'target smoothing coefficient(τ) (default: 0.005)'
        self.alpha = 0.20  # 'Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)'
        self.lr = 0.0003  # 'learning rate (default: 0.0003)'
        self.batch_size = 64  # 'batch size (default: 256)'
        self.policy = 'Gaussian'  # 'Policy Type: Gaussian | Deterministic (default: Gaussian)'
        self.target_update_interval = 1  # 'Value target update per no. of updates per step (default: 1)'
        self.replay_size = 1000000  # 'size of replay buffer (default: 1000000)'

        # Rollout Parameters
        self.beta = 0.7  # 'Rollout agent - adversarial sample ratio default 0.7'
        self.eta = 0.5  # 'Rollout agent - expert sample ratio default 0.5*(1-adversarial sample ratio)'


class MazeSafetyConfig(BaseSafetyConfig):
    """Configuration for the Maze environment."""

    def __init__(self):
        super().__init__()
        self.env_name = 'maze'  # 'Gym environment (default: maze)'
        # parser.add_argument('--saved_model_path', default='/AdvEx_RL_Trained_Models/Safety_policy/Jul-28-2022_04_15_AM_SafetyAgent_maze/Recovery_model/Best/Jul-28-2022_Best_Recovery_Model_safety_ratio1.0/recovery_policy', help='exterior log directory')
        self.saved_model_path = '/AdvEx_RL_Trained_Models/Safety_policy/Nov-08-2024_22_29_PM_SafetyAgent_maze/Recovery_model/Best/Nov-10-2024_Best_Recovery_Model_safety_ratio1.0/recovery_policy'  # 'exterior log directory'


class Nav1SafetyConfig(BaseSafetyConfig):
    """Configuration for the Nav1 environment."""

    def __init__(self):
        super().__init__()
        self.env_name = 'nav1'  # 'Gym environment (default: maze)'
        self.hidden_size = 512  # 'hidden size (default: 256)'
        self.saved_model_path = '/AdvEx_RL_Trained_Models/Safety_policy/Aug-04-2022_02_02_AM_SafetyAgent_nav1/Recovery_model/Interval/1060_Interval_Recovery_Model_safety_1.0/recovery_policy'  # 'exterior log directory'
        self.tau = 0.0005  # 'target smoothing coefficient(τ) (default: 0.005)'
        self.lr = 0.000003  # 'learning rate (default: 0.0003)'


class Nav2SafetyConfig(BaseSafetyConfig):
    """Configuration for the Nav2 environment."""

    def __init__(self):
        super().__init__()
        self.env_name = 'nav2'  # 'Gym environment (default: maze)'
        self.hidden_size = 512  # 'hidden size (default: 512)'
        self.saved_model_path = '/AdvEx_RL_Trained_Models/Safety_policy/Jul-28-2022_01_05_AM_SafetyAgent_nav2/Recovery_model/Best/Jul-28-2022_Best_Recovery_Model_safety_ratio1.0/recovery_policy'  # 'exterior log directory'

def get_safety_args(env_name):
    if env_name=='maze':
        return MazeSafetyConfig()
    elif env_name=='nav1':
        return Nav1SafetyConfig()
    elif env_name=='nav2':
        return Nav2SafetyConfig()

    