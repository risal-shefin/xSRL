class RecoveryRLConfig:
    """
    Configuration class for Recovery RL experiments.
    Util to compile command line arguments for core script to run experiments for Recovery RL (rrl_main.py).
    """

    def __init__(self):
        # Global Parameters
        self.configure_env = 'none'  # ''
        self.exp_data_dir = '/Experimental_Data/'  # 'Set experiment data location'
        self.env_change = 1.0  # 'multiplier for variation of env dynamics'
        self.env_name = 'maze'  # 'Gym environment (default: maze)'
        self.logdir = 'runs'  # 'exterior log directory'
        self.logdir_suffix = ''  # 'log directory suffix'
        self.cuda = False  # 'run on CUDA (default: False)'
        self.cnn = False  # 'visual observations (default: False)'
        self.lr = 0.0003  # 'learning rate (default: 0.0003)'
        self.updates_per_step = 1  # 'model updates per simulator step (default: 1)'
        self.start_steps = 100  # 'Steps sampling random actions (default: 100)'
        self.target_update_interval = 1  # 'Value target update per no. of updates per step (default: 1)'

        # Forward Policy (SAC)
        self.policy = 'Gaussian'  # 'Policy Type: Gaussian | Deterministic (default: Gaussian)'
        self.eval = True  # 'Evaluates a policy every 10 episode (default: True)'
        self.gamma = 0.99  # 'discount factor for reward (default: 0.99)'
        self.tau = 0.005  # 'target smoothing coefficient(τ) (default: 0.005)'
        self.alpha = 0.2  # 'Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)'
        self.automatic_entropy_tuning = False  # 'Automatically adjust α (default: False)'
        self.seed = 123456  # 'random seed (default: 123456)'
        self.batch_size = 256  # 'batch size (default: 256)'
        self.num_steps = 1000000  # 'maximum number of steps (default: 1000000)'
        self.num_eps = 1000000  # 'maximum number of episodes (default: 1000000)'
        self.hidden_size = 256  # 'hidden size (default: 256)'
        self.replay_size = 1000000  # 'size of replay buffer (default: 1000000)'
        self.task_demos = False  # 'use task demos to pretrain safety critic'
        self.num_task_transitions = 10000000  # 'number of task transitions'
        self.critic_pretraining_steps = 3000  # 'gradient steps for critic pretraining'

        # Q risk
        self.pos_fraction = -1  # 'fraction of positive examples for critic training'
        self.gamma_safe = 0.5  # 'discount factor for constraints (default: 0.9)'
        self.eps_safe = 0.1  # 'Qrisk threshold (default: 0.1)'
        self.tau_safe = 0.0002  # 'Qrisk target smoothing coefficient(τ) (default: 0.005)'
        self.safe_replay_size = 1000000  # 'size of replay buffer for Qrisk (default: 1000000)'
        self.num_unsafe_transitions = 10000  # 'number of unsafe transitions'
        self.critic_safe_pretraining_steps = 10000  # 'gradient steps for Qrisk pretraining'

        ################### Recovery RL ###################
        self.use_recovery = False  # 'use recovery policy'

        # Recovery RL MF Recovery
        self.MF_recovery = False  # 'model free recovery policy'
        self.Q_sampling_recovery = False  # 'sample actions over Qrisk for recovery'

        # Recovery RL MB Recovery (parameters for PETS)
        self.ctrl_arg = []  # 'Controller arguments, see https://github.com/kchua/handful-of-trials#controller-arguments'
        self.override = []  # 'Override default parameters, see https://github.com/kchua/handful-of-trials#overrides'
        self.recovery_policy_update_freq = 1  # 'Model updated with new transitions every recovery_policy_update_freq episodes'

        # Recovery RL Visual MB Recovery
        self.vismpc_recovery = False  # 'use model-based visual planning for recovery policy'
        self.load_vismpc = False  # 'load pre-trained visual dynamics model'
        self.model_fname = 'image_maze_dynamics'  # 'path to pre-trained visual dynamics model'
        self.beta = 10  # 'beta for training VAE for visual dynamics model (default: 10)'

        # Recovery RL Ablations
        self.disable_offline_updates = False  # 'only train Qrisk online'
        self.disable_online_updates = False  # 'only train Qrisk on offline data'
        self.disable_action_relabeling = False  # 'train task policy on recovery policy actions'
        self.add_both_transitions = False  # 'use both task and recovery transitions to train task policy'

        ################### Comparisons ###################

        # RP
        self.constraint_reward_penalty = 0  # 'reward penalty when a constraint is violated (default: 0)'

        # Lagrangian, RSPO
        self.DGD_constraints = False  # 'use dual gradient descent to jointly optimize for task rewards + constraints'
        self.use_constraint_sampling = False  # 'sample actions with task policy and filter with Qrisk'
        self.nu = 0.01  # 'penalty term in Lagrangian objective'
        self.update_nu = False  # 'update Lagrangian penalty term'
        self.nu_schedule = False  # 'use linear schedule for Lagrangian penalty term nu'
        self.nu_start = 1e3  # 'start value for nu (high)'
        self.nu_end = 0  # 'end value for nu (low)'

        # RCPO
        self.RCPO = False  # 'Use RCPO'
        self.lambda_RCPO = 0.01  # 'penalty term for RCPO (default: 0.01)'


def recRL_get_args():
    return RecoveryRLConfig()
