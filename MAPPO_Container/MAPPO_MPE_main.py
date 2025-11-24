import torch
import numpy as np
import argparse
from .normalization import Normalization, RewardScaling
from .replay_buffer import ReplayBuffer
from .mappo import MAPPO
from .make_env_pettingzoo import make_env
from gym.spaces import Box, Discrete


class Runner_MAPPO_MPE:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env = make_env(env_name, discrete=True)  # Discrete action space
        self.args.N = self.env.n  # The number of agents
        
        # Handle observation dimensions for each agent
        self.args.obs_dim_n = []
        for obs_space in self.env.observation_space:
            # Properly handle Box observation spaces from gymnasium
            if isinstance(obs_space, Box) or str(type(obs_space)) == "<class 'gymnasium.spaces.box.Box'>":
                self.args.obs_dim_n.append(obs_space.shape[0])
            else:
                print(f"Unexpected observation space type: {type(obs_space)}", flush=True)
                self.args.obs_dim_n.append(obs_space.shape[0])  # Try to use shape attribute anyway
        
        # Handle action dimensions for each agent
        self.args.action_dim_n = []
        for act_space in self.env.action_space:
            # Handle both gym and gymnasium Discrete spaces
            if isinstance(act_space, Discrete) or str(type(act_space)) == "<class 'gymnasium.spaces.discrete.Discrete'>":
                self.args.action_dim_n.append(act_space.n)
            else:
                print(f"Unexpected action space type: {type(act_space)}", flush=True)
                if hasattr(act_space, 'n'):
                    self.args.action_dim_n.append(act_space.n)
                else:
                    self.args.action_dim_n.append(act_space.shape[0])  # For continuous actions
        
        # Only for homogenous agents environments like Spread in MPE, all agents have the same dimension of observation space and action space
        self.args.obs_dim = self.args.obs_dim_n[0]  # The dimensions of an agent's observation space
        self.args.action_dim = self.args.action_dim_n[0]  # The dimensions of an agent's action space
        self.args.state_dim = np.sum(self.args.obs_dim_n)  # The dimensions of global state space (Sum of the dimensions of the local observation space of all agents)
        
        print("observation_space=", self.env.observation_space, flush=True)
        print("obs_dim_n={}".format(self.args.obs_dim_n), flush=True)
        print("action_space=", self.env.action_space, flush=True)
        print("action_dim_n={}".format(self.args.action_dim_n), flush=True)

        # Create N agents
        self.agent_n = MAPPO(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        self.total_steps = 0
        
        if self.args.use_reward_norm:
            print("------use reward norm------", flush=True)
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------", flush=True)
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, episode_steps = self.run_episode_mpe(evaluate=False)  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward), flush=True)
        
        return evaluate_reward

    def run_episode_mpe(self, evaluate=False):
        total_reward = 0
        obs_n = self.env.reset()  # Reset the environment
        action_masks = [np.ones(self.args.action_dim) for _ in range(self.args.N)]  # Dummy action masks for compatibility
        episode_steps = 0
        
        while True:
            # Get actions for all agents using the policy
            actions = []
            action_logprobs = []
            
            # Collect state for critic
            s = np.concatenate(obs_n)  # This assumes global state is concatenation of observations
            
            # Get values for the current state
            v = self.agent_n.get_value(s)
            
            for agent_id in range(self.args.N):
                obs = obs_n[agent_id]
                mask = action_masks[agent_id]
                if evaluate:
                    action = self.agent_n.select_action(obs, agent_id, evaluate=True, action_mask=mask)
                    actions.append(action)
                else:
                    # When training, we need both actions and their log probabilities
                    action, action_logprob = self.agent_n.choose_action([obs], False, [mask])
                    actions.append(action[0])  # choose_action returns a numpy array
                    action_logprobs.append(action_logprob[0])  # choose_action returns a numpy array

            q = None
            if not evaluate and self.args.use_central_q:
                q = self.agent_n.get_central_q(s, actions)

            # Take a step in the environment
            next_obs_n, reward_n, done_n, info_n = self.env.step(actions)

            # Store transitions in the replay buffer if not evaluating
            if not evaluate:
                self.replay_buffer.store_transition(
                    episode_step=episode_steps,
                    obs_n=obs_n,
                    s=s,
                    v_n=v,
                    a_n=np.array(actions),
                    a_logprob_n=np.array(action_logprobs),
                    r_n=np.array(reward_n),
                    done_n=np.array(done_n),
                    action_mask_n=np.array(action_masks),
                    q_n=q
                )
            
            # Calculate the total reward
            reward = sum(reward_n)
            total_reward += reward
            
            # Update the observations
            obs_n = next_obs_n
            
            episode_steps += 1
            done = all(done_n) or episode_steps >= self.args.episode_limit
            
            # End the episode if done
            if done:
                # If not evaluating, store the last value
                if not evaluate:
                    # Get the value of the last state, or zero if terminal
                    if all(done_n):  # If truly done (not just hitting episode limit)
                        v_last = np.zeros_like(v)
                    else:
                        # Calculate the next state for the critic and get its value
                        next_s = np.concatenate(next_obs_n)
                        v_last = self.agent_n.get_value(next_s)
                    
                    # Store the last value and increment episode counter
                    self.replay_buffer.store_last_value(episode_steps, v_last)
                
                break
        
        return total_reward, episode_steps


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=25, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=int, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=int, default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=False, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False, help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")
    parser.add_argument("--env_id", type=str, required=True, help="The name of the environment to run")
    parser.add_argument("--use_central_q", action="store_true", help="Enable centralized Q-function training")

    # Add output directory argument
    parser.add_argument("--output_dir", type=str, default="./runs", help="Directory to save all output files")

    args = parser.parse_args()
    runner = Runner_MAPPO_MPE(args, env_name=args.env_id, number=1, seed=42)
    runner.run()
