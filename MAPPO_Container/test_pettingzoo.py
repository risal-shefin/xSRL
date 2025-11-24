from collections import deque
from .make_env_pettingzoo import make_env
from datetime import datetime
import torch
import torch.nn as nn
import argparse
import gymnasium
import numpy as np
import os
import sys
import random
from torch.distributions import Categorical
import math
from tqdm import tqdm

from .MAPPO_MPE_main import Runner_MAPPO_MPE
from common.enums import PolicyEnum
from common.DictionarySummaryModel import DictionarySummaryModel, SummaryMethodEnum


def round_state(s):
    """Round state values to 5 decimal places for consistent hashing."""
    rounded_s = []
    for ele in s:
        ele = round(ele, 5)
        rounded_s.append(ele)
    return rounded_s


def run_eval_episode(env, runner: Runner_MAPPO_MPE, agent_id: int):
    """
    Run one evaluation episode and collect data for a specific agent.
    
    Args:
        env: PettingZoo environment
        runner: MAPPO runner with trained agent
        agent_id: ID of the agent to track for explanation (0-indexed)
    
    Returns:
        episode_data: Dictionary with states, actions, entropy, etc.
        all_state: Set of all unique rounded states visited
    """
    done = [False for _ in range(runner.args.N)]
    obs_n = env.reset(seed=runner.seed)
    episode_reward = 0
    epi_step_count = 0
    
    # Episode data structure matching test_dosing.py
    episode_data = {
        'states': [],           # Individual agent's state (rounded tuple)
        'actions': [],          # Individual agent's action
        'entropy': [],          # Entropy of agent's action distribution
        'dones': [],            # Done flags
        'rewards': [],          # Sum of all agent rewards
        'env_actions': [],      # Same as actions for discrete case
        'policies': [],         # PolicyEnum.TaskPolicy
        'task_critic_vals': [], # Critic value from centralized critic
        'safety_critc_vals': [] # Set to 0 (no safety agent for MARL)
    }
    
    all_state = set()
    
    while not all(done):
        epi_step_count += 1
        
        # Get actions for all agents
        actions = []
        entropies = []
        
        # Collect global state for critic
        s = np.concatenate(obs_n)
        
        # Get value from centralized critic
        v = runner.agent_n.get_value(s)
        task_critic_val = v[agent_id]  # Get value for the specific agent
        
        for id in range(runner.args.N):
            action, action_logprob = runner.agent_n.choose_action([obs_n[id]], evaluate=False, action_masks=[None])
            action = action[0]
            
            # Calculate entropy from the policy distribution
            with torch.no_grad():
                obs_tensor = torch.tensor(obs_n[id], dtype=torch.float32).unsqueeze(0)
                prob = runner.agent_n.actor(obs_tensor)
                dist = Categorical(probs=prob)
                entropy = dist.entropy().item()
            
            actions.append(action)
            entropies.append(entropy)
        
        # Store data for the specific agent we're tracking
        agent_state = obs_n[agent_id]
        rounded_state = tuple(round_state(agent_state))
        
        episode_data['states'].append(rounded_state)
        episode_data['actions'].append(actions[agent_id])
        episode_data['entropy'].append(entropies[agent_id])
        episode_data['env_actions'].append(actions[agent_id])
        episode_data['policies'].append(PolicyEnum.TaskPolicy)
        episode_data['task_critic_vals'].append(task_critic_val)
        episode_data['safety_critc_vals'].append(0.0)  # No safety agent for MARL
        
        all_state.add(rounded_state)
        
        # Step environment
        next_obs_n, reward_n, done, info_n = env.step(actions)
        
        # Calculate total reward
        total_reward = sum(reward_n)
        episode_reward += total_reward
        
        episode_data['dones'].append(int(all(done)))
        episode_data['rewards'].append(total_reward)
        
        obs_n = next_obs_n
        
        # Check episode termination
        if all(done) or epi_step_count >= runner.args.episode_limit:
            break
    
    print(f"Episode finished. Total Reward: {episode_reward:.2f}, Steps: {epi_step_count}")
    return episode_data, all_state


def run(env_name, args):
    """
    Main function to collect episode data from PettingZoo MPE environment.
    Matches the signature and return structure of test_dosing.py run() function.
    
    Args:
        env_name: Name of the PettingZoo environment (e.g., 'simple_spread_v3')
        args: Argument namespace with configuration
    
    Returns:
        Tuple of 13 values:
        - highlights_data: List of episode dictionaries
        - num_feats: Number of state features for the tracked agent
        - num_actions: Number of discrete actions available
        - runner: Runner_MAPPO_MPE object with trained agent
        - fail_dic: Empty dict (failure analysis skipped)
        - ts_dic: Empty dict (failure analysis skipped)
        - reward_dic: Empty dict (failure analysis skipped)
        - fail_dic_user: Empty dict (failure analysis skipped)
        - ts_dic_user: Empty dict (failure analysis skipped)
        - reward_dic_user: Empty dict (failure analysis skipped)
        - extra_dicts_list: List of DictionarySummaryModel objects
        - env: PettingZoo environment
        - safety_agent: None (no safety agent for MARL)
    """
    eval_epi_no = args.num_episodes if hasattr(args, 'num_episodes') else 3
    numpy_seed = args.numpy_seed if hasattr(args, 'numpy_seed') else 0
    agent_id = args.agent_id if hasattr(args, 'agent_id') else 0
    
    # Set random seeds
    np.random.seed(numpy_seed)
    torch.manual_seed(numpy_seed)
    random.seed(numpy_seed)
    
    # Create MAPPO-specific args namespace with required hyperparameters
    # These are needed by Runner_MAPPO_MPE but not provided by config.py
    mappo_args = argparse.Namespace(
        # Transfer args from config.py
        num_episodes=eval_epi_no,
        numpy_seed=numpy_seed,
        agent_id=agent_id,
        model_dir=args.model_dir if hasattr(args, 'model_dir') else '',
        
        # MAPPO-specific hyperparameters (using defaults from MAPPO_MPE_main.py)
        max_train_steps=int(3e6),
        episode_limit=25,
        evaluate_freq=5000,
        evaluate_times=3,
        batch_size=32,
        mini_batch_size=8,
        rnn_hidden_dim=64,
        mlp_hidden_dim=64,
        lr=5e-4,
        gamma=0.99,
        lamda=0.95,
        epsilon=0.2,
        K_epochs=15,
        use_adv_norm=True,
        use_reward_norm=True,
        use_reward_scaling=False,
        entropy_coef=0.01,
        use_lr_decay=True,
        use_grad_clip=True,
        use_orthogonal_init=True,
        set_adam_eps=True,
        use_relu=False,
        use_rnn=False,
        add_agent_id=False,
        use_value_clip=False,
        use_central_q=getattr(args, 'use_central_q', False)
    )
    
    # Create environment and runner
    env = make_env(env_name=env_name, discrete=True)
    runner = Runner_MAPPO_MPE(mappo_args, env_name=env_name, number=1, seed=numpy_seed)
    
    # Load trained model
    if mappo_args.model_dir:
        print(f"Loading model from: {mappo_args.model_dir}")
        runner.agent_n.load_model_from_directory(mappo_args.model_dir)
    else:
        raise ValueError("model_dir argument is required to load trained MAPPO agent")
    
    # Get environment dimensions
    num_feats = env.observation_space[agent_id].shape[0]
    num_actions = env.action_space[agent_id].n
    
    print(f"Environment: {env_name}")
    print(f"Tracking Agent ID: {agent_id}")
    print(f"Number of features: {num_feats}")
    print(f"Number of actions: {num_actions}")
    print(f"Running {eval_epi_no} episodes...")
    
    # Collect episode data
    highlights_data = []
    episode_all_state = set()
    
    for i in tqdm(range(eval_epi_no)):
        print(f'\n=== Episode {i+1}/{eval_epi_no} ===')
        episode_data, all_state = run_eval_episode(env, runner, agent_id)
        highlights_data.append(episode_data)
        episode_all_state = episode_all_state.union(all_state)
    
    print(f"\nTotal unique states collected: {len(episode_all_state)}")
    
    # Create dictionaries for critic values (similar to test_dosing.py)
    agent_critic_dict = {}
    safety_critic_dict = {}
    
    for episode_data in highlights_data:
        for i, state in enumerate(episode_data['states']):
            task_critic_val = episode_data['task_critic_vals'][i]
            safety_critic_val = episode_data['safety_critc_vals'][i]
            
            # Store the values (overwrite if state appears multiple times)
            agent_critic_dict[state] = task_critic_val
            safety_critic_dict[state] = safety_critic_val
    
    # Create extra_dicts_list with DictionarySummaryModel objects
    extra_dicts_list = [
        DictionarySummaryModel("Safety Critic Average (by AdvExRL Model)", 
                              safety_critic_dict, SummaryMethodEnum.Average),
        DictionarySummaryModel("Safety Critic Max (by AdvExRL Model)", 
                              safety_critic_dict, SummaryMethodEnum.Max),
        DictionarySummaryModel("Agent Task Critic Average", 
                              agent_critic_dict, SummaryMethodEnum.Average)
    ]
    
    # Return 13 values matching test_dosing.py structure
    fail_dic = {}
    ts_dic = {}
    reward_dic = {}
    fail_dic_user = {}
    ts_dic_user = {}
    reward_dic_user = {}
    safety_agent = None
    
    return (highlights_data, num_feats, num_actions, runner, 
            fail_dic, ts_dic, reward_dic, 
            fail_dic_user, ts_dic_user, reward_dic_user,
            extra_dicts_list, env, safety_agent)


if __name__ == '__main__':    
    parser = argparse.ArgumentParser("Test MAPPO in MPE environment for xSRL explanation")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=25, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

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
    
    # Add output directory argument
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save all output files")
    parser.add_argument("--env_id", type=str, required=True, help="Environment ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to load the trained model")
    parser.add_argument("--num_episodes", type=int, default=3, help="Number of episodes to collect")
    parser.add_argument("--numpy_seed", type=int, default=0, help="Numpy random seed")
    parser.add_argument("--agent_id", type=int, default=0, help="Agent ID to track for explanation generation (0-indexed)")
    parser.add_argument("--use_central_q", action="store_true", help="Enable centralized Q-function")

    args = parser.parse_args()
    
    # Run data collection
    exp_data = run(args.env_id, args)
    
    print("\n=== Data Collection Complete ===")
    print(f"Episodes collected: {len(exp_data[0])}")
    print(f"Features: {exp_data[1]}, Actions: {exp_data[2]}")
    # runner = Runner_MAPPO_MPE(args, env_name="simple_spread_v3", number=1, seed=0)
    # runner.run()
