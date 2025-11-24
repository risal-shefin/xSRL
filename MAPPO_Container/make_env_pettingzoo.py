import numpy as np
from gymnasium.spaces import Box, Discrete
import torch
import pettingzoo.mpe as mpe
import pettingzoo.sisl as sisl
import pettingzoo.atari as atari
import supersuit

def preprocess_env_atari(env):
    # as per openai baseline's MaxAndSKip wrapper, maxes over the last 2 frames
    # to deal with frame flickering
    env = supersuit.max_observation_v0(env, 2)
    # skip frames for faster processing and less control
    # to be compatible with gym, use frame_skip(env, (2,5))
    env = supersuit.frame_skip_v0(env, 4)
    # downscale observation for faster processing
    env = supersuit.resize_v1(env, 84, 84)
    # allow agent to see everything on the screen despite Atari's flickering screen problem
    env = supersuit.frame_stack_v1(env, 4)
    return env

class PettingZooWrapper:
    """
    Wrapper for PettingZoo environments to match the interface expected by MAPPO
    """
    def __init__(self, env_name="simple_spread_v3", continuous=False):
        try:
            env_func = getattr(mpe, env_name)
            if env_name == "simple_spread_v3":
                self.env = env_func.parallel_env(continuous_actions=continuous, render_mode='rgb_array', N=3)
            else:
                self.env = env_func.parallel_env(continuous_actions=continuous, render_mode='rgb_array')
        except:
            try:
                env_func = getattr(sisl, env_name)
                self.env = env_func.parallel_env(n_pursuers=5, render_mode='rgb_array') if env_name == 'waterworld_v4' else env_func.parallel_env(render_mode='rgb_array')
            except:
                env_func = getattr(atari, env_name)
                self.env = env_func.parallel_env(render_mode='rgb_array')
                self.env = preprocess_env_atari(self.env)

        obs, _ = self.env.reset(seed=42)  # Initialize with a seed for reproducibility
        self.n = len(self.env.agents)  # Number of agents
        self.agent_ids = list(self.env.agents)
        
        # Set up observation and action spaces
        self.observation_space = []
        self.action_space = []
        
        for agent in self.agent_ids:
            self.observation_space.append(self.env.observation_space(agent))
            self.action_space.append(self.env.action_space(agent))
    
    def reset(self, seed=None):
        observations, _ = self.env.reset(seed=seed) if seed is not None else self.env.reset()
        # Convert the dict to a list in the same order as agent_ids
        obs_list = [observations[agent] for agent in self.agent_ids]
        return obs_list
    
    def step(self, actions):
        # Convert list of actions to dict
        action_dict = {agent_id: action for agent_id, action in zip(self.agent_ids, actions)}
        
        # Execute actions
        observations, rewards, terminations, truncations, infos = self.env.step(action_dict)
        
        # Convert from dicts to lists
        obs_list = [observations[agent] for agent in self.agent_ids]
        reward_list = [rewards[agent] for agent in self.agent_ids]
        done_list = [terminations[agent] or truncations[agent] for agent in self.agent_ids]
        info_list = [infos[agent] if agent in infos else {} for agent in self.agent_ids]
        
        # Check if episode is done
        done = all(done_list)
        
        return obs_list, reward_list, done_list, info_list
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()


def make_env(env_name, discrete=True):
    """
    Create a wrapped PettingZoo environment.
    
    Args:
        env_name: The name of the environment
        discrete: Whether to use discrete action space
    
    Returns:
        A wrapped environment
    """
    # Convert discrete=True to continuous=False (and vice versa)
    continuous = not discrete
    return PettingZooWrapper(env_name, continuous=continuous)