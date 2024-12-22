/deac/csc/alqahtaniGrp/liut18/AdvExRL_Submission/AdvExRL_code/env/navigation2.py:62: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  self.observation_space = Box(-np.ones(2) * np.float('inf'),
/deac/csc/alqahtaniGrp/software/venvs/AdvExRL/lib/python3.9/site-packages/gym/core.py:200: DeprecationWarning: [33mWARN: Function `env.seed(seed)` is marked as deprecated and will be removed in the future. Please use `env.reset(seed=seed)` instead.[0m
  deprecation(
  0%|          | 0/10 [00:00<?, ?it/s]  0%|          | 0/10 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/deac/csc/alqahtaniGrp/liut18/XRL/run.py", line 146, in <module>
    data, num_feats, num_actions, fail_dic, ts_dic, agent = run_nav2('nav2', path)
  File "/deac/csc/alqahtaniGrp/liut18/AdvExRL_Submission/AdvExRL_code/test_nav2.py", line 287, in run
    episode_data, all_state = run_eval_episode(env, expert_agent, use_safety=True, aaa_atk = True, aaa_agent = adv_agent, atk_rate=0.5, shield_threshold=shield_threshold, safety_agent= safety_agent)
  File "/deac/csc/alqahtaniGrp/liut18/AdvExRL_Submission/AdvExRL_code/test_nav2.py", line 136, in run_eval_episode
    shield_val_tsk = safety_agent.get_shield_value(torchify(state), torchify(action_tsk))
  File "/deac/csc/alqahtaniGrp/liut18/AdvExRL_Submission/AdvExRL_code/AdvEx_RL/safety_agent.py", line 91, in get_shield_value
    q1, q2 = self.adv_critic.critic(state, action)
  File "/deac/csc/alqahtaniGrp/software/venvs/AdvExRL/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/deac/csc/alqahtaniGrp/liut18/AdvExRL_Submission/AdvExRL_code/AdvEx_RL/network.py", line 69, in forward
    x1 = F.relu(self.linear1(xu))
  File "/deac/csc/alqahtaniGrp/software/venvs/AdvExRL/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/deac/csc/alqahtaniGrp/software/venvs/AdvExRL/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 103, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument mat1 in method wrapper_addmm)
