/deac/csc/alqahtaniGrp/liut18/AdvExRL_Submission/AdvExRL_code/env/navigation2.py:62: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  self.observation_space = Box(-np.ones(2) * np.float('inf'),
/deac/csc/alqahtaniGrp/software/venvs/AdvExRL/lib/python3.9/site-packages/gym/core.py:200: DeprecationWarning: [33mWARN: Function `env.seed(seed)` is marked as deprecated and will be removed in the future. Please use `env.reset(seed=seed)` instead.[0m
  deprecation(
  0%|          | 0/10 [00:00<?, ?it/s] 10%|█         | 1/10 [00:00<00:01,  4.51it/s] 20%|██        | 2/10 [00:00<00:01,  4.18it/s] 30%|███       | 3/10 [00:00<00:01,  4.18it/s] 40%|████      | 4/10 [00:00<00:01,  4.22it/s] 50%|█████     | 5/10 [00:01<00:01,  4.23it/s] 60%|██████    | 6/10 [00:01<00:00,  4.32it/s] 70%|███████   | 7/10 [00:01<00:00,  4.25it/s] 80%|████████  | 8/10 [00:01<00:00,  4.45it/s] 90%|█████████ | 9/10 [00:02<00:00,  4.45it/s]100%|██████████| 10/10 [00:02<00:00,  4.48it/s]100%|██████████| 10/10 [00:02<00:00,  4.36it/s]
/deac/csc/alqahtaniGrp/liut18/AdvExRL_Submission/AdvExRL_code/Nav2/risk_estimation.py:4: DeprecationWarning: The symbol module is deprecated and will be removed in future versions of Python
  from symbol import try_stmt
Traceback (most recent call last):
  File "/deac/csc/alqahtaniGrp/liut18/XRL/run.py", line 146, in <module>
    data, num_feats, num_actions, fail_dic, ts_dic, agent = run_nav2('nav2', path)
  File "/deac/csc/alqahtaniGrp/liut18/AdvExRL_Submission/AdvExRL_code/test_nav2.py", line 302, in run
    fail_dic, ts_dic = estimate_agent_capability(env, expert_agent, adv_agent, safety_agent, F_trainer, model, episode_all_state)
  File "/deac/csc/alqahtaniGrp/liut18/AdvExRL_Submission/AdvExRL_code/Nav2/train_risk_estimation.py", line 275, in estimate_agent_capability
    failure_probabilities, fail_dic, ts_dic, fail_dic_user, ts_dic_user = risk_estimator.state_failure_probs(model, info_dic, n = 10)
  File "/deac/csc/alqahtaniGrp/liut18/AdvExRL_Submission/AdvExRL_code/Nav2/risk_estimation.py", line 77, in state_failure_probs
    failed, ts = self.rollout.evaluate(x, model, info_dic, half)
  File "/deac/csc/alqahtaniGrp/liut18/AdvExRL_Submission/AdvExRL_code/Nav2/risk_estimation.py", line 188, in evaluate
    shield_val_tsk = self.safety_agent.get_shield_value(self.torchify(state), self.torchify(action))
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
