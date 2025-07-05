## xSRL: Safety-Aware Explainable RL - Safety as a Product of Explainability
This is the code repository of the paper "xSRL: Safety-Aware Explainable RL - Safety as a Product of Explainability".<br>
Authors: Risal Shahriar Shefin, Md Asifur Rahman, Thai Le, Sarra Alqahtani<br>
Paper Link: https://arxiv.org/abs/2412.19311

## Setup
Tested Python Version: 3.10

Create & activate a virtual environment:
```shell
conda create --name venv-xsrl python=3.10
conda activate venv-xsrl
```
Clone this repository:
```shell
git clone https://github.com/risal-shefin/xSRL.git
cd xSRL
```
Now run the following commands to install the dependencies:
```shell
pip install -r requirements.txt
conda install -c conda-forge mesa glfw glew patchelf
conda install -c menpo osmesa
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mkdir -p ~/.mujoco
tar -xvzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
```
For Mujoco, we need to set the following environment variables (`./pkgs/usr/lib64` contains the libgcrypt library files):
```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia:$(pwd)/pkgs/usr/lib64
```
To run the code, perform the following command:
```shell
python run.py --env nav2    # maze is also tested
```
There are several command-line arguments you can use:
- `--num_episodes <int>` – Number of episodes to run.
- `--attack_rate <float>` – Probability of adversarial attack (range: `[0,1]`).
- `--use_safety <bool>` – Enables safety mechanisms. Default is AdvExRL safety agent if `--recovery_rl` is not set.
- `--recovery_rl <str>` – Specifies the recovery RL agent (`RRL_MF`, `SQRL` are tested. You can find other algo names [here](https://github.com/risal-shefin/xSRL/blob/6ddb6d044ae3f6ef5271288aaf4c8243c16dfbd4/AdvExRL_Recovery_Container/AdvExRL_Recovery_code/RecoveryRL/recRL_comparison_exp_aaa_atk.py#L372).). `--use_safety` must be **True** to use Recovery RL.
- `--numpy_seed <int>` – Sets the random seed for NumPy.
- `--max_height <int>` - Maximum height of the CAPS graph.
- `--calc_fidelity <bool>` – Enables fidelity calculations for evaluating specific components.
- `--ctf_action_method <enum>` – Defines the agent's counterfactual action selection strategy. One use case is [here](https://github.com/risal-shefin/xSRL/blob/6ddb6d044ae3f6ef5271288aaf4c8243c16dfbd4/AdvExRL_Recovery_Container/AdvExRL_Recovery_code/RecoveryRL/recRL_comparison_exp_aaa_atk.py#L84). (Options: 'RiskyAlways' or 'RiskyOnce'). May need more work.
- `--user_test <bool>` – Experimental. Used to observe data transitions from one scenario to another scenario. One use case is [here](https://github.com/risal-shefin/xSRL/blob/6ddb6d044ae3f6ef5271288aaf4c8243c16dfbd4/AdvExRL_Recovery_Container/AdvExRL_Recovery_code/test_nav_maze.py#L462). May need more work.

Example run command with some arguments:
```shell
python run.py --env nav2 --num_episodes 10 --attack_rate 0.5 --use_safety True --recovery_rl RRL_MF --max_height 3
```

## Configuration Instructions

- The AdvExRL trained models are placed inside the directory: `/xSRL/AdvExRL_Recovery_Container/AdvExRL_Recovery_code/AdvEx_RL_Trained_Models`. The corresponding configs are placed inside the directory: `/xSRL/AdvExRL_Recovery_Container/AdvExRL_Recovery_code/AdvEx_RL_config`.<br>
- The RecoveryRL trained models are placed inside the directory: `/xSRL/AdvExRL_Recovery_Container/AdvExRL_Recovery_code/RecoveryRL/RecoveryRL_Model` and the corresponding model directory with the best reward is automatically picked through the function `get_model_directories()` of `/xSRL/AdvExRL_Recovery_Container/AdvExRL_Recovery_code/RecoveryRL/recRL_comparison_exp_aaa_atk.py`.

## Acknowledgements
- CAPS: https://github.com/mccajl/CAPS
- AdvEx-RL: https://github.com/asifurrahman1/AdvEx-RL
- RecoveryRL: https://github.com/abalakrishna123/recovery-rl