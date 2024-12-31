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
For Mujoco, we need to set the following environment variables:
```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
```
To run the code, perform the following command:
```shell
python run.py
```

## Configuration Instructions

- To choose between Nav2 and Maze environment, you need to set the `args.env` variable in the `run.py`. Also, in the `set_nav_maze_args(args)` method of the `run.py`, you can also modify other configurations.<be>
- The AdvExRL trained models are placed inside the directory: `/xSRL/AdvExRL_Recovery_Container/AdvExRL_Recovery_code/AdvEx_RL_Trained_Models`. The corresponding configs are placed inside the directory: `/xSRL/AdvExRL_Recovery_Container/AdvExRL_Recovery_code/AdvEx_RL_config`.<br>
- The RecoveryRL trained models are placed inside the directory: `/xSRL/AdvExRL_Recovery_Container/AdvExRL_Recovery_code/RecoveryRL/RecoveryRL_Model` and the corresponding model directory with the best reward is automatically picked through the function `get_model_directories()` of `/xSRL/AdvExRL_Recovery_Container/AdvExRL_Recovery_code/RecoveryRL/recRL_comparison_exp_aaa_atk.py`.

## Acknowledgements
- CAPS: https://github.com/mccajl/CAPS
- AdvEx-RL: https://github.com/asifurrahman1/AdvEx-RL
- RecoveryRL: https://github.com/abalakrishna123/recovery-rl