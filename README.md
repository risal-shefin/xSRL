# Setup
Tested Python Version: 3.10.2

Create & activate an environment:
```shell
conda create --name venv-xsrl
conda activate venv-xsrl
```
Now run the following commands to install the dependencies:
```shell
pip install -r requirements.txt
conda install mesa glfw glew patchelf osmesa
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mkdir -p ~/.mujoco
tar -xvzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
```
To run the code, perform the following command:
```shell
python run.py
```
To choose between Nav2 and Maze environment, you need to set the `args.env` variable in the `run.py`. Also, in the `set_nav_maze_args(args)` method of the `run.py`, you can also modify other configurations.