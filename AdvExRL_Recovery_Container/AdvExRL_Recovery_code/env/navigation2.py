"""
A robot that can exert force in cardinal directions. The robot's goal is to
reach the origin and it experiences zero-mean Gaussian Noise. State
representation is (x, y). Action representation is (dx, dy).
"""

import os
import pickle

import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from gym import Env
from gym import utils
from gym.spaces import Box

from env.obstacle import Obstacle, ComplexObstacle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import io
import cv2
"""
Constants associated with the Navigation2 env.
"""

START_POS = [-50, 0]
END_POS = [0, 0]
GOAL_THRESH = 1.
START_STATE = START_POS
GOAL_STATE = END_POS

MAX_FORCE = 1
HORIZON = 100

NOISE_SCALE = 0.05          #Default 0.05
AIR_RESIST = 0.02          #Default 0.02   

HARD_MODE = False

OBSTACLE = [[[-30, -20], [-7.5, 7.5]]]

CAUTION_ZONE = [[[-32, -18], [-12, 12]]]

OBSTACLE = ComplexObstacle(OBSTACLE)
CAUTION_ZONE = ComplexObstacle(CAUTION_ZONE)


def process_action(a):
    return np.clip(a, -MAX_FORCE, MAX_FORCE)


class Navigation2(Env, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None
        self.A = np.eye(2)
        self.B = np.eye(2)
        self.horizon = HORIZON
        self.action_space = Box(-np.ones(2) * MAX_FORCE,
                                np.ones(2) * MAX_FORCE)
        self.observation_space = Box(-np.ones(2) * float('inf'),
                                     np.ones(2) * float('inf'))
        self._max_episode_steps = HORIZON
        self.obstacle = OBSTACLE
        self.caution_zone = CAUTION_ZONE
        # self.transition_function = get_offline_data
        self.goal = GOAL_STATE

        self.got_adv_reward = False
        self.state = None
        self.cost = []
        self.hist = []
        self.time = 0
    
    def set_curr_state(self, state):
        self.state = state
        self.hist = [self.state]
    
    def step(self, a):
        a = process_action(a)
        old_state = self.state.copy()
        next_state = self._next_state(self.state, a)
        cur_cost = self.step_cost(self.state, a)
        self.cost.append(cur_cost)
        self.state = next_state
        self.time += 1
        self.hist.append(self.state)
        self.done = cur_cost > -4 or self.obstacle(next_state)

        if self.obstacle(next_state) :
            adv_reward = 1
        else:
            adv_reward = -0.1
      
        return self.state, cur_cost, self.done, {
            "constraint": self.obstacle(next_state),
            "reward": cur_cost,
            "adv_reward": adv_reward,
            "state": old_state,
            "next_state": next_state,
            "action": a,
            "success": cur_cost>-4
        }

    def reset(self):
        self.state = START_STATE + np.random.randn(2)
        #print(f'in nav2.py, rest function, here reset state is {self.state}')
        self.time = 0
        self.cost = []
        self.done = False
        self.hist = [self.state]
        self.got_adv_reward = False
        return self.state

    def reset_state(self, state):
        self.state = state
        #print(f'in nav2.py, rest function, here reset state is {self.state}')
        self.time = 0
        self.cost = []
        self.done = False
        self.hist = [self.state]
        self.got_adv_reward = False
        return self.state

    def _next_state(self, s, a, override=False):
        if self.obstacle(s):
            # print("obs", s, a)
            return s
        return self.A.dot(s) + self.B.dot(a) + NOISE_SCALE * np.random.randn(
            len(s))

    def step_cost(self, s, a):
        if HARD_MODE:
            return int(
                np.linalg.norm(np.subtract(GOAL_STATE, s)) < GOAL_THRESH)
        return -np.linalg.norm(np.subtract(GOAL_STATE,
                                           s)) - self.obstacle(s) * 0.

    def values(self):
        return np.cumsum(np.array(self.cost)[::-1])[::-1]

    def sample(self):
        """
        samples a random action from the action space.
        """
        return np.random.random(2) * 2 * MAX_FORCE - MAX_FORCE

    def plot_trajectory(self, states=None):
        if states == None:
            states = self.hist
        states = np.array(states)
        plt.scatter(states[:, 0], states[:, 2])
        plt.show()

    # Returns whether a state is stable or not
    def is_stable(self, s):
        return np.linalg.norm(np.subtract(GOAL_STATE, s)) <= GOAL_THRESH


def get_offline_data(num_transitions, task_demos=False, save_rollouts=False):
    env = Navigation2()
    transitions = []
    rollouts = []
    done = False
    for i in range(num_transitions // 10 // 3):
        rollouts.append([])
        state = np.array(
            [np.random.uniform(-40, 10),
             np.random.uniform(-25, 25)])
        while env.obstacle(state):
            state = np.array(
                [np.random.uniform(-40, 10),
                 np.random.uniform(-25, 25)])
        for j in range(10):
            action = np.clip(np.random.randn(2), -1, 1)
            next_state = env._next_state(state, action, override=True)
            constraint = env.obstacle(next_state)
            reward = env.step_cost(state, action)
            transitions.append(
                (state, action, constraint, next_state, not constraint))
            rollouts[-1].append(
                (state, action, constraint, next_state, not constraint))
            state = next_state
            if constraint:
                break

    for i in range(num_transitions // 10 * 1 // 4):
        rollouts.append([])
        state = np.array(
            [np.random.uniform(-35, -30),
             np.random.uniform(-12, 12)])
        for j in range(10):
            action = np.clip(
                np.array([np.random.uniform(0.5, 1, 1),
                          np.random.randn(1)]), -1, 1).ravel()
            next_state = env._next_state(state, action, override=True)
            constraint = env.obstacle(next_state)
            reward = env.step_cost(state, action)
            transitions.append(
                (state, action, constraint, next_state, not constraint))
            rollouts[-1].append(
                (state, action, constraint, next_state, not constraint))
            state = next_state
            if constraint:
                break

    for i in range(num_transitions // 10 * 1 // 4):
        rollouts.append([])
        state = np.array(
            [np.random.uniform(-20, -15),
             np.random.uniform(-12, 12)])
        for j in range(10):
            action = np.clip(
                np.array([np.random.uniform(-1, -0.5, 1),
                          np.random.randn(1)]), -1, 1).ravel()
            next_state = env._next_state(state, action, override=True)
            constraint = env.obstacle(next_state)
            reward = env.step_cost(state, action)
            transitions.append(
                (state, action, constraint, next_state, not constraint))
            rollouts[-1].append(
                (state, action, constraint, next_state, not constraint))
            state = next_state
            if constraint:
                break

    for i in range(num_transitions // 10 * 1 // 4):
        rollouts.append([])
        state = np.array(
            [np.random.uniform(-30, -20),
             np.random.uniform(10, 15)])
        for j in range(10):
            action = np.clip(
                np.array([np.random.randn(1),
                          np.random.uniform(-1, -0.5, 1)]), -1, 1).ravel()
            next_state = env._next_state(state, action, override=True)
            constraint = env.obstacle(next_state)
            reward = env.step_cost(state, action)
            transitions.append(
                (state, action, constraint, next_state, not constraint))
            rollouts[-1].append(
                (state, action, constraint, next_state, not constraint))
            state = next_state
            if constraint:
                break

    for i in range(num_transitions // 10 * 1 // 4):
        rollouts.append([])
        state = np.array(
            [np.random.uniform(-30, -20),
             np.random.uniform(-15, -10)])
        for j in range(10):
            action = np.clip(
                np.array([np.random.randn(1),
                          np.random.uniform(0.5, 1, 1)]), -1, 1).ravel()
            next_state = env._next_state(state, action, override=True)
            constraint = env.obstacle(next_state)
            reward = env.step_cost(state, action)
            transitions.append(
                (state, action, constraint, next_state, not constraint))
            rollouts[-1].append(
                (state, action, constraint, next_state, not constraint))
            state = next_state
            if constraint:
                break

    if save_rollouts:
        return rollouts
    else:
        return transitions


x=[]
y=[]
def reset_trajectory():
    global x
    global y 
    x = []
    y = []

def render(loc, end = False):
    global x
    global y  
    def get_img_from_fig(ax, x, y, fig, dpi=180):
        #***********************************
        u = np.diff(x)
        v = np.diff(y)
        pos_x = x[:-1] + u/2
        pos_y = y[:-1] + v/2
        norm = np.sqrt(u**2+v**2) 
        # if len(x)>4:
        p = plt.plot(x[:-1],y[:-1],'--g')
        q =  ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid", minshaft=0.9)
        #***********************************
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    #*************************************************
    if end:
        x=[]
        y=[]
        return
    else:
        x.append(loc[0])
        y.append(loc[1])
    #*************************************************

    data_set = np.array([[.9, .9], [.85, 2.1], [1.2, 1.], [2.1,
                                                           .95], [3., 1.1],
                         [3.9, .7], [4., 1.4], [4.2, 1.8], [2.,
                                                            2.3], [3., 2.3],
                         [1.5, 1.8], [2., 1.5], [2.2, 2.], [2.6, 1.7],
                         [2.7, 1.85]])
    categories = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    color1 = (0.69411766529083252, 0.3490196168422699, 0.15686275064945221,
              1.0)
    color2 = (0.65098041296005249, 0.80784314870834351, 0.89019608497619629,
              1.0)
    colormap = np.array([color1, color2])
    fig = plt.figure()
    ax = fig.add_subplot(111)

    margin = .1
    min_f0, max_f0 = -30, -20
    min_f1, max_f1 = -7.5, 7.5
    width = max_f0 - min_f0
    height = max_f1 - min_f1

    ax.add_patch(
        patches.Rectangle(
            xy=(min_f0, min_f1),  # point of origin.
            width=width,
            height=height,
            linewidth=1,
            color='red',
            fill=True))
    
    circle = plt.Circle(loc, radius=1, color='green')
    ax.add_patch(circle)
    circle = plt.Circle((-50, 0), radius=1)
    ax.add_patch(circle)
    circle = plt.Circle((0, 0), radius=1)
    ax.add_patch(circle)
    label = ax.annotate("start", xy=(-50, 3), fontsize=10, ha="center")
    label = ax.annotate("goal", xy=(0, 3), fontsize=10, ha="center")

    plt.xlim(-60, 10)
    plt.ylim(-30, 30)

    ax.set_aspect('equal')
    ax.autoscale_view()
    # plt.savefig("pointbot0_cartoon.png")
    
    return get_img_from_fig(ax, x, y, fig)
