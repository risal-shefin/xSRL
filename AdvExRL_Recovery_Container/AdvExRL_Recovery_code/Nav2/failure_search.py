# Failure search implementation based off of Rigorous Agent Evaluation: An Adversarial
# Approach to Uncover Catastrophic Failures, Uesato et al 2018.

import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random

class Approximator(nn.Module):
    #Approximator network takes in a state instance and predicts the probability of agent failure from this state
    #Treated as binary classification
    def __init__(self, input_shape):
        super(Approximator, self).__init__()
        self.input_shape = input_shape
        self.layers = nn.Sequential(
            nn.Linear(self.input_shape[0], 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

class Approx_Buffer(object):
    #Buffer class for the failure search approximator
    def __init__(self, size=100000):
        self.buffer = deque(maxlen=size)
    
    def push(self, states, agent_infos, result):
        #At the end of each training episode, we push the states and agent_infos for the entire episode into the buffer,
        #along with a single result for the episode (1 for failure, 0 else)
        result_arr = np.array([result])
        result_arr = np.reshape(result_arr, [1, -1])
        for i in range(len(states)):
            state = np.expand_dims(states[i], 0)
            agent_info = np.reshape(agent_infos[i], [1, -1])
            self.buffer.append((state, agent_info, result_arr))

    def sample(self, batch_size):
        states, agent_infos, results = zip(
            *random.sample(self.buffer, batch_size)
        )
        states = np.concatenate(states)
        agent_infos = np.concatenate(agent_infos)
        results = np.concatenate(results)
        states = torch.from_numpy(states).float()
        agent_infos = torch.from_numpy(agent_infos).float()
        results = torch.from_numpy(results).float()
        return states, agent_infos, results

    def __len__(self):
        return len(self.buffer)


class Trainer(object):
    #Trainer class for the AVF
    def __init__(self, network, replay_buffer, lr=0.01, batch_size=32, training_iter=16, verbose=True):
        self.network = network
        self.replay_buffer = replay_buffer
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose
        self.training_iter = training_iter #How many iterations to train on each time we train the approximator
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.loss_fn = nn.BCELoss()
    

    def add_experience(self, states, agent_infos, result):
        self.replay_buffer.push(states, agent_infos, result)
    

    def train_step(self, states, agent_info, results):
        #Training step for a single batch: Take in batch size states, additional info, and result of trajectory (failures or success)
        #All the inputs should be pytorch tensors

        if agent_info.shape == (self.batch_size): #Expand dims if one dimensional
            agent_info = torch.unsqueeze(agent_info, 1)
        if results.shape == (self.batch_size):
            results = torch.unsqueeze(results, 1)
        
        net_input = torch.cat((states, agent_info), 1)
        predictions = self.network(net_input)

        loss = self.loss_fn(predictions, results)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        class_assignments = predictions.reshape(-1).detach().numpy().round()
        class_ground_truths = results.reshape(-1).detach().numpy()
        accuracy = (class_assignments == class_ground_truths).mean()

        return loss, accuracy
    
    def train(self):
        total_loss = []
        total_accuracy = []
        for i in range(self.training_iter):
            states, agent_info, results = self.replay_buffer.sample(self.batch_size)
            loss, accuracy = self.train_step(states, agent_info, results)
            total_loss.append(loss)
            total_accuracy.append(accuracy)

        if self.verbose:
            print("Approximator Loss: {} Accuracy: {}".format(sum(total_loss)/len(total_loss),
                                                              sum(total_accuracy)/len(total_accuracy)))

    def predict(self, x):
        x = torch.from_numpy(x).float()
        with torch.no_grad():
            return self.network(x).reshape(-1).detach().numpy().round().item()
    


