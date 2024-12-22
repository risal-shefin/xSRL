from ACORN.test import test
from ACORN.utils import load_graph
import sys
import torch
import numpy as np
import random
from highlights import Highlights
from torch.autograd import Variable
from abstract import Abstraction
from translation import AcornPredicates



def acorn_xrl(model_path, acorn_config):
    
    validation_graphs = load_graph('test')
    assert len(validation_graphs)

    single_graph = random.choice(list(validation_graphs.items()))
    single_graph = {single_graph[0]: single_graph[1]}
    print(single_graph)

    acorn_config['validation_graphs'] = single_graph
    

    highlights_data, agent1_model = test(model_path, **acorn_config)

    
    critical_trajectories = []
    num_episodes = len(highlights_data[0])
    for graph_num in range(len(highlights_data)):
        highlight_maker = Highlights(highlights_data[graph_num], num_episodes=num_episodes)
        critical_trajectories.append(highlight_maker.compute_trajectories())
    
    print(critical_trajectories[0]['critical_times'])
    
    def acorn_value_fn(obs):
        obs = np.reshape(obs, [1, 5])
        obs = Variable(torch.from_numpy(obs))
        seq_len = torch.Tensor([1])
        rnn_state = agent1_model.get_initial_state()
        _, _ = agent1_model.forward(input_dict={'obs':obs, 'obs_flat': obs}, state=rnn_state, seq_lens=seq_len)
        return agent1_model.value_function().detach().numpy()[0]

    num_features = len(np.squeeze(graph_info['trajectories'][0][0]))
    translator = AcornPredicates(num_features, num_bins=10)
    
    for graph_info in critical_trajectories:
        trajectories = graph_info['trajectories']
        abstract_trajectories = []
        for critical_traj in trajectories:
            states = [np.squeeze(t[0]) for t in critical_traj]
            actions = [t[1] for t in critical_traj]
            info = {'states': states, 'actions': actions}
            abstraction_helper = Abstraction(info, 4, 1, acorn_value_fn, translator)
            abstract_t, _ = abstraction_helper.compute_abstractions()
            for i, t in enumerate(abstract_t):
                print("Group {}: {}".format(i+1, t))
            abstract_trajectories.append(abstract_t)



    """
    for i in range(100):
        random_state = np.random.random(size=[1, 5])
        value = acorn_value_fn(random_state)
        print(value)
    """


if __name__ == '__main__':

    pass

