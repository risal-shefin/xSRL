from Gridworld.test_gridworld import test
from highlights import Highlights
import torch
from torch.autograd import Variable
import numpy as np
from abstract import Abstraction
from translation import GridworldPredicates
from data import Data
from data import InstanceData
from CLTree import CLTree
from ray.tune.registry import register_env
from cliffwalking import CliffWalkingEnv
from Gridworld.test_gridworld import calculate_fidelity
from Gridworld.test_gridworld import run_abstract_episode
import matplotlib.pyplot as plt
from explain_utils import graph_scores
from explain_utils import cluster_data


def gridworld_xrl(model_path, num_episodes=10, k=3):
    highlights_data, model = test(model_path, num_episodes)
    num_actions = 4

    highlight_maker = Highlights(highlights_data, num_episodes=num_episodes)
    #critical_trajectories = highlight_maker.compute_trajectories()


    def value_fn(obs):
        int_state = obs[0]
        obs = np.zeros(48)
        obs[int_state] = 1
        obs = np.reshape(obs, [1, -1])
        obs = Variable(torch.from_numpy(obs))
        _, _ = model.forward(input_dict={'obs': obs, 'obs_flat': obs}, state=model.get_initial_state(), seq_lens=torch.Tensor([1]))
        value = model.value_function().detach().numpy()[0]
        return value

    dataset = Data(highlights_data, value_fn)
    
    #trajectories = critical_trajectories['trajectories']
    abstract_trajectories = []
    translator = GridworldPredicates(num_feats=1)

    """
    info = {'states': states, 'actions': actions, 'next_states': next_states, 'dones': dones, 'entropies': entropies}
    abstraction_helper = Abstraction(info, 2, 1, value_fn, translator)
    abstract_t, abstract_t_binary = abstraction_helper.compute_abstractions()
    for i, t in enumerate(abstract_t):
        print("Group {}: {}".format(i+1, t))
    """

    abstraction_helper = Abstraction(None, num_actions, 1, value_fn, translator)

    attr_names = ['Position', 'State Value', 'Action']

    test_fidelity = False
    if test_fidelity:
        fidelity_fn = calculate_fidelity
    else:
        fidelity_fn = None
    
    lmbda = 1 #Should be the same as in the RL environment during training 
    alpha = 0.075
    all_clusters, best_heights, cluster_scores, value_scores, entropy_scores, fidelity_scores, lengths = cluster_data(translator,
                                                                                                                      abstraction_helper,
                                                                                                                      dataset,
                                                                                                                      attr_names,
                                                                                                                      alpha,
                                                                                                                      num_actions=num_actions,
                                                                                                                      lmbda=lmbda,
                                                                                                                      max_height=10,
                                                                                                                      k=k,
                                                                                                                      fidelity_fn=fidelity_fn,
                                                                                                                      model_path=model_path,
                                                                                                                      env='grid'
                                                                                                                    )
                                                                                                                



                                                                                                                

    """
    graph_scores('grid', alpha, lengths, 
                 cluster_scores=cluster_scores, 
                 value_scores=value_scores, 
                 entropy_scores=entropy_scores,
                 fidelity_scores=fidelity_scores)
    """

    print('Best Heights: ', best_heights)
    all_clusters = np.array(all_clusters)
    best_clusters = all_clusters[best_heights]

    test_cluster = best_clusters[0]
    fidelity = calculate_fidelity(model_path, test_cluster, dataset)

    return fidelity
    
    """
    for h, clusters in enumerate(best_clusters):

        print('Clusters at height {}'.format(best_heights[h]+1))

        c = 0
        cluster_state_indices = []
        for i, node in enumerate(clusters):
            
            c += node.getNrInstancesInNode()
            
            cluster_state_indices.append(node.getInstanceIds())

    
    


        

        #print(cluster_state_indices)
        
        abstract_state_groups = []
        abstract_binary_state_groups = []
        for cluster in cluster_state_indices:
            abs_t = []
            bin_t = []
            for idx in cluster:
                idx = int(idx)
                abs_t.append((dataset.states[idx], dataset.actions[idx], dataset.next_states[idx], dataset.dones[idx], dataset.entropies[idx]))
                binary = translator.state_to_binary(dataset.states[idx])
                bin_t.append((binary, dataset.actions[idx]))
            abstract_state_groups.append(abs_t)
            abstract_binary_state_groups.append(bin_t)

        

        
        #abstract_trajectories.append((abstract_state_groups, abstract_binary_state_groups))

        abs_t = abstract_state_groups
        bin_t = abstract_binary_state_groups

        use_qm_minimization = True

        #abstract_t is a list of length (# groups). Each group is a list of 1-n (state, action, next_state, done) tuples
        #abstract_t_binary is a list of length (# groups). Each group is a list of 1-n (binary_predicates, action) tuples
        
        critical_values, group_ent = abstraction_helper.get_critical_groups(abs_t)

        l, transitions, taken_actions = abstraction_helper.compute_graph_info(abs_t)
        #print(taken_actions)
        for j in range(len(transitions)):
            nonzero_idx = np.where(np.array(transitions[j])!=0)[0]
            for idx in nonzero_idx:
                if idx == len(transitions[j]) - 1:
                   print('Group {} to Terminal with p={} and action {}'.format(j+1, transitions[j][idx], int(np.mean(taken_actions[j][idx])))) 
                else:
                    print('Group {} to Group {} with p={} and action {}'.format(j+1, idx+1, transitions[j][idx], int(np.mean(taken_actions[j][idx]))))
                
        
        
        
        if use_qm_minimization:
            translations = translator.reduce_logic(bin_t)
            
            for j, t in enumerate(translations):
                print('Group {}: {}'.format(j+1, t))
                print('Critical value: {}. Entropy: {:.2f}'.format(critical_values[j], group_ent[j]))
        
        else:
            for j, t in enumerate(bin_t): #t is a single group of (binary_state, action) tuples
                bin_states = [np.array(c[0]) for c in t]
                condensed_state = abstraction_helper.condense_predicate_set(t)
                translation = translator.translate_state(condensed_state)
                print('Group {}: {}'.format(j+1, translation))
                print('Critical value: {}. Entropy: {:.2f}'.format(critical_values[j], group_ent[j]))
        

        print('----------------------------------------')
        feat_groups = [[0, 1, 2, 3, 4, 5], [6]]
        other_translations = translator.my_translation_algo(bin_t, feat_groups)
        for j, t in enumerate(other_translations):
            print('Group {}: {}'.format(j+1, t))
            print('Critical value: {}. Entropy: {:.2f}'.format(critical_values[j], group_ent[j]))
        print('----------------------------------------')
        
        #fidelity = calculate_fidelity(model_path, clusters, dataset)
        #print('Gridworld CLTree Clustering Fidelity: ', fidelity)
        
        #run_abstract_episode(clusters, dataset)
    """
        


