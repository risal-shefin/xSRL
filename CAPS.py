from common.DictionarySummaryModel import DictionarySummaryModel
from common.enums import PolicyEnum, SummaryMethodEnum
import torch
import numpy as np
import matplotlib.pyplot as plt
from explain_utils import graph_scores
from explain_utils import cluster_data



import networkx as nx 

def visualize(transitions, taken_actions, taken_randoms, translator, bin_t, height, critical_values):
  # declare the graph
  G = nx.MultiDiGraph()
  edge_labels = {}

  """
  translation_2 = translator.my_translation_algo(bin_t)
  for j, t in enumerate(translation_2):
      print('Group {}: {}'.format(j+1, t))
  """
  critical_idx = []
  for j in range(len(transitions)):
    nonzero_idx = np.where(np.array(transitions[j])!=0)[0]
    for idx in nonzero_idx:
      if critical_values[j] == 1.0:
        critical_idx.append(j+1)
      if j+1 in critical_idx:  
        G.add_edge(j+1,idx+1, text = '{:0.4f} a{}'.format(transitions[j][idx], int(np.mean(taken_actions[j][idx]))), color = 'r')
      else:
        G.add_edge(j+1,idx+1, text = '{:0.4f} a{}'.format(transitions[j][idx], int(np.mean(taken_actions[j][idx]))), color = 'b')
      
      #G.add_edge(j+1,idx+1, text = "{:0.2f} a{}".format(transitions[j][idx], int(np.mean(taken_actions[j][idx])), int(np.mean(taken_randoms[j][idx]))), color = 'b')
      #if (j+1, idx+1) not in edge_labels:
      #  edge_labels[(j+1,idx+1)] = ""
      edge_labels=dict([((u,v,),d['text'])for u,v,d in G.edges(data=True)])
      #edge_labels[(j+1,idx+1)] += "{:0.2f} to take \naction {}".format(transitions[j][idx], np.mean(taken_actions[j][idx]))
  colors = nx.get_edge_attributes(G,'color').values()
  pos=nx.spring_layout(G)

  color_map = []
  #print("critical index is {}".format(critical_idx))
  for node in G:
    #print(node)
    if node in critical_idx:
      color_map.append('red')
    else:
      color_map.append('yellow')
  #print(color_map)
    
  #nx.draw(G, pos, with_labels = True, node_color = "yellow", edge_color = colors, connectionstyle='arc3, rad = 0.22')
  nx.draw(G, pos, with_labels = True, node_color=color_map, edge_color = colors, connectionstyle='arc3, rad = 0.22')
  nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, label_pos = 0.3)
  nx.draw_networkx_labels(G, pos, font_size=11)
  import pylab as plt
  plt.figure(1, figsize=(15,15), dpi = 2000) 
  plt.savefig("test{}.png".format(height))
  plt.show()


"""
Env specific info:
run episode function
predicate class
number of actions
number of features
value function
env name
alpha
max height
lambda
feature groups (include in predicate class)
"""
def explain(args, dataset, model_path, translator, num_feats, num_actions, fidelity_fn=None, apg_baseline=None, mode="PPO", 
            fail=None, ts=None, fail_user = None, ts_user = None, user_test = None, reward=None, reward_user=None, extra_dicts: list[DictionarySummaryModel]=None):


    attr_names = translator.attr_names
    attr_names.append('State Value')
    attr_names.append('Action')
    
    num_runs = 1
    fidelities = []
    cluster_v_scores = []
    ls = []
    e_scores = []
    for run in range(num_runs):
        all_clusters, best_heights, cluster_scores, value_scores, entropy_scores, lengths = cluster_data(translator, 
                                                                                                        apg_baseline, 
                                                                                                        dataset,
                                                                                                        attr_names,
                                                                                                        args.alpha,
                                                                                                        num_actions=num_actions,
                                                                                                        lmbda=args.lmbda,
                                                                                                        k=args.k,
                                                                                                        max_height=args.max_height,
                                                                                                        model_path=model_path,
                                                                                                        env=args.env
                                                                                                    )

        

        """
        graph_scores('cart', alpha, lengths, 
                    cluster_scores=cluster_scores, 
                    value_scores=value_scores, 
                    entropy_scores=entropy_scores,
                    fidelity_scores=fidelity_scores)
        """

        all_clusters = np.array(all_clusters, dtype=object)
        best_clusters = all_clusters[best_heights]
        

        fidelity_scores = []
        cluster_v_scores.append(value_scores[best_heights[0]])
        print('information is here')
        fail[(0,0)] = 0
        ts[(0,0)] = 0
        # print(fail)
        # print(ts)
        print('start generating the graph')
        for h, clusters in enumerate(all_clusters):
        #for h, clusters in enumerate(best_clusters):
            # if h != 2: #rss-comment - to analyze the output.
            #   continue
            print('***********************************************')
            #print('Clusters at height {}'.format(best_heights[h]+1))
            print('Clusters at height {}'.format(h))
            c = 0
            cluster_state_indices = []
            for i, node in enumerate(clusters):
                
                c += node.getNrInstancesInNode()
                
                cluster_state_indices.append(node.getInstanceIds())

            if fidelity_fn is not None:
                print("\n\nCalculating {} Fidelity Score.....\n\n".format(args.env))
                # if h == 0:
                if args.env == 'nav2' or args.env == 'maze':
                    fidelity = fidelity_fn(args, clusters, dataset, 2000)
                else:
                    assert model_path is not None
                    fidelity = fidelity_fn(model_path, clusters, dataset, mode=mode)

                print('\nFidelity: {}\n'.format(fidelity))
                
                fidelities.append(fidelity)
                fidelity_scores.append(fidelity)

            abstract_state_groups = []
            abstract_binary_state_groups = []
            abstract_state = []
            for cluster in cluster_state_indices:
                #print("cluster is {}".format(cluster))
                abs_t = []
                bin_t = []
                s_set = set()
                for idx in cluster:
                    idx = int(idx)
                    abs_t.append((dataset.states[idx], dataset.actions[idx], dataset.next_states[idx], dataset.dones[idx], dataset.entropies[idx], dataset.rewards[idx], 
                                  dataset.policies[idx], dataset.task_critic_vals[idx], dataset.safety_critic_vals[idx]))
                    s_set.add(tuple(dataset.states[idx]))
                    binary = translator.state_to_binary(dataset.states[idx])
                    bin_t.append((binary, dataset.actions[idx]))
                    
                #print("abs_t is {}".format(abs_t))
                #print("bin_t is {}".format(bin_t))
                abstract_state_groups.append(abs_t)
                abstract_binary_state_groups.append(bin_t)
                abstract_state.append(s_set)
            
            abs_t = abstract_state_groups
            bin_t = abstract_binary_state_groups
            
            fail_ans = []
            ts_ans = []
            reward_ans = []
            """
            print("abstract state is {}".format(abstract_state))
            for state in abstract_state:
              print("A {}".format(list(state)[0]))
            """
            for state_group in abstract_state:
              fail_ind = 0
              ts_ind = 0
              reward_ind = 0
              counter = 0
              for state in list(state_group):
                # state = tuple([float("{:.1f}".format(num)) for num in state])
                #print("state before {}".format(state))
                state = tuple(state)
                #print("state after {}".format(state))
                
                # for safety grid, state is (x,) single tuple
                if state in fail and state in ts:
                  fail_ind += fail[state]
                  ts_ind += ts[state]
                  if reward is not None and state in reward:
                     reward_ind += reward[state]
                  #print(f'state is {state} with ts {ts[state]} and fp {fail[state]}')
                counter += 1
              fail_ind /= counter
              ts_ind /= counter
              reward_ind /= counter
              fail_ans.append(fail_ind)
              ts_ans.append(ts_ind)
              reward_ans.append(reward_ind)
            # print("fail is {}".format(fail_ans))
            # print("ts is {}".format(ts_ans))
            # print("reward is {}".format(reward_ans))
            

            if user_test:
              #print("CAPS fail user: ", fail_user)
              fail_ans_user = []
              ts_ans_user = []
              reward_ans_user = []
              for state_group in abstract_state:
                fail_ind = 0
                ts_ind = 0
                reward_ind = 0
                counter = 0
                for state in list(state_group):
                  #state = [float("{:.3f}".format(num)) for num in state]
                  #print("state before {}".format(state))
                  state = tuple(state)
                  # state = tuple([float("{:.1f}".format(num)) for num in state])
                  #print("state after {}".format(state))
                  if state in fail_user and state in ts_user:
                    fail_ind += fail_user[state]
                    ts_ind += ts_user[state]
                  if reward_user is not None and state in reward_user:
                    reward_ind += reward_user[state]
                  counter += 1
                #print("fail_ind and counter: ", fail_ind, counter)
                fail_ind /= counter
                ts_ind /= counter
                reward_ind /= counter
                fail_ans_user.append(fail_ind)
                ts_ans_user.append(ts_ind)
                reward_ans_user.append(reward_ind)
              print("fail_user is {}".format(fail_ans_user))
              print("ts_user is {}".format(ts_ans_user))
              print("reward_user is {}".format(reward_ans_user))

            #If there is any additional dictionary containing info like failure probs, timesteps, rewards etc.
            if type(extra_dicts) is type(list()):
              extra_dicts_ans = []
              for dict_model in extra_dicts:
                dict_ans = []
                for state_group in abstract_state:
                  calc_ind = 0
                  counter = 0
                  for state in list(state_group):
                    # state = tuple([float("{:.1f}".format(num)) for num in state])
                    state = tuple(state)
                    if state in dict_model.dictionary:
                      if dict_model.summary_method == SummaryMethodEnum.Average:
                        calc_ind += dict_model.dictionary[state]
                      elif dict_model.summary_method == SummaryMethodEnum.Max:
                        calc_ind = max(calc_ind, dict_model.dictionary[state])
                      elif dict_model.summary_method == SummaryMethodEnum.Min:
                        calc_ind = min(calc_ind, dict_model.dictionary[state])
                    counter += 1
                  if dict_model.summary_method == SummaryMethodEnum.Average:
                    calc_ind /= counter
                  dict_ans.append(calc_ind)
                extra_dicts_ans.append(dict_ans)
               

            critical_values, group_ent = apg_baseline.get_critical_groups(abs_t)

            l, transitions, taken_actions, taken_randoms, action_policies, action_task_critic_vals, action_safety_critic_vals = apg_baseline.compute_graph_info(abs_t, take_randoms=True)
            #print("length of transition is {}".format(len(transitions)))
            #print("length of abstract state is {}".format(len(abstract_state)))
            print("\n ****** CAPS GRAPH's EDGES: ***********")
            for j in range(len(transitions)):
                nonzero_idx = np.where(np.array(transitions[j])!=0)[0]
                #nonzero_idx = np.where(np.array(transitions[j]) >= 0.01)[0] # discarding low probability edges
                for idx in nonzero_idx:
                    
                    task_policy_count = 0
                    safety_policy_count = 0
                    for policy in action_policies[j][idx]:
                        if policy == PolicyEnum.TaskPolicy:
                            task_policy_count += 1
                        else:
                            safety_policy_count += 1

                    avg_action = int(np.mean(taken_actions[j][idx]))
                    if args.env in ['nav2', 'maze']:
                        # Map actions to angles (assuming 0 = east, 1 = northeast, ..., 7 = southeast)
                        angles = np.array(taken_actions[j][idx]) * (2 * np.pi / 8)  # Convert actions to radians
                        x = np.cos(angles).sum()  # Sum of x-components
                        y = np.sin(angles).sum()  # Sum of y-components
                        avg_angle = np.arctan2(y, x)  # Resultant angle
                        avg_action = int(round(avg_angle / (2 * np.pi / 8))) % 8  # Map back to action space

                    print('Group {} to Group {} with p={} and action {}, random {}, Task Policy Count {}, Safety Policy Count {}, Task Critic {}, Safety Critic {}'
                          .format(j+1, idx+1, transitions[j][idx], avg_action, int(np.mean(taken_randoms[j][idx])), task_policy_count, safety_policy_count, np.mean(action_task_critic_vals[j][idx]), np.mean(action_safety_critic_vals[j][idx])))
                    #print('Group {} to Group {} with p={} and action {}'.format(j+1, idx+1, transitions[j][idx], int(np.mean(taken_actions[j][idx]))))
            #visualize(transitions, taken_actions, taken_randoms, translator, bin_t, best_heights[h] + 1)
            visualize(transitions, taken_actions, taken_randoms, translator, bin_t, h, critical_values)
            if args.hayes_baseline: #Hayes and Shah baseline
                hayes_translations = translator.reduce_logic(bin_t)
            
            #CAPS explanation producer
            # print('----------------------------------------')
            # print('debug CAPS.py')
            # print(bin_t)
            translations = translator.my_translation_algo(bin_t)
            print("\n ****** CAPS GRAPH's ABSTRACT STATE INFORMATION: ***********")
            for j, t in enumerate(translations):
                if not user_test:
                  print('Group {}: {}, failure prob {}, expected ts {} and expected reward {}'.format(j+1, t, fail_ans[j], ts_ans[j], reward_ans[j]))
                else:
                  print('Group {}: {}, failure prob from {} to {}, expected ts from {} to {} and expected reward from {} to {}'
                        .format(j+1, t, fail_ans[j], fail_ans_user[j], ts_ans[j], ts_ans_user[j], reward_ans[j], reward_ans_user[j]))
                if args.hayes_baseline:
                    print('(Hayes) Group {}: {}'.format(j+1, hayes_translations[j]))
                print('Critical value: {}. Entropy: {:.2f}'.format(critical_values[j], group_ent[j]))

                if type(extra_dicts) is type(list()):
                  print("-----Additional Values:-----")
                  dict_counter = 0
                  for dict_ans in extra_dicts_ans:
                    print('Additional Data No. {}, Name: {}, Value = {}'.format(dict_counter, extra_dicts[dict_counter].name, dict_ans[j]))
                    dict_counter += 1
                  print("") # extra newline
            print('----------------------------------------')
            

