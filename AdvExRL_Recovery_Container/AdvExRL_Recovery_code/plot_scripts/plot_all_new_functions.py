import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pickle
# from plot_utils import *
from plot_scripts.plot_utils import *
import os

def get_stats(data):
    # minlen = min([len(d) for d in data])
    # data = [d[:minlen] for d in data]
    size = data.size
    mu = np.mean(data, axis=0)
    lb = mu - np.std(data, axis=0) / np.sqrt(size)
    ub = mu + np.std(data, axis=0) / np.sqrt(size)
    return mu, lb, ub

def moving_average(x, N):
    window_means = []
    for i in range(len(x) - N + 1):
        window = x[i:i + N]
        num_nans = np.count_nonzero(np.isnan(window))
        window_sum = np.nansum(window)
        if num_nans < N:
            window_mean = window_sum / (N - num_nans)
        else:
            window_mean = np.nan
        window_means.append(window_mean)
    return window_means

def get_our_safety_data(path, env_name):
    with open(path, 'rb') as f:
      loaded_data = pickle.load(f)

    max_epi_len = 100
    task_info_data = loaded_data['tsk_info']
    task_test_violations = []
    task_test_rewards = []
    task_last_rewards = []
    task_test_rewards_safe_list = []
    for traj_stats in task_info_data:
                task_test_violations.append([])
                task_test_rewards.append(0)
                last_reward = 0
                for step_stats in traj_stats:
                    task_test_violations[-1].append(step_stats['constraint'])
                    task_test_rewards[-1] += step_stats['reward']
                    last_reward = step_stats['reward']
                task_last_rewards.append(last_reward)
    #--------------------------------------------------
    task_safety = []
    for t in task_test_violations:
      if sum(t) > 0:
        task_safety.append(float(len(t)/max_epi_len))
      elif sum(t)==0:
        task_safety.append(1)
    #--------------------------------------------------    
    if env_name=='maze':
        max_dist = -0.03
        min_dist = -60.0
    elif env_name=='nav1':
        max_dist = -4.0
        min_dist = -100.0
    elif env_name=='nav2':
        max_dist = -4.0
        min_dist = -150.0
    tsk_agent_dist = np.array(task_last_rewards)
    task_success_rate = (((tsk_agent_dist-min_dist)/(max_dist-min_dist))*(task_safety))*100
   
    #****************************************************************************
    #****************************************************************************
    rec_info_data = loaded_data['tsk_rec_info']
    rec_test_violations = []
    rec_test_rewards = []
    rec_last_rewards = []
    rec_test_rewards_safe_list = []
    for traj_stats in rec_info_data:
                rec_test_violations.append([])
                rec_test_rewards.append(0)
                last_reward = 0
                for step_stats in traj_stats:
                    rec_test_violations[-1].append(step_stats['constraint'])
                    rec_test_rewards[-1] += step_stats['reward']
                    last_reward = step_stats['reward']
                rec_last_rewards.append(last_reward)
    
    #--------------------------------------------------
    rec_safety = []
    for t in rec_test_violations:
      if sum(t) > 0:
        rec_safety.append(float(len(t)/100))
      elif sum(t)==0:
        rec_safety.append(1)
    #--------------------------------------------------    
    rec_agent_dist = np.array(rec_last_rewards)
    rec_success_rate = (((rec_agent_dist-min_dist)/(max_dist-min_dist))*(rec_safety))*100
    # rec_success_rate = (((rec_agent_dist-min_dist)/(max_dist-min_dist)))*100
    #-------------------------------------------------- 
    task_safety_percentage = sum(task_safety)/len(task_safety)*100
    rec_safety_percentage = sum(rec_safety)/len(rec_safety)*100
    # print(task_safety_percentage)
    # print(rec_safety_percentage)
    #***************************************************************************
    data={'task_sfty_percentage': task_safety_percentage,
            'rec_sfty_percentage': rec_safety_percentage,
            'task_success_rate': task_success_rate , 
            'rec_success_rate':rec_success_rate,
            'percnt_tsk_policy': None,
            'percnt_rec_policy': None
            }
    return data

def get_agent_percentage_count(path):
    with open(path, 'rb') as f:
      loaded_data = pickle.load(f)
    tsk_cnt = np.array(loaded_data['epi_task_rec_tsk_cnt']).astype(float)
    rec_cnt = np.array(loaded_data['epi_task_rec_rec_cnt']).astype(float)
    total = tsk_cnt+rec_cnt
    
    percentage_tsk = tsk_cnt/ total
    if sum(percentage_tsk)==0:
      percentage_tsk=0.0
    else:
      percentage_tsk = sum(percentage_tsk)/len(percentage_tsk)

    percentage_rec = rec_cnt/ total
    if sum(percentage_rec)==0:
      percentage_rec=0.0
    else:
      percentage_rec = sum(percentage_rec)/len(percentage_rec)
    
    return round(percentage_tsk,2)*100, round(percentage_rec,2)*100
    
def env_safety_data_our_model(env_data_path, env_name):
    paths = os.listdir(env_data_path)
    Data ={}
    Data['rate']={}
    for path in paths:
        name = path.split('atk_rate')[-1]
        info = name.split('_eps_')
        # print(info)
        atk_rate = info[0]
        eps = info[1]
        data_path = os.path.join(env_data_path,path, 'Info_for_plotting_data.pkl')
        per_count_path = os.path.join(env_data_path,path, 'Exp_data.pkl')
        # print(data_path)
        Our_safety_data = get_our_safety_data(data_path, env_name)
        per_tsk, per_rec = get_agent_percentage_count(per_count_path)
        Our_safety_data['percnt_tsk_policy']= per_tsk
        Our_safety_data['percnt_rec_policy']= per_rec
        Data['rate'][eps] = Our_safety_data
    return Data

def RecRL_data(info_data, env_name):
    max_epi_len = 100
    test_violations = []
    test_rewards = []
    last_rewards = []
    violations_list = []
    safety_list = []
    for traj_stats in info_data:
            test_violations.append([])
            test_rewards.append(0)
            last_reward = 0
            for step_stats in traj_stats:
                test_violations[-1].append(step_stats['constraint'])
                test_rewards[-1] += step_stats['reward']
                last_reward = step_stats['reward']
            last_rewards.append(last_reward)
            violations_list.append(test_violations)
  
    for t in test_violations:
      if sum(t) > 0:
        safety_list.append(float(len(t)/max_epi_len))
      elif sum(t)==0:
        safety_list.append(1)
    
    #--------------------------------------------------    
    if env_name=='maze':
        max_dist = -0.03
        min_dist = -60.0
    elif env_name=='nav1':
        max_dist = -4.0
        min_dist = -100
    elif env_name=='nav2':
        max_dist = -4.0
        min_dist = -150.0
    agent_dist = np.array(last_rewards)
    success_rate = (((agent_dist-min_dist)/(max_dist-min_dist))*(safety_list))*100
    # success_rate = ((agent_dist-min_dist)/(max_dist-min_dist))*100
    #--------------------------------------------------  
    safety_percentage = sum(safety_list)/len(safety_list)*100
    return safety_percentage, success_rate

def get_recoveryRL_safety(path, env_name):
  with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
  safety_data = {}
  safety_data['algos'] = {}
  for algo in loaded_data['algos']:
      # if algo =='unconstrained':
      #   info = loaded_data['algos'][algo]['result']['task_agent']['info']
      #   rec_safety_percentage, rec_success_rate = RecRL_data(info, env_name)
      # else:
      info = loaded_data['algos'][algo]['result']['task_rec_agent']['info']
      rec_safety_percentage, rec_success_rate = RecRL_data(info, env_name)

      tsk_cnt = np.array(loaded_data['algos'][algo]['result']['task_rec_agent']['tsk_cnt']).astype(float)
      rec_cnt = np.array(loaded_data['algos'][algo]['result']['task_rec_agent']['rec_cnt']).astype(float)
      total = tsk_cnt+rec_cnt


      percentage_tsk = tsk_cnt/ total
      if sum(percentage_tsk)==0:
        percentage_tsk=0.0
      else:
        percentage_tsk = sum(percentage_tsk)/len(percentage_tsk)

      percentage_rec = rec_cnt/ total
      if sum(percentage_rec)==0:
        percentage_rec=0.0
      else:
        percentage_rec = sum(percentage_rec)/len(percentage_rec)

      data={'rec_sfty_percentage': rec_safety_percentage,
            'rec_success_rate':rec_success_rate,
            'percnt_tsk_policy': round(percentage_tsk,2)*100,
            'percnt_rec_policy': round(percentage_rec,2)*100
            }
      safety_data['algos'][algo]=data
      
  return safety_data

def get_all_Recovery_RL_data(logdir_recoveryRL, env_name):
    paths = os.listdir(logdir_recoveryRL)
    RecoveryRL_data = {}
    RecoveryRL_data['rate']={}
    for path in paths:
        recoveryRL_safety_data = None
        data_path = os.path.join(logdir_recoveryRL , path, 'saved_exp_data.pkl')
        name = path.split('Atk_rate')[-1]
        info = name.split('_eps')
        atk_rate = info[0]
        eps = info[1]
        
        recoveryRL_safety_data = get_recoveryRL_safety(data_path, env_name)
        
        name = path.split('Atk_rate')[-1]
        info = name.split('_eps')
        atk_rate = info[0]
        eps = info[1]

        RecoveryRL_data['rate'][eps]=recoveryRL_safety_data
    return RecoveryRL_data
#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------
def draw_safety_plot(recRL_data, our_data, env_name, atk_name, save_dir):
    safety_SQRL = []
    safety_RRL_MF = []
    safety_RCPO = []
    safety_LR = []
    safety_RSPO = []
    safety_RP = []
    safety_Our_model = []
    safety_SAC = []

    for r in recRL_data['rate']:
      safety_SQRL.append(recRL_data['rate'][r]['algos']['SQRL']['rec_sfty_percentage'])
      safety_RRL_MF.append(recRL_data['rate'][r]['algos']['RRL_MF']['rec_sfty_percentage'])
      safety_RCPO.append(recRL_data['rate'][r]['algos']['RCPO']['rec_sfty_percentage'])
      safety_LR.append(recRL_data['rate'][r]['algos']['LR']['rec_sfty_percentage'])
      safety_RSPO.append(recRL_data['rate'][r]['algos']['RSPO']['rec_sfty_percentage'])
      safety_RP.append(recRL_data['rate'][r]['algos']['RP']['rec_sfty_percentage'])
      safety_Our_model.append(our_data['rate'][r]['rec_sfty_percentage'])
      # safety_SAC.append(our_data['rate'][r]['task_sfty_percentage'])
      safety_SAC.append(recRL_data['rate'][r]['algos']['unconstrained']['rec_sfty_percentage'])
      
    # plt.figure().clear()
    size_spec = get_fig_size()
    fig, ax = plt.subplots(figsize=(int(size_spec['fig_size_x']), int(size_spec['fig_size_y'])), dpi=300)
    # plt.rcParams.update({'font.size': 50})

    labels = ['0', '25', '50', '75', '100']
    atk_rate = np.array([0,25,50,75,100])

    # epsilon = np.array([0.0, 0.25, 0.50, 0.75, 1.00]) #[0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    x = np.array([0,25,50,75,100])  # the label locations

    width = 1  # the width of the bars

    rects1 = ax.bar(x-3*width, safety_SAC, width, label='SAC',color='skyblue' )
    rects2 = ax.bar(x-2*width, safety_SQRL, width, label='SQRL',color='hotpink' )
    rects3 = ax.bar(x-1*width, safety_RCPO, width, label='RCPO',color='lightgray' )
    rects4 = ax.bar(x*width, safety_LR, width, label='LR',color='pink' )
    rects5 = ax.bar(x+1*width, safety_RSPO, width, label='RSPO',color='brown' )
    rects6 = ax.bar(x+2*width, safety_RRL_MF, width, label='RRL_MF',color='mediumpurple' )
    rects7 = ax.bar(x+3*width, safety_RP, width, label='RP',color='orange' )
    rects8 = ax.bar(x+4*width, safety_Our_model, width, label='AdvEx-RL',color='springgreen' )
    
    
    ax.set_xticks([0,25,50,75,100])
    ax.set_yticks([0,25,50,75,100])
    ax.tick_params(axis='x', labelsize=int(size_spec['ticks']))
    ax.tick_params(axis='y', labelsize=int(size_spec['ticks']))
    # if atk_name=="random":
    #   atk_name_title = "Random Action Perturbation"
    # else:
    #   atk_name_title = "Alternative Adversarial Action Perturbation"
    # if env_name=='maze':
    #   # ax.set_title('Maze ({})'.format(atk_name_title), fontsize=50)
    #   ax.set_title('Maze ({})'.format(atk_name_title), fontsize=int(size_spec['title']))
    # elif env_name=='nav1':
    #   # ax.set_title('Navigation 1 ({})'.format(atk_name_title), fontsize=50)
    #   ax.set_title('Navigation 1 ({})'.format(atk_name_title), fontsize=int(size_spec['title']))
    # elif env_name=='nav2':
    #   # ax.set_title('Navigation 2 ({})'.format(atk_name_title), fontsize=50)
    #   ax.set_title('Navigation 2 ({})'.format(atk_name_title), fontsize=int(size_spec['title']))
    
    ax.set_xlabel('Perturbation rate (%)', fontsize=int(size_spec['label']))
    ax.set_ylabel('Safety (%)', fontsize=int(size_spec['label']))
    # ax.set_xlabel('Perturbation rate (%)', fontsize=50)
    # ax.set_ylabel('Safety (%)', fontsize=50)

    plt.grid(color='gray', linewidth=1, axis='y', alpha=0.5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=int(size_spec['label']), fancybox=True, shadow=True, ncol=4)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=50, fancybox=True, shadow=True, ncol=8)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt_dir = os.path.join(save_dir, env_name, atk_name)
    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)
    plt.savefig(plt_dir+'/only_safety_percentage_plot.png',dpi=300,format='png',bbox_inches='tight')
    
    plt.show()
#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------
def draw_success_rate_plot(recRL_data, our_data ,env_name, atk_name, save_dir):
    
    mean_SQRL = []
    lb_SQRL = []
    ub_SQRL = []

    mean_RRL_MF = []
    lb_RRL_MF = []
    ub_RRL_MF = []
    
    mean_RCPO = []
    lb_RCPO = []
    ub_RCPO = []

    mean_LR = []
    lb_LR = []
    ub_LR = []

    mean_RSPO = []
    lb_RSPO = []
    ub_RSPO = []

    mean_RP = []
    lb_RP = []
    ub_RP = []

    mean_Our_model = []
    lb_Our_model = []
    ub_Our_model = []

    mean_SAC = []
    lb_SAC = []
    ub_SAC = []

    for r in recRL_data['rate']:
          mean, lb, ub = get_stats(recRL_data['rate'][r]['algos']['SQRL']['rec_success_rate'])
          mean_SQRL.append(mean)
          lb_SQRL.append(lb)
          ub_SQRL.append(ub)

          mean, lb, ub = get_stats(recRL_data['rate'][r]['algos']['RRL_MF']['rec_success_rate'])
          mean_RRL_MF.append(mean)
          lb_RRL_MF.append(lb)
          ub_RRL_MF.append(ub)

          mean, lb, ub = get_stats(recRL_data['rate'][r]['algos']['RCPO']['rec_success_rate'])
          mean_RCPO.append(mean)
          lb_RCPO.append(lb)
          ub_RCPO.append(ub)

          mean, lb, ub = get_stats(recRL_data['rate'][r]['algos']['LR']['rec_success_rate'])
          mean_LR.append(mean)
          lb_LR.append(lb)
          ub_LR.append(ub)

          mean, lb, ub = get_stats(recRL_data['rate'][r]['algos']['RSPO']['rec_success_rate'])
          mean_RSPO.append(mean)
          lb_RSPO.append(lb)
          ub_RSPO.append(ub)

          mean, lb, ub = get_stats(recRL_data['rate'][r]['algos']['RP']['rec_success_rate'])
          mean_RP.append(mean)
          lb_RP.append(lb)
          ub_RP.append(ub)

          # mean, lb, ub = get_stats(our_data['rate'][r]['task_success_rate'])
          mean, lb, ub = get_stats(recRL_data['rate'][r]['algos']['unconstrained']['rec_success_rate'])
          mean_SAC.append(mean)
          lb_SAC.append(lb)
          ub_SAC.append(ub)

          mean, lb, ub = get_stats(our_data['rate'][r]['rec_success_rate'])
          mean_Our_model.append(mean)
          lb_Our_model.append(lb)
          ub_Our_model.append(ub)
          # mean_SQRL.append(pd.DataFrame(np.array(recRL_data['rate'][r]['algos']['SQRL']['rec_success_rate'])))
    #*****************************************************************
    plt.figure()
    x = np.array([0,25,50,75,100])
    # bp=mean_SQRL[0].boxplot()


    # fig, axs = plt.subplots(1, figsize=(35, 15), dpi=300)
    # axs.tick_params(axis='both', which='major', labelsize=36)
    
    size_spec = get_fig_size()
    # fig, ax = plt.subplots(figsize=(35, 15), dpi=300)
    fig, axs = plt.subplots(figsize=(int(size_spec['fig_size_x']), int(size_spec['fig_size_y'])), dpi=300)

    axs.fill_between(x, ub_SAC, lb_SAC, color='skyblue', 
                     alpha=.25, label='SAC')
    axs.plot(x, mean_SAC, color='skyblue', linewidth='2')

    axs.fill_between(x, ub_SQRL, lb_SQRL, color='hotpink', 
                     alpha=.25, label='SQRL')
    axs.plot(x, mean_SQRL, color='hotpink', linewidth='2')
    
    axs.fill_between(x, ub_RCPO, lb_RCPO, color='lightgray', 
                     alpha=.25, label='RCPO')
    axs.plot(x,mean_RCPO, color='lightgray', linewidth='2')

    axs.fill_between(x, ub_LR, lb_LR, color='pink', 
                     alpha=.25, label='LR')
    axs.plot(x, mean_LR, color='pink', linewidth='2')

    axs.fill_between(x, ub_RSPO, lb_RSPO, color='brown', 
                     alpha=.25, label='RSPO')
    axs.plot(x,mean_RSPO, color='brown', linewidth='2')

    axs.fill_between(x, ub_RRL_MF, lb_RRL_MF, color='mediumpurple', 
                     alpha=.25, label='RRL_MF')
    axs.plot(x,mean_RRL_MF, color='mediumpurple', linewidth='2')

    axs.fill_between(x, ub_RP, lb_RP, color='orange', 
                     alpha=.25, label='RP')
    axs.plot(x,mean_RP, color='orange', linewidth='2')

    axs.fill_between(x, ub_Our_model, lb_Our_model, color='springgreen', 
                     alpha=.25, label='AdvEx-RL')
    axs.plot(x,mean_Our_model, color='lime', linewidth='2')
    if atk_name=="random":
      atk_name_title = "Random Action Perturbation"
    else:
      atk_name_title = "Alternative Adversarial Action Perturbation"
    labels = ['0', '25', '50', '75', '100']
    # axs.set_xticks(x, labels)

    #--------------------------------
    axs.tick_params(axis='x', labelsize=int(size_spec['ticks']))
    axs.tick_params(axis='y', labelsize=int(size_spec['ticks']))
    # if atk_name=="random":
    #   atk_name_title = "Random Action Perturbation"
    # else:
    #   atk_name_title = "Alternative Adversarial Action Perturbation"
    # if env_name=='maze':
    #   # ax.set_title('Maze ({})'.format(atk_name_title), fontsize=50)
    #   axs.set_title('Maze ({})'.format(atk_name_title), fontsize=int(size_spec['title']))
    # elif env_name=='nav1':
    #   # ax.set_title('Navigation 1 ({})'.format(atk_name_title), fontsize=50)
    #   axs.set_title('Navigation 1 ({})'.format(atk_name_title), fontsize=int(size_spec['title']))
    # elif env_name=='nav2':
    #   # ax.set_title('Navigation 2 ({})'.format(atk_name_title), fontsize=50)
    #   axs.set_title('Navigation 2 ({})'.format(atk_name_title), fontsize=int(size_spec['title']))
    
    axs.set_xlabel('Perturbation rate (%)', fontsize=int(size_spec['label']))
    axs.set_ylabel('Success-safety (%)', fontsize=int(size_spec['label']))
    # ax.set_xlabel('Perturbation rate (%)', fontsize=50)
    # ax.set_ylabel('Safety (%)', fontsize=50)

    plt.grid(color='gray', linewidth=1, axis='y', alpha=0.5)
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=int(size_spec['label']), fancybox=True, shadow=True, ncol=4)
    #--------------------------------

    plt_dir = os.path.join(save_dir, env_name, atk_name)
    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)
    
    plt.savefig(plt_dir+'/{}_{}_success_safety_percentage_plot.png'.format(env_name, atk_name),dpi=300,format='png',bbox_inches='tight')
    plt.show()


#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------
def draw_all_safety_with_policy_ratio_plot(recRL_data, our_data, env_name, atk_name, save_dir):

    task_ratio_RRL_MF = []
    rec_ratio_RRL_MF = []

    task_ratio_Our_model = []
    rec_ratio_Our_model = []

    safety_RCPO = []
    safety_LR = []
    safety_RSPO = []
    safety_RP = []
    safety_SAC = []
    safety_SQRL = []
    
    for r in recRL_data['rate']:
        
          safety = recRL_data['rate'][r]['algos']['RRL_MF']['rec_sfty_percentage']
          p_tsk = recRL_data['rate'][r]['algos']['RRL_MF']['percnt_tsk_policy']
          p_rec = recRL_data['rate'][r]['algos']['RRL_MF']['percnt_rec_policy']
          safety_wrt_tsk = (safety/100)*p_tsk
          safety_wrt_rec = (safety/100)*p_rec
          task_ratio_RRL_MF.append(safety_wrt_tsk)
          rec_ratio_RRL_MF.append(safety_wrt_rec)

          safety = our_data['rate'][r]['rec_sfty_percentage']
          p_tsk = our_data['rate'][r]['percnt_tsk_policy']
          p_rec = our_data['rate'][r]['percnt_rec_policy']
          safety_wrt_tsk = (safety/100)*p_tsk
          safety_wrt_rec = (safety/100)*p_rec
          task_ratio_Our_model.append(safety_wrt_tsk)
          rec_ratio_Our_model.append(safety_wrt_rec)
          #---------------------------------------------------------
          safety = recRL_data['rate'][r]['algos']['RSPO']['rec_sfty_percentage']
          p_tsk = recRL_data['rate'][r]['algos']['RSPO']['percnt_tsk_policy']
          p_rec = recRL_data['rate'][r]['algos']['RSPO']['percnt_rec_policy']
          safety_wrt_tsk = (safety/100)*p_tsk
          safety_wrt_rec = (safety/100)*p_rec
          # task_ratio_RSPO.append(safety_wrt_tsk)
          # rec_ratio_RSPO.append(safety_wrt_rec)
          # safety_RSPO.append(recRL_data['rate'][r]['algos']['RSPO']['rec_sfty_percentage'])
          
          safety_SQRL.append(recRL_data['rate'][r]['algos']['SQRL']['rec_sfty_percentage'])
          safety_RCPO.append(recRL_data['rate'][r]['algos']['RCPO']['rec_sfty_percentage'])
          safety_LR.append(recRL_data['rate'][r]['algos']['LR']['rec_sfty_percentage'])
          safety_RSPO.append(recRL_data['rate'][r]['algos']['RSPO']['rec_sfty_percentage'])
          safety_RP.append(recRL_data['rate'][r]['algos']['RP']['rec_sfty_percentage'])
          # safety_SAC.append(our_data['rate'][r]['task_sfty_percentage'])
          safety_SAC.append(recRL_data['rate'][r]['algos']['unconstrained']['rec_sfty_percentage'])
          #---------------------------------------------------------

    #-----------------------------------------------------
    # fig, ax = plt.subplots(figsize=(35, 15), dpi=300)
    # plt.rcParams.update({'font.size': 35})

    size_spec = get_fig_size()
    fig, ax = plt.subplots(figsize=(int(size_spec['fig_size_x']), int(size_spec['fig_size_y'])), dpi=300)

    labels = ['0', '25', '50', '75', '100']
    x = np.array([0,25,50,75,100])  # the label locations
    width = 1  # the width of the bars


    rects1 = ax.bar(x-3*width, safety_SAC, width, label='SAC',color='skyblue' )
    rects2 = ax.bar(x-2*width, safety_SQRL, width, label='SQRL',color='hotpink' )
    # rects2 = ax.bar(x-2*width, rec_ratio_SQRL, width, label='SQRL-Safety policy(%)',color='hotpink' )
    # rects2 = ax.bar(x-2*width, task_ratio_SQRL, width, bottom=rec_ratio_SQRL, label='SAC-task policy(%)',color='skyblue' )
    
    rects3 = ax.bar(x-1*width, safety_RCPO, width, label='RCPO',color='lightgray' )
    
    rects4 = ax.bar(x*width, safety_LR, width, label='LR',color='pink' )
    
    rects5 = ax.bar(x+1*width, safety_RSPO, width, label='RSPO',color='brown' )
  
    rects6 = ax.bar(x+2*width, rec_ratio_RRL_MF, width, label='RRL-MF-Safety policy(%)',color='mediumpurple' )
    rects6 = ax.bar(x+2*width, task_ratio_RRL_MF, width, bottom=rec_ratio_RRL_MF, label='RRL-MF:SAC-Task policy(%)',color='lavender' )
    
    rects7 = ax.bar(x+3*width, safety_RP, width, label='RP',color='orange' )
    
    rects8 = ax.bar(x+4*width, rec_ratio_Our_model, width, label='AdvEx-RL-Safety policy(%)',color='springgreen' )
    rects8 = ax.bar(x+4*width, task_ratio_Our_model, width, bottom=rec_ratio_Our_model, label='AdvEx-RL: SAC-task policy(%)',color='yellow' )



    #--------------------------------
    ax.tick_params(axis='x', labelsize=int(size_spec['ticks']))
    ax.tick_params(axis='y', labelsize=int(size_spec['ticks']))
    # if atk_name=="random":
    #   atk_name_title = "Random Action Perturbation"
    # else:
    #   atk_name_title = "Alternative Adversarial Action Perturbation"
    # if env_name=='maze':
    #   # ax.set_title('Maze ({})'.format(atk_name_title), fontsize=50)
    #   ax.set_title('Maze ({})'.format(atk_name_title), fontsize=int(size_spec['title']))
    # elif env_name=='nav1':
    #   # ax.set_title('Navigation 1 ({})'.format(atk_name_title), fontsize=50)
    #   ax.set_title('Navigation 1 ({})'.format(atk_name_title), fontsize=int(size_spec['title']))
    # elif env_name=='nav2':
    #   # ax.set_title('Navigation 2 ({})'.format(atk_name_title), fontsize=50)
    #   ax.set_title('Navigation 2 ({})'.format(atk_name_title), fontsize=int(size_spec['title']))
    
    ax.set_xlabel('Perturbation rate (%)', fontsize=int(size_spec['label']))
    ax.set_ylabel('Safety (%)', fontsize=int(size_spec['label']))
    # ax.set_xlabel('Perturbation rate (%)', fontsize=50)
    # ax.set_ylabel('Safety (%)', fontsize=50)

    plt.grid(color='gray', linewidth=1, axis='y', alpha=0.5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=int(size_spec['label']), fancybox=True, shadow=True, ncol=3)
    #--------------------------------



    # ax.set_xticks([0,25,50,75,100])
    # ax.set_yticks([0,25,50,75,100])
    # ax.tick_params(axis='x', labelsize=40)
    # ax.tick_params(axis='y', labelsize=40)
    # if atk_name=="random":
    #   atk_name_title = "Random Action Perturbation"
    # else:
    #   atk_name_title = "Alternative Adversarial Action Perturbation"
    # if env_name=='maze':
    #   ax.set_title('Maze ({})'.format(atk_name_title), fontsize=50)
    # elif env_name=='nav1':
    #   ax.set_title('Navigation 1 ({})'.format(atk_name_title), fontsize=50)
    # elif env_name=='nav2':
    #   ax.set_title('Navigation 2 ({})'.format(atk_name_title), fontsize=50)
    
    # ax.set_xlabel('Perturbation rate (%)', fontsize=40)
    # ax.set_ylabel('Safety (%)', fontsize=40)
    # plt.grid(color='gray', linewidth=1, axis='y', alpha=0.5)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=40, fancybox=True, shadow=True, ncol=8)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt_dir = os.path.join(save_dir, env_name, atk_name)
    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)
    
    plt.savefig(plt_dir+'/all_safety_with_tsk_safety_ratio.png',dpi=300,format='png',bbox_inches='tight')
    plt.show()