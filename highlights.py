import numpy as np

class Highlights:
    def __init__(self, info, k=5, l=10, post_states=5, num_episodes=1, min_dist=5):
        """
        Parameters:
        info: array of dicts with length num_episodes.
        dict of form {'states': states, 'actions': actions, 'values': values, 'entropy': entropy}
        all dict values should be np arrays
        values are included if highlights should use V(s) as measure of importance
        entropy is included if highlights should use entropy of policy as measure of importance
        k: number of trajectories to be displayed in explanation
        l: length of each trajectory
        post_states: number of (state, action) pairs included after the critical state
        num_episodes: number of episodes included in info
        min_dist: minimum distance between trajectories
        """
        self.info = info
        self.k = k
        self.l = l
        self.post_states = post_states
        self.pre_states = self.l - self.post_states - 1
        self.num_episodes = num_episodes
        self.min_dist = min_dist
        self.trajectories = {'trajectories': [], 'critical_times': [], 'critical_indicies': []}
        self.cur_episode_len = 0

        if 'values' in self.info[0]:
            self.value_measure = True
        elif 'entropy' in self.info[0]:
            self.value_measure = False
        else:
            raise ValueError('Include values or entropy measures in info')


    def identify_critical_states(self):
        traj_info = []
        for episode in range(self.num_episodes):
            states = self.info[episode]['states']
            actions = self.info[episode]['actions']
            num_steps = len(actions)
            num_critical = int(num_steps / 10) #Top 10% ordered by importance are labeled critical
            if not self.value_measure:
                entropy = self.info[episode]['entropy']
                sorted_idx = np.argsort(entropy)
                top_indices = sorted_idx[num_steps-num_critical:]
                criticals = np.zeros(num_steps)
                criticals[top_indices] = 1

                episode_info = []
                for i in range(num_steps):
                    episode_info.append((states[i], actions[i], entropy[i]))

                traj_info.append(episode_info)


            else:
                raise NotImplementedError
            
        return traj_info


    def compute_trajectories(self):
        
        self.trajectories = {'trajectories': [], 'critical_times': [], 'critical_indices': []}

        for episode in range(self.num_episodes):
            curr_trajectory = []
            self.states = self.info[episode]['states']
            self.actions = self.info[episode]['actions']
            if self.value_measure:
                values = self.info[episode]['values']
                self.importance = np.amax(values, 1) - np.amin(values, 1)
            else:
                self.importance = self.info[episode]['entropy']
            
            self.cur_episode_len = len(self.actions)

            min_importance, replace_ind = self.calc_min_importance()

            for t in range(self.cur_episode_len):

                if len(curr_trajectory) == self.l:
                    curr_trajectory.pop(0)
                
                curr_trajectory.append((self.states[t], self.actions[t], self.importance[t], t))

                if self.value_measure:
                    if self.importance[t] > min_importance:
                        critical = True
                    else:
                        critical = False
                else:
                    if self.importance[t] < min_importance:
                        critical = True
                    else:
                        critical = False


                if critical:
                    
                    dist, ind = self.closest_traj(t)
                    if dist < self.min_dist:
                        
                        if self.value_measure and self.importance[t] > self.trajectories['trajectories'][ind][self.trajectories['critical_indices'][ind]][2]:
                            pre, post, crit = self.get_surrounding_traj(curr_trajectory)
                            replace_ind = ind
                            self.add_trajectory(pre, post, crit, replace_ind)
                        elif not self.value_measure and self.importance[t] < self.trajectories['trajectories'][ind][self.trajectories['critical_indices'][ind]][2]:
                            pre, post, crit = self.get_surrounding_traj(curr_trajectory)
                            replace_ind = ind
                            self.add_trajectory(pre, post, crit, replace_ind)
                    
                    else:
                        pre, post, crit = self.get_surrounding_traj(curr_trajectory)
                        self.add_trajectory(pre, post, crit, replace_ind)
                    
                    min_importance, replace_ind = self.calc_min_importance()
        
        return self.trajectories



    def closest_traj(self, t):
        dist = float('inf')
        ind = None
        for i, traj in enumerate(self.trajectories['trajectories']):
            crit_ind = self.trajectories['critical_times'][i]
            if abs(crit_ind - t) < dist:
                dist = abs(crit_ind - t)
                ind = i
        
        return dist, ind
    
    def calc_min_importance(self):
        
        if self.value_measure:
            min_importance = float('inf')
        else:
            min_importance = -1 * float('inf')
        
        ind = None

        if len(self.trajectories['trajectories']) == self.k:
            for t, traj in enumerate(self.trajectories['trajectories']):
                critical_index = self.trajectories['critical_indices'][t]
                max_traj_importance = traj[critical_index][2]
                if self.value_measure:
                    if max_traj_importance < min_importance:
                        min_importance = max_traj_importance
                        ind = t
                else:
                    if max_traj_importance > min_importance:
                        min_importance = max_traj_importance
                        ind = t
        
        if len(self.trajectories['trajectories']) < self.k:
            min_importance = -1 * min_importance
        
        return min_importance, ind


    def get_surrounding_traj(self, trajectory):

        critical_step = trajectory[-1][3]
        prev_index = critical_step - self.pre_states
        post_index = critical_step + self.post_states
        if prev_index < 0:
            diff = -1 * prev_index
            prev_index = 0
            post_index = post_index + diff
        
        if post_index >= self.cur_episode_len:
            diff = post_index - self.cur_episode_len + 1
            post_index = self.cur_episode_len - 1
            prev_index = prev_index - diff
        
        assert post_index - prev_index + 1 == self.l, 'Correct trajectory length cannot be formed. Adjust parameters l and post_states. '

        return prev_index, post_index, critical_step


    def add_trajectory(self, pre, post, crit, ind_to_replace=None):
        
        indices = np.arange(pre, post+1)
        traj = [(self.states[i], self.actions[i], self.importance[i], i) for i in indices]
        if ind_to_replace is None:
            self.trajectories['trajectories'].append(traj)
            self.trajectories['critical_times'].append(crit)
            self.trajectories['critical_indices'].append(crit-pre)
        else:
            self.trajectories['trajectories'][ind_to_replace] = traj
            self.trajectories['critical_times'][ind_to_replace] = crit
            self.trajectories['critical_indices'][ind_to_replace] = crit-pre
        

if __name__ == '__main__':
    
    info = []
    for i in range(3):
        sample_states = np.random.uniform(0, 1, [100, 5])
        sample_actions = np.random.randint(0, 4, [100])
        sample_values = np.random.uniform(0, 1, [100, 4])
        info.append({'states': sample_states, 'actions': sample_actions, 'values': sample_values})
    
    hlights = Highlights(info, num_episodes=3)
    trajectories = hlights.compute_trajectories()
    print(trajectories['critical_times'])



                