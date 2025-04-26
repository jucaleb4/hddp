import os
import numpy as np
from enum import IntEnum

def save_logs(folder, k, total_time_arr, fwd_time_arr, select_time_arr, 
              eval_time_arr, comm_time_arr, lb_arr, ub_arr, scen_arr):

    all_times_arr = np.zeros((k, 5), dtype=float)
    all_times_arr[:,0] = total_time_arr[:k]
    all_times_arr[:,1] = fwd_time_arr[:k]
    all_times_arr[:,2] = select_time_arr[:k]
    all_times_arr[:,3] = eval_time_arr[:k]
    all_times_arr[:,4] = comm_time_arr[:k]
    fname = os.path.join(folder, "elpsed_times.csv")
    np.savetxt(fname, all_times_arr, delimiter=',', header='total_time,fwd_time,select_time,eval_time,comm_time')

    all_bounds_arr = np.hstack((np.atleast_2d(lb_arr[:k]).T, np.atleast_2d(ub_arr[:k]).T))
    fname = os.path.join(folder, "bounds.csv")
    np.savetxt(fname, all_bounds_arr, delimiter=',', header='lower,upper')

    fname = os.path.join(folder, "scenarios.csv")
    np.savetxt(fname, np.atleast_2d(scen_arr[:k]).T, delimiter=',', header='lower,upper')

class Mode(IntEnum):
    INF_EDDP = 0
    CE_INF_EDDP = 1
    GAP_INF_EDDP = 2
    INF_SDDP = 3
    EDDP = 4
    GCE_INF_EDDP = 5
    SDDP = 6
    G_INF_SDDP = 7
    P_SDDP = 8

class SaturatedSet:

    def __init__(self, n, x_lb_arr, x_ub_arr, eps, T, **kwargs):
        self.dim = n
        self.x_lb_arr = np.array(x_lb_arr)
        self.x_ub_arr = np.array(x_ub_arr)

        self.x_range_arr = self.x_ub_arr - self.x_lb_arr # max width of domain
        assert np.amin(self.x_range_arr) > 0
        self.D = np.amax(self.x_range_arr)
        self.num_parts = int(np.floor(self.D/eps + 1))
        self.S = {}
        self.T = T
        # self.S = (params['T']-1)*np.ones((self.num_parts,)*n, dtype=np.ushort)

    def get_idx(self, x):
        """ Returns index of point @x
        
        Args:
            x (np.array): point
        """
        return np.floor((x-self.x_lb_arr)/(self.x_range_arr) * self.num_parts).astype('int')

    def get(self, x):
        """ Returns saturation level of point @x 
        
        Args:
            x (np.array): point

        Returns:
            sat_lvl (int): saturation level
        """
        idx_arr = self.get_idx(x)
        idx = ','.join(idx_arr.astype('str'))

        if idx not in self.S.keys():
            self.S[idx] = self.T-1
        sat_lvl = self.S[idx]

        return sat_lvl

    def update(self, x, lvl):
        """ Updates saturation level of point @x
        
        Args:
            x (np.array): point
            lvl (int): saturation level
        """
        idx_arr = self.get_idx(x)
        idx = ','.join(idx_arr.astype('str'))
        self.S[idx] = lvl

    def largest_sat_lvl(self, agg_x, rng, prioritize_zero):
        """ Gets highest saturation lvl (and index) 
        
        Args:
            agg_x (np.array): aggregate points
            prioritize_zero (bool): select 0 if it is argmin
        
        Returns:
            dist_x (np.array): point with highest saturation level
            sat_lvls[argmax_idx] (np.array): saturation level 
            argmax_idx (int): index of point with highest saturation level
        """
        sat_lvls = np.array([self.get(agg_x[i]) for i in range(len(agg_x))])
        # randomize over ties (proritize x_0)
        if prioritize_zero and self.get(agg_x[0]) == np.max(sat_lvls):
            argmax_idx = 0
        else:
            argmax_idx = rng.choice(np.flatnonzero(sat_lvls == np.max(sat_lvls)))
        # argmax_idx = np.argmax(sat_lvls)
        dist_x = agg_x[argmax_idx]
        return dist_x, sat_lvls[argmax_idx], argmax_idx
