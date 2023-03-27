import numpy as np

class SaturatedSet:

    def __init__(self, params):
        self.dim = params['n']
        self.L = params['L']
        self.R = params['R']
        eps = params['eps']

        self.W = self.R - self.L # max width of domain
        assert np.amin(self.W) > 0
        self.D = np.amax(self.W)
        self.num_parts = np.int(np.floor(self.D/eps + 1))
        self.S = {}
        self.T = params['T']
        # self.S = (params['T']-1)*np.ones((self.num_parts,)*n, dtype=np.ushort)

    def get_idx(self, x):
        """ Returns index of point @x
        
        Args:
            x (np.array): point
        """
        return np.floor((x-self.L)/(self.W) * self.num_parts).astype('int')

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

    # TODO: Can we get rid of the two functions below
    def smallest_sat_lvl(self, agg_x):
        """ Gets lowest saturation lvl (and index) 
        
        Args:
            agg_x (np.array): aggregate points
        
        Returns:
            dist_x (np.array): point with highest saturation level
            sat_lvls[argmax_idx] (np.array): saturation level 
        """
        sat_lvls = np.array([self.get(agg_x[i]) for i in range(len(agg_x))])
        argmin_idx = np.argmin(sat_lvls)
        dist_x = agg_x[argmin_idx]
        return dist_x, sat_lvls[argmin_idx]

    def rand_sat_lvl(self, agg_x):
        """ Gets random saturation lvl (and index)

        Args:
            agg_x (np.array): aggregate points

        Returns:
            dist_x (np.array): point with highest saturation level
            sat_lvls[argmax_idx] (np.array): saturation level
        """
        i = np.random.randint(low=0, high=len(agg_x))
        return agg_x[i], self.get(agg_x[i])

