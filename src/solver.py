import numpy as np
import gurobipy as gp
import re

class Cut:
    def __init__(self, val, grad, x_center):
        self.val = val
        self.grad = grad
        self.x_center = x_center
        self.last_iter = 0
        self.num_uses = 0

class UnderApproxValue:
    """ Underapproximation of value function """

    def __init__(self, n):
        """ @n is dim of variable/gradient 
        
        Args:
            n (int): dimension of variable/gradient
        """
        # Custom datatype for store cuts
        self.cuts = np.array([], dtype=Cut)


    def add_cut(self, avg_val, avg_grad, x_center):
        """ Given N solutions, construct cut 
        
        Args:
            avg_val (float): average value of N solutions
            avg_grad (np.array): average gradient of N solutions
            x_center (np.array): average state of N solutions
        """
        new_cut = Cut(avg_val, avg_grad, x_center)
        self.cuts = np.append(self.cuts, new_cut)

    def last_N_created_cuts(self, N=-1):
        """ Returns @N (N=-1 means all) most recently created cuts

        Args:
            N (int): number of cuts to return

        Returns:
            num_cuts (int): number of cuts returned
            val_arr (np.array): array of values
            grad_arr (np.array): array of gradients
            x_center_arr (np.array): array of states
        """
        num_cuts = len(self.cuts)

        if num_cuts == 0:
            return 0, None, None, None

        if N==-1: 
            N = num_cuts

        val_arr = [self.cuts[i].val for i in range(num_cuts-N, num_cuts)]
        grad_arr = [self.cuts[i].grad for i in range(num_cuts-N, num_cuts)]
        x_center_arr = [self.cuts[i].x_center for i in range(num_cuts-N, num_cuts)]

        return len(val_arr), val_arr, grad_arr, x_center_arr

    def get_num_cuts(self):
        return len(self.cuts)

class LowerBoundModel:
    def __init__(self, n, initial_lb=None):
        """ Creates lower bound model

        Args:
            n (int): dimension of variable
            initial_lb (float): initial lower bound. If None, does not set initial LB
        """
        self.md = gp.Model("lower_bound")
        self.md.setParam('OutputFlag', 0)
        self.n = n
        self.x = self.md.addMVar(n, lb=-float('inf'), name="x_lb")
        self.t = self.md.addMVar(1, lb=-float('inf'), name="t_lb")
        self.md.setObjective(self.t, gp.GRB.MINIMIZE)
        self.is_bounded = False

        # Gurobi naming conventions is different for n=1
        if n == 1:
            self.md.addConstr(self.x == np.zeros(n), name="dummy[0]")
        else:
            self.md.addConstr(self.x == np.zeros(n), name="dummy")

        if initial_lb is not None:
            self.md.addConstr(self.t >= initial_lb, name="initial_lb")
            self.is_bounded = True

        self.md.update()

    def add_cut(self, val, grad, x_center):
        self.md.addConstr(self.t - grad @ self.x >= val - grad @ x_center)
        self.is_bounded = True

    def evaluate_lb(self, x):
        if not self.is_bounded:
            print("Lower bound model is not bounded, returning -inf")
            return -float('inf')
        else:
            for i in range(self.n):
                self.md.setAttr(
                    "RHS", 
                    self.md.getConstrByName("dummy[{}]".format(i)), 
                    x[i]
                )
            self.md.optimize()
            return self.md.objVal

class UpperBoundModel:
    def __init__(self, md, scenarios, lam, M_0, now_var_name_arr, max_val):
        """ Creates upper bound model 

        Args:
            md (gurobipy.Model): model of original problem
            scenarios (list): list of scenarios
            lam (float): discount factor
            M_0 (float): Lipschitz value
            now_var_names (list): list of names of now state variables in model
            max_val (float): average maximum value across scenarios
        """
        md.update() # update before copying
        self.md_1 = md.copy()
        self.md_1.setParam('OutputFlag', 0)

        now_vars_list = [self.md_1.getVarByName(vname) for vname in now_var_name_arr]
        self.now_vars = gp.MVar(now_vars_list)
        self.n = len(now_var_name_arr)
        self.scenarios = scenarios
        self.N = len(self.scenarios)
        self.lam = lam
        self.V_0 = 1/(1-self.lam) * max_val
        self.M_0 = M_0

        # self.hat_vs contains all hat_v's across iterations and 
        # scenarios. We store it naively in a 1D array, where hat v's
        # in the same iteration (but different scenarios) are stored
        # in `N` consecutive elements. 
        # TODO: Better data structure
        self.hat_vs = np.array([])

        # e's and f's 
        self.es = []
        self.fs = []
        # add new variables and include objective coeffs 
        for l in range(self.N):
            self.es.append(self.md_1.addMVar(
                self.n, obj=self.lam*self.M_0/self.N, lb=0, name="e_{}".format(l)
            ))
            self.fs.append(self.md_1.addMVar(
                self.n, obj=self.lam*self.M_0/self.N, lb=0, name="f_{}".format(l)
            ))

        # affine span constraint
        self.affine_span_constrs = []
        self.sum_pi_constrs = []
        for l in range(self.N):
            new_affine_constr = self.md_1.addConstr(
                -self.now_vars + (self.es[l] - self.fs[l]) == np.zeros(self.n)
            )
            self.affine_span_constrs.append(new_affine_constr)

        # second model
        self.md_2 = gp.Model("upper_bound2")
        self.md_2.setParam('OutputFlag', 0)
        self.mu = self.md_2.addMVar(1, name="mu", lb=-float("inf"))
        self.rho = self.md_2.addMVar(self.n, name="rho", lb=-float("inf"))
        self.md_2.addConstrs(self.rho[i] <= M_0 for i in range(self.n))
        self.md_2.addConstrs(-self.rho[i] <= M_0 for i in range(self.n))
        self.md_2_cut_constr_arr = []
        self.num_iters = 0
        # new objective each time (relies on input `x`), so we set it on the fly

    def _solve_model_1_and_update_hat_vs(self, x):
        """ 
        Solves model 1 for all scenarios. Updates self.hat_vs with new solutions.
        This function is called from `add_search_point_to_ub_model`.

        Args:
            x (np.array): previous search point
        """
        new_hat_vs = np.array([])
        for ver in range(self.N):
            for i in range(self.n):
                # set x^{k-1}
                self.md_1.setAttr(
                    "RHS", 
                    self.md_1.getConstrByName("dummy[{}]".format(i)), 
                    x[i]
                )

            # RHS
            for i in range(len(self.scenarios[ver])):
                self.md_1.setAttr(
                    "RHS", 
                    self.md_1.getConstrByName("rand[{}]".format(i)),
                    self.scenarios[ver][i]
                )

            self.md_1.optimize()
            if self.num_iters == 0: # replace penalty (M_0) with V_0 cost to go
                x_sol = self.now_vars.X
                new_hat_vs = np.append(new_hat_vs, 
                    self.md_1.objVal 
                    + self.lam * self.V_0 
                    - self.lam * self.M_0 * np.sum(np.abs(x_sol)))
            else:
                new_hat_vs = np.append(new_hat_vs, self.md_1.objVal)

        # form one big array
        self.hat_vs = np.append(self.hat_vs, new_hat_vs)

    def add_search_point_to_ub_model(self, x):
        """ 
        Adds search point to upper bound model. First processes search point.
        To be called at the end of each iteration of DDP.

        Args:
            x (np.array): previous search point
        """
        # Compute \hat{v} (aka, solve Model 1)
        self._solve_model_1_and_update_hat_vs(x)

        # Update model 1: add new pi_i^{k-1} for each scenario {i}
        # Recall self.hat_vs stored hat_v's across iterations and scenarios.
        # Values in same iteration are grouped together in consecutive N elements,
        # hence, we offset by `N * num_iters` to dtermine which group of N
        # elemennts to use. 
        one_appended_with_x = np.append(1, x).tolist()
        self.md_1.update()
        for l in range(self.N): # (l == ver)
            # TODO Can we make this one call rather than if/else? Reason we need if/else
            # is because initializing constraint with sum pi = 1 (when pi doesn't exist
            # yet) leads to a wear initial constraint of 0 <= -1...
            if self.num_iters == 0:
                col_l = gp.Column(x, self.affine_span_constrs[l].tolist())
                new_pi_var = self.md_1.addVar(
                    lb=0,
                    obj=(self.lam/self.N) * self.hat_vs[self.N * self.num_iters + l], 
                    name="pi_{}^{}".format(l, self.num_iters), 
                    column=col_l,
                )

                # add sum pi = 1 constraint
                # for l in range(self.N):
                self.sum_pi_constrs.append(
                    self.md_1.addConstr(
                        new_pi_var == 1, 
                        name="sum_pi[{}]".format(l)
                    )
                )

            else:
                constr_list_l = [self.sum_pi_constrs[l]] \
                                + self.affine_span_constrs[l].tolist()
                col_l = gp.Column(one_appended_with_x, constr_list_l)
                self.md_1.addVar(
                    lb=0,
                    obj=(self.lam/self.N) * self.hat_vs[self.N * self.num_iters + l], 
                    name="pi_{}^{}".format(l, self.num_iters), 
                    column=col_l,
                )

        # TODO: How can we update this without looping?
        # self.md.setAttr("RHS", self.md.getConstrByName("dummy"), x_prev)

        # Update Model 2: Add a new constraint. RHS of 0 is placeholder value
        self.md_2.addConstr(
            self.mu + x @ self.rho <= 0, 
            name="md_2_cut_constr[{}]".format(self.num_iters)
        )

        self.md_2.update()

        self.num_iters += 1

    def evaluate_ub(self, x):
        """ Evaluates upper bound model at search point `x`

        Args:
            x (np.array): search point

        Returns:
            float: upper bound
        """
        if self.num_iters == 0:
            return self.V_0

        ub_model_sum = 0

        self.md_2.setObjective(self.mu + self.rho @ x, gp.GRB.MAXIMIZE)
        self.md_2.update()
        # Update hat v's to appropriate scenario
        for ver in range(1, self.N+1):
            for k in range(self.num_iters):
                constr_ptr = self.md_2.getConstrByName(
                    "md_2_cut_constr[{}]".format(k)
                )
                self.md_2.setAttr(
                    "RHS", 
                    constr_ptr,
                    self.hat_vs[(ver-1) + k*self.N],
                )

            self.md_2.optimize()

            ub_model_sum += self.md_2.objVal

        return ub_model_sum / self.N

class GurobiSolver:
    # TODO: Allow multiple names for now and past state variables
    def __init__(
        self, 
        md, 
        x, 
        var_names,
        lam, 
        scenarios,
        min_val = 0,
        max_val = 0,
        past_state_for_min_val=None,
        past_state_for_max_val=None,
    ): 
        """ Adds cost-to-go-function. Defaults to minimize 
        
        Args:
            md (gurobipy.Model): model
            x (gurobipy.Var): variable
            var_names (list): list of variable names 
            lam (float): discount factor
            scenarios (np.array): array of scenarios 
            min_val (float): minimum value of cost-to-go function
            max_val (float): maximum value of cost-to-go function
            past_state_for_min_val (str): name of past state variable for min value. Will override min_val
            past_state_for_max_val (str): name of past state variable for max value. Will override max_val
        """

        md.setParam('OutputFlag', 0)
        md.update()
        self.md_copy = md.copy()

        self.md = md
        self.x = x
        self.var_names = var_names
        self.n = len(x.tolist())
        self.N = len(scenarios)
        self.scenarios = scenarios
        self.cost_to_go_initialized = False
        self.solve_ct = 0

        # compute minimum and maximum value
        if past_state_for_min_val is not None:
            min_val = 0
            for i in range(1,self.N):
                [_, val, _, _] = self.solve(past_state_for_min_val, i)
                min_val += val
            min_val = min_val / self.N

        if past_state_for_max_val is not None:
            max_val = 0
            # convert to maximization 
            # self.md.setAttr(gp.GRB.Attr.ModelSense, -1)
            for i in range(1,self.N):
                [_, val, _, _] = self.solve(past_state_for_max_val, i)
                max_val += val
            max_val = max_val / self.N
            # revert to minimization
            # self.md.setAttr(gp.GRB.Attr.ModelSense, 1)

        # add cost to go
        self.t = self.md.addMVar(1, lb=-float("inf"), obj=lam, name="cost_to_go")
        self.md.addConstr(self.t >= min_val/(1-lam))

        self.cost_to_go_initialized = True
        self.solve_ct = 0

        self.md.update()

        self.min_val = min_val
        self.max_val = max_val

        print(">> min_val={:.2e} max_val={:.2e}".format(min_val, max_val))

    def get_gurobi_model(self):
        return self.md_copy

    def get_scenarios(self):
        return self.scenarios

    def get_var_names(self):
        return self.var_names

    def get_extrema_values(self):
        """
        Returns the estimated min and max values computed during construction

        Returns:
            min_val (float): mean of minimum values
            max_val (float): mean of maximum values
        """
        return [self.min_val, self.max_val]

    def get_num_scenarios(self):
        return len(self.scenarios)  

    def solve(self, x_prev, ver):
        """ Solves LP by adding most recent cut 

        Args:
            x_prev (np.array): past state
            ver (int): scenario version

        Returns:
            x_sol (np.array): solution
            val (float): value of solution
            grad (np.array): gradient of solution
            ctg (float): cost-to-go
        """

        # Setup past state variable
        for i in range(self.n):
            self.md.setAttr("RHS", 
                    self.md.getConstrByName("dummy[{}]".format(i)), 
                    x_prev[i])
        # TODO: How can we update this without looping?
        # self.md.setAttr("RHS", self.md.getConstrByName("dummy"), x_prev)

        # Setup uncertainity RHS
        for i in range(len(self.scenarios[ver])):
            self.md.setAttr("RHS", 
                    self.md.getConstrByName("rand[{}]".format(i)),
                    self.scenarios[ver][i])

        # Solve
        self.md.optimize()

        # check feasible
        if self.md.status != gp.GRB.OPTIMAL:
            raise Exception("LP did not terminate as optimal")

        # Get solution
        x_sol = self.x.X
        y_sol_dummy = np.array([self.md.getAttr("Pi", 
            [self.md.getConstrByName("dummy[{}]".format(i))]) 
            for i in range(self.n)])
        y_sol_rand = np.array([self.md.getAttr("Pi", 
            [self.md.getConstrByName("rand[{}]".format(i))]) 
            for i in range(len(self.scenarios[ver]))])
        y_sol = y_sol_dummy
        # y_sol = y_sol_rand
        # y_sol = np.zeros(len(y_sol_dummy))
        # y_sol = y_sol_rand # - y_sol_dummy
        # y_sol = y_sol_dummy
        val = self.md.objVal # getObjective().getValue()

        grad = y_sol 

        self.solve_ct += 1
        self.md.update()

        if self.cost_to_go_initialized:
            ctg = self.t.X[0]
        else:
            ctg = 0

        return [x_sol, val, grad, ctg]

    def add_newest_cut(self, x_prev, under_V):

        # obtain most recent cuts and add as constraint
        num_cuts, cut_val, cut_grad, cut_x = under_V.last_N_created_cuts(1)

        # if we have any cuts
        if num_cuts > 0:
            cut_val = cut_val[0]
            cut_grad = cut_grad[0]
            cut_x = cut_x[0]

            self.md.addConstr(self.t - cut_grad@self.x >= cut_val-cut_grad@x_prev)

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

    def largest_sat_lvl(self, agg_x):
        """ Gets highest saturation lvl (and index) 
        
        Args:
            agg_x (np.array): aggregate points
        
        Returns:
            dist_x (np.array): point with highest saturation level
            sat_lvls[argmax_idx] (np.array): saturation level 
            argmax_idx (int): index of point with highest saturation level
        """
        sat_lvls = np.array([self.get(agg_x[i]) for i in range(len(agg_x))])
        # randomize over ties (proritize x_0)
        if self.get(agg_x[0]) == np.max(sat_lvls):
            argmax_idx = 0
        else:
            argmax_idx = np.random.choice(np.flatnonzero(sat_lvls == np.max(sat_lvls)))
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

