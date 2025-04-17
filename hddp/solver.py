import numpy as np
import gurobipy as gp
import re
import os
import numpy.linalg as la
import time

from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional

def get_dual_proj_bounds_from_sense(sense):
    dt_lb = {"=": -np.inf, ">": -np.inf, "<": 0}
    dt_ub = {"=": np.inf, ">": 0, "<": np.inf}
    # dt_lb = {"=": -np.inf, "<": -np.inf, ">": 0}
    # dt_ub = {"=": np.inf, "<": 0, ">": np.inf}
    lb_arr = np.array([dt_lb[s] for s in sense])
    ub_arr = np.array([dt_ub[s] for s in sense])
    return lb_arr, ub_arr

class GenericSolver(ABC):
    """
    Generic solver. Solver must implement certain methods to be used in the
    hierarchical dual dynamic programming (HDDP) algorithm. 
    """

    def __init__(self, M_h, x_lb_arr, x_ub_arr, h_min_val, h_max_val):
        """ 
        :param M_h: single stage Lipschitz value estimate
        :param x_lb_arr: lower bound on x values
        :param x_ub_arr: upper bound on x values
        :param h_min_val: min of single stage cost h
        :param h_max_val: max of single stage cost h
        """
        self.M_h = M_h
        self.x_lb_arr = x_lb_arr
        self.x_ub_arr = x_ub_arr
        self.h_min_val = h_min_val
        self.h_max_val = h_max_val

        n = len(x_lb_arr)
        m = 1024
        self.ct = 0
        self.val_arr = np.zeros(m, dtype=float)
        self.grad_arr = np.zeros((m,n), dtype=float)
        self.x_prev_arr = np.zeros((m,n), dtype=float)
        self.time_arr = np.zeros(m, dtype=float)
        self.s_time = time.time()
    
    @abstractmethod
    def solve(self, x_prev: np.ndarray, scenario_id: int) \
            -> Tuple[np.ndarray, float, np.ndarray, float]:
        """ Solves subproblem @scenario_id using @x_prev as the previous point.
        Returns a cut, or an (approximately) optimal objective value, primal,
        dual variable, and cost to go.

        :returns x_sol: optimal primal solution
        :returns val: optimal solution value
        :returns grad: subgradient
        :return ctg: cost to go
        """
        pass

    @abstractmethod
    def add_cut(self, val, grad, x_prev) -> None:
        raise NotImplementedError

    def load_cuts(self, val_arr, grad_arr, x_prev_arr) -> None:
        """ Loads multiple cuts at once. """
        self.val_arr = val_arr
        self.grad_arr = grad_arr
        self.x_prev_arr = x_prev_arr
        self.ct = len(self.x_prev_arr)
        self.time_arr = np.zeros(self.ct)

    def save_cut(self, val, grad, x_prev) -> None:
        """ Adds cut to cost-to-go function, in the form of
              l_f(x) := val + <grad, x - x_prev>
        """
        if self.ct == len(self.val_arr):
            self.val_arr = np.append(self.val_arr, np.zeros(self.val_arr.shape, dtype=float))
            self.grad_arr = np.vstack((self.grad_arr, np.zeros(self.grad_arr.shape, dtype=float)))
            self.x_prev_arr = np.vstack((self.x_prev_arr, np.zeros(self.x_prev_arr.shape, dtype=float)))
            self.time_arr = np.append(self.time_arr, np.zeros(self.val_arr.shape, dtype=float))

        self.val_arr[self.ct] = val
        self.grad_arr[self.ct] = grad
        self.x_prev_arr[self.ct] = x_prev
        self.time_arr[self.ct] = time.time() - self.s_time
        self.ct += 1

    def save_cuts_to_file(self, folder):
        np.savetxt(os.path.join(folder, "vals.csv"), self.val_arr[:self.ct], delimiter=',')
        np.savetxt(os.path.join(folder, "grad.csv"), self.grad_arr[:self.ct], delimiter=',')
        np.savetxt(os.path.join(folder, "x_prev.csv"), self.x_prev_arr[:self.ct], delimiter=',')
        np.savetxt(os.path.join(folder, "time.csv"), self.time_arr[:self.ct], delimiter=',')

    def get_scenarios(self):
        raise NotImplementedError

    def set_scenario_idx(self, i):
        pass

    def get_single_stage_lbub(self) -> Tuple[float, float]:
        """
        Returns estimate of min/max value of single stage cost
        """
        raise NotImplementedError

    def get_num_scenarios(self):
        raise NotImplementedError

    def get_gurobi_model(self):
        raise NotImplementedError

    def get_var_names(self):
        raise NotImplementedError

class EmptySolver(GenericSolver):
    """ Solver that returns zero value and gradient """
    def __init__(self, n):
        self.n = n

    def solve(self, x_prev, ver):
        return np.zeros(self.n), 0, np.zeros(self.n), 0

    def add_cut(self, val, grad, x_prev):
        pass

    def load_cuts(self, val_arr, grad_arr, x_prev_arr):
        pass

class GurobiSolver(GenericSolver):
    def __init__(
        self, 
        model, 
        x, 
        var_names,
        lam, 
        scenarios,
        M_h,
        x_lb_arr,
        x_ub_arr,
        h_min_val = 0,
        h_max_val = 0,
        past_state_for_min_val=None,
        past_state_for_max_val=None,
    ): 
        """ Makes copy of model and adds cost to go function
        
        :param model (gurobipy.Model): model
        :param x (gurobipy.Var): variable
        :param var_names (list): list of variable names 
        :param lam (float): discount factor
        :param scenarios (np.array): array of scenarios 
        :param min_val (float): minimum value of cost-to-go function
        :param max_val (float): maximum value of cost-to-go function
        :param past_state_for_min_val (str): name of past state variable for min value. Will override min_val
        :param past_state_for_max_val (str): name of past state variable for max value. Will override max_val
        """
        super().__init__(M_h, x_lb_arr, x_ub_arr, 0, 0)
        model.setParam('OutputFlag', 0)
        model.update()
        self.model_copy = model.copy()

        self.model = model
        self.x = x
        self.var_names = var_names
        self.n = len(x.tolist())
        self.N = len(scenarios)
        self.scenarios = scenarios
        self.cost_to_go_initialized = False
        self.solve_ct = 0

        # compute minimum and maximum value
        if past_state_for_min_val is not None:
            h_min_val = 0
            for i in range(1,self.N):
                [_, val, _, _] = self.solve(past_state_for_min_val, i)
                h_min_val += val
            h_min_val = h_min_val / self.N

        if past_state_for_max_val is not None:
            h_max_val = 0
            # convert to maximization 
            # self.model.setAttr(gp.GRB.Attr.ModelSense, -1)
            for i in range(1,self.N):
                [_, val, _, _] = self.solve(past_state_for_max_val, i)
                h_max_val += val
            h_max_val = h_max_val / self.N
            # revert to minimization
            # self.model.setAttr(gp.GRB.Attr.ModelSense, 1)

        # add cost to go
        self.t = self.model.addMVar(1, lb=-float("inf"), obj=lam, name="cost_to_go")
        self.model.addConstr(self.t >= h_min_val/(1.-lam))

        self.cost_to_go_initialized = True
        self.solve_ct = 0

        self.model.update()

        self.h_min_val = h_min_val
        self.h_max_val = h_max_val

        # TODO: Add test to check `self.model.getConstrByName("dummy[{}]".format(i))`  does not return None (i.e., setup Gurobi problem right)
        # TODO: Check x is MVar type (and not numpy array) 

    def get_gurobi_model(self):
        return self.model_copy

    def get_scenarios(self):
        return self.scenarios

    def get_var_names(self):
        return self.var_names

    def get_single_stage_lbub(self):
        """
        :return h_min_val: minimum value of single stage cost
        :return h_max_val: maximum value of single stage cost
        """
        return [self.h_min_val, self.h_max_val]

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
            self.model.setAttr("RHS", 
                    self.model.getConstrByName("dummy[{}]".format(i)), 
                    x_prev[i])
        # TODO: How can we update this without looping?
        # self.model.setAttr("RHS", self.model.getConstrByName("dummy"), x_prev)

        # Setup uncertainity RHS
        for i in range(len(self.scenarios[ver])):
            self.model.setAttr("RHS", 
                    self.model.getConstrByName("rand[{}]".format(i)),
                    self.scenarios[ver][i])

        # Solve
        self.model.optimize()

        # check feasible
        if self.model.status != gp.GRB.OPTIMAL:
            raise Exception("LP did not terminate as optimal")

        # Get solution
        x_sol = self.x.X
        y_sol_dummy = np.array([self.model.getAttr("Pi", 
            [self.model.getConstrByName("dummy[{}]".format(i))]) 
            for i in range(self.n)])
        # y_sol_rand = np.array([self.model.getAttr("Pi", 
        #     [self.model.getConstrByName("rand[{}]".format(i))]) 
        #     for i in range(len(self.scenarios[ver]))])
        y_sol = y_sol_dummy
        # y_sol = y_sol_rand
        # y_sol = np.zeros(len(y_sol_dummy))
        # y_sol = y_sol_rand # - y_sol_dummy
        # y_sol = y_sol_dummy
        val = self.model.objVal # getObjective().getValue()

        grad = y_sol 

        self.solve_ct += 1
        self.model.update()

        if self.cost_to_go_initialized:
            ctg = self.t.X[0]
        else:
            ctg = 0

        return [x_sol, val, grad, ctg]

    def add_cut(self, val, grad, x_prev):
        self.save_cut(val, grad, x_prev)
        self.model.addConstr(self.t - grad@self.x >= val-grad@x_prev)

    def load_cuts(self, val_arr, grad_arr, x_prev_arr):
        super().load_cuts(val_arr, grad_arr, x_prev_arr)

        # TODO: This may not work
        self.model.addConstr(self.t - grad_arr@self.x >= val_arr-np.diag(grad_arr@x_prev_arr.T))

class PDSASolverForLPs(GenericSolver):
    """ 
    Primal dual stochastic approximation solver for solving 2-stage SP.

    :params lam: discount factor (for ctg)
    :params (c1,A1,b1,B2): data for (min c'x : A1x + B2u [sense1_arr]  b1)
    :params (lb1_arr,ub1_arr): lower and upper bounds on x
    :params sense1_arr: constraint sense (=, >=, <=)
    :params scenarios: 2d array, where i-th row is for scenario i
    :params rand1_idx_arr: subset of rhs rows corresponding to data in scenario
    :params (B2, lb2_arr, ub2_arr, sense2_arr): similar to stage 1
    :params get_stochastic_lp2_params: function that returns random data (c2,A2,b2) for stage 2
    :params k1,k2: number of iterations to run for first stage and second stage
    :params eta1_scale, tau1_scale: step size scaling factor for stage 1
    :params eta2_scale: (also tau2_scale) step size scaling factor for stage 2
    """
    def __init__(self, lam,
            c1, A1, b1, B1, state1_idx_arr, x1_bnd_arr, sense1_arr, scenarios, rand1_idx_arr, # first-stage
            get_stochastic_lp2_params, B2, x2_bnd_arr, sense2_arr, # second-stage
            k1, k2, eta1_scale, tau1_scale, eta2_scale, has_ctg, ctg_bnds, # hyperparameters
    ): 
        # ignore non-state variables
        M_h = la.norm(c1)
        x_lb_arr = x1_bnd_arr[0][state1_idx_arr]
        x_ub_arr = x1_bnd_arr[1][state1_idx_arr]
        super().__init__(M_h, x_lb_arr, x_ub_arr, 0, 0)

        # save first-stage settings
        self.A1 = np.asarray(A1)
        self.B1 = B1
        self.c1 = c1
        self.b1 = b1
        self.state1_idx_arr = state1_idx_arr
        self.x1_bnd_arr = x1_bnd_arr
        y_lb_arr, y_ub_arr = get_dual_proj_bounds_from_sense(sense1_arr)
        self.y1_bnd_arr = np.vstack((y_lb_arr, y_ub_arr))
        self.scenarios = scenarios
        self.rand1_idx_arr = rand1_idx_arr
        cx_bnds = np.vstack((np.multiply(c1, x1_bnd_arr[0]), np.multiply(c1, x1_bnd_arr[1])))

        # setup some useful attributes
        self.lam = lam
        self.h_min_val = np.sum(np.min(cx_bnds, axis=0))
        self.h_max_val = np.sum(np.max(cx_bnds, axis=0))

        # save second-stage settings
        self.get_stochastic_lp2_params = get_stochastic_lp2_params
        self.B2 = B2
        self.x2_bnd_arr = x2_bnd_arr
        y2_lb_arr, y2_ub_arr = get_dual_proj_bounds_from_sense(sense2_arr)
        self.y2_bnd_arr = np.vstack((y2_lb_arr, y2_ub_arr))

        # append cost to go to cuts (and update variables)
        self.has_ctg = has_ctg
        if has_ctg:
            self.A1 = np.hstack((self.A1, np.zeros((self.A1.shape[0], 1))))
            self.c1 = np.append(self.c1, lam)
            self.B2 = np.hstack((self.B2, np.zeros((self.B2.shape[0], 1))))
            self.x1_bnd_arr = np.hstack((self.x1_bnd_arr, np.atleast_2d(ctg_bnds).T))
            self.ctg_constr_starting_idx = self.A1.shape[0]
            self.ctg_idx = self.A1.shape[1]-1

        # estimate some hyperparameters and set parameters
        M2 = -np.inf
        oB2 = np.max(la.svd(B2, compute_uv=False))
        uA2 = np.inf
        for i in range(10):
            # arbitrary choose initial scenario to estimate
            (c2,A2,_) = get_stochastic_lp2_params()
            uA2 = min(np.min(la.svd(A2, compute_uv=False)), uA2)
            M2  = max(M2, 2*la.norm(c2))
        if uA2 < 1e-2:
            print("uA2 is small (%.4e), manually increasing to 1e-2)" % uA2)
            uA2 = 1e-2
        Omega_1 = np.sqrt(0.5*la.norm(x1_bnd_arr[1] - x1_bnd_arr[0])**2)
        Omega_2 = np.sqrt(0.5*la.norm(x2_bnd_arr[1] - x2_bnd_arr[0])**2)
        barG    = oB2*(2*M2/max(1e-1, uA2) + 2*Omega_2) # assume zero initial dual vector

        self.k1 = k1
        self.k2 = k2
        self.eta1 = eta1_scale * np.sqrt(2)*la.norm(self.A1)
        self.tau1 = max(self.eta1, tau1_scale*np.sqrt(6*k1*barG**2)/Omega_1)
        # TEMP: Simpler dual stepsize
        self.tau1 = self.eta1
        self.eta2_scale = eta2_scale
        self.t = 0

    def get_gurobi_model(self):
        return None

    def get_var_names(self):
        return []

    def get_scenarios(self):
        return self.scenarios

    def set_scenario_idx(self, i):
        self.i = i

    def get_single_stage_lbub(self) -> Tuple[float, float]:
        return self.h_min_val, self.h_max_val

    def solve(self, u, ver, stage=1) -> Tuple[np.ndarray, float, np.ndarray, float]:
        """ Solves subproblem and returns approx. optimal value, primal, and
        dual variable.

        :params u: previous state variable
        :params ver: subproblem id
        :returns objval: computed objective value
        :returns barx: computed primal solution
        :returns subgrad: computed subgradient w.r.t. last state variable
        """
        theta = 1.
        if stage == 1:
            A,B,c,b = self.A1, self.B1, self.c1, self.b1
            x_bnd_arr, y_bnd_arr = self.x1_bnd_arr, self.y1_bnd_arr
            k, eta, tau = self.k1, self.eta1, self.tau1
            for i,j in enumerate(self.rand1_idx_arr):
                b[j] = self.scenarios[ver][i]
            x = np.zeros(A.shape[1])
        elif stage == 2:
            B = self.B2
            c,A,b = self.get_stochastic_lp2_params()
            x_bnd_arr, y_bnd_arr = self.x2_bnd_arr, self.y2_bnd_arr
            k = self.k2
            tau = eta = self.eta2_scale * np.sqrt(2) * la.norm(A) # theory says tau = eta
            x = np.zeros(A.shape[1])
        elif stage == 3:
            return (0, 0, 0, 0)
        else:
            raise Exception("Code only supports 2-stage, received %d" % stage)

        # add dummy variable constraint
        null_ver = 0 # no version for 2nd stage
        b_prime = b - B@u
        y = np.zeros(A.shape[0]) 
        y_prev = y.copy()
        barx = np.zeros(len(x))
        bary = np.zeros(len(y))
        barval = 0.

        for t in range(k):
            # get next stage's subgradient
            [_, val_t, G_t, _] = self.solve(x, null_ver, stage+1)

            d = y + theta * (y - y_prev) 
            x = self.primal_prox(x, d, c, G_t, tau, x_bnd_arr, A)
            y_prev = y
            y = self.dual_prox(y, x, b_prime, eta, y_bnd_arr, A)

            alpha = 1./(t+1)
            barx = (1.-alpha)*barx + alpha*x
            bary = (1.-alpha)*bary + alpha*y
            barval = (1.-alpha)*barval + alpha*val_t

        # fix ctg
        # if self.ct > 0:
        #     x_for_ctg = barx[self.state1_idx_arr]
        #     linear_approx = self.val_arr[:self.ct] + np.einsum("ij,ij->i", self.grad_arr[:self.ct], x_for_ctg - self.x_prev_arr[:self.ct])
        #     correct_ctg = np.min(linear_approx)
        #     barx[self.ctg_idx] = correct_ctg

        # compute final output
        objval = np.dot(c, barx) + barval
        subgrad = B.T.dot(bary)
        ctg = 0
        if stage == 1:
            # for 1-stage problem, we only want primal
            if self.has_ctg:
                ctg = barx[self.ctg_idx]
            barx = barx[self.state1_idx_arr]

        return barx, objval, subgrad, ctg

    def primal_prox(self, x_prev, d, c, G_k, tau, bnd_arr, A):
        # Paper says to use -A'd, but I think we want A'd
        x = x_prev - (1./tau) * (G_k + c + A.T.dot(d))
        x = np.clip(x, bnd_arr[0], bnd_arr[1])
        return x

    def dual_prox(self, y_prev, x, b, eta, bnd_arr, A):
        # Paper says to use "- (1./eta) ...", but I think it should be "+ (1./eta) ..."
        y = y_prev + (1./eta) * (A.dot(x) - b)
        y = np.clip(y, bnd_arr[0], bnd_arr[1])
        return y

    def add_cut(self, val, grad, x_prev):
        # add constraint 
        #   theta >= val + <grad, x-x_prev>
        # New cut means new constraint and a corresponding dual variable 
        # (since >= constraint, then dual cone is < 0, or y in [-inf,0])
        self.save_cut(val, grad, x_prev)

        new_row = np.zeros(self.A1.shape[1])
        new_row[self.state1_idx_arr] = -grad
        new_row[self.ctg_idx] = 1
        self.b1 = np.append(self.b1, val - np.dot(grad, x_prev))
        self.A1 = np.vstack((self.A1, new_row))
        self.B1 = np.vstack((self.B1, np.zeros(self.B1.shape[1])))
        self.y1_bnd_arr = np.hstack((self.y1_bnd_arr, np.array([[-np.inf,0]]).T))

    def load_cuts(self, val_arr, grad_arr, x_prev_arr):
        super().load_cuts(val_arr, grad_arr, x_prev_arr)

        # TODO: Add to A matrix

class PDSAEvalSA(GenericSolver):
    """ 
    Uses SA-type evaluation for PDSA solution.

    :params lam: discount factor (for ctg)
    :params (c1,A1,b1,B2): data for (min c'x : A1x + B2u [sense1_arr]  b1)
    :params (lb1_arr,ub1_arr): lower and upper bounds on x
    :params sense1_arr: constraint sense (=, >=, <=)
    :params scenarios: 2d array, where i-th row is for scenario i
    :params rand1_idx_arr: subset of rhs rows corresponding to data in scenario
    :params (B2, lb2_arr, ub2_arr, sense2_arr): similar to stage 1
    :params get_stochastic_lp2_params: function that returns random data (c2,A2,b2) for stage 2
    :params num_second_stage: number of scenarios to pre-samples for second stage
    """
    def __init__(self, lam, mdl, state1_idx_arr, x1_bnd_arr, scenarios, rand1_idx_arr, dummy1_idx_arr,
            # c1, A1, b1, B1, state1_idx_arr, x1_bnd_arr, sense1_arr, scenarios, rand1_idx_arr, dummy1_idx_arr, # first-stage
            # get_stochastic_lp2_params, B2, x2_bnd_arr, sense2_arr, # second-stage
            # seed, num_second_stage,
    ): 
        M_h = 0. # la.norm(c1)
        x_lb_arr = x1_bnd_arr[0][state1_idx_arr]
        x_ub_arr = x1_bnd_arr[1][state1_idx_arr]
        super().__init__(M_h, x_lb_arr, x_ub_arr, 0, 0)

        self.mdl = mdl
        self.ctg = self.mdl.addVar(lb=-np.inf, obj=lam, name="ctg") 
        self.mdl.setParam('LogToConsole', 0)

        # x1_lb_arr = x1_bnd_arr[0]
        # x1_ub_arr = x1_bnd_arr[1]
        # x2_lb_arr = x2_bnd_arr[0]
        # x2_ub_arr = x2_bnd_arr[1]
        # n_prev = B1.shape[1]
        # n1 = len(x1_lb_arr)
        # n2 = len(x2_lb_arr)
        # self.mdl_arr = []
        # self.x_arr = []
        # self.ctg_arr = []

        # self.rng = np.random.default_rng(seed)
        # self.state1_idx_arr = state1_idx_arr
        self.scenarios = scenarios
        self.state1_idx_arr = state1_idx_arr
        self.dummy1_idx_arr = dummy1_idx_arr
        self.rand1_idx_arr = rand1_idx_arr
        assert self.scenarios.shape[1] == len(self.rand1_idx_arr), \
            "Input scenario dim (%d) does not match rand dim (%d)" % (self.scenarios.shape[1], len(self.rand1_idx_arr))

    def get_gurobi_model(self):
        return None

    def get_var_names(self):
        return []

    def get_scenarios(self):
        return self.scenarios

    def set_scenario_idx(self, i):
        self.i = i

    def get_single_stage_lbub(self) -> Tuple[float, float]:
        return self.h_min_val, self.h_max_val

    def solve(self, x_prev, ver) -> Tuple[np.ndarray, float, np.ndarray, float]:
        assert len(self.dummy1_idx_arr) == len(x_prev), "Input state dim (%d) does not match dummy dim (%d)" % (len(x_prev), len(self.dummy1_idx_arr))

        for i, idx in enumerate(self.dummy1_idx_arr):
            self.mdl.setAttr("RHS", 
                    self.mdl.getConstrs()[idx],
                    x_prev[i])

        # Setup uncertainity RHS
        for i, idx in enumerate(self.rand1_idx_arr):
            self.mdl.setAttr("RHS", 
                    self.mdl.getConstrs()[idx],
                    self.scenarios[ver][i])

        # Solve
        self.mdl.update() 
        self.mdl.optimize()

        if self.mdl.status != gp.GRB.OPTIMAL:
            raise Exception("LP did not terminate as optimal, got %s" % self.mdl.status)

        x_vars = self.mdl.getVars()
        x_sol = np.array([x_vars[i].X for i in self.state1_idx_arr])
        val = self.mdl.objVal # getObjective().getValue()
        ctg = self.ctg.X
        grad = np.zeros(len(x_sol))

        return [x_sol, val, grad, ctg]

    def add_cut(self, val, grad, x_prev):
        self.save_cut(val, grad, x_prev)
        x = gp.MVar.fromlist(self.mdl.getVars())
        self.mdl.addConstr(self.ctg - grad@x[self.state1_idx_arr] >= val-grad@x_prev)

    def load_cuts(self, val_arr, grad_arr, x_prev_arr):
        super().load_cuts(val_arr, grad_arr, x_prev_arr)
        x = gp.MVar.fromlist(self.mdl.getVars())
        self.mdl.addConstr(self.ctg - grad_arr@x[self.state1_idx_arr] >= val_arr-np.einsum('ij,ij->i', grad_arr, x_prev_arr))

class FixedEval(GenericSolver):
    """ 
    Uses SA-type evaluation for PDSA solution.

    :params lam: discount factor (for ctg)
    :params (c1,A1,b1,B2): data for (min c'x : A1x + B2u [sense1_arr]  b1)
    :params (lb1_arr,ub1_arr): lower and upper bounds on x
    :params sense1_arr: constraint sense (=, >=, <=)
    :params scenarios: 2d array, where i-th row is for scenario i
    :params rand1_idx_arr: subset of rhs rows corresponding to data in scenario
    :params (B2, lb2_arr, ub2_arr, sense2_arr): similar to stage 1
    :params get_stochastic_lp2_params: function that returns random data (c2,A2,b2) for stage 2
    :params num_second_stage: number of scenarios to pre-samples for second stage
    """
    def __init__(
            self, lam, mdl, state1_idx_arr, x1_bnd_arr, scenarios, rand1_idx_arr, dummy1_idx_arr, ctrl1_idx_arr,
            # c1, A1, b1, B1, state1_idx_arr, x1_bnd_arr, sense1_arr, scenarios, rand1_idx_arr, dummy1_idx_arr, ctrl1_idx_arr, # first-stage
            # get_stochastic_lp2_params, B2, x2_bnd_arr, sense2_arr, # second-stage
            # seed, num_second_stage, 
            use_pid, target_s, kp, ki, kd,
    ): 
        # setup fixed control
        self.t = 0
        self.use_pid = use_pid
        self.target_s = target_s
        assert (not use_pid or (target_s is not None)), "PID must be given non-empty target_s, received None"
        self.prev_error = 0
        self.integral = 0
        self.dt = 0.1
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.err_history = []

        M_h = 0. # la.norm(c1)
        x_lb_arr = x1_bnd_arr[0][state1_idx_arr]
        x_ub_arr = x1_bnd_arr[1][state1_idx_arr]
        super().__init__(M_h, x_lb_arr, x_ub_arr, 0, 0)

        self.mdl = mdl
        self.mdl.setParam('LogToConsole', 0)
        self.scenarios = scenarios
        self.state1_idx_arr = state1_idx_arr
        self.dummy1_idx_arr = dummy1_idx_arr
        self.rand1_idx_arr = rand1_idx_arr
        self.ctrl1_idx_arr = ctrl1_idx_arr
        assert self.scenarios.shape[1] == len(self.rand1_idx_arr), \
            "Input scenario dim (%d) does not match rand dim (%d)" % (self.scenarios.shape[1], len(self.rand1_idx_arr))

    def get_gurobi_model(self):
        return None

    def get_var_names(self):
        return []

    def get_scenarios(self):
        return self.scenarios

    def set_scenario_idx(self, i):
        self.i = i

    def get_single_stage_lbub(self) -> Tuple[float, float]:
        return self.h_min_val, self.h_max_val

    def solve(self, x_prev, ver) -> Tuple[np.ndarray, float, np.ndarray, float]:
        assert len(self.dummy1_idx_arr) == len(x_prev), "Input state dim (%d) does not match dummy dim (%d)" % (len(x_prev), len(self.dummy1_idx_arr))

        # Setup previous variable
        for i, idx in enumerate(self.dummy1_idx_arr):
            self.mdl.setAttr("RHS", 
                    self.mdl.getConstrs()[idx],
                    x_prev[i])

        # Setup uncertainity RHS
        for i, idx in enumerate(self.rand1_idx_arr):
            self.mdl.setAttr("RHS", 
                    self.mdl.getConstrs()[idx],
                    self.scenarios[ver][i])

        # Set the control variable
        x = self.mdl.getVars()
        if self.use_pid:
            # compute pid controller
            error = self.target_s - x_prev
            self.t += 1
            self.err_history += [la.norm(error)]
            self.integral += error * self.dt
            derivative = (error - self.prev_error)/self.dt
            u = self.kp*error + self.ki*self.integral + self.kd * derivative

            u = np.clip(u, -25-(x_prev-self.scenarios[ver]), 50-(x_prev-self.scenarios[ver]))
            for i, idx in enumerate(self.ctrl1_idx_arr):
                x[idx].setAttr("LB", u[i])
                x[idx].setAttr("UB", u[i])
                # x[idx].lb = u[i] <- this does not set lower bound
                # x[idx].ub = u[i]

        # Solve
        self.mdl.update()
        self.mdl.optimize()

        if self.mdl.status != gp.GRB.OPTIMAL:
            raise Exception("LP did not terminate as optimal, got %s" % self.mdl.status)

        x_sol = np.array([x[i].X for i in self.state1_idx_arr]) # x.X[self.state1_idx_arr]
        val = self.mdl.objVal # getObjective().getValue()
        ctg = 0
        grad = np.zeros(len(x_sol))

        return [x_sol, val, grad, ctg]

    def add_cut(self, val, grad, x_prev):
        self.save_cut(val, grad, x_prev)

    def load_cuts(self, val_arr, grad_arr, x_prev_arr):
        super().load_cuts(val_arr, grad_arr, x_prev_arr)

class LowerBoundModel:
    def __init__(self, n, initial_lb=None):
        """ Creates lower bound model

        Args:
            n (int): dimension of variable
            initial_lb (float): initial lower bound. If None, does not set initial LB
        """
        self.model = gp.Model("lower_bound")
        self.model.setParam('OutputFlag', 0)
        self.n = n
        self.x = self.model.addMVar(n, lb=-float('inf'), name="x_lb")
        self.t = self.model.addMVar(1, lb=-float('inf'), name="t_lb")
        self.model.setObjective(self.t, gp.GRB.MINIMIZE)
        self.is_bounded = False

        # Gurobi naming conventions is different for n=1
        # if n == 1:
        #     self.model.addConstr(self.x == np.zeros(n), name="dummy[0]")
        # else:
        #     self.model.addConstr(self.x == np.zeros(n), name="dummy")
        self.model.addConstr(self.x == np.zeros(n), name="dummy")

        if initial_lb is not None:
            self.model.addConstr(self.t >= initial_lb, name="initial_lb")
            self.is_bounded = True

        self.model.update()

    def add_cut(self, val, grad, x_center):
        self.model.addConstr(self.t - grad @ self.x >= val - grad @ x_center)
        self.is_bounded = True

    def evaluate_lb(self, x):
        if not self.is_bounded:
            print("Lower bound model is not bounded, returning -inf")
            return -float('inf')
        else:
            for i in range(self.n):
                self.model.setAttr(
                    "RHS", 
                    self.model.getConstrByName("dummy[{}]".format(i)), 
                    x[i]
                )
            self.model.optimize()
            return self.model.objVal

class UpperBoundModel:
    def __init__(self, model, scenarios, lam, M, now_var_name_arr, h_max_val):
        """ Creates two upper bound models:

        1:- Solves hat{v}_i^k(bar{x}):=min{h + lam*bar{V}} (we need to take dual of $bar{v}_i^k to convert it from max to min)
        2:- Evaluates bar{v}_i^k (for upper bound evaluation)

        The aforemtioned dual LP is

        min  hat{v}'pi + M*r
        s.t. 1'pi= 1
             Xa + e - f = bar{x}
             -(e+f) + 1*r = 0
             alpha,e,f,r >= 0

        :param model (gurobipy.Model): model of original problem
        :param scenarios (list): list of scenarios
        :param lam (float): discount factor
        :param M (float): Lipschitz value
        :param now_var_names (list): list of names of now state variables in model
        :param h_max_val (float): average maximum value across scenarios for single stage
        """
        self.has_gp_mdl = True
        self.lam = lam
        self.V_0 = h_max_val/(1-self.lam) 

        if not isinstance(model, gp._model.Model):
            self.has_gp_mdl = False
            print("Not given gurobi model, upper bound model neglected")
            return

        model.update() # update before copying
        self.model_1 = model.copy()
        self.model_1.setParam('OutputFlag', 0)

        now_vars_list = [self.model_1.getVarByName(vname) for vname in now_var_name_arr]
        self.now_vars = gp.MVar(now_vars_list)
        self.n = len(now_var_name_arr)
        self.scenarios = scenarios
        self.N = len(self.scenarios)
        self.M = M

        # self.hat_vs contains all hat_v's across iterations and 
        # scenarios. We store it naively in a 1D array, where hat v's
        # in the same iteration (but different scenarios) are stored
        # in `N` consecutive elements. 
        # TODO: Better data structure
        self.hat_vs = np.array([])

        # e's and f's 
        self.es = [] # vector residual (+)
        self.fs = [] # vector residual (-)
        self.rs = [] # scalar residual
        # add new variables and include objective coeffs 
        for l in range(self.N+1):
            self.es.append(self.model_1.addMVar(
                # self.n, obj=self.lam*self.M/self.N, lb=0, name="e_{}".format(l)
                self.n, obj=0, lb=0, name="e_{}".format(l)
            ))
            self.fs.append(self.model_1.addMVar(
                # self.n, obj=self.lam*self.M/self.N, lb=0, name="f_{}".format(l)
                self.n, obj=0, lb=0, name="f_{}".format(l)
            ))
            self.rs.append(self.model_1.addVar(obj=self.lam*self.M/self.N, lb=0, name="r_{}".format(l)))

        # affine span constraint
        self.affine_span_constrs = []
        self.sum_pi_constrs = []
        for l in range(self.N):
            # -bar{x} + (e-f) + Xpi = 0
            new_affine_constr = self.model_1.addConstr(
                -self.now_vars + (self.es[l] - self.fs[l]) == np.zeros(self.n)
            )
            # -(e+f) + 1*r = 0
            self.affine_span_constrs.append(new_affine_constr)
            self.model_1.addConstr(-self.es[l] - self.fs[l] + self.rs[l] == 0)

        # second model
        self.model_2 = gp.Model("upper_bound2")
        self.model_2.setParam('OutputFlag', 0)
        self.mu = self.model_2.addMVar(1, name="mu", lb=-float("inf"))
        self.rho = self.model_2.addMVar(self.n, name="rho", lb=-float("inf"))
        self.abs_rho = self.model_2.addMVar(self.n, name="abs_rho")
        self.model_2.addConstr(self.abs_rho >= self.rho)
        self.model_2.addConstr(self.abs_rho >= -self.rho)
        self.model_2.addConstr(gp.quicksum(self.abs_rho) <= M)
        self.model_2_cut_constr_arr = []
        self.num_iters = 0
        # new objective each time (relies on input `x`), so we set it on the fly

    def _solve_model_1_and_update_hat_vs(self, x, k):
        """ 
        Solves model 1 for all scenarios. Updates self.hat_vs with new solutions.
        This function is called from `add_search_point_to_ub_model`.

        Args:
            x (np.array): previous search point
        """
        if not self.has_gp_mdl:
            return 

        new_hat_vs = np.array([])
        for ver in range(self.N):
            for i in range(self.n):
                # set x^{k-1}
                self.model_1.setAttr(
                    "RHS", 
                    self.model_1.getConstrByName("dummy[{}]".format(i)), 
                    x[i]
                )

            # RHS
            for i in range(len(self.scenarios[ver])):
                self.model_1.setAttr(
                    "RHS", 
                    self.model_1.getConstrByName("rand[{}]".format(i)),
                    self.scenarios[ver][i]
                )

            self.model_1.optimize()
            if self.num_iters == 0: # replace penalty (M) with V_0 cost to go
                x_sol = self.now_vars.X
                new_hat_vs = np.append(new_hat_vs, 
                    # self.model_1.objVal 
                    # + self.lam * self.V_0 
                    # - self.lam * self.M * np.sum(np.abs(x_sol)))
                    self.V_0
                )
            else:
                new_hat_vs = np.append(new_hat_vs, self.model_1.objVal)

        # form one big array
        self.hat_vs = np.append(self.hat_vs, new_hat_vs)

    def add_search_point_to_ub_model(self, x):
        """ 
        Adds search point to upper bound model. First processes search point.
        To be called at the end of each iteration of DDP.

        Args:
            x (np.array): previous search point
        """
        if not self.has_gp_mdl:
            return 

        # Compute \hat{v} (aka, solve Model 1)
        self._solve_model_1_and_update_hat_vs(x, self.num_iters)

        # Update model 1: add new pi_i^{k-1} for each scenario {i}
        # Recall self.hat_vs stored hat_v's across iterations and scenarios.
        # Values in same iteration are grouped together in consecutive N elements,
        # hence, we offset by `N * num_iters` to dtermine which group of N
        # elemennts to use. 
        self.model_1.update()
        for l in range(self.N): # (l == ver)
            # TODO Can we make this one call rather than if/else? Reason we need if/else
            # is because initializing constraint with sum pi = 1 (when pi doesn't exist
            # yet) leads to a wear initial constraint of 0 <= -1...
            if self.num_iters == 0:
                col_l = gp.Column(x, self.affine_span_constrs[l].tolist())
                new_pi_var = self.model_1.addVar(
                    lb=0,
                    obj=(self.lam/self.N) * self.hat_vs[self.N * self.num_iters + l], 
                    name="pi_{}^{}".format(l, self.num_iters), 
                    column=col_l,
                )

                # add sum pi = 1 constraint
                self.sum_pi_constrs.append(
                    self.model_1.addConstr(
                        new_pi_var == 1, 
                        name="sum_pi[{}]".format(l)
                    )
                )

            else:
                # add new pi variable for 1'pi=1 and Xpi + (e-f) = bar{x}
                constr_list_l = [self.sum_pi_constrs[l]] \
                                + self.affine_span_constrs[l].tolist()
                one_appended_with_x = np.append(1, x).tolist()
                col_l = gp.Column(one_appended_with_x, constr_list_l)
                self.model_1.addVar(
                    lb=0,
                    obj=(self.lam/self.N) * self.hat_vs[self.N * self.num_iters + l], 
                    name="pi_{}^{}".format(l, self.num_iters), 
                    column=col_l,
                )

        # TODO: How can we update this without looping?
        # self.model.setAttr("RHS", self.model.getConstrByName("dummy"), x_prev)

        # Update Model 2: Add a new constraint. RHS of 0 is placeholder value
        self.model_2.addConstr(
            self.mu + x @ self.rho <= 0, 
            name="model_2_cut_constr[{}]".format(self.num_iters)
        )

        self.model_2.update()

        self.num_iters += 1

    def evaluate_ub(self, x):
        """ Evaluates upper bound model at search point `x`

        Args:
            x (np.array): search point

        Returns:
            float: upper bound
        """
        if not self.has_gp_mdl:
            return self.V_0

        if self.num_iters == 0:
            return self.V_0

        ub_model_sum = 0

        self.model_2.setObjective(self.mu + self.rho @ x, gp.GRB.MAXIMIZE)
        self.model_2.update()
        # Update hat v's to appropriate scenario
        for ver in range(self.N):
            for k in range(self.num_iters):
                constr_ptr = self.model_2.getConstrByName(
                    # "model_2_cut_constr[{}]".format(k)
                    "model_2_cut_constr[{}][0]".format(k)
                )
                assert constr_ptr is not None, "Cannot find constraint %s in ub's model_2" % "model_2_cut_constr[{}]".format(k)
                self.model_2.setAttr(
                    "RHS", 
                    constr_ptr,
                    self.hat_vs[(ver) + k*self.N],
                )

            self.model_2.update()
            self.model_2.optimize()

            ub_model_sum += self.model_2.objVal
            # if ver == 0:
            #     print(len(self.hat_vs[ver::self.N]), self.hat_vs[ver::self.N][-5:])

        return ub_model_sum / self.N
