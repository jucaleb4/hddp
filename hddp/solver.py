import numpy as np
import gurobipy as gp
import re
import os
import numpy.linalg as la

from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional

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

    def save_cut(self, val, grad, x_prev) -> None:
        """ Adds cut to cost-to-go function, in the form of
        \[
            l_f(x) := val + <grad, x - x_prev>
        \]
        """
        self.val_arr[self.ct] = val
        self.grad_arr[self.ct] = grad
        self.x_prev_arr[self.ct] = x_prev
        self.ct += 1

        if self.ct == len(self.val_arr):
            self.val_arr = np.append(self.val_arr, np.zeros(self.val_arr.shape, dtype=float))
            self.grad_arr = np.vstack((self.grad_arr, np.zeros(self.grad_arr.shape, dtype=float)))
            self.x_prev_arr = np.vstack((self.x_prev_arr, np.zeros(self.x_prev_arr.shape, dtype=float)))

    def save_cuts_to_file(self, folder):
        np.savetxt(os.path.join(folder, "vals.csv"), self.val_arr[:self.ct], delimiter=',')
        np.savetxt(os.path.join(folder, "grad.csv"), self.grad_arr[:self.ct], delimiter=',')
        np.savetxt(os.path.join(folder, "x_prev.csv"), self.x_prev_arr[:self.ct], delimiter=',')

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

class PDSASolverForLPs(GenericSolver):
    """ 
    Primal dual stochastic approximation solver for solving the stochastic
    LP
    \[
        \min c.x + lam uV(x) + E[ v_2(x) ] s.t. x \in X(u)
        s.t. X(u) := {x : Ax-Bu-b=0, x_i >= 0 for i \in I},
    \]
    where $uV$ is a piecewise linear function and $v_2(z)$ is a blackbox where
    one can obtain approximate/exact optimal value, primal, and dual variable
    by calling @subproblem_solver.solve(), and $I$ is a subset of indices that
    need projection (defined by @primal_projection_var_idxs). We reformulate
    the LP to a saddle point problem
    \[
        \min_x \max_y c.x + <y, Ax - Bu - b> + lam * uV(x) + E[ v_2(x) ]
        s.t. x_i >= i \in I,
    \]
    which is a solved by a primal-dual stochastic approximation method. 

    :params A: Constraint matrix for primal variable
    :params b: Initial RHS vector
    :params c: Cost vector
    :params lam: discount factor
    :params subproblem_solver: subproblem solver that given an input @x,
                               returns the approximate/optimal value, primal,
                               and dual variable
    :params scenarios: RHS from SAA problem
    :params primal_var_idx_to_return: which primal variables correspond to state variables
    :params subprob_var_idx_list: list of np.ndarray of indices subproblem corresponds to
    :params primal_projection_var_idxs: Projection onto non-negative orthodant 
    :params dummy_cons_idx: constraints corresponding to past state variable (dummy vars)
    :params rand_rhs_idx: constraint indices w.r.t. random variables
    :params h_min_val: lower bound on cost to go: lower bound for cost to go function
    :params max_val: upper bound on cost to go: upper bound for cost to go function
    :params seed: seed for random scenario selector
    :params warm_start_x: warm start primal solution
    """
    def __init__(
            self, 
            model: Any,
            var_names: Any,
            A: np.ndarray,
            b: np.ndarray,
            c: np.ndarray,
            lam: float,
            subproblem_solver: GenericSolver, 
            scenarios: np.ndarray,
            primal_var_idx_to_return: np.ndarray,
            subprob_var_idx_list: list,
            primal_projection_var_idxs: np.ndarray,
            dummy_cons_idx: np.ndarray,
            rand_rhs_idx: np.ndarray,
            h_min_val: float,
            max_val: Optional[float]=None,
            seed: Optional[int]=None,
            warm_start_x: Optional[np.ndarray]=None,
            gsolver=None # Gurobi solver
    ): 
        self.model = model.copy()
        self.var_names = var_names
        self.gsolver = gsolver

        self.A = A
        self.b = b
        self.c = c
        self.subproblem_solver = subproblem_solver
        self.scenarios = scenarios
        self.primal_var_idx_to_return = primal_var_idx_to_return
        self.subprob_var_idx_list = subprob_var_idx_list
        self.primal_projection_var_idxs = primal_projection_var_idxs
        self.dummy_cons_idx = dummy_cons_idx
        self.rand_rhs_idx = rand_rhs_idx

        # Initialize DS for cuts and lower bound
        self.num_cuts = 1
        self.cut_capacity = 16
        self.cut_vals = np.zeros(self.cut_capacity)
        self.cut_vals[0] = h_min_val/(1.-lam)
        self.cut_grads = np.zeros((self.cut_capacity, len(primal_var_idx_to_return)))
        self.lam = lam

        self.rng = np.random.default_rng(seed)

        if warm_start_x is not None:
            N = len(self.scenarios)-1
            self.prev_x = np.outer(np.ones(N+1), warm_start_x)
            # ensure we start with a feasible solution
            assert np.min(warm_start_x[primal_projection_var_idxs]) >= 0, "Did not warm start with feasible solution"
        else:
            self.prev_x = np.zeros((N+1, A.shape[1]))

        # Estimate parameters
        barG = 100
        D_X  = 1000
        W_norm = la.norm(A.todense(), ord=2)
        a_X = 1

        N = 50 # manually tune
        self.n_iters = 1000 # manually tune
        self.w = 1
        self.theta = 1
        self.tau = max(barG * (3*N)**0.5/D_X, 2**0.5 * W_norm)/a_X**0.5
        self.eta = 2**0.5 * W_norm/a_X**0.5 

        self.i = 0

    def get_gurobi_model(self):
        return self.model

    def get_var_names(self):
        return self.var_names

    def get_scenarios(self):
        return self.scenarios

    def set_scenario_idx(self, i):
        self.i = i

    def get_single_stage_lbub(self) -> Tuple[float, float]:
        return self.min_val, self.max_val

    def solve(self, u, ver) -> Tuple[np.ndarray, float, np.ndarray, float]:
        """ Solves subproblem and returns approx. optimal value, primal, and
        dual variable.

        :params u: previous state variable
        :params ver: subproblem id
        :returns objval: computed objective value
        :returns barx: computed primal solution
        :returns subgrad: computed subgradient w.r.t. last state variable
        """
        # Do not warm start
        x = np.zeros(len(self.prev_x[self.i]))# self.prev_x[self.i] 
        # x = self.prev_x[ver]
        y = np.zeros(self.A.shape[0]) 
        y_prev = y.copy()
        sumx = np.zeros(len(x))
        sumy = np.zeros(len(y))

        # update RHS for (upper level) scenario
        b = self.b
        for i,j in enumerate(self.rand_rhs_idx):
            b[j] = self.scenarios[ver][i]
        for i,j in enumerate(self.dummy_cons_idx):
            b[j] = u[i]

        # number of lower level scenarios
        N = len(self.subproblem_solver.get_scenarios())

        for k in range(self.n_iters):

            # get unbiased subgradient
            i_k = self.rng.integers(0, N)
            [_, G_k, _] = self.get_stochastic_subgradient_of_v2(x, i_k)

            d = self.theta * (y - y_prev) + y
            x = self.primal_prox(x, d, self.c, G_k, self.tau, 
                                 self.primal_projection_var_idxs)
            y_prev = y
            y = self.dual_prox(y, x, b, self.eta)

            sumx += self.w * x
            sumy += self.w * y

        # compute final output
        sumw = self.w * self.n_iters
        barx = (1/sumw) * sumx
        bary = (1/sumw) * sumy

        # warm start for next solve
        self.prev_x[ver] = barx

        subgrad = bary[self.dummy_cons_idx]
        objval = np.dot(self.c, barx)
        for i in range(N):
            # evaluate the output
            [val, _, ctg] = self.get_stochastic_subgradient_of_v2(barx, i)
            objval += (1/N) * val
        objval += self.lam * ctg

        # _, _, ggrad, _= self.gsolver.solve(u, ver)

        barx_to_return = barx[self.primal_var_idx_to_return]

        return barx_to_return, objval, subgrad, ctg

    def get_stochastic_subgradient_of_v2(self, x_k, i_k):
        """ Computes stochastic subgradient, which is sum of cost to go's (with
        a discount factor) and v2 subproblem's subgradient. Since subproblem
        gradient and cost-to-go only corresponds to some parts of the primal
        solution, we project onto the correct portion.
        """
        # gradient from subproblem
        subprob_var_idxs = self.subprob_var_idx_list[i_k]
        x_k_subprob_proj = x_k[subprob_var_idxs]
        [_, val, G_k, _] = self.subproblem_solver.solve(x_k_subprob_proj, i_k)
        grad = np.zeros(len(x_k))
        grad[subprob_var_idxs] = G_k

        # gradient from cost-to-go
        nc = self.num_cuts
        x_k_primal_proj = x_k[self.primal_var_idx_to_return]
        cut_evals = self.cut_vals[:nc] \
                    + np.dot(self.cut_grads[:nc], x_k_primal_proj)
        tight_cut_idx = np.argmax(cut_evals)
        ctg = cut_evals[tight_cut_idx]
        primal_idxs = self.primal_var_idx_to_return
        grad[primal_idxs] += self.lam * self.cut_grads[tight_cut_idx]

        return val, grad, ctg

    def primal_prox(self, x_prev, d, c, G_k, tau, projection_var_idxs):
        # TODO: Transpose multiply
        x = x_prev - (1./tau) * (G_k + c - self.A.T.dot(d))
        if len(projection_var_idxs):
            x[projection_var_idxs] = np.maximum(x[projection_var_idxs], 0)
        return x

    def dual_prox(self, y_prev, x, b, eta):
        y = y_prev - (1./eta) * (self.A.dot(x) - b)
        return y

    def add_cut(self, val, grad, x_prev):
        # self.gsolver.save_cut(val, grad, x_prev)
        self.save_cut(val, grad, x_prev)
        self.gsolver.add_cut(val, grad, x_prev)

        self.cut_vals[self.num_cuts] = val - np.dot(grad, x_prev)
        self.cut_grads[self.num_cuts] = grad

        # double capacity if full
        self.num_cuts += 1
        if self.num_cuts >= self.cut_capacity:
            self.cut_vals = np.append(self.cut_vals, np.zeros(self.cut_capacity))
            zeros_matrix = np.zeros((self.cut_capacity, len(self.primal_var_idx_to_return)))
            self.cut_grads = np.vstack((self.cut_grads, zeros_matrix))
            self.cut_capacity *= 2 

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
        if n == 1:
            self.model.addConstr(self.x == np.zeros(n), name="dummy[0]")
        else:
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
        model.update() # update before copying
        self.model_1 = model.copy()
        self.model_1.setParam('OutputFlag', 0)

        now_vars_list = [self.model_1.getVarByName(vname) for vname in now_var_name_arr]
        self.now_vars = gp.MVar(now_vars_list)
        self.n = len(now_var_name_arr)
        self.scenarios = scenarios
        self.N = len(self.scenarios)
        self.lam = lam
        self.V_0 = h_max_val/(1-self.lam) 
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
