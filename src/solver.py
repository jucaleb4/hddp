import numpy as np
import gurobipy as gp
import re
import numpy.linalg as la
import scipy.sparse.linalg as spla

from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional

class GenericSolver(ABC):
    """
    Generic solver. Solver must implement certain methods to be used in the
    hierarchical dual dynamic programming (HDDP) algorithm. 
    """
    
    @abstractmethod
    def solve(self, x_prev: np.ndarray, scenario_id: int) \
            -> Tuple[float, np.ndarray, np.ndarray]:
        """ Solves subproblem @scenario_id using @x_prev as the previous point.
        Returns a cut, or an (approximately) optimal objective value, primal,
        and dual variable.
        """
        pass

    @abstractmethod
    def add_cut(self, val, grad, x_prev) -> None:
        """ Adds cut to cost-to-go function, in the form of
        \[
            l_f(x) := val + <grad, x - x_prev>
        \]
        """
        pass

    def get_scenarios(self):
        raise NotImplementedError

    def get_extrema_values(self) -> Tuple[float, float]:
        """
        Returns the estimated min and max values computed during construction

        Returns:
            min_val (float): mean of minimum values
            max_val (float): mean of maximum values
        """
        raise NotImplementedError

    def get_num_scenarios(self):
        raise NotImplementedError

class EmptySolver(GenericSolver):
    """ Solver that returns zero value and gradient """
    def __init__(self, n):
        self.n = n

    def solve(self, x_prev, ver):
        return [0, np.zeros(self.n), np.zeros(self.n)]

    def add_cut(self, val, grad, x_prev) -> None:
        pass

class GurobiSolver(GenericSolver):
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
        super().__init__()
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
        # y_sol_rand = np.array([self.md.getAttr("Pi", 
        #     [self.md.getConstrByName("rand[{}]".format(i))]) 
        #     for i in range(len(self.scenarios[ver]))])
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

    def add_cut(self, val, grad, x_prev):
        self.md.addConstr(self.t - grad@self.x >= val-grad@x_prev)

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
    :params primal_projection_var_idxs: Projection onto non-negative orthodant 
    :params dummy_cons_idx: constraints corresponding to past state variable (dummy vars)
    :params rand_rhs_idx: constraint indices w.r.t. random variables
    :params min_val: lower bound on cost to go: lower bound for cost to go function
    :params max_val: upper bound on cost to go: upper bound for cost to go function
    :params seed: seed for random scenario selector
    :params warm_start_x: warm start primal solution
    """
    def __init__(
            self, 
            A: np.ndarray,
            b: np.ndarray,
            c: np.ndarray,
            lam: float,
            subproblem_solver: GenericSolver, 
            scenarios: np.ndarray,
            primal_var_idx_to_return: np.ndarray,
            primal_projection_var_idxs: np.ndarray,
            dummy_cons_idx: np.ndarray,
            rand_rhs_idx: np.ndarray,
            min_val: float,
            max_val: Optional[float]=None,
            seed: Optional[int]=None,
            warm_start_x: Optional[np.ndarray]=None,
    ): 
        self.A = A
        self.b = b
        self.c = c
        self.primal_var_idx_to_return = primal_var_idx_to_return
        self.primal_projection_var_idxs = primal_projection_var_idxs
        self.dummy_cons_idx = dummy_cons_idx
        self.scenarios = scenarios
        self.rand_rhs_idx = rand_rhs_idx
        self.subproblem_solver = subproblem_solver

        # Initialize DS for cuts and lower bound
        self.num_cuts = 1
        self.cut_capacity = 16
        self.cut_vals = np.zeros(self.cut_capacity)
        self.cut_vals[0] = min_val
        self.cut_grads = np.zeros((self.cut_capacity, len(primal_var_idx_to_return)))
        self.lam = lam

        self.min_val = min_val
        self.max_val = max_val

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
        D_X  = 5000
        W_norm = la.norm(A.todense(), ord=2)
        a_X = 1

        N = 20000 # manually tune
        self.n_iters = 20000 # manually tune
        self.ws = np.ones(N)
        self.thetas = np.ones(N)
        self.taus = max(barG * (3*N)**0.5/D_X, 2**0.5 * W_norm)/a_X**0.5 * np.ones(N)
        self.etas = 2**0.5 * W_norm/a_X**0.5 * np.ones(N)

    def get_extrema_values(self) -> Tuple[float, float]:
        return self.min_val, self.max_val

    def solve(self, u, ver) -> Tuple[float, np.ndarray, np.ndarray]:
        """ Solves subproblem and returns approx. optimal value, primal, and
        dual variable.

        :params u: previous state variable
        :params ver: subproblem id
        :returns objval: computed objective value
        :returns barx: computed primal solution
        :returns subgrad: computed subgradient w.r.t. last state variable
        """

        x = self.prev_x[ver] 
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
        N = len(self.scenarios)

        for k in range(self.n_iters):

            # get unbiased subgradient
            i_k = self.rng.integers(0, N)
            G_k = self.get_stochastic_subgradient_of_v2(x, i_k)

            d = self.thetas[k] * (y - y_prev) + y
            x = self.primal_prox(x, d, self.c, G_k, self.taus[k], 
                                 self.primal_projection_var_idxs)
            y_prev = y
            y = self.dual_prox(y, x, b, self.etas[k])

            sumx += self.ws[k] * x
            sumy += self.ws[k] * y

        # compute final output
        sumw = float(np.sum(self.ws[:self.n_iters]))
        barx = (1./sumw) * sumx
        bary = (1./sumw) * sumy

        self.prev_x[ver] = barx

        subgrad = bary[self.dummy_cons_idx]
        objval = np.dot(self.c, barx)
        for i in range(N):
            # evaluate the output
            [val, _, _] = self.subproblem_solver.solve(barx, i)
            objval += (1/N) * val

        barx_subset = barx[self.primal_var_idx_to_return]
        return objval, barx, subgrad

    def get_stochastic_subgradient_of_v2(self, x_k, i_k):
        """ Computes stochastic subgradient, which is sum of cost to go's (with
        a discount factor) and v2 subproblem's subgradient.
        """
        [_, G_k, _] = self.subproblem_solver.solve(x_k, i_k)

        nc = self.num_cuts
        cut_evals = self.cut_vals[:nc] + np.dot(self.cut_grads[:nc], x_k)
        tight_cut_idx = np.argmax(cut_evals)

        primal_idxs = self.primal_var_idx_to_return
        # ensure state variable length matches gradient from cut
        assert len(primal_idxs) == len(self.cut_grads[tight_cut_idx])
        G_k[primal_idxs] += self.lam * self.cut_grads[tight_cut_idx]

        return G_k

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

        self.cut_vals[self.num_cuts] = val - np.dot(grad, x_prev)
        self.cut_grads[self.num_cuts] = grad

        # double capacity if full
        self.num_cuts += 1
        if self.num_cuts >= self.cut_capacity:
            self.cut_vals = np.append(self.cut_vals, np.zeros(self.cut_capacity))
            zeros_matrix = np.zeros(self.cut_capacity, len(self.primal_var_idx_to_return))
            self.cut_grads = np.vstack((self.cut_grads, zeros_matrix))
            self.cut_capacity *= 2 

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

