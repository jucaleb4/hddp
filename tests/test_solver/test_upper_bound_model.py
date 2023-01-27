from src.solver import UpperBoundModel
import numpy as np
import gurobipy as gp

def test_upper_bound_constructor():
    """
    Test upper bound produces correct initial upper bound.
    """
    n = 1
    md = gp.Model()
    x = md.addMVar(n, lb=-float('inf'), name="x")
    old_x = md.addMVar(n, lb=-float('inf'), name="old_x")
    md.setObjective(x[0], sense=gp.GRB.MINIMIZE)
    md.addConstr(x[0] + old_x[0] >= -100, name="rand[0]")
    md.addConstr(old_x[0] == 0, name="dummy[0]")
    md.update()

    N = 5
    scenarios = np.array([ [-102 + i] for i in range(N) ])
    lam = 0.9
    M_0 = 100
    now_var_name = ["x[0]"]
    # made up values
    min_vals_arr = np.ones(N)
    max_val = 10

    ubm = UpperBoundModel(
        md, 
        scenarios,
        lam, 
        M_0, 
        now_var_name, 
        min_vals_arr, 
        max_val
    )

    x = np.zeros(n)
    ub_eval = ubm.evaluate_ub(x)

    true_ub = max_val/(1-lam)

    assert abs(ub_eval - true_ub) < 1e-6

def test_upper_bound_model_constructor():
    """
    Add just one point to upper bound model and ensure it is correct.
    We used the definition of BFS (where an OPT lives) to analytically
    compute the upper bound.

    Should know closed form solution...
    """
    n = 1
    md = gp.Model()
    x = md.addMVar(n, lb=-float('inf'), name="x")
    old_x = md.addMVar(n, lb=-float('inf'), name="old_x")
    md.setObjective(x[0], sense=gp.GRB.MINIMIZE)
    md.addConstr(x[0] + old_x[0] >= -100, name="rand[0]")
    md.addConstr(old_x[0] == 0, name="dummy[0]")
    md.update()

    N = 5
    scenarios = np.array([ [-102 + i] for i in range(N) ])
    lam = 0.9
    M_0 = 100
    now_var_name = ["x[0]"]
    # [-98, -99, -100, -101, -102]
    min_vals_arr = -np.arange(98, 103) 
    max_val = 10

    ubm = UpperBoundModel(
        md, 
        scenarios,
        lam, 
        M_0, 
        now_var_name, 
        min_vals_arr, 
        max_val
    )

    x = np.zeros(n)
    ubm.add_search_point_to_ub_model(x)

    x = np.ones(n)
    ub_eval = ubm.evaluate_ub(x)

    true_ub = np.mean(min_vals_arr) + lam * max_val/(1-lam) + M_0

    assert abs(ub_eval - true_ub) < 1e-6

def test_upper_bound_monotone():
    """
    Add multiple points to ensure upper bound is monotone (w.r.t. number of points)
    """
    n_trials = 1
    n_iters = 10

    for _ in range(n_trials):
        n = 10
        md = gp.Model()
        x = md.addMVar(n, lb=-float('inf'), name="x")
        old_x = md.addMVar(n, lb=-float('inf'), name="old_x")
        md.setObjective(np.ones(n)@x, sense=gp.GRB.MINIMIZE)
        md.addConstr(x + old_x >= np.zeros(n), name="rand")
        md.addConstr(old_x == np.zeros(n), name="dummy")
        md.update()

        N = 5
        scenarios = [ i*np.ones(n) for i in range(N) ]
        lam = 0.9
        M_0 = 1000
        now_var_name = ["x[{}]".format(i) for i in range(n)]
        min_vals_arr = -100*np.zeros(n)
        max_val = 100*np.zeros(n)

        ubm = UpperBoundModel(
            md, 
            scenarios,
            lam, 
            M_0, 
            now_var_name, 
            min_vals_arr, 
            max_val
        )

        # number of points
        x = 100*(1-2*np.random.random(n))
        curr_eval_ub = np.inf

        for _ in range(n_iters):
            x_center = 10*(1-2*np.random.random(n))
            ubm.add_search_point_to_ub_model(x_center)

            new_eval_ub = ubm.evaluate_ub(x)

            assert new_eval_ub <= curr_eval_ub

            curr_eval_ub = new_eval_ub

def test_upper_bound_solution_for_1d_problem():
    """
    Test upper bound produces correct upper bounds for a simple 1D problem.
    """
    lam = 3/4
    mu  = 2/3
    n = 1
    md = gp.Model()
    x = md.addMVar(n, lb=-float('inf'), name="x")
    old_x = md.addMVar(n, lb=-float('inf'), name="old_x")
    md.setObjective(x[0], sense=gp.GRB.MINIMIZE)
    md.addConstr(x[0] - mu * old_x[0] == 0, name="rand[0]")
    md.addConstr(old_x[0] == 0, name="dummy[0]")
    md.update()

    scenarios = [[0]]
    now_var_name = ["x[0]"]
    min_vals_arr = np.zeros(1)
    max_val = 10

    # Note: Make sure M_0 not too large, otherwise we will always get initial UB
    M_0 = 10

    ubm = UpperBoundModel(
        md, 
        scenarios,
        lam, 
        M_0, 
        now_var_name, 
        min_vals_arr, 
        max_val
    )

    # 1) First eval is 1/(1-lam) * max_val
    assert ubm.evaluate_ub([1]) == 1/(1-lam) * max_val # == 40

    # 2) Second eval is based on min + lam * max
    ubm.add_search_point_to_ub_model(np.array([1])) 
    hatv_1 = 90/3 # \hat{v}_1 = 30 (lower bound + lam * upper bound)
    assert abs(ubm.evaluate_ub(np.array([1])) - hatv_1) < 1e-2

    # 3) Third eval involves solving hatv_2 but since it is smaller than initial UB, attains it
    ubm.add_search_point_to_ub_model(np.array([1])) 
    hatv_2 = (2/3)*(1) + (3/4)*(M_0*1/3 + hatv_1) # include penalty term
    assert abs(ubm.evaluate_ub(np.array([1])) - hatv_2) < 1e-2

    # 4) Fourth eval requires writing and solving LP by hand/computer
    ubm.add_search_point_to_ub_model(np.array([3/2])) 
    assert abs(ubm.evaluate_ub(np.array([1])) - 101/4) < 1e-2