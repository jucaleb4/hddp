from solver import UpperBoundModel
from opt_setup import opt_rareflow
import gurobipy as gp
import numpy as np

def a():
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

    x = np.arange(n)
    ubm.add_search_point_to_ub_model(x)
    x = np.arange(n)[::-1]
    ubm.add_search_point_to_ub_model(x)
    x = np.ones(n)
    ubm.add_search_point_to_ub_model(x)
    ubm.evaluate_ub(x)

def b():
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
    ubm.add_search_point_to_ub_model(x)

    x = np.ones(n)
    ub_eval = ubm.evaluate_ub(x)

    true_ub = np.mean(
        np.array([-102 + i for i in range(N) ]) + lam * max_val/(1-lam) + M_0
    )

    assert abs(ub_eval - true_ub) < 1e-6

def c():
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

def d():
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

    # arbitrary value
    M_0 = 10 # TODO: Is this too big?

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

    # 2) Second eval is 
    ubm.add_search_point_to_ub_model(np.array([1])) # \hat{v}_1 = 90/3
    hatv_1 = 90/3
    assert abs(ubm.evaluate_ub(np.array([1])) - hatv_1) < 1e-2

    # 3) Third eval is
    ubm.add_search_point_to_ub_model(np.array([1])) # \hat{v}_1 = 30 (lower bound + lam * upper bound)
    hatv_2 = (2/3)*(1) + (3/4)*(M_0*1/3 + hatv_1)
    assert abs(ubm.evaluate_ub(np.array([1])) - hatv_2) < 1e-2

    # 4) Fourth eval does not use {f}, solve by LP
    ubm.add_search_point_to_ub_model(np.array([3/2])) 
    hatv_3 = (2/3)*(3/2) + (3/4)*(hatv_2) # = 81/4
    print(hatv_3)
    assert abs(ubm.evaluate_ub(np.array([1])) - 101/4) < 1e-2

if __name__ == "__main__":
    """ Test if opt_rareflow setup works """
    lam = 0.9
    N = 10
    import ipdb; ipdb.set_trace()
    m = gp.Model()
    n = 1
    now = m.addMVar(n, lb=0, name="now")
    past = m.addMVar(n, lb=0, name="past")
    ctrl = m.addMVar(n, lb=0)
    t = m.addMVar(1, lb=-float("inf"), name="t") 
    m.addConstr(now[0] <= 10)
    m.addConstr(ctrl[0] <= 1)
    m.addConstr(now[0] - past[0] - ctrl[0] == 0, name="rand[0]")
    m.addConstr(past[0] == 0, name="dummy[0]")
    m.setObjective(-now[0] + ctrl[0] + lam*t, gp.GRB.MINIMIZE)
    scenarios = np.append(
        1,
        -0.9*np.ones(N)
    )
    scenarios = np.reshape(scenarios, (N+1, 1))