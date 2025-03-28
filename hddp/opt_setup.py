import gurobipy as gp
import numpy as np
import scipy.sparse as spp
import numpy.linalg as la
import pandas
import time
import re
from hddp import solver

from typing import Tuple, Any

def create_inventory_gurobi_model(N, lam, seed, has_ctg=True):
    """ Setups basic inventory problem

    State (dim=1) x_t: inventory at start of ordering at time t, before demand realized
    Ctrl (dim=5) y_t,u_t,x_t^+,x_t^-: dummy of past, order amount, max(0,x_t), max(0,-x_t)

    min  c*u_t + h*(x_t^+) + b*(x_t^-)
    s.t. x_t = y_t + u_t - D_t
         y_t = x_{t-1}
         x_t^+ >= max(0, x_t)
         x_t^- >= max(0, -x_t)
         u_t >= 0

    """
    h = 0.2
    b = 2.8
    c = np.cos(np.pi/3) + 1.5
    d = 10.0
    phi = 1.6
    rng = np.random.default_rng(seed)

    # define gurobi model
    model = gp.Model()
    now = model.addMVar(1, lb=-100, ub=100, name="now")
    x_lb_arr = np.array([-100])
    x_ub_arr = np.array([100])
    now_var_name_arr = ["now[0]"]
    past_state_for_max_val = np.array([-100])

    past = model.addMVar(1, lb=-100)
    y = model.addMVar(1, lb=-float('inf')) # quantity we order
    x_past_pos = model.addMVar(1, lb=0) # over-supply
    x_past_neg = model.addMVar(1, lb=0) # under-supply
    u = model.addMVar(1, lb=-float('inf')) # order amount

    model.addConstr(-now + (past + u) == 0, name='rand') # .values()
    model.addConstr(x_past_pos - past >= 0, name='x_pos') # .values()
    model.addConstr(x_past_neg + past >= 0, name='x_neg') # .values()
    dummy_constrs = model.addConstr(np.eye(1)@past == np.zeros(1), name="dummy") # .values()

    model.setObjective(c*u + b*x_past_neg + h*x_past_pos, gp.GRB.MINIMIZE)
    model.update()
    M_h = c + b + h

    scenarios = np.append([5.5], phi*rng.normal(loc=d, scale=phi, size=N))
    scenarios = np.atleast_2d(scenarios).T
    inventory_solver = solver.GurobiSolver(
        model, 
        now, 
        now_var_name_arr,
        lam, 
        scenarios,
        M_h,
        x_lb_arr,
        x_ub_arr,
        h_min_val=0,
        past_state_for_max_val=past_state_for_max_val if has_ctg else None,
    )

    x_0 = np.array([0])

    return inventory_solver, x_0

def create_hydro_thermal_gurobi_model(N, lam, seed, has_ctg=True):
    """ (Single stage) hydro problem. 

    Based from: https://optimization-online.org/2019/09/7367/

    :param N: number of scenarios
    :param lam: discount factor
    :param seed: 
    :param has_ctg: whether to include cost-to-go function

    :return solver: solver object. See `solver.py` for details
    :return x_0: initial state
    """

    rng = np.random.default_rng(seed)

    n_regions = 4
    fname = "./data/"
    
    hydro_ = pandas.read_csv(fname + "hydro.csv", index_col=0)
    demand = pandas.read_csv(fname + "demand.csv", index_col=0)
    deficit_ = pandas.read_csv(fname + "deficit.csv", index_col=0)
    exchange_ub = pandas.read_csv(fname + "exchange.csv", index_col=0)
    exchange_cost = pandas.read_csv(fname + "exchange_cost.csv", index_col=0)
    thermal_ = [pandas.read_csv(fname + "thermal_{}.csv".format(i),
        index_col=0) for i in range(n_regions)]

    start_time = time.time()

    # historical rainfall data
    hist = [pandas.read_csv(fname + "hist_{}.csv".format(i), sep=";") for i in range(n_regions)]
    hist = pandas.concat(hist, axis=1)
    hist.dropna(inplace=True)
    hist.drop(columns='YEAR', inplace=True)
    scenarios = [hist.iloc[:,12*i:12*(i+1)].transpose().values for i in range(n_regions)]
    # [region][month][year]
    scenarios = np.array(scenarios)
    scenarios = np.mean(scenarios, axis=1)

    assert n_regions == scenarios.shape[0]

    means  = np.mean(scenarios, axis=1)
    sigmas = np.std(scenarios, axis=1)

    lognorm_sigmas= np.sqrt(np.log(np.power(np.divide(sigmas,means), 2) + 1))
    lognorm_means = np.log(means) - np.square(lognorm_sigmas)/2

    scenarios = np.array([rng.lognormal(
                            mean=lognorm_means[i], 
                            sigma=lognorm_sigmas[i], 
                            size=N,
                          ) 
                        for i in range(n_regions)])
    # TEMP
    scenario_0 = np.array([hydro_['INITIAL'][n_regions:2*n_regions].to_numpy()]).T 
    scenarios = np.hstack((scenario_0, scenarios)).T

    demand = demand.to_numpy()
    # get monthly avg
    demand = np.mean(demand, axis=0)

    # define Gurobi model
    model = gp.Model()

    # stored_now = m.addMVar(n_regions, ub=hydro_['UB'][:n_regions], name="stored")
    now_var_name = "stored"
    stored_now = model.addMVar(n_regions, name=now_var_name)
    now_var_name_arr = ["{}[{}]".format(now_var_name, i) for i in range(n_regions)]
    model.addConstrs(stored_now[i] <= hydro_['UB'][i] for i in range(n_regions))
    x_ub_arr = hydro_['UB'].to_numpy()[:4]
    x_lb_arr = np.zeros(len(x_ub_arr))

    # x_prev's that help achieve the maximum and minimum cost
    past_state_for_max_val = np.zeros(len(now_var_name))
    past_state_for_min_val = hydro_['UB']

    # 59419.3  5874.9 12859.2  5271.5
    # stored_now = m.addMVar(n_regions, ub=[100000,100000,100000,100000], name="stored")
    stored_past= model.addMVar(n_regions, name="stored_past")
    spill = model.addMVar(n_regions, name="spill")
    c_spill = 0.001 * np.ones(n_regions)
    # hydro = m.addMVar(n_regions, ub=hydro_['UB'][-4:], name="hydro")    
    hydro = model.addMVar(n_regions, name="hydro")    
    model.addConstrs(hydro[i] <= hydro_['UB'][-i] for i in range(n_regions))

    c_deficit = np.array([[deficit_['OBJ'][j] for i in range(4)] for j in range(4)])
    deficit = model.addMVar((n_regions,n_regions),
                        # ub = [demand[i] * deficit_['DEPTH'][j] 
                        # for i in range(n_regions) for j in range(n_regions)],
                        name = "deficit")
    model.addConstrs(deficit[i][j] <= demand[i] * deficit_['DEPTH'][j] 
                for i in range(n_regions) for j in range(n_regions))

    thermal = [None] * n_regions
    c_thermal = [None] * n_regions
    for i in range(n_regions):
        thermal[i] = model.addMVar(
            len(thermal_[i]), 
            # ub=thermal_[i]['UB'], 
            # lb=thermal_[i]['LB'], 
            name="thermal_{}".format(i)
        )
        model.addConstrs(thermal[i][j] <= thermal_[i]['UB'][j] for j in range(len(thermal_[i])))
        model.addConstrs(-thermal[i][j] <= -thermal_[i]['LB'][j] for j in range(len(thermal_[i])))
        c_thermal[i] = np.array(thermal_[i]['OBJ'])

    c_exchange = exchange_cost.values
    exchange = model.addMVar((n_regions+1,n_regions+1), 
                          # ub=exchange_ub.values.flatten(), 
                          ub=exchange_ub.values, 
                          name="exchange")    
    thermal_sum = model.addMVar(n_regions, name="thermal_sum")

    model.addConstrs(thermal_sum.tolist()[i] == sum(thermal[i].tolist()) for i in range(n_regions))
    
    model.addConstrs(
            thermal_sum[i] 
            + sum(deficit[i,j] for j in range(n_regions)) 
            + hydro[i] 
            - sum(exchange[i,j] for j in range(n_regions+1))
            + sum(exchange[j,i] for j in range(n_regions+1)) 
            == demand[i]
            for i in range(n_regions)
        )
    model.addConstr(
            sum(exchange[j,n_regions] for j in range(n_regions+1)) 
            - sum(exchange[n_regions,j] for j in range(n_regions+1)) 
            == 0
    )
        
    model.addConstr(
            stored_now + spill + hydro - stored_past == np.zeros(n_regions), 
            name="rand")

    model.setObjective(c_spill@spill 
                   + sum(c_deficit[i]@deficit[i] for i in range(n_regions))
                   + sum(c_exchange[i]@exchange[i] for i in range(n_regions+1))
                   + sum(c_thermal[i]@thermal[i] for i in range(n_regions)),
                   gp.GRB.MINIMIZE)

    # Last state variable constraint as dummy variable
    model.addConstr(np.eye(n_regions)@stored_past == np.zeros(n_regions), name="dummy")

    model.setObjective(c_spill@spill 
                   + sum(c_deficit[i]@deficit[i] for i in range(n_regions))
                   + sum(c_exchange[i]@exchange[i] for i in range(n_regions+1))
                   + sum(c_thermal[i]@thermal[i] for i in range(n_regions)),
                   gp.GRB.MINIMIZE)

    M_h = la.norm(c_spill, ord=1)
    for i in range(n_regions):
        M_h += la.norm(c_deficit[i], ord=1) 
        M_h += la.norm(c_exchange[i], ord=1) 
        M_h += la.norm(c_thermal[i], ord=1)
    M_h += la.norm(c_exchange[n_regions], ord=1)

    # TEMP
    # m.setParam(gp.GRB.Param.FeasibilityTol, 1e-2)
    # m.setParam(gp.GRB.Param.IntFeasTol, 1e-1)

    model.update()

    x_0 = hydro_['INITIAL'][:n_regions].to_numpy()

    hydro_solver = solver.GurobiSolver(
        model, 
        stored_now, 
        now_var_name_arr,
        lam, 
        scenarios,
        M_h,
        x_lb_arr,
        x_ub_arr,
        h_min_val=0,
        past_state_for_min_val=past_state_for_min_val if has_ctg else None,
        past_state_for_max_val=past_state_for_max_val if has_ctg else None
    )

    return hydro_solver, x_0

def get_gurobi_var_name_idx(mdl, key):
    name_arr = [var.VarName for (idx, var) in enumerate(mdl.getVars()) if (key in var.VarName)]
    idx_arr = [idx for (idx, var) in enumerate(mdl.getVars()) if (key in var.VarName)]
    return name_arr, idx_arr

def get_gurobi_constr_name_idx(mdl, key):
    name_arr = [constr.ConstrName for (idx, constr) in enumerate(mdl.getConstrs()) if (key in constr.ConstrName)]
    idx_arr = [idx for (idx, constr) in enumerate(mdl.getConstrs()) if (key in constr.ConstrName)]
    return name_arr, idx_arr

def get_cAb_model(mdl):
    A = mdl.getA().todense()
    c = np.array([var.Obj for var in mdl.getVars()])
    rhs = np.copy(mdl.rhs)
    sense_arr = list(map(lambda constr : constr.sense, mdl.getConstrs()))
    lb_arr = [var.lb for var in mdl.getVars()]
    ub_arr = [var.ub for var in mdl.getVars()]

    return (c,A,rhs,sense_arr,lb_arr,ub_arr)

def test_gurobi_model_cost(mdl, cost):
    assert abs(mdl.objVal - cost)/(1e-8 + min(abs(mdl.objVal), abs(cost))) <= 1e-12, "mdl.objVal=%.4e != %.4e (expected cost)" % (mdl.objVal, cost)

def test_gurobi_var_val(mdl, idx_arr, val_arr):
    var_arr = np.array([mdl.getVars()[i].X for i in idx_arr])
    assert len(var_arr) == len(val_arr), "Number of vars with key (%d) does not match given val_arr len (%d)" % (len(var_arr), len(val_arr))
    val_arr = np.array(val_arr)
    assert la.norm(val_arr-var_arr)/(1e-8 + min(la.norm(val_arr), la.norm(var_arr))) <= 1e-12, "val_arr=%s and var_arr=%s are not equal" % (val_arr, var_arr)

def create_hierarchical_inventory_gurobi_model(
        N, lam, seed, k1, k2, eta1_scale, tau1_scale, eta2_scale, warm_start_x=False, has_ctg=True
    ):
    """ Creates basic hierarchical inventory problem """

    n = 4 # number of high-level items
    m = 6 # number of lower-level distributors

    # for testing
    is_rnd = 1
    if not is_rnd:
        n = m = 2
        seed = 0

    rng = np.random.default_rng(seed)
    c_arr = 1 + np.cos(np.pi/3) + is_rnd*rng.random(size=n)
    b_arr = 2.3 + is_rnd*rng.random(size=n)
    h_arr = 0.1 + is_rnd*0.2*rng.uniform(size=n)
    phi = 1.6
    scenarios = np.zeros((N+1,n), dtype=float)
    scenarios[0] = np.arange(9,9+n)
    scenarios[1:,:] = (5.5 + is_rnd*10*phi*rng.uniform(size=(N,n)))

    # first stage
    mdl = gp.Model()
    x_t = mdl.addMVar(n, lb=0, ub=50, name="state")
    u_t = mdl.addMVar(n, lb=0, ub=50, name="ctrl")
    y_t = mdl.addMVar(n, lb=-25, ub=50, name="next")
    z_t = mdl.addMVar(n, lb=0, ub=50)
    mdl.addConstr(x_t - (y_t + u_t) == 0)
    mdl.addConstr(z_t - y_t == 0, name="rand")
    mdl.addConstr(z_t == 0, name="dummy")
    mdl.setObjective(c_arr@u_t, gp.GRB.MINIMIZE)
    x_0 = 5*rng.uniform(size=n)
    mdl.update()
    (c,A,rhs,sense_arr,lb_arr,ub_arr) = get_cAb_model(mdl)
    _, dummy_idx_arr = get_gurobi_constr_name_idx(mdl, "dummy")
    _, rand_constr_idx_arr = get_gurobi_constr_name_idx(mdl, "rand") 
    B = np.zeros((A.shape[0], n), dtype=float)
    B[dummy_idx_arr,:] = -np.eye(n)

    if not is_rnd:
        rhs[rand_constr_idx_arr] = scenarios[0]
        test_mdl = gp.Model()
        x = test_mdl.addMVar(len(lb_arr), lb=lb_arr, ub=ub_arr, obj=c)
        x_prev = np.ones(2)
        for i in range(len(rhs)):
            if sense_arr[i] == '=':
                test_mdl.addConstr(A[i]@x + B[i]@x_prev == rhs[i])
            elif sense_arr[i] == '>':
                test_mdl.addConstr(A[i]@x + B[i]@x_prev >= rhs[i])
            else:
                test_mdl.addConstr(A[i]@x + B[i]@x_prev <= rhs[i])
        _, state_idx_arr = get_gurobi_var_name_idx(mdl, "state") 
        test_mdl.addConstr(x[state_idx_arr] >= 1.5)
        test_mdl.update()
        test_mdl.optimize()
        test_gurobi_model_cost(test_mdl, 30.)
        _, ctrl_var_idx_arr = get_gurobi_var_name_idx(mdl, "ctrl") 
        test_gurobi_var_val(test_mdl, ctrl_var_idx_arr, [9.5,10.5])

    # second stage 
    mdl2 = gp.Model()
    p = mdl2.addMVar(n, lb=0, ub=50, name="parts")
    s = mdl2.addMVar(n, lb=0, ub=25, name="slack")
    x = mdl2.addMVar(n, lb=-25, ub=50)
    v = mdl2.addMVar(m, lb=0, ub=50, name="prods")
    mdl2.addConstrs((p[i] == x[i] + s[i] + gp.quicksum(v) for i in range(n)), name="convert")
    mdl2.addConstr(x == 0, name="dummy")
    mdl2.addConstr(v <= 0, name="rand")
    mdl2.setObjective(gp.quicksum(v) + gp.quicksum(p) + gp.quicksum(s)) 
    mdl2.update()
    (c2,A2,rhs2,sense2_arr,lb2_arr,ub2_arr) = get_cAb_model(mdl2)
    n_prev = len(mdl.getVars())
    B2 = np.zeros((A2.shape[0], n_prev), dtype=float)
    _, dummy_idx_arr = get_gurobi_constr_name_idx(mdl2, "dummy")
    _, next_idx_arr  = get_gurobi_var_name_idx(mdl, "next")
    B2[dummy_idx_arr,next_idx_arr] = -np.ones(len(next_idx_arr))

    # get columns of random components
    _, p_var_idx = get_gurobi_var_name_idx(mdl2, "parts")
    _, s_var_idx = get_gurobi_var_name_idx(mdl2, "slack")
    _, v_var_idx = get_gurobi_var_name_idx(mdl2, "prods")
    _, convert_constr_idx = get_gurobi_constr_name_idx(mdl2, "convert")
    v_var_idx = np.array(v_var_idx)
    convert_constr_idx = np.array(convert_constr_idx)
    _, rand_constr_idx = get_gurobi_constr_name_idx(mdl2, "rand")
    mu = -0.5*np.ones(m)
    L  = rng.normal(size=(m,m))
    Sig = L@L.T
    def get_stochastic_lp_params():
        tmp = rng.multivariate_normal(mean=mu, cov=Sig) if is_rnd else mu
        c2[v_var_idx] = l = np.minimum(9, np.maximum(-10, tmp))
        c2[p_var_idx] = h = 0.1 + is_rnd*0.2*rng.uniform(size=n)
        c2[s_var_idx] = b = 10 + is_rnd*5*rng.uniform(size=n)

        a1 = rng.binomial(n=1,p=1./(m+n),size=(len(convert_constr_idx), len(v_var_idx)))
        a2 = 2./m*rng.uniform(size=a1.shape)
        # https://stackoverflow.com/questions/22927181/selecting-specific-rows-and-columns-from-numpy-array
        A2[np.ix_(convert_constr_idx,v_var_idx)] = -np.multiply(a1,a2) if is_rnd else -1./m*np.ones(a2.shape)
        A2[convert_constr_idx,v_var_idx[:n]] = -1./m

        rhs2[rand_constr_idx] = d = np.minimum(50, rng.poisson(5, size=m)) if is_rnd else 5
        return (c2, A2, rhs2)

    if not is_rnd:
        test_mdl = gp.Model()
        (c2, A2, rhs2) = get_stochastic_lp_params()
        x = test_mdl.addMVar(len(lb2_arr), lb=lb2_arr, ub=ub2_arr, obj=c2)
        y_t = rng.uniform(size=len(mdl.getVars()))
        y_t[next_idx_arr] = np.array([1,-1])
        for i in range(len(rhs2)):
            if sense2_arr[i] == '=':
                test_mdl.addConstr(A2[i]@x + B2[i]@y_t == rhs2[i])
            elif sense2_arr[i] == '>':
                test_mdl.addConstr(A2[i]@x + B2[i]@y_t >= rhs2[i])
            else:
                test_mdl.addConstr(A2[i]@x + B2[i]@y_t <= rhs2[i])
        test_mdl.update()
        test_mdl.optimize()
        test_gurobi_model_cost(test_mdl, -4.)
        _, parts_var_idx_arr = get_gurobi_var_name_idx(mdl2, "parts") 
        _, prods_var_idx_arr = get_gurobi_var_name_idx(mdl2, "prods") 
        _, slack_var_idx_arr = get_gurobi_var_name_idx(mdl2, "slack") 
        test_gurobi_var_val(test_mdl, parts_var_idx_arr, [6,4])
        test_gurobi_var_val(test_mdl, prods_var_idx_arr, [5,5])
        test_gurobi_var_val(test_mdl, slack_var_idx_arr, [0,0])

    hier_inv_solver = solver.PDSASolverForLPs(
        lam, c, A, b, B, lb_arr, ub_arr, sense_arr, scenarios, rand_idx_arr,
        B2, lb2_arr, ub2_arr, sense2_arr, get_stochastic_lp_params, 
        k1, k2, eta1_scale, tau1_scale, eta2_scale, warm_start_x,
    )
    x_0 = rng.uniform(n)

    return hier_inv_solver, x_0

if __name__ == '__main__':
    create_hierarchical_inventory_gurobi_model(10, 0.9, 0)
