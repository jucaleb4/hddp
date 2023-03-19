import gurobipy as gp
import numpy as np
import pandas
import time
from solver import GurobiSolver, GenericSolver

from typing import Tuple, Any

def opt_rareflow(N, lam):
    """
    1D Optimization problem models that models moving up stream, and the 
    likelihood of moving up is rare
    """
    m = gp.Model()
    n = 1

    now_var_name = "now"
    now = m.addMVar(n, lb=0, name=now_var_name)
    now_var_name_arr = np.array(["{}[{}]".format(now_var_name, i) for i in range(n)])
    past = m.addMVar(n, lb=0, name="past")
    # What's going on here...
    past_state_for_min_val = 9*np.ones(n)
    past_state_for_max_val = np.zeros(n)
    ctrl = m.addMVar(2, lb=-float("inf"), name="ctrl")

    m.addConstr(now[0] <= 10)
    m.addConstr(ctrl[0] >= -0)
    m.addConstr(ctrl[0] <= 1)
    m.addConstr(now[0] - past[0] - ctrl[0] == 0, name="rand[0]")
    m.addConstr(past[0] == 0, name="dummy[0]")
    m.setObjective(-now[0] + 0.1*ctrl[0], gp.GRB.MINIMIZE)
    scenarios = np.append(
        1,
        -0.9*np.ones(N)
    )
    scenarios = np.reshape(scenarios, (N+1, 1))
    solver = GurobiSolver(
        m, 
        now, 
        now_var_name_arr,
        lam, 
        scenarios,
        past_state_for_min_val=past_state_for_min_val,
        past_state_for_max_val=past_state_for_max_val,
    )

    x_0 = np.array([0])
    return solver, x_0

def opt_setup_inventory_basic(N, lam, rng):
    """ Setups basic inventory problem

    State (dim=1) is the amount of inventory
    Ctrl (dim=3) are (u_k,v_k,w_k)=(buy now, undersupply, oversupply)

    min  \sum\limits_{k=1}^∞ γ^k (cu_k + bv_k + hw_k)

    s.t. x^{k+1} - x^k - u_k = -D_k
         v_k >= D_k - (x^k+u_k)
         w_k >= (x^k+u_k) - D_k

         Bounds
         ------
         u_k,v_k,w_k >= 0
         x^k free
    """
    h = 0.2
    b = 2.8
    c = np.cos(np.pi/3) + 1.5
    d = 9.0
    phi = 0.6

    # define gurobi model
    m = gp.Model()
    now = m.addMVar(1, lb=-1000, name="now")
    now_var_name_arr = ["now[0]"]
    past_state_for_min_val = np.array([10])
    past_state_for_max_val = np.array([-1000])

    past = m.addMVar(1, lb=-1000)
    # control variables
    u = m.addMVar(1, lb=0) # quantity we order
    v = m.addMVar(1, lb=0) # over-supply
    w = m.addMVar(1, lb=0) # under-supply

    rand_constrs = [None] * 3
    rand_constrs[0] = m.addConstr(-now + (past + u) == 0, name='rand[0]') # .values()
    rand_constrs[1] = m.addConstr(v + (past + u) >= 0, name='rand[1]') # .values()
    rand_constrs[2] = m.addConstr(-w + (past + u) <= 0, name='rand[2]') # .values()
    dummy_constrs = m.addConstr(np.eye(1)@past == np.zeros(1), name="dummy[0]") # .values()

    m.setObjective(c*u + b*v + h*w, gp.GRB.MINIMIZE)

    scenarios = np.append([5.5], d*np.ones(N) + phi*rng.random(N))
    avg = np.mean(scenarios[1:])
    # scenarios = np.append([5.5], d*np.ones(N) + phi*0.5)
    scenarios = np.reshape(np.repeat(scenarios, 3), newshape=(-1,3))
    solver = GurobiSolver(
        m, 
        now, 
        now_var_name_arr,
        lam, 
        scenarios,
        past_state_for_min_val=past_state_for_min_val,
        past_state_for_max_val=past_state_for_max_val,
    )

    x_0 = np.array([10])

    return solver, x_0

def opt_simple(N, lam):
    """
    Simple optimiztion problem we want to make sure lb == ub.
    """
    m = gp.Model()
    n = 10

    now_var_name = "now"
    now = m.addMVar(n, lb=-100, name=now_var_name)
    now_var_name_arr = np.array(["{}[{}]".format(now_var_name, i) for i in range(n)])
    past = m.addMVar(n, lb=-100, name="past")
    past_state_for_min_val = np.zeros(n)
    past_state_for_max_val = n*np.ones(n)

    m.addConstr(now - past == np.zeros(n), name="rand")
    m.addConstr(past == np.zeros(n), name="dummy")
    m.setObjective(np.ones(n)@now, gp.GRB.MINIMIZE)
    scenarios = -0*np.ones((N+1)*n)
    scenarios = np.reshape(scenarios, (N+1, n))
    solver = GurobiSolver(
        m, 
        now, 
        now_var_name_arr,
        lam, 
        scenarios,
        past_state_for_min_val=past_state_for_min_val,
        past_state_for_max_val=past_state_for_max_val,
    )

    x_0 = np.arange(1,n+1)
    return solver, x_0

def opt_electricity_price_setup_v2(
            N_1: int, N_2: int, lam: float,
            rng: np.random._generator.Generator)  \
            -> Tuple[GenericSolver, np.ndarray]:
    """ Setsups basic 2 stage electricity price with focus on implementing via
    matrix form

    :params N_1: number of scenarios for first stage problem
    :params N_2: number of scenarios for second stage problem
    :params lam: discount factor
    :params rng: random number generator for settling the constraints
    :returns c: cost vector
    :returns A: constraint matrix
    :returns b: rhs vector
    :returns x_0: initial feasible (w.r.t. easy constraints) solution
    :returns proj_idx: which vectors need projection onto positive orthant
    :returns dummy_cons_idx: which constraints corresponds to previous state variable
    :returns scenarios: matrix of scenarios
    :returns tildeD: array of random demands for second stage problem
    """
    # TODO: Make these parameters?
    n = 10 # number of generators
    r = 4  # number of demands
    mrkt_cost = 25 # cost to generate power in first stage problem
    self_cost = 100 # cost to generate power in second stage problem
    beta = 0.8 + 0.2*rng.random() # battery degredation
    assert n >= r

    # define some parameters and randomness for the problem 
    battery_lb_arr = np.ones(r) # 5*rng.random(r)
    battery_ub_arr = 50 * 10*rng.random(r)
    generator_lb_arr = 200*rng.random(n)
    generator_ub_arr = 500 + 200*rng.random(n)

    scenarios = np.zeros((N_1+1, r))
    scenarios[0] = 50 + 150*rng.random(r)
    scenarios[1] = 75 + 125*rng.random(r)
    for i in range(1,N_1):
        scenarios[i+1] = (125 - i/50) + (200 - 125 + i/50)*rng.random(r)

    # demands for second stage problem
    tildeD = 150 + 50 * rng.random(N_2)

    m = gp.Model()

    # List of variables and constraints
    dummy_cons_idx = np.array([], dtype=int)
    rand_rhs_idx = np.array([], dtype=int)
    con_idx_ct = 0

    # define variables
    b = m.addMVar(r, name="battery_now")
    b_0 = m.addMVar(r, name="battery_past")
    c = mrkt_cost * np.ones(n)
    g = m.addMVar(n, obj=c, name="generator")
    hs = []

    # generator in second stage problem for scenarios in SAA
    for i in range(N_1):
        # projection
        h_i = m.addMVar(r, lb=0, name="generator_2nd_stage_{}".format(i))
        hs.append(h_i)

    # add constraints (initialize with lower bound)
    m.addConstr(b_0 == battery_lb_arr, "dummy")
    dummy_cons_idx = np.append(dummy_cons_idx, np.arange(r))
    con_idx_ct += r

    # demand (initalize with scenario 0)
    gen_split = np.array_split(np.arange(n), r)
    for i in range(r):
        idxs = gen_split[i]
        m.addConstr(
            sum(g[j] for j in idxs) 
            - b[i] + beta*b_0[i] 
            - sum(hs[j][i] for j in range(N_1)) 
            == scenarios[0][i], 
            name="rand[{}]".format(i)
        )
        rand_rhs_idx = np.append(rand_rhs_idx, con_idx_ct)
        con_idx_ct += 1

    # upper and lower bounds (with slack variables)
    s_b_lb = m.addMVar(r, name="battery_slack_lb")
    s_b_ub = m.addMVar(r, name="battery_slack_ub")
    m.addConstr(-b + s_b_lb == -battery_lb_arr, name="battery lb")
    m.addConstr(b + s_b_ub == battery_ub_arr, name="battery ub")
    con_idx_ct += 2*r

    s_g_lb = m.addMVar(n, name="generator_slack_lb")
    s_g_ub = m.addMVar(n, name="generator_slack_ub")
    m.addConstr(-g + s_g_lb == -generator_lb_arr, name="generator lb")
    m.addConstr(g + s_g_ub == generator_ub_arr, name="generator ub")
    con_idx_ct += 2*n

    s_secondstage_generator_ub_arr = 500 + 200*rng.random(r)

    # TODO: Provide option to make two stage problem as a single LP
    for i in range(N_1):
        s_h_i_lb = m.addMVar(r, name="secondstage_gen_slack_ub")
        m.addConstr(hs[i] + s_h_i_lb == s_secondstage_generator_ub_arr, name="secondstage_gen_ub")
        con_idx_ct += r

    # Set as minimization problem
    m.setAttr('ModelSense', 1)
    m.update()

    # TODO: Remove this
    if True:
        m.optimize()
        print("Optimal battery: {}".format(b.X))
        print("Optimal solution: {}".format([x.X for x in m.getVars()]))

    # get [c,A,b] for LP of problem
    c = m.getAttr('Obj', m.getVars())
    A = m.getA() # returns sparse array
    b = m.getAttr('RHS', m.getConstrs())

    # initial feasible solution for easy constraints
    x_0 = np.zeros(len(c))
    proj_idx = np.arange(len(c))
    # now batteries
    x_0[:r] = (battery_lb_arr + battery_ub_arr)/2
    # later batteries
    x_0[r:2*r] = (battery_lb_arr + battery_ub_arr)/2
    x_0[2*r:2*r+n] = (generator_lb_arr + generator_ub_arr)/2

    return c, A, b, x_0, proj_idx, dummy_cons_idx, rand_rhs_idx, scenarios, tildeD

def opt_electricity_price_setup(N, lam, rng):
    """
    Sets up basic 2 stage electricity price model.

    We will denote 
    - D : number of (distinct) regions with demand
    - R : number of regions with generators
    - N : number of scenarios
    """
    D = 10
    R = 9

    # Define constants
    c = 25 * np.ones(R)
    # TODO: set appropriate ranges
    thermal_ub = 500 + 200*rng.random(R)
    battery_ub = 50 * 10*rng.random(D)
    battery_degredation = 0.8 + 0.2*rng.random()

    scenarios = np.zeros((N+1, D))
    scenarios[0] = 50 + 150*rng.random(D)
    scenarios[1] = 75 + 125*rng.random(D)
    for i in range(1,N):
        scenarios[i+1] = (125 - i/50) + (200 - 125 + i/50)*rng.random(D)
    print(">> scenarios: \n{}\n".format(scenarios))

    harvest_demand = 150 + 50 * rng.random(N)
    # do we want to make this random for each scenario?
    harvest_limit = (200/D) + (150/D) * rng.random(D)
    transport_degredation = 0.7 + 0.3 * rng.random(D)

    m = gp.Model("electrictiy prices")

    # state variables
    now_var_name = "now_battery"
    now = m.addMVar(D, lb=0, name=now_var_name)
    now_var_name_arr = ["{}[{}]".format(now_var_name, i) for i in range(D)]
    past = m.addMVar(D, lb=0)
    past_state_for_min_val = battery_ub
    past_state_for_max_val = np.zeros(D)

    # control variables
    thermal = m.addMVar(R, lb=0, name="thermal")
    thermal_dist = m.addMVar(D*R, lb=0, name="thermal_dist")
    alloc_1_arr = []
    alloc_not_1_arr = []
    for i in range(N):
        # how much to allocate to region 1
        alloc_1 = m.addMVar(D, lb=0, name="alloc_in_region_1")
        alloc_not_1 = m.addMVar(D-1, lb=0, name="alloc_not_in_region_1")
        alloc_1_arr.append(alloc_1)
        alloc_not_1_arr.append(alloc_not_1)

    # constraints for 1st stage problem (first constraint is random RHS representing demand)
    ones_R = np.ones(R)
    ones_D_less = np.ones(D-1)
    m.addConstrs((ones_R@thermal_dist[i*R:(i+1)*R]
                  - sum(alloc_1_arr[s][i] for s in range(N))
                  + past[i] - battery_degredation * now[i] 
                  == scenarios[0][i] for i in range(D)), name="rand") 
    m.addConstrs((ones_R@thermal_dist[i::D] - thermal[i] == 0 for i in range(R)), name="demand_distribution")
    m.addConstr(thermal <= thermal_ub, name="thermal_limit")
    m.addConstr(past == np.zeros(D), name="dummy")
    m.addConstr(now <= battery_ub, name="battery_limit")

    # constraints for 2nd stage problem
    for i in range(N):
        m.addConstr(alloc_1_arr[i][0] + ones_D_less@alloc_not_1_arr[i] == harvest_demand[i], 
                    name="harvest[{}]".format(i)
                )
        m.addConstr(alloc_1_arr[i] <= harvest_limit, name="harvest_limit_{}".format(i))
        m.addConstrs((-transport_degredation[j-1]*alloc_1_arr[i][j] + alloc_not_1_arr[i][j-1] 
                      == 0 for j in range(1,D)), 
                      name="harvest_transport_degredation_{}".format(i)
                )

    m.setObjective(c@thermal, gp.GRB.MINIMIZE)
    # if lowerr:
    #     m.setParam(gp.GRB.Param.FeasibilityTol, 1e-2)
    #     m.setParam(gp.GRB.Param.IntFeasTol, 1e-1)
    # m.update()

    solver = GurobiSolver(
        m, 
        now, 
        now_var_name_arr,
        lam, 
        scenarios,
        past_state_for_min_val=past_state_for_min_val,
        past_state_for_max_val=past_state_for_max_val,
    )

    x_0 = 0.5 * battery_ub

    return solver, x_0

def opt_setup_hydro_basic(N, lam, rng):
    """ Setups basic hydro problem. 

    Args:
        N (int): number of scenarios
        lam (float): discount factor of cost-to-go
        rng (np.random.Generator)

    Returns:
        solver (GurobiSolver): solver object. See `solver.py` for details
        x_0 (np.array): initial state
    """

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
    print(">> scenarios: \n{}\n".format(scenarios))
    scenario_0 = np.array([hydro_['INITIAL'][n_regions:2*n_regions].to_numpy()]).T 
    scenarios = np.hstack((scenario_0, scenarios)).T

    demand = demand.to_numpy()
    # get monthly avg
    demand = np.mean(demand, axis=0)

    # define Gurobi model
    m = gp.Model()

    # stored_now = m.addMVar(n_regions, ub=hydro_['UB'][:n_regions], name="stored")
    now_var_name = "stored"
    stored_now = m.addMVar(n_regions, name=now_var_name)
    now_var_name_arr = ["{}[{}]".format(now_var_name, i) for i in range(n_regions)]
    m.addConstrs(stored_now[i] <= hydro_['UB'][i] for i in range(n_regions))

    # x_prev's that help achieve the maximum and minimum cost
    past_state_for_max_val = np.zeros(len(now_var_name))
    past_state_for_min_val = hydro_['UB']

    # 59419.3  5874.9 12859.2  5271.5
    # stored_now = m.addMVar(n_regions, ub=[100000,100000,100000,100000], name="stored")
    stored_past= m.addMVar(n_regions, name="stored_past")
    spill = m.addMVar(n_regions, name="spill")
    c_spill = 0.001 * np.ones(n_regions)
    # hydro = m.addMVar(n_regions, ub=hydro_['UB'][-4:], name="hydro")    
    hydro = m.addMVar(n_regions, name="hydro")    
    m.addConstrs(hydro[i] <= hydro_['UB'][-i] for i in range(n_regions))

    c_deficit = np.array([[deficit_['OBJ'][j] for i in range(4)] for j in range(4)])
    deficit = m.addMVar((n_regions,n_regions),
                        # ub = [demand[i] * deficit_['DEPTH'][j] 
                        # for i in range(n_regions) for j in range(n_regions)],
                        name = "deficit")
    m.addConstrs(deficit[i][j] <= demand[i] * deficit_['DEPTH'][j] 
                for i in range(n_regions) for j in range(n_regions))

    thermal = [None] * n_regions
    c_thermal = [None] * n_regions
    for i in range(n_regions):
        thermal[i] = m.addMVar(
            len(thermal_[i]), 
            # ub=thermal_[i]['UB'], 
            # lb=thermal_[i]['LB'], 
            name="thermal_{}".format(i)
        )
        m.addConstrs(thermal[i][j] <= thermal_[i]['UB'][j] for j in range(len(thermal_[i])))
        m.addConstrs(-thermal[i][j] <= -thermal_[i]['LB'][j] for j in range(len(thermal_[i])))
        c_thermal[i] = np.array(thermal_[i]['OBJ'])

    c_exchange = exchange_cost.values
    exchange = m.addMVar((n_regions+1,n_regions+1), 
                          ub=exchange_ub.values.flatten(), 
                          name="exchange")    
    thermal_sum = m.addMVar(n_regions, name="thermal_sum")

    m.addConstrs(thermal_sum.tolist()[i] == sum(thermal[i].tolist()) for i in range(n_regions))
    
    m.addConstrs(
            thermal_sum[i] 
            + sum(deficit[i,j] for j in range(n_regions)) 
            + hydro[i] 
            - sum(exchange[i,j] for j in range(n_regions+1))
            + sum(exchange[j,i] for j in range(n_regions+1)) 
            == demand[i]
            for i in range(n_regions)
        )
    m.addConstr(
            sum(exchange[j,n_regions] for j in range(n_regions+1)) 
            - sum(exchange[n_regions,j] for j in range(n_regions+1)) 
            == 0
    )
        
    m.addConstr(
            stored_now + spill + hydro - stored_past == np.zeros(n_regions), 
            name="rand")

    m.setObjective(c_spill@spill 
                   + sum(c_deficit[i]@deficit[i] for i in range(n_regions))
                   + sum(c_exchange[i]@exchange[i] for i in range(n_regions+1))
                   + sum(c_thermal[i]@thermal[i] for i in range(n_regions)),
                   gp.GRB.MINIMIZE)

    # Last state variable constraint as dummy variable
    m.addConstr(np.eye(n_regions)@stored_past == np.zeros(n_regions), name="dummy")

    m.setObjective(c_spill@spill 
                   + sum(c_deficit[i]@deficit[i] for i in range(n_regions))
                   + sum(c_exchange[i]@exchange[i] for i in range(n_regions+1))
                   + sum(c_thermal[i]@thermal[i] for i in range(n_regions)),
                   gp.GRB.MINIMIZE)

    # TEMP
    # m.setParam(gp.GRB.Param.FeasibilityTol, 1e-2)
    # m.setParam(gp.GRB.Param.IntFeasTol, 1e-1)

    m.update()

    print("Setup time: {}s".format(time.time() - start_time))

    x_0 = hydro_['INITIAL'][:n_regions].to_numpy()
    print(">> x_0={}".format(x_0))

    solver = GurobiSolver(
        m, 
        stored_now, 
        now_var_name_arr,
        lam, 
        scenarios,
        min_val=0,
        past_state_for_min_val=past_state_for_min_val,
        past_state_for_max_val=past_state_for_max_val,
    )


    return solver, x_0

def opt_setup_hydro_homo(N, lam):
    """ Same as opt_setup_hydro, but scales to make more homogenous """
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

    ubs = hydro_['UB'][:n_regions]
    avg = np.mean(ubs)
    scale = np.diag(np.divide(avg, ubs))

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

    scenarios = np.array([np.random.lognormal(
                          mean =lognorm_means[i], 
                          sigma=lognorm_sigmas[i], size=N) 
                          for i in range(n_regions)])
    # TEMP
    # scenarios = np.diag(means) @ np.ones(scenarios.shape) 
    scenario_0 = np.array([hydro_['INITIAL'][n_regions:2*n_regions].to_numpy()]).T
    scenarios = np.hstack((scenario_0, scenarios)).T

    demand = demand.to_numpy()
    # get monthly avg
    demand = np.mean(demand, axis=0)

    demand = scale@demand
    scenario = scenarios@scale
    ubs = scale@ubs

    # define Gurobi model
    m = gp.Model()

    # TEMP
    stored_now = m.addMVar(n_regions, ub=ubs, name="stored")
    # 59419.3  5874.9 12859.2  5271.5
    # stored_now = m.addMVar(n_regions, ub=[100000,100000,100000,100000], name="stored")
    stored_past = m.addMVar(n_regions, name="stored_past")
    spill = m.addMVar(n_regions, name="spill")
    c_spill = 0.001 * np.ones(n_regions)
    hydro = m.addMVar(n_regions, ub=hydro_['UB'][-4:], name="hydro")    

    c_deficit = np.array([[deficit_['OBJ'][j] for i in range(4)] for j in range(4)])
    deficit = m.addMVar((n_regions,n_regions),
                        ub = [demand[i] * deficit_['DEPTH'][j]*scale[i,i]*scale[i,j] 
                        for i in range(n_regions) for j in range(n_regions)],
                        name = "deficit")

    thermal = [None] * n_regions
    c_thermal = [None] * n_regions
    for i in range(n_regions):
        thermal[i] = m.addMVar(
            len(thermal_[i]), 
            ub=thermal_[i]['UB']*scale[i,i], 
            lb=thermal_[i]['LB']*scale[i,i], 
            name="thermal_{}".format(i)
        )
        c_thermal[i] = np.array(thermal_[i]['OBJ'])

    c_exchange = exchange_cost.values
    exchange = m.addMVar((n_regions+1,n_regions+1), 
                          ub=exchange_ub.values.flatten(), 
                          name="exchange")    
    thermal_sum = m.addMVar(n_regions, name="thermal_sum")

    m.addConstrs(thermal_sum.tolist()[i] == sum(thermal[i].tolist()) for i in range(n_regions))
    
    # Lots of time spent in the constraints here
    m.addConstrs(
            thermal_sum[i] 
            + sum(deficit[i,j] for j in range(n_regions)) 
            + hydro[i] 
            - sum(exchange[i,j] for j in range(n_regions+1))
            + sum(exchange[j,i] for j in range(n_regions+1)) 
            == demand[i]
            for i in range(n_regions)
        )
    m.addConstr(
            sum(exchange[j,n_regions] for j in range(n_regions+1)) 
            - sum(exchange[n_regions,j] for j in range(n_regions+1)) 
            == 0
    )

        
    m.addConstr(
            stored_now + spill + hydro - stored_past 
            == np.zeros(n_regions), name='rand')

    # get initial upper and lower bounds
    m.setObjective(c_spill@spill 
                   + sum(c_deficit[i]@deficit[i] for i in range(n_regions))
                   + sum(c_exchange[i]@exchange[i] for i in range(n_regions+1))
                   + sum(c_thermal[i]@thermal[i] for i in range(n_regions)),
                   gp.GRB.MINIMIZE)
    min_val = m.objVal
    m.setObjective(c_spill@spill 
                   + sum(c_deficit[i]@deficit[i] for i in range(n_regions))
                   + sum(c_exchange[i]@exchange[i] for i in range(n_regions+1))
                   + sum(c_thermal[i]@thermal[i] for i in range(n_regions)),
                   gp.GRB.MAXIMIZE)
    max_val = m.objVal 

    t = m.addMVar(1, lb=0)

    m.setObjective(c_spill@spill 
                   + sum(c_deficit[i]@deficit[i] for i in range(n_regions))
                   + sum(c_exchange[i]@exchange[i] for i in range(n_regions+1))
                   + sum(c_thermal[i]@thermal[i] for i in range(n_regions)) 
                   + lam*t,
                   gp.GRB.MINIMIZE)

    # Last state variable constraint as dummy variable
    m.addConstr(np.eye(n_regions)@stored_past == np.zeros(n_regions), name="dummy")

    print("Setup time: {}s".format(time.time() - start_time))

    solver = GurobiSolver(m, stored_now, t, scenarios)
    # solver = GurobiSolver(m, stored_now, t, scenarios, hydro, deficit, 
    #                       exchange, thermal, stored_past)
    
    x_0 = hydro_['INITIAL'][:n_regions].to_numpy()
    x_0 = scale@x_0

    return solver, x_0, min_val, max_val

if __name__ == '__main__':
    opt_setup_hydro_basic(10, 0.9)
