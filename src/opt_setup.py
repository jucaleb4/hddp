import gurobipy as gp
import numpy as np
import pandas
import time
from solver import GurobiSolver

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
