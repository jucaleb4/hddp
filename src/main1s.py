from hddp import HDDP_multiproc
from solver import GurobiSolver
import gurobipy as gp
import numpy as np
import pandas
import time

def opt_setup_inventory_basic(N, lam):
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
    solvers = []
    h = 0.2
    b = 2.8
    c = np.cos(np.pi/3) + 1.5
    d = 9.0
    phi = 0.6

    # define gurobi model
    m = gp.Model()
    now = m.addMVar(1, lb=-1000)
    past = m.addMVar(1, lb=-1000)
    # control variables
    u = m.addMVar(1, lb=0) # quantity we order
    v = m.addMVar(1, lb=0) # over-supply
    w = m.addMVar(1, lb=0) # under-supply
    # cost-to-go
    t = m.addMVar(1, lb=0)

    rand_constrs = [None] * 3
    rand_constrs[0] = m.addConstr(-now + (past + u) == 0, name='rand[0]') # .values()
    rand_constrs[1] = m.addConstr(v + (past + u) >= 0, name='rand[1]') # .values()
    rand_constrs[2] = m.addConstr(-w + (past + u) <= 0, name='rand[2]') # .values()
    dummy_constrs = m.addConstr(np.eye(1)@past == np.zeros(1), name="dummy[0]") # .values()

    m.setObjective(c*u + b*v + h*w + lam*t, gp.GRB.MINIMIZE)

    np.random.seed(1)
    scenarios = np.append([5.5], d*np.ones(N) + phi*np.random.uniform(size=N))
    avg = np.mean(scenarios[1:])
    # scenarios = np.append([5.5], d*np.ones(N) + phi*0.5)
    scenarios = np.reshape(np.repeat(scenarios, 3), newshape=(-1,3))
    solver = GurobiSolver(m, now, t, scenarios)

    x_0 = np.array([10])
    return solver, x_0, avg

def opt_setup_hydro_basic(N, lam):
    solvers = []

    discount = lam
    nr = 4
    fname = "./data/"
    
    hydro_ = pandas.read_csv(fname + "hydro.csv", index_col=0)
    demand = pandas.read_csv(fname + "demand.csv", index_col=0)
    deficit_ = pandas.read_csv(fname + "deficit.csv", index_col=0)
    exchange_ub = pandas.read_csv(fname + "exchange.csv", index_col=0)
    exchange_cost = pandas.read_csv(fname + "exchange_cost.csv", index_col=0)
    thermal_ = [pandas.read_csv(fname + "thermal_{}.csv".format(i),
        index_col=0) for i in range(nr)]

    start_time = time.time()

    # historical rainfall data
    hist = [pandas.read_csv(fname + "hist_{}.csv".format(i), sep=";") for i in range(nr)]
    hist = pandas.concat(hist, axis=1)
    hist.dropna(inplace=True)
    hist.drop(columns='YEAR', inplace=True)
    scenarios = [hist.iloc[:,12*i:12*(i+1)].transpose().values for i in range(nr)]
    # [region][month][year]
    scenarios = np.array(scenarios)
    scenarios = np.mean(scenarios, axis=1)

    assert nr == scenarios.shape[0]

    means  = np.mean(scenarios, axis=1)
    sigmas = np.std(scenarios, axis=1)

    lognorm_sigmas= np.sqrt(np.log(np.power(np.divide(sigmas,means), 2) + 1))
    lognorm_means = np.log(means) - np.square(lognorm_sigmas)/2

    scenarios = np.array([np.random.lognormal(
                          mean =lognorm_means[i], 
                          sigma=lognorm_sigmas[i], size=N) 
                          for i in range(nr)])
    # TEMP
    # scenarios = np.diag(means) @ np.ones(scenarios.shape) 
    scenario_0 = np.array([hydro_['INITIAL'][nr:2*nr].to_numpy()]).T
    scenarios = np.hstack((scenario_0, scenarios)).T

    demand = demand.to_numpy()
    # get monthly avg
    demand = np.mean(demand, axis=0)

    # define Gurobi model
    m = gp.Model()

    # TEMP
    stored_now = m.addMVar(nr, ub=hydro_['UB'][:nr], name="stored")
    # 59419.3  5874.9 12859.2  5271.5
    # stored_now = m.addMVar(nr, ub=[100000,100000,100000,100000], name="stored")
    stored_past= m.addMVar(nr, name="stored_past")
    spill = m.addMVar(nr, name="spill")
    c_spill = 0.001 * np.ones(nr)
    hydro = m.addMVar(nr, ub=hydro_['UB'][-4:], name="hydro")    

    c_deficit = np.array([[deficit_['OBJ'][j] for i in range(4)] for j in range(4)])
    deficit = m.addMVar((nr,nr),
                        ub = [demand[i] * deficit_['DEPTH'][j] 
                        for i in range(nr) for j in range(nr)],
                        name = "deficit")

    thermal = [None] * nr
    c_thermal = [None] * nr
    for i in range(nr):
        thermal[i] = m.addMVar(
            len(thermal_[i]), 
            ub=thermal_[i]['UB'], 
            lb=thermal_[i]['LB'], 
            name="thermal_{}".format(i)
        )
        c_thermal[i] = np.array(thermal_[i]['OBJ'])

    c_exchange = exchange_cost.values
    exchange = m.addMVar((nr+1,nr+1), 
                          ub=exchange_ub.values.flatten(), 
                          name="exchange")    
    thermal_sum = m.addMVar(nr, name="thermal_sum")

    m.addConstrs(thermal_sum.tolist()[i] == sum(thermal[i].tolist()) for i in range(nr))
    
    # Lots of time spent in the constraints here
    m.addConstrs(
            thermal_sum[i] 
            + sum(deficit[i,j] for j in range(nr)) 
            + hydro[i] 
            - sum(exchange[i,j] for j in range(nr+1))
            + sum(exchange[j,i] for j in range(nr+1)) 
            == demand[i]
            for i in range(nr)
        )
    m.addConstr(
            sum(exchange[j,nr] for j in range(nr+1)) 
            - sum(exchange[nr,j] for j in range(nr+1)) 
            == 0
    )

        
    m.addConstr(
            stored_now + spill + hydro - stored_past 
            == np.zeros(nr), name='rand')

    t = m.addMVar(1, lb=0)

    m.setObjective(c_spill@spill 
                   + sum(c_deficit[i]@deficit[i] for i in range(nr))
                   + sum(c_exchange[i]@exchange[i] for i in range(nr+1))
                   + sum(c_thermal[i]@thermal[i] for i in range(nr)) 
                   + lam*t,
                   gp.GRB.MINIMIZE)

    # Last state variable constraint as dummy variable
    m.addConstr(np.eye(nr)@stored_past == np.zeros(nr), name="dummy")

    print("Setup time: {}s".format(time.time() - start_time))

    solver = GurobiSolver(m, stored_now, t, scenarios)
    # solver = GurobiSolver(m, stored_now, t, scenarios, hydro, deficit, 
    #                       exchange, thermal, stored_past)
    
    x_0 = hydro_['INITIAL'][:nr].to_numpy()

    return solver, x_0

def opt_setup_hydro_homo(N, lam):
    solvers = []

    discount = lam
    nr = 4
    fname = "./data/"
    
    hydro_ = pandas.read_csv(fname + "hydro.csv", index_col=0)
    demand = pandas.read_csv(fname + "demand.csv", index_col=0)
    deficit_ = pandas.read_csv(fname + "deficit.csv", index_col=0)
    exchange_ub = pandas.read_csv(fname + "exchange.csv", index_col=0)
    exchange_cost = pandas.read_csv(fname + "exchange_cost.csv", index_col=0)
    thermal_ = [pandas.read_csv(fname + "thermal_{}.csv".format(i),
        index_col=0) for i in range(nr)]

    start_time = time.time()

    ubs = hydro_['UB'][:nr]
    avg = np.mean(ubs)
    scale = np.diag(np.divide(avg, ubs))

    # historical rainfall data
    hist = [pandas.read_csv(fname + "hist_{}.csv".format(i), sep=";") for i in range(nr)]
    hist = pandas.concat(hist, axis=1)
    hist.dropna(inplace=True)
    hist.drop(columns='YEAR', inplace=True)
    scenarios = [hist.iloc[:,12*i:12*(i+1)].transpose().values for i in range(nr)]
    # [region][month][year]
    scenarios = np.array(scenarios)
    scenarios = np.mean(scenarios, axis=1)

    assert nr == scenarios.shape[0]

    means  = np.mean(scenarios, axis=1)
    sigmas = np.std(scenarios, axis=1)

    lognorm_sigmas= np.sqrt(np.log(np.power(np.divide(sigmas,means), 2) + 1))
    lognorm_means = np.log(means) - np.square(lognorm_sigmas)/2

    scenarios = np.array([np.random.lognormal(
                          mean =lognorm_means[i], 
                          sigma=lognorm_sigmas[i], size=N) 
                          for i in range(nr)])
    # TEMP
    # scenarios = np.diag(means) @ np.ones(scenarios.shape) 
    scenario_0 = np.array([hydro_['INITIAL'][nr:2*nr].to_numpy()]).T
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
    stored_now = m.addMVar(nr, ub=ubs, name="stored")
    # 59419.3  5874.9 12859.2  5271.5
    # stored_now = m.addMVar(nr, ub=[100000,100000,100000,100000], name="stored")
    stored_past= m.addMVar(nr, name="stored_past")
    spill = m.addMVar(nr, name="spill")
    c_spill = 0.001 * np.ones(nr)
    hydro = m.addMVar(nr, ub=hydro_['UB'][-4:], name="hydro")    

    c_deficit = np.array([[deficit_['OBJ'][j] for i in range(4)] for j in range(4)])
    deficit = m.addMVar((nr,nr),
                        ub = [demand[i] * deficit_['DEPTH'][j]*scale[i,i]*scale[i,j] 
                        for i in range(nr) for j in range(nr)],
                        name = "deficit")

    thermal = [None] * nr
    c_thermal = [None] * nr
    for i in range(nr):
        thermal[i] = m.addMVar(
            len(thermal_[i]), 
            ub=thermal_[i]['UB']*scale[i,i], 
            lb=thermal_[i]['LB']*scale[i,i], 
            name="thermal_{}".format(i)
        )
        c_thermal[i] = np.array(thermal_[i]['OBJ'])

    c_exchange = exchange_cost.values
    exchange = m.addMVar((nr+1,nr+1), 
                          ub=exchange_ub.values.flatten(), 
                          name="exchange")    
    thermal_sum = m.addMVar(nr, name="thermal_sum")

    m.addConstrs(thermal_sum.tolist()[i] == sum(thermal[i].tolist()) for i in range(nr))
    
    # Lots of time spent in the constraints here
    m.addConstrs(
            thermal_sum[i] 
            + sum(deficit[i,j] for j in range(nr)) 
            + hydro[i] 
            - sum(exchange[i,j] for j in range(nr+1))
            + sum(exchange[j,i] for j in range(nr+1)) 
            == demand[i]
            for i in range(nr)
        )
    m.addConstr(
            sum(exchange[j,nr] for j in range(nr+1)) 
            - sum(exchange[nr,j] for j in range(nr+1)) 
            == 0
    )

        
    m.addConstr(
            stored_now + spill + hydro - stored_past 
            == np.zeros(nr), name='rand')

    t = m.addMVar(1, lb=0)

    m.setObjective(c_spill@spill 
                   + sum(c_deficit[i]@deficit[i] for i in range(nr))
                   + sum(c_exchange[i]@exchange[i] for i in range(nr+1))
                   + sum(c_thermal[i]@thermal[i] for i in range(nr)) 
                   + lam*t,
                   gp.GRB.MINIMIZE)

    # Last state variable constraint as dummy variable
    m.addConstr(np.eye(nr)@stored_past == np.zeros(nr), name="dummy")

    print("Setup time: {}s".format(time.time() - start_time))

    solver = GurobiSolver(m, stored_now, t, scenarios)
    # solver = GurobiSolver(m, stored_now, t, scenarios, hydro, deficit, 
    #                       exchange, thermal, stored_past)
    
    x_0 = hydro_['INITIAL'][:nr].to_numpy()
    x_0 = scale@x_0

    return solver, x_0

def opt_setup_simple(N, lam):
    solvers = []

    # define gurobi model
    m = gp.Model()
    n = 4
    now = m.addMVar(n, ub=2*np.ones(4))
    past = m.addMVar(n)

    # control variables
    n_u = 100
    u = m.addMVar(n_u) 
    # cost-to-go
    t = m.addMVar(1, lb=0)

    np.random.seed(0)
    n_c = 20
    A = 10 * (2*np.random.rand(n_c,n)-1)
    B = 100 * (2*np.random.rand(n_c,n)-1)
    C = 1000 * (2*np.random.rand(n_c, n_u)-1)

    m.addConstr(A@now + B@past + C@u <= np.zeros(n_c), name="rand")
    m.addConstr(np.eye(n)@past == np.zeros(n), name="dummy")

    c = 2*np.random.rand(n_u)-1
    M = 4*np.random.rand(n_u, n_u)-2
    M = M.T@M
    m.setObjective((u@M@u) + c@u + lam*t, gp.GRB.MINIMIZE)

    scenarios = 1000 * (2*np.random.rand(N+1,n_c)-1)
    solver = GurobiSolver(m, now, t, scenarios, None, None)

    x_0 = 1.5*np.ones(n)
    return solver, x_0

def main():
    np.random.seed(1)
    N = 10
    nprocs = 1

    prob_id = 0

    solvers = [None]*nprocs
    for i in range(nprocs):
        if prob_id == 0:
            print("hydro")
            discount = 0.8
            solver, x_0 = opt_setup_hydro_basic(N, discount)
            params = {
                'L' : 0*np.zeros(len(x_0)),
                # 'R' : 201000*np.ones(len(x_0)),
                'R' : 100000*np.ones(len(x_0)),
                'T' : 50,
                'N' : N,
                'n' : len(x_0),
                'eps': 10000,
                'lam': discount,
                'fwd_T': 0, 
                'max_iter': 3000,
                'sddp_mode': True,
            }
            opt = 0

        elif prob_id == 1:
            # TODO: Changed demand and discount
            print("inventory")
            discount = 0.8
            solver, x_0, avg = opt_setup_inventory_basic(N, discount)
            params = {
                'L' : -11,
                'R' : 11,
                'T' : 100,
                'N' : N,
                'n' : len(x_0),
                'eps': 1,
                'lam': discount,
                'fwd_T': 10,
                'max_iter': 100,
                'sddp_mode': False
            }
            opt = (4.5*0.2) + discount*((avg-4.5)*2) + discount**2/(1-discount)*avg*2

        elif prob_id == 2:
            print("pathological")
            discount = 1.0-1e-5
            solver, x_0 = opt_setup_simple(N, discount)
            params = {
                'L' : 0,
                'R' : 3,
                'T' : 1000,
                'N' : N,
                'n' : len(x_0),
                'eps': 1,
                'lam': discount,
                'fwd_T': 1,
                'max_iter': -1,
                'sddp_mode': False
            }
            opt = 0

        elif prob_id == 3:
            print("hydro homo")
            discount = 0.8
            solver, x_0 = opt_setup_hydro_homo(N, discount)
            params = {
                'L' : np.zeros(len(x_0)),
                'R' : 100000*np.ones(len(x_0)),
                'T' : 50,
                'N' : N,
                'n' : len(x_0),
                'eps': 25000,
                'lam': discount,
                'fwd_T': 0, 
                'max_iter': -1,
                'sddp_mode': False,
            }
            opt = 0
        else: 
            print("undefined problem...")
            return 

        solvers[i] = solver

    # x, val = HDDP(x_0, eps, params, solver)
    x, val = HDDP_multiproc(x_0, params, solvers, nprocs)

    print("x_0:", x_0)
    print("F(x^*):", opt)
    print("Solution of x={} with value {}".format(x, val))

if __name__ == '__main__':
    main()
