from hddp import HDDP_multiproc, EDDP_ONLY_LB_MODE, EDDP_ONLY_LB_MODE_LINEAR, \
    EDDP_UB_AND_LB_MODE, SDDP_MODE, ESDDP_MODE
from solver import GurobiSolver
import opt_setup
import gurobipy as gp
import numpy as np
import argparse

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

def run_electricity_pricing(scen_seed, N, nprocs, mode, niters, perturb, select_seed=0):
    lam = 0.95
    n = 10
    T = 60
    eps = 1

    M_h = 25
    M = 25
    uM = 10000
    D = 100000
    eps_lvls = np.zeros(T)
    eps_lvls[T-1] = M_h*D/(1-lam)
    _eps = 0.01
    for t in range(T-2,-1,-1):
        eps_lvls[t] = lam**(T-t-1) * eps_lvls[T-1]
        for tau in range(t, T-1):
            eps_lvls[t] += (M + uM)*_eps*lam**(tau-t)

    params = {
        'L' : 0*np.zeros(n),
        # 'R' : 201000*np.ones(len(x_0)),
        'R' : 25*np.ones(n),
        'T' : T,
        'N' : N,
        'n' : n,
        'eps': eps,
        'lam': lam,
        'M_0': 500, 
        'fwd_T': 0, 
        'max_iter': niters,
        'mode': mode,
        'perturb': perturb,
        'eps_lvls': eps_lvls,
        'evaluate_lb': True,
        'evaluate_ub': True,
    }

    # Create multiple solvers for each proc
    solvers = [None]*nprocs
    for i in range(nprocs):
        np.random.seed(scen_seed + i)
        solver, x_0 = opt_setup.opt_electricity_price_setup(N, lam)
        if n != len(x_0):
            print("Error: n != len(x_0) ({} != {}), exitting...".format(n, len(x_0)))
            exit(0)
        solvers[i] = solver

    np.random.seed(select_seed)

    # x, val = HDDP(x_0, eps, params, solver)
    x, val = HDDP_multiproc(x_0, params, solvers, nprocs)

    print("x_0:", x_0)
    # print("F(x^*):", opt)
    print("Solution of x={} with value F(x)={}".format(x, val))

def run_hydro_basic(scen_seed, N, nprocs, mode, niters, perturb, select_seed=0):
    """
    Runs SDDP on hydro problem 
    
    Args:
        scne_seed (int): seed for random number generator
        N (int): number of scenarios
        nprocs (int): number of processes to run in parallel
        mode (int): mode of SDDP (SDDP_MODE, ESDDP_MODE, EDDP_ONLY_LB_MODE, EDDP_UB_AND_LB_MODE
        niters (int): number of iterations to run SDDP for
        perturb (bool): whether to perturb the solutions
        select_seed (int): seed for scenario selection
    """
    N = 10
    lam = 0.9906
    nprocs = 1
    n = 4
    T = 120
    eps = 10000

    M_h = 25
    M = 25
    uM = 10000
    D = 100000
    eps_lvls = np.zeros(T)
    eps_lvls[T-1] = M_h*D/(1-lam)
    _eps = 0.01
    for t in range(T-2,-1,-1):
        eps_lvls[t] = lam**(T-t-1) * eps_lvls[T-1]
        for tau in range(t, T-1):
            eps_lvls[t] += (M + uM)*_eps*lam**(tau-t)

    params = {
        'L' : 0*np.zeros(n),
        # 'R' : 201000*np.ones(len(x_0)),
        'R' : 100000*np.ones(n),
        'T' : T,
        'N' : N,
        'n' : n,
        'eps': eps,
        'lam': lam,
        'M_0': 500, 
        'fwd_T': 0, 
        'max_iter': niters,
        'mode': mode,
        'perturb': perturb,
        'eps_lvls': eps_lvls,
        'evaluate_lb': False,
        'evaluate_ub': False,
    }

    # Create multiple solvers for each proc
    solvers = [None]*nprocs
    # np.random.seed(scen_seed)
    rng = np.random.default_rng(scen_seed)
    for i in range(nprocs):
        solver, x_0 = opt_setup.opt_setup_hydro_basic(N, lam, rng)
        if n != len(x_0):
            print("Error: n != len(x_0) ({} != {}), exitting...".format(n, len(x_0)))
            exit(0)
        solvers[i] = solver

    np.random.seed(select_seed)

    # x, val = HDDP(x_0, eps, params, solver)
    x, val = HDDP_multiproc(x_0, params, solvers, nprocs)

    print("x_0:", x_0)
    # print("F(x^*):", opt)
    print("Solution of x={} with value F(x)={}".format(x, val))

if __name__ == '__main__':

    def nonnegative_type(x):
        x = int(x)
        if x < 0:
            raise argparse.ArgumentTypeError("Input argument must be nonnegative integer")
        return x

    def positive_type(x):
        x = int(x)
        if x < 1:
            raise argparse.ArgumentTypeError("Input argument must be positive integer")
        return x

    parser = argparse.ArgumentParser()
    parser.add_argument("--N",      type=positive_type,    help="Number of scenarios", default=10)
    parser.add_argument("--scen_seed", type=nonnegative_type, help="Seed for scenario generation", default=0)
    parser.add_argument("--sel_seed", type=nonnegative_type, help="Seed for scenario selection", default=0)
    parser.add_argument("--prob",   type=nonnegative_type, help="Problem ID. (hydro=1, price=2)", default=1)
    parser.add_argument("--nprocs", type=positive_type,    help="Number of processors", default=1)
    parser.add_argument("--mode",   type=nonnegative_type, help="SDDP mode (see header hddp.py)", default=0)
    parser.add_argument("--niters", type=positive_type,    help="Max numner of iterations", default=1000)
    parser.add_argument("--perturb", action="store_true", help="Perturb solution (with sig=0.1)", default=False)

    args = parser.parse_args()

    if args.prob == 1:
        run_hydro_basic(args.scen_seed, args.N, args.nprocs, args.mode, args.niters, args.perturb, args.sel_seed)
    elif args.prob == 2:
        run_electricity_pricing(args.scen_seed, args.N, args.nprocs, args.mode, args.niters, args.perturb, args.sel_seed)
    else:
        print("invalid problem id {}".format(args.prob))
