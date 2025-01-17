from hddp import solver
from hddp import utils 
import numpy as np
import numpy.linalg as la
from multiprocessing import Process, JoinableQueue, Queue
import time

def get_epsilon_ts(M, eps, eps_T, T, lam, **kwargs):
    eps_ts_arr = np.zeros(T, dtype=float)
    eps_ts_arr[-1] = eps_T
    for t in range(T-2,-1,-1):
        eps_ts_arr[t] = 2*M*eps + lam*eps_ts_arr[t+1]
    return eps_ts_arr

def fill_in_settings(x, settings, solver_arr):
    """ Fill in missing parameters based on problem (in solvers) """
    settings["M"] = solver_arr[0].M_h/(1.-settings['lam'])
    settings["eps_T"] = (solver_arr[0].h_max_val-solver_arr[0].h_min_val)/(1.-settings['lam'])
    settings["eps_lvls"] = get_epsilon_ts(**settings)
    settings["n"] = len(x)
    settings["x_lb_arr"] = solver_arr[0].x_lb_arr
    settings["x_ub_arr"] = solver_arr[0].x_ub_arr

def HDDP_multiproc(x_0, settings, solver_arr, n_procs):
    setup_qs = False
    fill_in_settings(x_0, settings, solver_arr)

    try:
        [q_host, q_child, ps, settings] = init_workers(x_0, settings, solver_arr, n_procs)
        setup_qs = True
        _HDDP_multiproc(
            x_0, 
            settings, 
            solver_arr[0], 
            n_procs, 
            q_host, 
            q_child, 
            is_host=True
        )
    except (KeyboardInterrupt, SystemExit):
        pass

    if setup_qs:
        for i in range(n_procs-1):
            ps[i].join()
        q_host.close()
        q_child.close()

def _HDDP_multiproc(x_0, settings, prob_solver, n_procs, q_host, q_child, is_host):
    """
    Parallel variant of hierarchical EDDP. The host is processor p1, and the
    remaining are p[2,...,@n_procs]. The host and children communicate with each
    other using multiprocessor's Queue to relay information.

    We write x_curr = x^k and x_next = x^{k+1}

    :param x_0: initial point
    :param settings: settings
    :param solver: 
    :param n_procs: 
    :param q_host:
    :param q_child:
    :param is_host:
    """
    S = utils.SaturatedSet(**settings)
    x_curr = np.copy(x_0)
    [start_scenario_idx, end_scenario_idx] = settings['scenario_endpts']
    [single_stage_min, single_stage_max] = prob_solver.get_single_stage_lbub()
    beta = 1./(1.-settings['lam'])
    lb_model = solver.LowerBoundModel(settings['n'], single_stage_min*beta)
    ub_model = solver.UpperBoundModel(prob_solver.get_gurobi_model(), 
        prob_solver.get_scenarios()[1:], 
        settings["lam"], 
        prob_solver.M_h*beta, 
        prob_solver.get_var_names(), 
        single_stage_max*beta,
    )

    settings["rng"] = np.random.default_rng(settings.get("alg_seed", None))

    s0_time = time.time()
    # elapsed time since genesis
    total_time_arr = np.zeros(settings['max_iter']+1)
    fwd_time_arr = np.zeros(settings['max_iter']+1)
    select_time_arr = np.zeros(settings['max_iter']+1)
    eval_time_arr = np.zeros(settings['max_iter']+1)
    comm_time_arr = np.zeros(settings['max_iter']+1)
    lb_arr = np.zeros(settings['max_iter'], dtype=float)
    ub_arr = np.zeros(settings['max_iter'], dtype=float)

    for k in range(1, settings['max_iter']+1):
        # forward phase: solve subproblems
        s_time = time.time()
        temp = solve_scenarios(prob_solver, x_0, x_curr, start_scenario_idx, end_scenario_idx)
        [agg_x, agg_val, agg_grad, agg_ctg] = temp

        # check termination
        x_0_sol = agg_x[0]
        if (k % settings['T'] == 1) and S.get(x_0_sol) <= 1:
            break
        fwd_time_arr[k] = fwd_time_arr[k-1] + time.time() - s_time

        if is_host:
            s_time = time.time()
            [agg_x, agg_val, agg_grad] = host_forward_get(n_procs, q_host, agg_x, agg_val, agg_grad)
            comm_time_arr[k] = comm_time_arr[k-1] + time.time() - s_time
            s_time = time.time()
            if settings['mode'] == utils.Mode.GAP_INF_EDDP:
                update_S_with_ub_and_lb(S, agg_x, lb_model, ub_model, settings["eps_lvls"])
            temp = get_cut_and_x_next(agg_x, agg_val, agg_grad, S, k, x_0, settings)
            [x_next, z_next, avg_val, avg_grad] = temp
            S.update(x_curr, min(S.get(x_curr), S.get(z_next) - 1))
            select_time_arr[k] = select_time_arr[k-1] + time.time() - s_time

            # evaluate
            s_time = time.time()
            lb_model.add_cut(avg_val, avg_grad, x_curr)
            ub_model.add_search_point_to_ub_model(x_curr)
            lb_arr[k-1], ub_arr[k-1] = evaluate_bounds(x_0_sol, agg_val[0], 
                                                      agg_ctg[0], ub_model, 
                                                      settings['lam'], k, 
                                                      time.time() - s0_time, 
                                                      k % 100 == 0)
            eval_time_arr[k] = eval_time_arr[k-1] + time.time() - s_time

            s_time = time.time()
            outmail = [x_next, avg_val, avg_grad]
            for _ in range(n_procs-1):
                q_child.put(outmail) 
            comm_time_arr[k] += time.time() - s_time

        elif not is_host:
            outmail = [agg_x, agg_val, agg_grad]
            q_host.put(outmail) 
            [x_next, avg_val, avg_grad] = q_child.get() 

        s_time = time.time()
        prob_solver.add_cut(avg_val, avg_grad, x_curr)
        x_curr = x_next
        select_time_arr[k] += time.time() - s_time

        total_time_arr[k] = time.time() - s0_time
        if total_time_arr[k] >= settings['time_limit']:
            break

    utils.save_logs(settings['log_folder'], total_time_arr, fwd_time_arr, select_time_arr, eval_time_arr, comm_time_arr, lb_arr, ub_arr) 

def evaluate_bounds(x_0_sol, val_0, ctg_0, ub_model, lam, k, elpsd_time, print_progress=True):
    lb = val_0
    ub_eval = ub_model.evaluate_ub(x_0_sol)
    ub = val_0 - lam*(ctg_0 - ub_eval)

    if print_progress:
        print("k=%d | %.1es | %.2e | %.2e" % (k, elpsd_time, lb, ub))

    return lb, ub

def update_S_with_ub_and_lb(S, agg_x, lb_model, ub_model, eps_lvls):
    """
    Update saturation data structure with upper and lower bounds
    """
    for x_i in agg_x:
        x_lb = lb_model.evaluate_lb(x_i)
        x_ub = ub_model.evaluate_ub(x_i)
        gap = x_ub - x_lb

        if gap < 0:
            print("WARNING: x_i_lb > x_i_ub ({} > {}). Consider increasing Lipschitz constant M".format(
                x_lb, x_ub
            ))
            continue

        eps_lvls_with_endpt = np.append(np.append(-1, eps_lvls), np.inf)
        lvl_based_on_gap = np.argmax(np.logical_and(
            eps_lvls_with_endpt[:-1] <= gap,
            gap < eps_lvls_with_endpt[1:]
        ))
        lvl_based_on_saturation = S.get(x_i)
        if lvl_based_on_gap < lvl_based_on_saturation:
            print("Updating S(x_i)={} -> {} (gap={})".format(
                lvl_based_on_saturation, 
                lvl_based_on_gap,
                gap
            ))
            S.update(x_i, lvl_based_on_gap)

def host_forward_get(n_procs, q_host, agg_x, agg_val, agg_grad):
    """ Gathers all subproblem solutions from the forward phase """
    # gather all solutions
    for _ in range(n_procs-1):
        [proc_i_agg_x, proc_i_agg_val, proc_i_agg_grad] = q_host.get()
        agg_x = np.vstack((agg_x, proc_i_agg_x))
        agg_val = np.append(agg_val, proc_i_agg_val)
        agg_grad = np.vstack((agg_grad, proc_i_agg_grad))

    return [agg_x, agg_val, agg_grad]

def get_cut_and_x_next(agg_x, agg_val, agg_grad, S, k, x_0, settings):
    """
    Averages the cuts and compute next search point
    :param agg_x:
    :param agg_val (np.array): aggregate values
    :param agg_grad (np.array): aggregate gradients
    :param S (Saturation): saturation object
    :param k: iteration index (for INF-EDDP)
    :param x_0: initial point (for INF-EDDP)
    :param settings: settings

    :return x_next (np.array): next select point
    :return z_next (np.array): most distinguishable point
    :return avg_val (float): average value across scenarios
    :return avg_grad (np.array): average gradient across scenarios
    """
    # compute average cut
    avg_val = np.mean(agg_val[1:])
    avg_grad = np.mean(agg_grad[1:], axis=0)

    if settings['mode'] == utils.Mode.INF_EDDP:
        # [x_next, sat_lvl, idx] = S.largest_sat_lvl(agg_x[1:], settings["rng"], prioritize_zero=False)
        [z_next, _, _] = S.largest_sat_lvl(agg_x[1:], settings["rng"], prioritize_zero=False)
    elif settings['mode'] == utils.Mode.CE_INF_EDDP:
        [z_next, _, _] = S.largest_sat_lvl(agg_x[:], settings["rng"], prioritize_zero=False)
    elif settings['mode'] == utils.Mode.GAP_INF_EDDP:
        [z_next, _, _] = S.largest_sat_lvl(agg_x[:], settings["rng"], prioritize_zero=False)
    else:
        i_rand = settings["rng"].integers(0, settings["N"], endpoint=True)
        z_next = agg_x[i_rand]

    x_next = z_next
    if settings['mode'] == utils.Mode.INF_EDDP and (k % settings['T'] == 0):
        x_next = x_0

    return [x_next, z_next, avg_val, avg_grad]

def init_workers(x_0, settings, solver_arr, n_procs):
    q_host = Queue() # JoinableQueue()
    q_child= Queue() # JoinableQueue()
    ps = [None]*n_procs

    N = settings['N']

    # N+1 where (+1) is for 0-th stage
    scen_split = np.array_split(np.arange(N+1), n_procs)
    # settings['scenario_endpts'] = [1+scen_split[0][0], 1+scen_split[0][-1]+1]
    # needs +1 since not inclusive
    settings['scenario_endpts'] = [scen_split[0][0], scen_split[0][-1]+1]

    for i in range(n_procs-1):
        settings_i = settings.copy()
        settings_i['scenario_endpts'] = [scen_split[i+1][0], scen_split[i+1][-1]+1] # OMG!
        ps[i] = Process(target=_HDDP_multiproc, 
                args=(x_0, settings_i, solver_arr[i], n_procs, q_host, q_child, False))
        ps[i].start()

    return [q_host, q_child, ps, settings]

def solve_scenarios(prob_solver, x_0, x_curr, start, end):
    """
    :param prob_solver:
    :param x_0: initial search point
    :param x_curr: current search point
    :param start: starting scenario index (inclusive)
    :param end: ending scenario index (inclusive)
    """
    agg_x = np.array([], dtype=float)
    agg_val = np.array([], dtype=float)
    agg_grad = np.array([], dtype=float)
    agg_ctg = np.array([], dtype=float)

    for i in range(start, end):
        if i == 0:
            (x, val, grad, ctg) = prob_solver.solve(x_0, 0) 
        else:
            (x, val, grad, ctg) = prob_solver.solve(x_curr, i) 

        agg_x = np.append(agg_x, x)
        agg_val = np.append(agg_val, val)
        agg_grad = np.append(agg_grad, grad)
        agg_ctg = np.append(agg_ctg, ctg)

    n = len(x_curr)
    agg_x = np.reshape(agg_x, newshape=(-1,n))
    agg_grad = np.reshape(agg_grad, newshape=(-1,n))

    return [agg_x, agg_val, agg_grad, agg_ctg]
