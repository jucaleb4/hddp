from multiprocessing import Process, JoinableQueue, Queue
import time
import os

import numpy as np
import numpy.linalg as la
import pandas as pd

from hddp import solver
from hddp import utils 
from hddp import opt_setup

def get_epsilon_ts(M, eps, eps_T, T, lam, **kwargs):
    eps_ts_arr = np.zeros(T, dtype=float)
    eps_ts_arr[-1] = eps_T
    for t in range(T-2,-1,-1):
        eps_ts_arr[t] = 2*M*eps + lam*eps_ts_arr[t+1]
    if np.max(eps_ts_arr[:-2] - eps_ts_arr[1:-1]) >= 0.0:
        print("eps_ts_arr is not monotone decreasing, consider decreasing eps")
    return eps_ts_arr.tolist()

def fill_in_settings(x_0, settings, prob_solver):
    """ Fill in missing parameters based on problem (in solvers) """
    [h_min_val, h_max_val] = prob_solver.get_single_stage_lbub()
    settings["M"] = float((h_max_val-h_min_val)/(1.-settings['lam']))
    settings['n'] = len(x_0)
    settings["eps_T"] = (prob_solver.h_max_val-prob_solver.h_min_val)/(1.-settings['lam'])
    settings["eps_lvls"] = get_epsilon_ts(**settings)
    settings["x_lb_arr"] = prob_solver.x_lb_arr.tolist()
    settings["x_ub_arr"] = prob_solver.x_ub_arr.tolist()

# def HDDP_multiproc(x_0, settings, solver_arr, n_procs):
def HDDP_multiproc(settings, n_procs):
    setup_qs = False

    if settings['mode'] in [utils.Mode.EDDP, utils.Mode.SDDP]:
        [q_host, q_child, ps, settings] = init_workers(settings, n_procs)
        setup_qs = True
        _EDDP(settings, n_procs, q_host, q_child)
        return

    try:
        # [q_host, q_child, ps, settings] = init_workers(x_0, settings, solver_arr, n_procs)
        [q_host, q_child, ps, settings] = init_workers(settings, n_procs)
        setup_qs = True
        _HDDP_multiproc(
            # x_0, 
            settings, 
            # solver_arr[0], 
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

def get_problem(settings):
    """ Returns instance of prob_solver """
    if settings['prob_name'] == 'hydro':
        return opt_setup.create_hydro_thermal_gurobi_model(settings['N'], settings['lam'], settings['prob_seed'])
    elif settings['prob_name'] == 'portfolio':
        return opt_setup.create_portfolio_gurobi_model(settings['N'], settings['lam'], settings['prob_seed'])
    elif settings['prob_name'] == 'inventory':
        return opt_setup.create_inventory_gurobi_model(settings['N'], settings['lam'], settings['prob_seed'])
    elif settings['prob_name'] == 'hierarchical_inventory':
        return opt_setup.create_hierarchical_inventory_gurobi_model(seed=settings['prob_seed'], **settings)
    elif settings['prob_name'] == 'hierarchical_test':
        return opt_setup.create_hierarchical_test_gurobi_model(seed=settings['prob_seed'], **settings)
    else:
        raise Exception("Unknown prob_name %s" % settings['prob_name'])

def get_eddp_problem(settings):
    """ Returns T instance of prob_solver (for each stage) """
    prob_solver_arr = [None,] * settings['T']
    for t in range(len(prob_solver_arr)):
        if settings['prob_name'] == 'hydro':
            prob_solver_arr[t], x_0 = opt_setup.create_hydro_thermal_gurobi_model(
                settings['N'], 
                settings['lam'], 
                settings['prob_seed'],
                has_ctg=t<settings['T']-1,
            )
        elif settings['prob_name'] == 'portfolio':
            prob_solver_arr[t], x_0 = opt_setup.create_portfolio_gurobi_model(
                settings['N'], 
                settings['lam'], 
                settings['prob_seed'],
                has_ctg=t<settings['T']-1,
            )
        elif settings['prob_name'] == 'inventory':
            prob_solver_arr[t], x_0 = opt_setup.create_inventory_gurobi_model(
                settings['N'], 
                settings['lam'], 
                settings['prob_seed'],
                has_ctg=t<settings['T']-1,
            )
        elif settings['prob_name'] == 'hierarchical_inventory':
            prob_solver_arr[t], x_0 = opt_setup.create_hierarchical_inventory_gurobi_model(
                settings['N'], 
                settings['lam'], 
                settings['prob_seed'],
                settings['k1'], 
                settings['k2'], 
                settings['eta1_scale'], 
                settings['tau1_scale'], 
                settings['eta2_scale'], 
                has_ctg=t<settings['T']-1, 
            )
        else:
            raise Exception("Unknown prob_name %s" % settings['prob_name'])
    return prob_solver_arr, x_0

def _HDDP_multiproc(settings, n_procs, q_host, q_child, is_host):
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
    prob_solver, x_0 = get_problem(settings)
    fill_in_settings(x_0, settings, prob_solver)

    S = utils.SaturatedSet(**settings)
    x_curr = np.copy(x_0)
    [start_scenario_idx, end_scenario_idx] = settings['scenario_endpts']
    [single_stage_min, single_stage_max] = prob_solver.get_single_stage_lbub()
    beta = 1./(1.-settings['lam'])
    lb_model = solver.LowerBoundModel(settings['n'], single_stage_min*beta)
    ub_model = solver.UpperBoundModel(prob_solver.get_gurobi_model(), 
        prob_solver.get_scenarios()[1:], 
        settings["lam"], 
        settings['M'], # prob_solver.M_h*beta, 
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
    scenario_arr = np.zeros(settings['max_iter'], dtype=float)
    reached_opt_sat = False

    n_iters_ran = settings['max_iter']
    for k in range(1, settings['max_iter']+1):
        # forward phase: solve subproblems
        s_time = time.time()
        temp = solve_scenarios(prob_solver, x_0, x_curr, start_scenario_idx, end_scenario_idx)
        [agg_x, agg_val, agg_grad, agg_ctg] = temp

        # check termination
        x_0_sol = agg_x[0]
        if (not reached_opt_sat) and S.get(x_0_sol) <= 1:
            reached_opt_sat = True
            print("Optimal solution reached saturation level 1, consider choosing smaller epsilon")
            # n_iters_ran = k
            # break

        fwd_time_arr[k] = fwd_time_arr[k-1] + time.time() - s_time

        if is_host:
            s_time = time.time()
            [agg_x, agg_val, agg_grad] = host_forward_get(n_procs, q_host, agg_x, agg_val, agg_grad)
            comm_time_arr[k] = comm_time_arr[k-1] + time.time() - s_time
            s_time = time.time()
            if settings['mode'] == utils.Mode.GAP_INF_EDDP:
                update_S_with_ub_and_lb(S, agg_x, lb_model, ub_model, settings["eps_lvls"])
            temp = get_cut_and_x_next(agg_x, agg_val, agg_grad, S, k, x_0, settings)
            [x_next, z_next, scenario_arr[k-1], avg_val, avg_grad] = temp
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
            n_iters_ran = k
            break

    if is_host:
        utils.save_logs(settings['log_folder'], n_iters_ran, total_time_arr, fwd_time_arr, select_time_arr, eval_time_arr, comm_time_arr, lb_arr, ub_arr, scenario_arr) 
        prob_solver.save_cuts_to_file(settings['log_folder'])

def _EDDP(settings, n_procs, q_host, q_child):
    """
    Parallel variant of Lan's EDDP. See `_HDDP_multiproc for more details.
    """
    prob_solver_arr, x_0 = get_eddp_problem(settings)
    fill_in_settings(x_0, settings, prob_solver_arr[0])

    S = utils.SaturatedSet(**settings)
    x_curr = np.copy(x_0)
    [start_scenario_idx, end_scenario_idx] = settings['scenario_endpts']
    [single_stage_min, single_stage_max] = prob_solver_arr[0].get_single_stage_lbub()
    beta = 1./(1.-settings['lam'])
    # only for the first stage
    lb_model = solver.LowerBoundModel(settings['n'], single_stage_min*beta)
    ub_model = solver.UpperBoundModel(prob_solver_arr[0].get_gurobi_model(), 
        prob_solver_arr[0].get_scenarios()[1:], 
        settings["lam"], 
        settings['M'], # prob_solver.M_h*beta, 
        prob_solver_arr[0].get_var_names(), 
        single_stage_max*beta,
    )

    settings["rng"] = np.random.default_rng(settings.get("alg_seed", None))

    s0_time = time.time()
    T = settings['T']
    T2 = 2*T-1
    x_from_fwd = np.zeros((T, settings['n']), dtype=float)
    grad_from_fwd = np.zeros((T, settings['n']), dtype=float)
    val_from_fwd = np.zeros(T, dtype=float)
    # elapsed time since genesis
    max_iter = T2 * (1+settings['max_iter']//T2)
    print("Running for %d (requested %d)" % (max_iter, settings["max_iter"]))
    total_time_arr = np.zeros(max_iter+1)
    fwd_time_arr = np.zeros(max_iter+1)
    select_time_arr = np.zeros(max_iter+1)
    eval_time_arr = np.zeros(max_iter+1)
    comm_time_arr = np.zeros(max_iter+1)
    lb_arr = np.zeros(max_iter+1, dtype=float)
    ub_arr = np.zeros(max_iter+1, dtype=float)
    scenario_arr = np.zeros(settings['max_iter'], dtype=float)
    n_iters_ran = max_iter

    # first evaluate just to fill the bounds
    temp = solve_scenarios(prob_solver_arr[0], x_0, x_0, 0, 1)
    [agg_x, agg_val, agg_grad, agg_ctg] = temp
    lb, ub = evaluate_bounds(agg_x[0], agg_val[0], agg_ctg[0], ub_model, 
                             settings['lam'], 0, 0, 0)
    lb_arr[0] = lb
    ub_arr[0] = ub

    for n_iters in range(max_iter//T2):
        lb_arr[n_iters*T2:(n_iters+1)*T2] = lb_arr[n_iters*T2]
        ub_arr[n_iters*T2:(n_iters+1)*T2] = ub_arr[n_iters*T2]
        eval_time_arr[n_iters*T2:(n_iters+1)*T2] = eval_time_arr[n_iters*T2]

        # forward phase: solve subproblems
        x_curr = x_0
        for t in range(T):
            s_time = time.time()
            temp = solve_scenarios(prob_solver_arr[t], x_0, x_curr, start_scenario_idx, end_scenario_idx)
            [agg_x, agg_val, agg_grad, agg_ctg] = temp
            fwd_time_arr[n_iters*T2+t+1] = fwd_time_arr[n_iters*T2+t] + time.time() - s_time

            s_time = time.time()
            temp = host_forward_get(n_procs, q_host, agg_x, agg_val, agg_grad)
            [agg_x, agg_val, agg_grad] = temp
            temp = get_cut_and_x_next(agg_x, agg_val, agg_grad, S, 0, x_0, settings)
            x_next = agg_x[0] if t == 0 else temp[0] 
            x_from_fwd[t,:] = x_next
            x_curr = x_next
            select_time_arr[n_iters*T2+t+1] = select_time_arr[n_iters*T2+t] + time.time() - s_time
            total_time_arr[n_iters*T2+t+1] = time.time() - s0_time

        # backward step
        for t_0, t in enumerate(range(T-1,0,-1)):
            s_time = time.time()
            x_t_prev = x_from_fwd[t-1]
            x_t = x_from_fwd[t]
            if S.get(x_t) <= t:
                S.update(x_t_prev, min(S.get(x_t_prev), S.get(x_t) - 1))
            select_time_arr[n_iters*T2+T+t_0+1] = select_time_arr[n_iters*T2+T+t_0] + time.time() - s_time

            s_time = time.time()
            temp = solve_scenarios(prob_solver_arr[t], x_0, x_t_prev, start_scenario_idx, end_scenario_idx)
            fwd_time_arr[n_iters*T2+T+t_0+1] = fwd_time_arr[n_iters*T2+T+t_0] + time.time() - s_time

            s_time = time.time()
            [agg_x, agg_val, agg_grad, agg_ctg] = temp
            temp = get_cut_and_x_next(agg_x, agg_val, agg_grad, S, 0, x_0, settings)
            [_, _, _, avg_val, avg_grad] = temp
            prob_solver_arr[t-1].add_cut(avg_val, avg_grad, x_t_prev)
            select_time_arr[n_iters*T2+T+t_0+1] += time.time() - s_time
            total_time_arr[n_iters*T2+T+t_0+1] = time.time() - s0_time

        # evaluate
        # these values are for first stage
        s_time = time.time()
        lb_model.add_cut(avg_val, avg_grad, x_t_prev)
        ub_model.add_search_point_to_ub_model(x_t_prev)

        temp = solve_scenarios(prob_solver_arr[0], x_0, x_0, 0, 1)
        [agg_x, agg_val, agg_grad, agg_ctg] = temp
        lb, ub = evaluate_bounds(agg_x[0], agg_val[0], agg_ctg[0], ub_model, 
                                 settings['lam'], n_iters*T2+T+t_0-1, 
                                 time.time() - s0_time, t==1)
        lb_arr[(n_iters+1)*T2] = lb
        ub_arr[(n_iters+1)*T2] = ub
        eval_time_arr[(n_iters+1)*T2] = eval_time_arr[(n_iters+1)*T2-1] + time.time() - s_time
        total_time_arr[(n_iters+1)*T2] = time.time() - s0_time

        if total_time_arr[(n_iters-1)*T2] >= settings['time_limit']:
            n_iters_ran = (n_iters+1)*T2
            break

    utils.save_logs(settings['log_folder'], n_iters_ran, total_time_arr, fwd_time_arr, select_time_arr, eval_time_arr, comm_time_arr, lb_arr, ub_arr, scenario_arr) 
    # to simulate rolling horizon basis, only save first-stage cutting plane (since time homogenous)
    prob_solver_arr[0].save_cuts_to_file(settings['log_folder'])

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
    :return i (int): scenario index
    :return avg_val (float): average value across scenarios
    :return avg_grad (np.array): average gradient across scenarios
    """
    # compute average cut
    avg_val = np.mean(agg_val[1:])
    avg_grad = np.mean(agg_grad[1:], axis=0)
    i_select = None

    if settings['mode'] == utils.Mode.INF_EDDP:
        # [x_next, sat_lvl, idx] = S.largest_sat_lvl(agg_x[1:], settings["rng"], prioritize_zero=False)
        [z_next, _, i_select] = S.largest_sat_lvl(agg_x[1:], settings["rng"], prioritize_zero=False)
        i_select += 1
    elif settings['mode'] == utils.Mode.CE_INF_EDDP:
        # [z_next, _, i] = S.largest_sat_lvl(agg_x[:], settings["rng"], prioritize_zero=False)
        [z_next, lvl, i_select] = S.largest_sat_lvl(agg_x[:], settings["rng"], prioritize_zero=True)
    elif settings['mode'] == utils.Mode.GAP_INF_EDDP:
        [z_next, _, i_select] = S.largest_sat_lvl(agg_x[:], settings["rng"], prioritize_zero=False)
    elif settings['mode'] == utils.Mode.INF_SDDP:
        i_select = i_rand = settings["rng"].integers(0, settings["N"], endpoint=True)
        z_next = agg_x[i_rand]
    elif settings['mode'] == utils.Mode.GCE_INF_EDDP:
        if settings.get('last_reset', 0) == 2*settings['T']:
            i_select = 0
            z_next = agg_x[i_select]
        else:
            [z_next, _, i_select] = S.largest_sat_lvl(agg_x[:], settings["rng"], prioritize_zero=False)
        settings['last_reset'] = settings.get('last_reset', 0) + 1 if i_select > 0 else 0
    elif settings['mode'] == utils.Mode.EDDP:
        [z_next, lvl, i_select] = S.largest_sat_lvl(agg_x[1:], settings["rng"], prioritize_zero=True)
        i_select += 1
    else:
        i_select = i_rand = settings["rng"].integers(1, settings["N"], endpoint=True)
        z_next = agg_x[i_rand]

    x_next = z_next
    if settings['mode'] == utils.Mode.INF_EDDP and (k % settings['T'] == 0):
        x_next = x_0
    # if settings['mode'] == utils.Mode.CE_INF_EDDP and (k % (2*settings['T']) == 0):
    #     x_next = x_0

    return [x_next, z_next, i_select, avg_val, avg_grad]

# def init_workers(x_0, settings, solver_arr, n_procs):
def init_workers(settings, n_procs):
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
                # args=(x_0, settings_i, solver_arr[i], n_procs, q_host, q_child, False))
                args=(settings_i, n_procs, q_host, q_child, False))
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

def HDDP_eval(settings):
    """ 
    Evaluates solution of EDDP for settings['eval_T'] steps.

    We assume settings['policy_folder'] specifies folder of policy.
    """
    assert "eval_fname" in settings, "Key 'eval_fname' not found in settings"
    prob_solver, x_0 = get_problem(settings)
    x = x_0

    # retrieve and load last policy 
    folder = settings["log_folder"]
    if not settings["fixed_eval"]:
        val_arr    = np.squeeze(pd.read_csv(os.path.join(folder, "vals.csv")).to_numpy())
        grad_arr   = pd.read_csv(os.path.join(folder, "grad.csv")).to_numpy()
        x_prev_arr = pd.read_csv(os.path.join(folder, "x_prev.csv")).to_numpy()

        time_limit = settings.get("policy_by_time", np.inf)
        if time_limit == 0:
            n = grad_arr.shape[1]
            prob_solver.add_cut(0, np.zeros(n), np.zeros(n))
        else:
            elpsed_time_arr = pd.read_csv(os.path.join(folder, "elpsed_times.csv"))["# total_time"].to_numpy()
            time_idx = len(elpsed_time_arr)-1 # since includes time 0
            if np.max(elpsed_time_arr) > time_limit:
                time_idx = np.argmax(elpsed_time_arr >= time_limit)

            max_cuts = min(len(val_arr), time_idx)
            prob_solver.load_cuts(val_arr[:max_cuts], grad_arr[:max_cuts], x_prev_arr[:max_cuts])

    cum_cost_arr = np.zeros(settings['eval_T'], dtype=float)
    prev_cum_cost = 0
    i_selector = np.random.default_rng(settings.get('prob_seed',0)+1)
    i = 0
    for t in range(settings['eval_T']):
        (x, val, _, ctg) = prob_solver.solve(x, i) 
        curr_cost = val - settings['lam']*ctg
        cum_cost_arr[t] = curr_cost + settings['lam'] * prev_cum_cost 
        prev_cum_cost = cum_cost_arr[t]
        i = i_selector.integers(1, settings['N'], endpoint=True)

    # save in file
    print("Final cumulative score over %d periods with discount %.4f: %.6e" % (settings['eval_T'], settings['lam'], cum_cost_arr[-1]))
    np.savetxt(os.path.join(folder, settings['eval_fname']), np.atleast_2d(cum_cost_arr).T, delimiter=',')
