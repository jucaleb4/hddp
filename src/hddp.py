from solver import SaturatedSet, UnderApproxValue, LowerBoundModel, UpperBoundModel
import numpy as np
import numpy.linalg as la
from multiprocessing import Process, JoinableQueue, Queue
import time

# TODO: Develop structs for data being sent

NORMAL_STATUS = 3
GENESIS_STATUS = 4
FWD_EVAL_STATUS = 5
FIN_STATUS = 6

EDDP_ONLY_LB_MODE = 0
EDDP_ONLY_LB_MODE_LINEAR = 1
EDDP_UB_AND_LB_MODE = 2
SDDP_MODE = 3
ESDDP_MODE = 4

def HDDP_multiproc(x_0, params, solvers, nprocs):
    s_time = time.time()
    setup_qs = False
    [dist_x, last_val] = [None, None]

    # should only have one
    if params["mode"] == SDDP_MODE:
        print("SDDP only requires one subproblem solver, setting nprocs=1")
        nprocs = 1

    try:
        [q_host, q_child, ps, params] = initiate_workers(x_0, params, solvers, nprocs)
        setup_qs = True
        [dist_x, last_val] = _HDDP_multiproc(
            x_0, 
            params, 
            solvers[0], 
            nprocs, 
            q_host, 
            q_child, 
            is_host=True
        )
    except (KeyboardInterrupt, SystemExit):
        print('\n' + '-'*64)
        print("Early exiting...")

    print("Total time: {:.2f}s".format(time.time() - s_time))
    if setup_qs:
        for i in range(nprocs-1):
            ps[i].join()
        q_host.close()
        q_child.close()

    return [dist_x, last_val] 

def _HDDP_multiproc(x_0, params, solver, nprocs, q_host, q_child, is_host):
    """
    Parallel variant of hierarchical EDDP. The host is ''called'' p1, and the
    remaining [2,...,@nprocs] are separate processes. All work is independent
    of one another except for two syncrhonization points: 
        1. forward evaluation (occurs T times)
        2. selecting the most distguishable point
    The host and children communicate with each other using multiprocessor's
    Queue to relay information.

    Some notation:
    - We write x_curr = x^k and x_next = x^{k+1}

    Parameters
    ----------
    x_0 : np.array
        - initial solution
    eps : float
        - tolerance

    Return
    ----------
    """
    n       = len(x_0)
    under_V = UnderApproxValue(n)
    S       = SaturatedSet(params)
    x_curr  = x_0
    x_curr_inf = x_0
    iter_ct = 1
    last_x = last_val = None
    maxiter = params['max_iter']

    print('-'*64)
    np.set_printoptions(precision=1, suppress=False)
    end_next = False

    [start_scenario_idx, end_scenario_idx] = params['scenario_endpts']

    lb_model = None
    ub_model = None
    # initialize lower bound and upper bound model
    if params.get("evaluate_lb", False):
        [initial_lb, _] = solver.get_extrema_values()
        # initial_lb *= 1/(1-params["lam"])
        lb_model = LowerBoundModel(n, initial_lb)

    if params.get("evaluate_ub", False):
        [_, max_val] = solver.get_extrema_values()
        ub_model = UpperBoundModel(
            solver.get_gurobi_model(), 
            solver.get_scenarios()[1:], 
            params.get("lam"), 
            params.get("M_0"), 
            solver.get_var_names(), 
            max_val
        )

    if is_host:
        params["rng"] = np.random.default_rng(params.get("sel_seed", None))
    use_x_0 = params["mode"] == EDDP_ONLY_LB_MODE_LINEAR or params["mode"] == EDDP_UB_AND_LB_MODE
    use_x_0 = False

    # TEMP
    print("I am solving [{}-{}]".format(start_scenario_idx, end_scenario_idx))

    while 1:
        s_time = time.time()

        [agg_x, agg_val, agg_grad, _] = solve_scenarios(
            solver, 
            x_0,
            x_curr, 
            start_scenario_idx, 
            end_scenario_idx,
            use_x_0
        )

        # TEMP
        if use_x_0:
            [add_agg_x, _, _, _] = solve_scenarios(
                solver, 
                None,
                x_curr_inf, 
                start_scenario_idx, 
                end_scenario_idx,
                use_x_0=False
            )
            agg_x[1:] = add_agg_x[1:]

        # select next search point and calculates value and gradients
        if is_host:
            tupl = select_subproblem(
                q_host, 
                q_child, 
                nprocs, 
                agg_x, 
                agg_val, 
                agg_grad, 
                S, 
                lb_model, 
                ub_model, 
                params,
            )
            [x_next, val, grad, avg_val, avg_grad, x_next_sat_lvl] = tupl

            # select next point from only last set of points
            if use_x_0:
                _tupl = select_subproblem(
                    q_host, 
                    q_child, 
                    nprocs, 
                    add_agg_x[1:], 
                    agg_val, 
                    agg_grad, 
                    S, 
                    lb_model, 
                    ub_model, 
                    params,
                )
                [x_curr_inf, _, _, _, _, _] = _tupl

            if params.get("perturb", False):
                print(">> perturbing")
                val += np.random.normal(0, 0.1)
                grad += np.random.normal(0, 0.1, n)
                x_next += np.random.normal(0, 0.1, n)

            # TEMP (sanity check to ensure LP returning something reasonable)
            if np.sum(np.abs(grad)) <= 0:
                print("[!!] grad is zero, grad={}".format(grad))

            [last_x, last_val] = [x_next, val]
        elif not is_host:
            outmail = [agg_x, agg_val, agg_grad]
            q_host.put(outmail) 
            [x_next, val, grad, x_next_sat_lvl] = q_child.get() 

        # Select and add new cut, update saturation data structure
        under_V.add_cut(val, grad, x_curr)
        solver.add_newest_cut(x_curr, under_V)
        S.update(x_curr, max(0, min(S.get(x_curr), x_next_sat_lvl - 1)))

        # evaluate approximate upper and lower bounds
        if is_host:
            if params.get("evaluate_lb", False):
                lb_model.add_cut(avg_val, avg_grad, x_curr)
            if params.get("evaluate_ub", False): # and iter_ct >= 2:
                ub_model.add_search_point_to_ub_model(x_curr)

            ub = 0
            [x_sol, lb, _, ctg] = solver.solve(x_0, 0) 
            # remove ctg from lb
            f_eval_at_x_sol = lb - params["lam"]*ctg
            x_0_sat_lvl = S.get(x_sol)

            # compute lower and upper bound (if requested)
            if params.get("evaluate_lb", False):
                lb_eval = lb_model.evaluate_lb(x_sol)
                lb = f_eval_at_x_sol + params["lam"]*lb_eval
            if params.get("evaluate_ub", False):
                ub_eval = ub_model.evaluate_ub(x_sol)
                ub = f_eval_at_x_sol + params["lam"]*ub_eval

            S_idx = S.get_idx(x_curr)

            print('[{:<.1e}s]  {})  _F(x_0):[[{:.2f}, {:.2f}]] \
                \n\t\tS(x_0)={}  S{}={} (idx:{})'.format(
                    time.time() - s_time, 
                    iter_ct, 
                    lb, 
                    ub, 
                    x_0_sat_lvl,
                    x_next, 
                    x_next_sat_lvl, 
                    ','.join(S_idx.astype('str'))
                ) 
            )

        if end_next or (maxiter > 0 and iter_ct >= maxiter):
            break
        
        if params['mode'] == EDDP_ONLY_LB_MODE \
            or params['mode'] == SDDP_MODE \
            or params['mode'] == ESDDP_MODE:
            if (iter_ct + 1) % params['T'] == 0:
                x_curr = x_0
            else:
                x_curr = x_next
        elif params['mode'] == EDDP_ONLY_LB_MODE_LINEAR or \
            params['mode'] == EDDP_UB_AND_LB_MODE:
            x_curr = x_next

        iter_ct += 1

    if is_host:
        print('-'*64,"\n")

    return last_x, last_val

def update_S_with_ub_and_lb(S, agg_x, lb_model, ub_model, eps_lvls):
    """
    Update saturation data structure with upper and lower bounds
    """
    for x_i in agg_x:
        x_lb = lb_model.evaluate_lb(x_i)
        x_ub = ub_model.evaluate_ub(x_i)
        gap = x_ub - x_lb

        if gap < 0:
            print("WARNING: x_i_lb > x_i_ub ({} > {}). Consider increasing M_0 or modifying code".format(
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

def select_subproblem(q_host, q_child, nprocs, agg_x, agg_val, agg_grad, S, 
        lb_model, ub_model, params):
    """
    Reduce and broadcast the most distinguishable state
    
    Args:
        q_host (Queue): queue for host to child
        q_child (Queue): queue for child to host
        nprocs (int): number of processes
        agg_x (np.array): aggregate states
        agg_val (np.array): aggregate values
        agg_grad (np.array): aggregate gradients
        S (Saturation): saturation object
        lb_model (LowerBoundModel): -
        ub_model (UpperBoundModel): -
        eps_lvls (np.ndarray): level of epsilons {\epsilon_t}
        params (dict): dictionary of parameters

    Returns:
        x (np.array): next select point
        val (float): function value for lower bound
        grad (np.array): gradient for lower bound
        avg_val (float): average value across scenarios
        avg_grad (np.array) average gradient across scenarios
        sat_lvl (int): saturation level of selected point
    """
    # gather all solutions
    for _ in range(nprocs-1):
        [proc_i_agg_x, proc_i_agg_val, proc_i_agg_grad] = q_host.get()

        agg_x = np.vstack((agg_x, proc_i_agg_x))
        agg_val = np.append(agg_val, proc_i_agg_val)
        agg_grad = np.vstack((agg_grad, proc_i_agg_grad))

    mode = params['mode']
    if mode == EDDP_UB_AND_LB_MODE:
        if lb_model is None or ub_model is None:
            print("EDDP_UB_AND_LB_MODE requires evaluate_lb and evaluate_ub=True")
            exit(0)
        eps_lvls = params.get("eps_lvls")
        update_S_with_ub_and_lb(S, agg_x, lb_model, ub_model, eps_lvls)

    # select the subproblem
    if mode == EDDP_ONLY_LB_MODE_LINEAR or mode == EDDP_UB_AND_LB_MODE:
        [x, sat_lvl, idx] = S.largest_sat_lvl(agg_x, params["rng"], prioritize_zero=False)
    elif mode == EDDP_ONLY_LB_MODE:
        [x, sat_lvl, idx] = S.largest_sat_lvl(agg_x[:], params["rng"], prioritize_zero=False)
        # [x, sat_lvl, idx] = S.largest_sat_lvl(agg_x[1:], params["rng"], prioritize_zero=False)
        idx += 1
    elif mode == ESDDP_MODE or mode == SDDP_MODE:
        i_rand = params["rng"].integers(0, params["N"], endpoint=True)
        # i_rand = params["rng"].integers(1, params["N"], endpoint=True)
        x = agg_x[i_rand]
        sat_lvl = S.get(x)
        idx = i_rand
    else:
        raise Exception("Unknown params mode {}".format(params['mode']))

    avg_val = np.mean(agg_val[1:])
    avg_grad = np.mean(agg_grad[1:], axis=0)
    # Compute the cut
    if mode == SDDP_MODE:
        val = agg_val[i_rand]
        grad = agg_grad[i_rand]
    else:
        val = avg_val
        grad = avg_grad

    # send subproblem to all processes
    outmail = [x, val, grad, sat_lvl]
    for _ in range(nprocs-1):
        q_child.put(outmail) 

    return [x, val, grad, avg_val, avg_grad, sat_lvl]

    # [x_next, val, grad, avg_val, avg_grad, x_next_sat_lvl] = tupl

def initiate_workers(x_0, params, solvers, nprocs):
    q_host = Queue() # JoinableQueue()
    q_child= Queue() # JoinableQueue()
    ps = [None]*nprocs

    N = params['N']

    # N+1 where (+1) is for 0-th stage
    scen_split = np.array_split(np.arange(N+1), nprocs)
    # params['scenario_endpts'] = [1+scen_split[0][0], 1+scen_split[0][-1]+1]
    # needs +1 since not inclusive
    params['scenario_endpts'] = [scen_split[0][0], scen_split[0][-1]+1]

    for i in range(nprocs-1):
        params_i = params.copy()
        params_i['scenario_endpts'] = [scen_split[i+1][0], scen_split[i+1][-1]+1] # OMG!
        ps[i] = Process(target=_HDDP_multiproc, 
                args=(x_0, params_i, solvers[i], nprocs, q_host, q_child, False))
        ps[i].start()

    return [q_host, q_child, ps, params]

def solve_scenarios(solver, x_0, x_curr, start, end, use_x_0=False):
    agg_x = np.array([], dtype=float)
    agg_val = np.array([], dtype=float)
    agg_grad = np.array([], dtype=float)
    agg_ctg = np.array([], dtype=float)

    for i in range(start, end):
        if i == 0 and use_x_0:
            (x, val, grad, ctg) = solver.solve(x_0, 0) 
        else:
            (x, val, grad, ctg) = solver.solve(x_curr, i) 

        agg_x = np.append(agg_x, x)
        agg_val = np.append(agg_val, val)
        agg_grad = np.append(agg_grad, grad)
        agg_ctg = np.append(agg_ctg, ctg)

    n = len(x_curr)
    agg_x = np.reshape(agg_x, newshape=(-1,n))
    agg_grad = np.reshape(agg_grad, newshape=(-1,n))

    return [agg_x, agg_val, agg_grad, agg_ctg]
