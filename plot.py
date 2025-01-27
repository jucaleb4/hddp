import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import colorsys

log_folder = "/Users/calebju/Code/hddp/logs/"
# list of linestyles
lss_arr = [
     'solid',
     'dotted',
     'dashed',
     'dashdot',
     (0, (1, 10)),
     (0, (1, 5)),
     (0, (1, 1)),
     (5, (10, 3)),
     (0, (5, 10)),
     (0, (5, 5)),
     (0, (5, 1)),
     (0, (3, 10, 1, 10)),
     (0, (3, 5, 1, 5)),
     (0, (3, 1, 1, 1)),
     (0, (3, 5, 1, 5, 1, 5)),
     (0, (3, 10, 1, 10, 1, 10)),
     (0, (3, 1, 1, 1, 1, 1))]
hatch_arr = ['/', '//', '///', '\\/...' '*', 'o', '0', '.']

def read_logs(folder_arr):
    """
    Read in files from folder 

    :return bounds_arr: array of shape (n_algs, n_iters, 2) holding lb/ubs
    :return elpsed_times_arr: array of shape (n_algs, n_iters, 5) holding total, fwd, select, val, and comm time
    :return size_arr: array of shape (n_algs) holding the n_iters for each alg 
    """
    n_folders = len(folder_arr)
    bounds_arr = None
    elpsed_times_arr = None
    size_arr = np.zeros(n_folders) # num elements in files from each folder (size may be different)

    for i, folder in enumerate(folder_arr):
        bounds = pd.read_csv(os.path.join(folder, 'bounds.csv'), header='infer')
        # ignore time 0
        elpsed_times = pd.read_csv(os.path.join(folder, 'elpsed_times.csv'), header='infer')[1:]
        n_elems = min(len(bounds), len(elpsed_times))
        bounds = bounds[:n_elems]
        elpsed_times = elpsed_times[:n_elems]
        size_arr[i] = n_elems

        if bounds_arr is None:
            bounds_arr = np.zeros((n_folders, n_elems, 2)) # lb, ub
            elpsed_times_arr = np.zeros((n_folders, n_elems, 5)) # total_time,fwd_time,select_time,eval_time
        elif n_elems > bounds_arr.shape[1]:
            old_n_elems = bounds_arr.shape[1]
            # resize
            new_bounds_arr = np.zeros((n_folders, n_elems, 2))
            new_elpsed_times_arr = np.zeros((n_folders, n_elems, 5)) 
            for j in range(i):
                new_bounds_arr[j,:old_n_elems,:] = bounds_arr[j,:,:]
                new_elpsed_times_arr[j,:old_n_elems,:] = elpsed_times_arr[j,:,:]

            # copy over old data
            bounds_arr = new_bounds_arr
            elpsed_times_arr = new_elpsed_times_arr

        # add new data
        bounds_arr[i,:n_elems,:] = bounds
        elpsed_times_arr[i,:n_elems,:] = elpsed_times

    return bounds_arr, elpsed_times_arr, size_arr

def plot_bound_gap(date, prob_name, large_lam, fname=None):
    """
    Plots upper and lower bound newsvendor gap

    :param large_lam: lam=0.9906 if 1 else 0.8
    """
    i_0 = 0 if large_lam else 24
    gamma = 0.9906 if large_lam else 0.8

    color_arr = ['red', 'black', 'blue', 'green', 'orange', 'purple']
    plt.style.use('ggplot')
    _, ax = plt.subplots(figsize=(5, 4))

    experiment_log_folder = os.path.join(log_folder, "%s/exp_0" % date)
    name_arr = ["EDDP", "Inf-EDDP", "CE-Inf-EDDP", "Gap-Inf-EDDP", "Inf-SDDP", "SDDP"]
    idx_arr = [i_0+13] + [i_0+0, i_0+1, i_0+2] + list(range(i_0+3,i_0+13)) + list(range(i_0+14, i_0+24))
    folder_arr = [os.path.join(experiment_log_folder, "run_%d" % i) for i in idx_arr]
    bounds_arr, elpsed_times_arr, size_arr = read_logs(folder_arr)

    print('-'*100)
    for i in range(4):
        n_iters = int(size_arr[i])
        xs = np.arange(n_iters)

        ax.plot(xs, bounds_arr[i,:n_iters,0], linestyle=lss_arr[0], label=name_arr[i], color=color_arr[i])
        ax.plot(xs, bounds_arr[i,:n_iters,1], linestyle=lss_arr[1], color=color_arr[i])
        if bounds_arr.shape[1] >= 500:
            print("%s iter=500, lb=%.1e ub=%.1e (%.3f)" % (name_arr[i], bounds_arr[i,500-1,0], bounds_arr[i,500-1,1], 
                 (bounds_arr[i,500-1,1] - bounds_arr[i,500-1,0])/bounds_arr[i,500-1,0]))
        if bounds_arr.shape[1] >= 1000:
            print("%s iter=1000, lb=%.1e ub=%.1e (%.3f)" % (name_arr[i], bounds_arr[i,1000-1,0], bounds_arr[i,1000-1,1],
                 (bounds_arr[i,1000-1,1] - bounds_arr[i,1000-1,0])/bounds_arr[i,1000-1,0]))
        print("%s iter=2000, lb=%.1e ub=%.1e (%.3f)" % (name_arr[i], bounds_arr[i,n_iters-1,0], bounds_arr[i,n_iters-1,1],
              (bounds_arr[i,n_iters-1,1] - bounds_arr[i,n_iters-1,0])/bounds_arr[i,n_iters-1,0]))

    # create and plot quantiles for Inf-SDDP (there is off-by-one error, so drop last iterate for SDDP)
    sddp_j_0_arr = [3, 14] # where the randon experiments start
    sddp_color_idx_arr = [4,5]
    for k,j_0 in zip(sddp_color_idx_arr,sddp_j_0_arr):
        n_iters = int(size_arr[j_0])-1
        inf_sddp_lb_arr = np.sort(bounds_arr[j_0:j_0+10,:n_iters,0], axis=0)
        inf_sddp_ub_arr = np.sort(bounds_arr[j_0:j_0+10,:n_iters,1], axis=0)
        xs = np.arange(n_iters)
        # median (average of 5 and 6th best)
        ax.plot(xs, np.mean(inf_sddp_lb_arr[4:6], axis=0), linestyle=lss_arr[0], label=name_arr[k], color=color_arr[k])
        ax.plot(xs, np.mean(inf_sddp_ub_arr[4:6], axis=0), linestyle=lss_arr[1], color=color_arr[k])
        # 10% and 90% quantile
        ax.fill_between(xs, inf_sddp_lb_arr[1], inf_sddp_lb_arr[8], color=color_arr[k], alpha=0.1)
        ax.fill_between(xs, inf_sddp_ub_arr[1], inf_sddp_ub_arr[8], color=color_arr[k], alpha=0.1)
        if n_iters >= 500:
            print("%s iter=500, median lb=%.1e ub=%.1e (%.3f)" % (
                    name_arr[k], 
                    np.mean(inf_sddp_lb_arr[4:6,500-1]), 
                    np.mean(inf_sddp_ub_arr[4:6,500-1]),
                   (np.mean(inf_sddp_ub_arr[4:6,500-1]) - np.mean(inf_sddp_lb_arr[4:6,500-1]))/np.mean(inf_sddp_lb_arr[4:6,500-1])
            ))
        if n_iters >= 1000:
            print("%s iter=1000, median lb=%.1e ub=%.1e (%.3f)" % (
                    name_arr[k], 
                    np.mean(inf_sddp_lb_arr[4:6,1000-1]), np.mean(inf_sddp_ub_arr[4:6,1000-1]),
                    (np.mean(inf_sddp_ub_arr[4:6,1000-1]) - np.mean(inf_sddp_lb_arr[4:6,1000-1]))/np.mean(inf_sddp_lb_arr[4:6,1000-1])
            ))
        print("%s iter=%d, median lb=%.1e ub=%.1e (%.3f)" % (
                    name_arr[k], 
                    n_iters,
                    np.mean(inf_sddp_lb_arr[4:6,-1]), 
                    np.mean(inf_sddp_ub_arr[4:6,-1]),
                    (np.mean(inf_sddp_ub_arr[4:6,-1]) - np.mean(inf_sddp_lb_arr[4:6,-1]))/np.mean(inf_sddp_lb_arr[4:6,-1])
        ))
        print("%s .10 lb=%.8f ub=%.8f" % (name_arr[k], inf_sddp_lb_arr[1,-1], inf_sddp_ub_arr[1,-1]))
        print("%s .90 lb=%.8f ub=%.8f" % (name_arr[k], inf_sddp_lb_arr[8,-1], inf_sddp_ub_arr[8,-1]))

    print('-'*100)

    # ax.legend(loc=1)
    ax.legend()
    ax.set(
        yscale='symlog', 
        xlabel="Number of sub-problems solved",
        title="Upper and lower bound gaps\n" + r"for {} ($\gamma={}$)".format(prob_name, gamma),
    )
    plt.tight_layout()
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, dpi=240)

def plot_bound_gap_hydrothermal(fname=None):
    """
    Plots upper and lower bound newsvendor gap

    :param large_lam: lam=0.9906 if 1 else 0.8
    """
    date = "2025_01_15"
    large_lam = False
    i_0 = 0 if large_lam else 24
    gamma = 0.9906 if large_lam else 0.8

    color_arr = ['red', 'black', 'blue', 'cyan', 'green', 'orange', 'purple']
    plt.style.use('ggplot')
    _, ax = plt.subplots(figsize=(5, 4))

    experiment_log_folder = os.path.join(log_folder, "%s/exp_0" % date)
    name_arr = ["EDDP", "Inf-EDDP", "CE-Inf-EDDP", "GCE-Inf-EDDP", "Gap-Inf-EDDP", "Inf-SDDP", "SDDP"]
    idx_arr = [i_0+13] + list(range(i_0,i_0+2)) + [i_0+14] + [i_0+2] + list(range(i_0+3,i_0+13)) + list(range(i_0+15,i_0+25))
    folder_arr = [os.path.join(experiment_log_folder, "run_%d" % i) for i in idx_arr]
    bounds_arr, elpsed_times_arr, size_arr = read_logs(folder_arr)

    for i in range(5):
        n_iters = int(size_arr[i])
        xs = np.arange(n_iters)

        ax.plot(xs, bounds_arr[i,:n_iters,0], linestyle=lss_arr[0], label=name_arr[i], color=color_arr[i])
        ax.plot(xs, bounds_arr[i,:n_iters,1], linestyle=lss_arr[1], color=color_arr[i])
        if bounds_arr.shape[1] >= 100:
            print("%s iter=100, lb=%.1e ub=%.1e (%.3f)" % (name_arr[i], bounds_arr[i,100-1,0], bounds_arr[i,100-1,1], 
                 (bounds_arr[i,100-1,1] - bounds_arr[i,100-1,0])/bounds_arr[i,100-1,0]))
        if bounds_arr.shape[1] >= 500:
            print("%s iter=500, lb=%.1e ub=%.1e (%.3f)" % (name_arr[i], bounds_arr[i,500-1,0], bounds_arr[i,500-1,1], 
                 (bounds_arr[i,500-1,1] - bounds_arr[i,500-1,0])/bounds_arr[i,500-1,0]))
        if bounds_arr.shape[1] >= 1000:
            print("%s iter=1000, lb=%.1e ub=%.1e (%.3f)" % (name_arr[i], bounds_arr[i,1000-1,0], bounds_arr[i,1000-1,1],
                 (bounds_arr[i,1000-1,1] - bounds_arr[i,1000-1,0])/bounds_arr[i,1000-1,0]))
        print("%s iter=%d, lb=%.1e ub=%.1e (%.3f)" % (name_arr[i], n_iters, bounds_arr[i,n_iters-1,0], bounds_arr[i,n_iters-1,1],
              (bounds_arr[i,n_iters-1,1] - bounds_arr[i,n_iters-1,0])/bounds_arr[i,n_iters-1,0]))

    # create and plot quantiles for Inf-SDDP (there is off-by-one error, so drop last iterate for SDDP)
    sddp_j_0_arr = [3, 15] # where the randon experiments start
    sddp_color_idx_arr = [5,6]
    for k,j_0 in zip(sddp_color_idx_arr, sddp_j_0_arr):
        n_iters = int(size_arr[j_0])-1
        xs = np.arange(n_iters)
        inf_sddp_lb_arr = np.sort(bounds_arr[j_0:j_0+10,:n_iters,0], axis=0)
        inf_sddp_ub_arr = np.sort(bounds_arr[j_0:j_0+10,:n_iters,1], axis=0)
        # median (average of 5 and 6th best)
        ax.plot(xs, np.mean(inf_sddp_lb_arr[4:6], axis=0), linestyle=lss_arr[0], label=name_arr[k], color=color_arr[k])
        ax.plot(xs, np.mean(inf_sddp_ub_arr[4:6], axis=0), linestyle=lss_arr[1], color=color_arr[k])
        # 10% and 90% quantile
        ax.fill_between(xs, inf_sddp_lb_arr[1], inf_sddp_lb_arr[8], color=color_arr[k], alpha=0.15)
        ax.fill_between(xs, inf_sddp_ub_arr[1], inf_sddp_ub_arr[8], color=color_arr[k], alpha=0.15)
        if n_iters >= 100:
            print("%s iter=100, median lb=%.1e ub=%.1e (%.3f)" % (
                    name_arr[k], 
                    np.mean(inf_sddp_lb_arr[4:6,100-1]), 
                    np.mean(inf_sddp_ub_arr[4:6,100-1]),
                   (np.mean(inf_sddp_ub_arr[4:6,100-1]) - np.mean(inf_sddp_lb_arr[4:6,100-1]))/np.mean(inf_sddp_lb_arr[4:6,100-1])
            ))
        if n_iters >= 500:
            print("%s iter=500, median lb=%.1e ub=%.1e (%.3f)" % (
                    name_arr[k], 
                    np.mean(inf_sddp_lb_arr[4:6,500-1]), 
                    np.mean(inf_sddp_ub_arr[4:6,500-1]),
                   (np.mean(inf_sddp_ub_arr[4:6,500-1]) - np.mean(inf_sddp_lb_arr[4:6,500-1]))/np.mean(inf_sddp_lb_arr[4:6,500-1])
            ))
        if n_iters >= 1000:
            print("%s iter=1000, median lb=%.1e ub=%.1e (%.3f)" % (
                    name_arr[k], 
                    np.mean(inf_sddp_lb_arr[4:6,1000-1]), np.mean(inf_sddp_ub_arr[4:6,1000-1]),
                    (np.mean(inf_sddp_ub_arr[4:6,1000-1]) - np.mean(inf_sddp_lb_arr[4:6,1000-1]))/np.mean(inf_sddp_lb_arr[4:6,1000-1])
            ))
        print("%s iter=%d, median lb=%.1e ub=%.1e (%.3f)" % (
                    name_arr[k], 
                    n_iters,
                    np.mean(inf_sddp_lb_arr[4:6,-1]), 
                    np.mean(inf_sddp_ub_arr[4:6,-1]),
                    (np.mean(inf_sddp_ub_arr[4:6,-1]) - np.mean(inf_sddp_lb_arr[4:6,-1]))/np.mean(inf_sddp_lb_arr[4:6,-1])
        ))
        print("%s .10 lb=%.8f ub=%.8f" % (name_arr[k], inf_sddp_lb_arr[1,-1], inf_sddp_ub_arr[1,-1]))
        print("%s .90 lb=%.8f ub=%.8f" % (name_arr[k], inf_sddp_lb_arr[8,-1], inf_sddp_ub_arr[8,-1]))

    # ax.legend(loc=1)
    ax.legend()
    ax.set(
        yscale='symlog', 
        xlabel="Number of sub-problems solved",
        title="Upper and lower bound gaps\n" + r"for {} ($\gamma={}$)".format("hydro-thermal", gamma),
    )
    plt.tight_layout()
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, dpi=240)

def plot_newsvendor_bound_gap(large_lam, fname=None):
    date = "2025_01_16"
    plot_bound_gap(date, 'newsvendor', large_lam, fname)

def plot_hydro_bound_gap(large_lam, fname=None):
    date = "2025_01_15"
    plot_bound_gap(date, 'hydro-thermal', large_lam, fname)

def get_all_total_elpsed_times(elpsed_times_arr, size_arr, i):
    """ Returs all total elapsed time (without evaluation ) """
    size = int(size_arr[i])
    return elpsed_times_arr[i,:size,0] - elpsed_times_arr[i,:size,3]

def get_times(elpsed_times_arr, size_arr, i):
    """
    Selects final times from:
        - Sub-problem solve (col 1)
        - Next-point selection and model update (col 2)
        - Total time without evaluation: (col 0 - col 3) 

    :param elpsed_times_arr: 3d array storing times (see above)
    :param size_arr: duration of run_i
    :param i: run index
    """
    size = int(size_arr[i])
    return [
        elpsed_times_arr[i, size-1, 1], 
        elpsed_times_arr[i, size-1, 2], 
        elpsed_times_arr[i, size-1, 0]-elpsed_times_arr[i, size-1, 3], 
        # elpsed_times_arr[i, size-1, 0]
    ]

def plot_runtimes(date, prob_name, large_lam, fname=None):
    """
    Compare runtimes between all EDDP algs on.
    """
    i_0 = 0 if large_lam else 13
    gamma = 0.9906 if large_lam else 0.8

    categories = ['Sub-problem', 'Selection and update', 'Total']  # Categories
    experiment_log_folder = os.path.join(log_folder, "%s/exp_0" % date)
    run_id_arr = list(range(i_0,i_0+12)) + list(range(i_0+14,i_0+24))
    folder_arr = [os.path.join(experiment_log_folder, "run_%d" % i) for i in run_id_arr]
    name_arr = ["Inf-EDDP", "CE-Inf-EDDP", "Gap-Inf-EDDP", "Inf-SDDP", "SDDP"]
    bounds_arr, elpsed_times_arr, size_arr = read_logs(folder_arr)
    total_times_arr = np.zeros((len(folder_arr), 3), dtype=float)
    for i in range(len(run_id_arr)):
        total_times_arr[i,:] = get_times(elpsed_times_arr, size_arr, i)
    # take the average for Inf-SDDP
    total_times_arr[3,:] = np.mean(total_times_arr[3:13,:], axis=0)
    total_times_arr[4,:] = np.mean(total_times_arr[13:23,:], axis=0) * np.min(size_arr)/size_arr[13]

    # Set the positions of the bars on the x-axis
    width = 1./(len(categories)+2.5)  # The width of the bars
    x = np.arange(len(categories))  # The x positions for each category

    # Create the figure and axis
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(7, 4))

    # Plot bars for each variable
    label_arr = ['',] * 12
    for i in range(5):
        ax.bar(x - (i-2)*width, total_times_arr[i,:], width, label=name_arr[i], hatch=hatch_arr[i])
        label_arr[i*3:(i+1)*3] = ['%.1fs' % t for t in total_times_arr[i,:]]

    # add value at top (https://stackoverflow.com/questions/28931224/how-to-add-value-labels-on-a-bar-chart)
    rect_arr = ax.patches
    for rect, label in zip(rect_arr, label_arr):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom"
        )

    ax.set(
        xlabel='Component',
        ylabel='Times (seconds)',
        title=r"Runtime for %s ($\gamma=%s$) over %d subprobs" % (prob_name, gamma, 1+np.min(size_arr)),
        yscale='log',
    )
    ax.set_xticks(x+0.5*width)  
    ax.set_xticklabels(categories)  
    ax.legend()  

    plt.tight_layout()  
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, dpi=240)

def plot_hydro_runtimes(large_lam, fname=None):
    date = "2025_01_15"
    plot_runtimes(date, 'hydro-thermal', large_lam, fname)

def plot_newsvendor_runtimes(large_lam, fname=None):
    date = "2025_01_16"
    plot_runtimes(date, 'newsvendor', large_lam, fname)

def print_optimality_gap(large_lam):
    """ 
    Print optimality gap for thermal problem 
    """

    date = "2025_01_15" 
    if large_lam:
        i_long_0 = 2
        gamma = 0.9906
        name_arr = ["Inf-EDDP", "CE-Inf-EDDP", "Gap-Inf-EDDP", "Inf-SDDP", "EDDP", "SDDP"]
        run_id_arr = list(range(24))
        iter_arr = [500, 2000, 96000]
    else:
        i_long_0 = 0
        i_0 = 24
        gamma = 0.8
        name_arr = ["Inf-EDDP", "CE-Inf-EDDP", "Gap-Inf-EDDP", "Inf-SDDP", "EDDP", "SDDP", "GCE-Inf-EDDP"]
        run_id_arr = list(range(i_0,i_0+2)) + list(range(i_0+2,i_0+14)) + list(range(i_0+15,i_0+25)) + [i_0+14]
        iter_arr = [150, 1000, 4800]

    experiment_log_folder = os.path.join(log_folder, "%s/exp_0" % date)
    folder_arr = [os.path.join(experiment_log_folder, "run_%d" % i) for i in run_id_arr]
    bounds_arr, elpsed_times_arr, size_arr = read_logs(folder_arr)

    # orders the times based on `name_arr`. Start with Inf-EDDP, CE-Inf-EDDP, and Gap-Inf-EDDP. 
    reordered_bounds_arr = np.zeros((len(name_arr), bounds_arr.shape[1], 2), dtype=float)
    reordered_size_arr = np.zeros(len(name_arr), dtype=float)
    for i in range(3):
        reordered_bounds_arr[i,:,:] = bounds_arr[i,:,:]
        reordered_size_arr[i] = size_arr[i]
    # EDDP
    reordered_bounds_arr[4,:,:] = bounds_arr[13,:,:]
    reordered_size_arr[4] = size_arr[13]
    if not large_lam:
        # GCE-Inf-EDDP
        reordered_bounds_arr[6,:,:] = bounds_arr[24,:,:]
        reordered_size_arr[6] = size_arr[24]

    # take the average for Inf-SDDP (Inf-SDDP and SDDP)
    reordered_bounds_arr[3,:,:] = np.median(bounds_arr[3:13,:,:], axis=0)
    reordered_bounds_arr[5,:,:] = np.median(bounds_arr[14:24,:,:], axis=0)
    reordered_size_arr[3] = size_arr[3]
    reordered_size_arr[5] = size_arr[14]

    # use extended experiments for EDDP and SDDP
    date = "2025_01_22" 
    long_experiment_log_folder = os.path.join(log_folder, "%s/exp_0" % date)
    long_folder_arr = [os.path.join(long_experiment_log_folder, "run_%d" % i) for i in range(i_long_0, i_long_0+2)]
    long_bounds_arr, long_elpsed_times_arr, long_size_arr = read_logs(long_folder_arr)

    # copy over old data
    old_n = reordered_bounds_arr.shape[1]
    new_bounds_arr = np.zeros((len(name_arr), int(np.max(long_size_arr)), 2), dtype=float)
    assert old_n <= np.max(long_size_arr)
    for i in range(len(name_arr)):
        new_bounds_arr[i,:old_n,:] = reordered_bounds_arr[i,:,:]
    reordered_bounds_arr = new_bounds_arr

    # replace with extneded EDDP and SDDP
    reordered_bounds_arr[4,:,:] = long_bounds_arr[0,:,:]
    reordered_bounds_arr[5,:,:] = long_bounds_arr[1,:,:]
    reordered_size_arr[4] = long_size_arr[0]
    reordered_size_arr[5] = long_size_arr[1]
    reordered_size_arr = reordered_size_arr.astype(int)

    print("-"*50)
    print("Gamma=%s" % (0.9906 if large_lam else 0.8))
    print("-"*50)
    for i, name in enumerate(name_arr):
        print("Alg %s" % name)
        for t in iter_arr:
            if reordered_size_arr[i] < t-1:
                print("Skipping %s's size_arr[i]=%d since t=%d" % (name, reordered_size_arr[i], t))
                lb = reordered_bounds_arr[i,:reordered_size_arr[i],0]
                ub = reordered_bounds_arr[i,:reordered_size_arr[i],1]
                gap = np.divide(ub-lb, lb)
                t_star = np.argmin(gap)
                print("(Last) iter %d: lb=%.2f ub=%.2f (gap=%.1f percent)" % (
                    reordered_size_arr[i], lb[t_star], ub[t_star], gap[t_star]*100.))
                continue
            lb = reordered_bounds_arr[i,:t-1,0]
            ub = reordered_bounds_arr[i,:t-1,1]
            gap = np.divide(ub-lb, lb)
            t_star = np.argmin(gap)
            print("Iter %d: lb=%.2f ub=%.2f (gap=%.1f percent)" % (t, lb[t_star], ub[t_star], gap[t_star]*100.))

    print("-"*50)

def read_oos_logs(folder_arr, n_seeds):
    """
    Read in out-of-sample files from folder. 
    Assumes there are `<folder_arr>/eval_seed=%d.csv` files storing cumulative costs with "%d" ranging from [0,n_seeds).

    :return perf_arr: array of shape (n_algs, n_seeds) holding final performance
    """
    n_folders = len(folder_arr)
    perf_arr = np.zeros((n_folders, n_seeds), dtype=float)

    for i, folder in enumerate(folder_arr):
        for s in range(n_seeds):
            perf_arr[i,s] = np.squeeze(pd.read_csv(os.path.join(folder, 'eval_seed=%d.csv' % s), header=None).to_numpy())[-1]

    return perf_arr

def print_oos_performance(prob_name):
    """ 
    Prints in/out-of-sample performance for both inventory and hydro with gamma=0.9906
    """
    if prob_name == 'inventory':
        date = "2025_01_16"
    elif prob_name == 'hydro':
        date = "2025_01_15"
    else:
        raise "Unknown prob_name=%s" % prob_name

    run_id_arr = []
    name_arr = ["Inf-EDDP", "CE-Inf-EDDP", "Gap-Inf-EDDP", "Inf-SDDP(0)", "SDDP(0)"]
    run_id_arr = [0,1,2,3,14]

    oos_log_folder = os.path.join(log_folder, "%s/exp_0" % date)
    folder_arr = [os.path.join(oos_log_folder, "run_%d" % i) for i in run_id_arr]
    perf_arr = read_oos_logs(folder_arr, n_seeds=30)

    print("-"*50)
    print("Prob_name=%s" % prob_name)
    print("-"*50)
    for i, name in enumerate(name_arr):
        print("alg=%s" % name)
        # in-sample
        if perf_arr[i,0] > 1e5:
            print("In-sample: %.3e" % perf_arr[i,0])
        else:
            print("In-sample: %.1f" % perf_arr[i,0])

        # out-sample
        oos_perf_arr = perf_arr[i,1:]
        if perf_arr[i,0] > 1e5:
            print("oos-sample: %.3e +/ %.2e" % (np.mean(oos_perf_arr), np.std(oos_perf_arr)))
        else:
            print("oos-sample: %.1f +/ %.2f" % (np.mean(oos_perf_arr), np.std(oos_perf_arr)))
        print("-"*30)

if __name__ == '__main__':
    plot_bound_gap_hydrothermal('hydro_gap_gammasmall_guided.png')
    # plot_newsvendor_bound_gap(False, 'newsvendor_gap_gammasmall.png')
    plot_newsvendor_bound_gap(True, 'newsvendor_gap_gammalarge.png')
    plot_hydro_bound_gap(True, 'hydro_gap_gammalarge.png')

    plot_newsvendor_runtimes(1, 'newsvendor_runtime_gammalarge.png')

    print_optimality_gap(True)
    print_optimality_gap(False)

    print_oos_performance('inventory')
    print_oos_performance('hydro')
