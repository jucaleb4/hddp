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

    :return bounds_arr:
    :return elpsed_times_arr:
    :return size_arr:
    """
    n_folders = len(folder_arr)
    bounds_arr = None
    elpsed_times_arr = None
    size_arr = np.zeros(n_folders) # num elements in files from each folder (size may be different)

    for i, folder in enumerate(folder_arr):
        bounds = pd.read_csv(os.path.join(folder, 'bounds.csv'), header='infer')
        elpsed_times = pd.read_csv(os.path.join(folder, 'elpsed_times.csv'), header='infer')
        n_elems = len(bounds)
        size_arr[i] = n_elems

        if bounds_arr is None:
            bounds_arr = np.zeros((n_folders, n_elems, 2)) # lb, ub
            elpsed_times_arr = np.zeros((n_folders, n_elems, 5)) # total_time,fwd_time,select_time,eval_time
        elif n_elems > bounds_arr.shape[1]:
            # resize
            new_bounds_arr = np.zeros((n_folders, n_elems, 2))
            new_elpsed_times_arr = np.zeros((n_folders, n_elems, 5)) 
            for j in range(i):
                new_bounds_arr[i,:,:] = bounds_arr[i,:,:]
                new_elpsed_times_arr[i,:,:] = elpsed_times_arr[i,:,:]

            # copy over old data
            bounds_arr = new_bounds_arr
            elpsed_times_arr = new_bounds_arr

        # add new data
        bounds_arr[i,:,:] = bounds
        elpsed_times_arr[i,:,:] = elpsed_times

    return bounds_arr, elpsed_times_arr, size_arr

def plot_inventory_bound_gap(large_lam, fname=None):
    """
    Plots upper and lower bound inventory gap

    :param large_lam: lam=0.9906 if 1 else 0.8
    """
    i_0 = 0 if large_lam else 13
    gamma = 0.9906 if large_lam else 0.8

    N = 4
    color_arr = [(0,0,0)] + list([colorsys.hsv_to_rgb(x*1./N, 0.75, 0.5+x*0.5/N) for x in range(N)])
    plt.style.use('ggplot')
    _, ax = plt.subplots(figsize=(7, 6))

    experiment_log_folder = os.path.join(log_folder, "2025_01_16/exp_0")
    folder_arr = [os.path.join(experiment_log_folder, "run_%d" % i) for i in range(i_0,i_0+12)]
    name_arr = ["Inf-EDDP", "CE-Inf-EDDP", "Gap-Inf-EDDP", "Inf-SDDP"]
    bounds_arr, _, size_arr = read_logs(folder_arr)

    for i in range(3):
        n_iters = int(size_arr[i])
        xs = np.arange(n_iters)
        ax.plot(xs, bounds_arr[i,:n_iters,0], linestyle=lss_arr[0], label=name_arr[i], color=color_arr[i])
        ax.plot(xs, bounds_arr[i,:n_iters,1], linestyle=lss_arr[1], color=color_arr[i])
        print("%s lb=%.8f ub=%.8f" % (name_arr[i], bounds_arr[i,n_iters-1,0], bounds_arr[i,n_iters-1,0]))

    # create and plot quantiles for Inf-SDDP
    inf_sddp_lb_arr = np.sort(bounds_arr[3:12,:,0], axis=0)
    inf_sddp_ub_arr = np.sort(bounds_arr[3:12,:,1], axis=0)
    n_iters = int(size_arr[3])
    xs = np.arange(n_iters)
    # median (average of 5 and 6th best)
    ax.plot(xs, np.mean(inf_sddp_lb_arr[4:6], axis=0), linestyle=lss_arr[0], label=name_arr[3], color=color_arr[3])
    ax.plot(xs, np.mean(inf_sddp_ub_arr[4:6], axis=0), linestyle=lss_arr[1], color=color_arr[3])
    # 10% and 90% quantile
    ax.fill_between(xs, inf_sddp_lb_arr[1], inf_sddp_lb_arr[8], color=color_arr[3])
    ax.fill_between(xs, inf_sddp_ub_arr[1], inf_sddp_ub_arr[8], color=color_arr[3])
    print("%s median lb=%.8f ub=%.8f" % (name_arr[3], np.mean(inf_sddp_lb_arr[4:6,-1]), np.mean(inf_sddp_ub_arr[4:6,-1])))
    print("%s .10 lb=%.8f ub=%.8f" % (name_arr[3], inf_sddp_lb_arr[1,-1], inf_sddp_ub_arr[1,-1]))
    print("%s .90 lb=%.8f ub=%.8f" % (name_arr[3], inf_sddp_lb_arr[8,-1], inf_sddp_ub_arr[8,-1]))

    ax.legend()
    ax.set(
        yscale='log', 
        xlabel="Iterations (num. sub-problems solved)",
        title=r"Upper and lower bound gaps for inventory ($\gamma={}$)".format(gamma),
    )
    plt.tight_layout()
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, dpi=240)

def get_times(elpsed_times_arr, size_arr, i):
    """
    Selects times
        - Sub-problem solve (col 1)
        - Next-point selection and model update (col 2)
        - Total time without evaluation: (col 0 - col 3) 
    """
    size = int(size_arr[i])
    return [
        elpsed_times_arr[i, size-1, 1], 
        elpsed_times_arr[i, size-1, 2], 
        elpsed_times_arr[i, size-1, 0]-elpsed_times_arr[i, size-1, 3], 
        # elpsed_times_arr[i, size-1, 0]
    ]

def plot_inventory_runtimes(large_lam, fname=None):
    """
    Compare runtimes between all EDDP algs on.
    """
    i_0 = 0 if large_lam else 13
    gamma = 0.9906 if large_lam else 0.8

    categories = ['Sub-problem', 'Selection and update', 'Total']  # Categories
    experiment_log_folder = os.path.join(log_folder, "2025_01_16/exp_0")
    folder_arr = [os.path.join(experiment_log_folder, "run_%d" % i) for i in range(i_0,i_0+12)]
    name_arr = ["Inf-EDDP", "CE-Inf-EDDP", "Gap-Inf-EDDP", "Inf-SDDP"]
    bounds_arr, elpsed_times_arr, size_arr = read_logs(folder_arr)
    total_times_arr = np.zeros((len(folder_arr), 3), dtype=float)
    for i in range(i_0,i_0+12):
        total_times_arr[i,:] = get_times(elpsed_times_arr, size_arr, i)
    # take the average for Inf-SDDP
    total_times_arr[3,:] = np.mean(total_times_arr[3:,:], axis=0)

    # Set the positions of the bars on the x-axis
    width = 1./(len(categories)+2)  # The width of the bars
    x = np.arange(len(categories))  # The x positions for each category

    # Create the figure and axis
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot bars for each variable
    label_arr = ['',] * 12
    for i in range(4):
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
        xlabel='Runtime',
        ylabel='Times (seconds)',
        title=r"Breakdown of runtime for inventory ($\gamma=%s$)" % gamma,
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

if __name__ == '__main__':
    plot_inventory_bound_gap(1)
    # plot_inventory_runtimes(0, 'inventory_runtime_gammasmall.png')
