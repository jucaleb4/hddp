import numpy as np
import matplotlib.pyplot as plt

def parse_lb_ub(fname):
    fp = open(fname, "r")

    ct = 0
    lb_arr = np.zeros(128)
    ub_arr = np.zeros(128)

    for unproc_line in fp.readlines():
        line = unproc_line.rstrip()
        idx = line.find("[[")
        if idx >= 0:
            line = line[idx+2:]
            idx = line.find(", ")
            lb_arr[ct] = float(line[:idx])

            line = line[idx+2:]
            idx = line.find("]]")
            ub_arr[ct] = float(line[:idx])

            ct += 1
            arr_too_small = (ct >= len(lb_arr))
            if arr_too_small:
                lb_arr = np.append(lb_arr, np.zeros(len(lb_arr)))
                ub_arr = np.append(ub_arr, np.zeros(len(ub_arr)))

    return lb_arr[:ct], ub_arr[:ct]

def plot_ddp_comparisons():
    """ Compares EDDP and SDDP variations for hydro-thermal planning.  """

    eddp_lb, eddp_ub = parse_lb_ub("output/run_0.txt")
    eddp_fast_lb, eddp_fast_ub = parse_lb_ub("output/run_1.txt")
    eddp_lu_lb, eddp_lu_ub = parse_lb_ub("output/run_2.txt")

    # Gather 10 trials of SDDP and compute 10th, 50th, 90th percentile
    esddp_lbs = np.zeros((10, len(eddp_lb)))
    esddp_ubs = esddp_lbs.copy()
    for i in range(10):
        esddp_lb, esddp_ub = parse_lb_ub("output/run_{}.txt".format(i+3))
        esddp_lbs[i,:] = esddp_lb
        esddp_ubs[i,:] = esddp_ub

    esddp_lbs_sorted = np.sort(esddp_lbs, axis=0)
    esddp_ubs_sorted = np.sort(esddp_ubs, axis=0)

    xs = np.arange(len(eddp_lb))

    _, ax = plt.subplots(figsize=(7,5))

    colors = ["black", "gray", "red", "blue", "red"]

    ax.plot(xs, eddp_lb, label="EDDP", color=colors[0])
    ax.plot(xs, eddp_ub, color=colors[0], linestyle="dotted")
    ax.plot(xs, eddp_fast_lb, label="EDDPFast", color=colors[1])
    ax.plot(xs, eddp_fast_ub, color=colors[1], linestyle="dotted")
    ax.plot(xs, eddp_lu_lb, label="EDDPLU", color=colors[2])
    ax.plot(xs, eddp_lu_ub, color=colors[2], linestyle="dotted")

    ax.plot(xs, esddp_lbs_sorted[4,:], label="SDDP*", color=colors[3])
    ax.plot(xs, esddp_ubs_sorted[4,:], color=colors[3], linestyle="dotted")
    ax.fill_between(
        xs, esddp_lbs_sorted[1,:], esddp_lbs_sorted[8,:], color=colors[3], alpha=0.15
    )
    ax.fill_between(
        xs, esddp_ubs_sorted[1,:], esddp_ubs_sorted[8,:], color=colors[3], alpha=0.15
    )

    ax.legend()
    ax.grid(color="gray",linestyle=(0,(2,5)))

    ax.set(
        title="Lower and upper bounds for various algorithms",
        ylabel=r"$\min_{x \in X(x_0, \xi_0)} F(x)$",
        xlabel="Iteration count",
    )

    plt.tight_layout()
    # plt.show()
    plt.savefig("eddp_fig1.png", dpi=240)

def plot_hddp():
    hddp_lb, hddp_ub = parse_lb_ub("output/run_13.txt")
    hddp_approx_lb, hddp_approx_ub = parse_lb_ub("output/run_15.txt")

    n_0 = 10
    n = 500
    hddp_lb = hddp_lb[n_0:n]
    hddp_ub = hddp_ub[n_0:n]
    hddp_approx_lb = hddp_approx_lb[n_0:n]
    hddp_approx_ub = hddp_approx_ub[n_0:n]
    kernel_size = 10
    kernel = np.ones(kernel_size) / kernel_size
    hddp_approx_ub_conv = np.convolve(hddp_approx_ub, kernel, mode='same') 

    xs = n_0 + np.arange(len(hddp_lb))

    _, ax = plt.subplots(figsize=(7,5))

    colors = ["black", "red", "gray", "blue", "red"]
    ax.plot(xs, hddp_approx_lb, label="HDDP (with error)", color=colors[1])
    # ax.plot(xs, hddp_approx_ub, color=colors[1], linestyle="dotted")
    ax.plot(xs[:-kernel_size+1], hddp_approx_ub_conv[:-kernel_size+1], color=colors[1], linestyle="dotted")
    ax.plot(xs, hddp_lb, label="HDDP (exact)", color=colors[0])
    ax.plot(xs, hddp_ub, color=colors[0], linestyle="dotted")

    ax.legend()
    ax.grid(color="gray",linestyle=(0,(2,5)))

    ax.set(
        title="Lower and upper bounds for HDDP with exact and inexact solvers",
        ylabel=r"$\min_{x \in X(x_0, \xi_0)} F(x)$",
        xlabel="Iteration count",
    )

    plt.tight_layout()
    # plt.show()
    plt.savefig("eddp_fig2.png", dpi=240)

if __name__ == "__main__":
    plot_hddp()