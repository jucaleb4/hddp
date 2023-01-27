import numpy as np
from parse import get_rainfall
from scipy.stats import skewnorm
import matplotlib.pyplot as plt

STD_CUTOFF = 0.675

def gen_rainfall(N, agg_size):
    """ Generates skew normal rainfall distribution

    Parameters
    ----------
    N : int 
        - Number of periods to generate
    agg_size : int
        - Aggregate size (must divide 12)
    """
    assert N>=1
    arr = get_rainfall()

    num_months = arr.shape[1]
    assert num_months % agg_size == 0
    gen_arr = np.zeros((4, N*(num_months//agg_size)), dtype=float)

    # 4 regions: N -> NE -> S -> SE
    for i in range(4):
        # every 2 rows store the 75% and 25% percentile rainfall, resp
        bot_quart = arr[2*i:2*i+1, :]
        top_quart = arr[2*i+1:2*(i+1), :]

        mean = (bot_quart+top_quart).flatten()/2
        std  = (mean-bot_quart).flatten()/STD_CUTOFF

        # stores this regions rainfall month by month
        local_arr = np.zeros(N*num_months, dtype=float)
        for j in range(num_months):
            data = np.random.normal(loc=mean[j], scale=std[j], size=N)
            if np.amin(data) < 0:
                print('found negative value, setting to small noise')
                # data[data < 0] = np.exp(np.random.normal(np.sum(data < 0)))
                data = np.maximum(data, 0)
            local_arr[j::num_months] = data

        # aggregate sums 
        local_arr = local_arr.reshape(len(local_arr)//agg_size, agg_size).sum(
                axis=1)
        gen_arr[i,:] = local_arr/agg_size

    return gen_arr

data = gen_rainfall(3,1)
# plt.plot(np.arange(data.shape[1]), data[0])
# plt.show()
