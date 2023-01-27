from parse import get_rainfall
from scipy.stats import skewnorm
import matplotlib.pyplot as plt

STD_CUTOFF = 0.675

def gen_rainfall(agg_size):
    """ Generates skew normal rainfall distribution

    Parameters
    ----------
    agg_size : int
        - Aggregate size (must divide 12)
    """
    arr = get_rainfall()
    assert arr.shape[1] % agg_size == 0
    gen_arr = np.zeros((4,arr.shape[1]//agg_size))

    # 4 regions
    for i in range(4):
        bot_quart = arr[2*i+0:2*i+1, :]
        top_quart = arr[2*i+1:2*i+2, :]

        mean = (bot_quart+top_quart)/2
        std  = (bot_quart+mean)/STD_CUTOFF

        print(mean)
        print(std)

    # r = skewnorm.rvs(a, size=1000)

gen_rainfall()
