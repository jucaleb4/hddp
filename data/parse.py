import numpy as np

def get_rainfall():
    """ Parses rainfall data from data.txt and returns as 2D array, where rows
        are each region's 25% or 75% quartile, and columns are jan-dec rainfall
    """
    f = open('data.txt')
    arr = np.array([], dtype=float)

    for l in f.readlines():
        line = l.rstrip()
        # Skip lines that are comments
        if len(l) == 0 or l[0]=='%':
            continue
        else:
            val = float(line)

        arr = np.append(arr, val)

    assert len(arr) == 12*8
    arr = np.reshape(arr, newshape=(8,12))

    return arr
