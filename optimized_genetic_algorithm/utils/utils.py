import numpy as np
from numba import jit


@jit(nopython=True)
def shuffle_list(list_to_shuffle):
    np.random.shuffle(list_to_shuffle)





