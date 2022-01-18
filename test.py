from numba import jit, cuda, njit
import numpy as np
# to measure exec time
from timeit import default_timer as timer

from numpy import vectorize


def func(a):
    for i in range(10000000):
        a[i] += 1

    # function optimized to run on gpu


@jit
def func2(a):
    for i in range(10000000):
        a[i] += 1

@jit(nopython=True)
def func3(a):
    for i in range(10000000):
        a[i] += 1

@cuda.jit
def func4(a):
    for i in range(10000000):
        a[i] += 1

@cuda.jit(device=True)
def func5(a:np.array) -> None:
    for i in range(10000000):
        a[i] += 1

if __name__ == '__main__':
    # https://towardsdatascience.com/speed-up-your-algorithms-part-2-numba-293e554c5cc1
    n = 10000000
    a = np.ones(n, dtype=np.float64)
    b = np.ones(n, dtype=np.float32)

    # start = timer()
    # func(a)
    # print("without GPU:", timer() - start)
    #
    # start = timer()
    # func(a)
    # print("without GPU:", timer() - start)

    start = timer()
    func2(a)
    print("Classic with GPU:", timer() - start)

    start = timer()
    func2(a)
    print("Classic with GPU:", timer() - start)

    start = timer()
    func3(a)
    print("Classic v2 with GPU:", timer() - start)

    start = timer()
    func3(a)
    print("Classic v2 with GPU:", timer() - start)

    a = cuda.to_device(a)
    start = timer()
    func4[1, 1](a)
    print("Cuda with GPU:", timer() - start)

    start = timer()
    func4[1,1](a)
    print("Cuda with GPU:", timer() - start)

    a = cuda.to_device(a)
    start = timer()
    func5(a)
    print("Cuda with GPU:", timer() - start)

    start = timer()
    func5[1,1](a)
    print("Cuda with GPU:", timer() - start)