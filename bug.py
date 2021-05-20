impot os
os.system("conda install numba -y")
import numpy as np
from numba import njit, prange
@njit(parallel=True)
def case_parallel(size,case):
    result = np.zeros((size,))
    if case == 1:
        for i in prange(size):
            result[i] += 1
    else:
        for i in prange(size):
            result[i] += 2
    return result.mean()
@njit(parallel=False)
def case_serial(size,case):
    result = np.zeros((size,))
    if case == 1:
        for i in prange(size):
            result[i] += 1
    else:
        for i in prange(size):
            result[i] += 2
    return result.mean()
for size in [1,10,100]:
    print(f'serial  (size={size:3}), should be 1.0: {case_serial(size=size,case=1)}')
    print(f'parallel(size={size:3}), should be 1.0: {case_parallel(size=size,case=1)}')
print(threading_layer())
