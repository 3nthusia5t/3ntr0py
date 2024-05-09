import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
import warnings
import math

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


@cuda.jit
def hamming_distance(src, dst, result):

    stride = cuda.gridsize(1)
    idx = cuda.grid(1)

    for i in range(idx, len(src), stride):
        tmp = src[i] ^ dst[i]
        cuda.syncthreads()
        while(tmp):
            result[i] += tmp & 1
            tmp >>= 1

@cuda.jit
def find_strings(src, result):
    stride = cuda.gridsize(1)
    idx = cuda.grid(1)
    for i in range(idx, len(src), stride):
        cuda.syncthreads()
        y = 1
        cuda.syncthreads()
        if 33 <= src[i] <= 126:
            if not(33 <= src[i-1] <= 126):
                result[i][0] = src[i]
                if i + y < len(src): 
                    while (33 <= src[i + y] <= 126):
                        result[i][y] = src[i + y]
                        y += 1