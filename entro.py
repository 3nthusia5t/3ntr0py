import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
import warnings
import math

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


@cuda.jit
def calculate_histogram(data, hist_out):
    # Initialize shared memory for local histogram
    local_hist = cuda.shared.array(256, dtype=np.uint32)

    #The Thread id is supposed to be from 0-256. (256 threads per block) 
    tx = cuda.threadIdx.x

    local_hist[tx] = 0

    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(idx, data.shape[0], stride):
        #count 
        cuda.atomic.add(local_hist, data[i], 1)
    cuda.syncthreads()

    #local_hist is shared memory, the other threads will handle other indexes. 
    cuda.atomic.add(hist_out, tx, local_hist[tx])


@cuda.jit
def calculate_entropy(hist, data_size, entropy_out):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(idx, hist.shape[0], stride):
        prob = hist[i] / data_size
        if prob != 0:
            entropy_out[i] = -prob * math.log2(prob)
        else:
            # Some small, not important number
            entropy_out[i] = -0.000001 * math.log2(0.000001)


# TODO: implement it properly.
@cuda.jit
def sum_array(arr, result):
    local_mem = cuda.shared.array(256, dtype=np.float32)

    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x

    i = bid * bdim + tid

    local_mem[tid] = arr[i]
    cuda.syncthreads()

    s = bdim // 2
    while s > 0:
        if tid < s and i < arr.shape[0]:
            local_mem[tid] += local_mem[tid + s]
        cuda.syncthreads()
        s //= 2

    if tid == 0:
        result[bid] = local_mem[0]

    cuda.syncthreads()


def entropy_with_cuda(data):
    # The entropy value for empty set is 0. 
    # However, the main goal of the function is to calculate file entropy, so raising an exception is more practical in this case.
    if not data:
        raise ValueError('The list is empty and cannot be processed.')
    
    data_size = len(data)
    data_gpu = cuda.to_device(np.frombuffer(data, dtype=np.uint8))
    
    #Initialize hist with 0. For some reason numba.cuda.device_array didnt work as expected.
    hist_host = np.zeros(256, dtype=np.uint32)
    hist_out = cuda.to_device(hist_host)
    
    threadsperblock_hist = 256
    blockspergrid_hist = min((len(data) + (threadsperblock_hist - 1)) // threadsperblock_hist, 1024)
    calculate_histogram[blockspergrid_hist, threadsperblock_hist](data_gpu, hist_out)
       
    
    del data_gpu
    cuda.synchronize()

    entropy_out_gpu = cuda.device_array(256, dtype=np.float32)
    

    threadsperblock_entropy = 256
    blockspergrid_entropy = min((hist_out.size + (threadsperblock_entropy - 1)) // threadsperblock_entropy, 1024)
    calculate_entropy[blockspergrid_entropy, threadsperblock_entropy](hist_out, data_size, entropy_out_gpu)

    cuda.synchronize()
    del hist_out

    local_entropies = entropy_out_gpu.copy_to_host()

    #todo: remove sum() make it parrarel
    return local_entropies.sum()

def is_supported_cuda():
    return cuda.is_available() and cuda.detect()