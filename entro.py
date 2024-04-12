import numpy as np
from numba import cuda
import math
@cuda.jit
def calculate_histogram(data, hist_out):
    # Initialize shared memory for local histogram
    local_hist = cuda.shared.array(256, dtype=np.uint32)
    tx = cuda.threadIdx.x

    local_hist[tx] = 0
    cuda.syncthreads()


    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(idx, data.shape[0], stride):
        cuda.atomic.add(local_hist, data[i], 1)
    cuda.syncthreads()


    cuda.atomic.add(hist_out, tx, local_hist[tx])


@cuda.jit
def calculate_entropy(hist, total_pixels, entropy_out):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(idx, hist.shape[0], stride):
        prob = hist[i] / total_pixels
        if prob != 0:
            entropy_out[i] = -prob * math.log2(prob)
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

    total_pixels = len(data)
    data_gpu = cuda.to_device(np.frombuffer(data, dtype=np.uint8))

    cuda.synchronize()
    hist_host = np.zeros(256, dtype=np.uint32)
    #hist_out = cuda.device_array(256, dtype=np.uint32)
    # Initialize histogram array to zeros
    #cuda.device_array_like(hist_out, fill_value=0)
    
    hist_out = cuda.to_device(hist_host)
    cuda.synchronize()
    threadsperblock_hist = 256
    blockspergrid_hist = min((len(data) + (threadsperblock_hist - 1)) // threadsperblock_hist, 1024)
    calculate_histogram[blockspergrid_hist, threadsperblock_hist](data_gpu, hist_out)
       

    del data_gpu
    cuda.synchronize()

    entropy_out_gpu = cuda.device_array(256, dtype=np.float32)
    

    threadsperblock_entropy = 256
    blockspergrid_entropy = min((hist_out.size + (threadsperblock_entropy - 1)) // threadsperblock_entropy, 1024)
    calculate_entropy[blockspergrid_entropy, threadsperblock_entropy](hist_out, total_pixels, entropy_out_gpu)

    cuda.synchronize()
    del hist_out

    result = cuda.device_array(blockspergrid_entropy, dtype=np.float32)
    
    cuda.synchronize()

    sum_array[blockspergrid_entropy, threadsperblock_entropy](entropy_out_gpu, result)


    cuda.synchronize()
    del entropy_out_gpu

 
    entropy_sum = result.copy_to_host()

    del result


    cuda.synchronize()

    return entropy_sum.sum()

def is_supported_cuda():
    return cuda.is_available() and cuda.detect()