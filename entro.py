import numpy as np
from numba import cuda
import math

@cuda.jit
def count_values(arr, counts):
    idx = cuda.grid(1)
    if idx < 256:
        counts[idx] = 0

    cuda.syncthreads()

    for i in range(arr.shape[0]):
        if arr[i] == idx:
            cuda.atomic.add(counts, idx, 1)

def count_values_with_cuda(arr):
    counts = np.zeros(256, dtype=np.int32)
    threadsperblock = 256
    blockspergrid = (threadsperblock + len(counts) - 1) // threadsperblock
    count_values[blockspergrid, threadsperblock](arr, counts)
    return counts



@cuda.jit
def calculate_histogram(data, hist_out):
    # Calculate histogram using CUDA
    x = cuda.grid(1)
    if x < data.size:
        cuda.atomic.add(hist_out, data[x], 1)

@cuda.jit
def calculate_entropy(hist, total_pixels, entropy_out):
    # Calculate entropy using CUDA
    x = cuda.grid(1)
    if x < hist.size:
        prob = hist[x] / total_pixels
        if prob != 0:
            entropy_out[x] = -prob * math.log2(prob)

def entropy_with_cuda(data):
    # Convert input data to numpy array
    data_np = np.array(data)

    counts = count_values_with_cuda(data)
    # Determine unique values and their counts
    #unique_values, counts = np.unique(data_np, return_counts=True)

    # Total number of pixels
    total_pixels = data_np.size

    # Compute histogram on GPU
    hist_out = np.zeros_like(range(0,255))
    threadsperblock = 256
    blockspergrid = (data_np.size + (threadsperblock - 1)) // threadsperblock
    calculate_histogram[blockspergrid, threadsperblock](data_np, hist_out)

    # Compute entropy on GPU
    entropy_out = np.zeros_like(hist_out, dtype=np.float32)
    threadsperblock = 256
    blockspergrid = (hist_out.size + (threadsperblock - 1)) // threadsperblock
    calculate_entropy[blockspergrid, threadsperblock](hist_out, total_pixels, entropy_out)

    # Sum the entropy values to get the total entropy
    entropy = np.sum(entropy_out)

    return entropy