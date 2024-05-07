import numpy as np
from numba import cuda
import unittest
import math
from scipy.stats import entropy
from numba.core.errors import NumbaPerformanceWarning

# Functions to test
from entro import calculate_histogram, calculate_entropy

class TestCalculateHistogram(unittest.TestCase):
    
    def test_histogram_calculation(self):
        import warnings
        warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

        data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        hist_out = np.zeros(256, dtype=np.uint32)

        expected_hist = np.histogram(data, bins=256, range=(0, 255))[0]

        # Run the function
        calculate_histogram[1, 256](data, hist_out)

        np.testing.assert_array_equal(hist_out[:10], expected_hist[:10], "Histograms do not match")

def entropy_from_histogram(hist):
    """
    Calculate entropy from a histogram.

    Parameters:
        hist (array_like): 1-D array representing the histogram.

    Returns:
        float: Entropy value.
    """
    # Normalize histogram to obtain probability distribution
    prob_dist = hist / np.sum(hist)

    # Remove zero probabilities to avoid logarithm of zero
    prob_dist = prob_dist[prob_dist != 0]

    # Calculate entropy
    entropy = -np.sum(prob_dist * np.log2(prob_dist))

    return entropy

class TestCalculateEntropy(unittest.TestCase):
    def test_entropy_calculation(self):
        import warnings
        warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
        
        # Test data
        hist = np.array([10, 20, 30, 40, 50, 20, 30, 20, 30])

        # Expected value
        expected_entropy = entropy(hist, base=2)
 

        # Actual value
        cuda.synchronize()
        entropy_out_gpu = cuda.device_array(hist.size, dtype=np.float32)
        threadsperblock_entropy = hist.sum()
        calculate_entropy[1, 1](hist, hist.sum(), entropy_out_gpu)
        result = entropy_out_gpu.copy_to_host().sum()
        # Assert with some error
        np.testing.assert_allclose(result, expected_entropy, rtol=1e-6, atol=1e-6, err_msg="Entropies do not match")


    def test_random_histograms(self):
        import warnings
        warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
        
        for i in range(0, 1000):
            # Test data
            hist = np.random.randint(1, 1000, size=np.random.randint(1, 2000))
            # Expected value
            expected_entropy = entropy(hist, base=2)

            # Actual value
            cuda.synchronize()
            entropy_out_gpu = cuda.device_array(hist.size, dtype=np.float32)
            threadsperblock_entropy = hist.size
            blockspergrid_entropy = min((hist.size + (threadsperblock_entropy - 1)) // threadsperblock_entropy, 1024)
            calculate_entropy[threadsperblock_entropy, blockspergrid_entropy](hist, hist.sum(), entropy_out_gpu)
            result = entropy_out_gpu.copy_to_host().sum()
            
            # Assert with some error
            np.testing.assert_allclose(result, expected_entropy, rtol=1e-4, atol=1e-4, err_msg="Entropies do not match")
            del result
            del entropy_out_gpu


if __name__ == '__main__':
    unittest.main()