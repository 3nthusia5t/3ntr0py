## Utilizing CUDA + Numba to calculate entropy.

Normally entropy is calculated using the solution below. The Numba + CUDA solution is around 10% faster than this for single file and up to 300 times faster for multiple files  (on my equipment - NVIDIA 3060). 

```python

from scipy.stats import entropy
import numpy as np

def entropy(labels, base=None):
  labels = np.frombuffer(labels, dtype=np.uint8)

  value,counts = np.unique(labels, return_counts=True)
  return entropy(counts, base=2)

```

Still in development


## Goal

Quickly calculate entropy of over 200k (110 GB) malware samples without using any CPU multiprocessing. 
It took 10522.091444253922 seconds to complete the processing of all 200k malware samples (110GB). 
The malware was stored on a network attached storage, which has greatly impacted the I/O performance. 

By applying CPU multiprocessing I was able to maximize usage of my computer resources and process the 110 GB in around 3600 seconds (1 hour).
The data was stored on the network attached storage.

## Testing

Currently, tests cannot be performed on the Github actions as there is no Nvidia GPU available.
If it will be possible, I will create a self-hosted runner in the future.

## Remarks

Code is not optimized and cleaned yet.