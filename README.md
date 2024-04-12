## Utilizing CUDA + Numba to calculate entropy.

Around 10% faster than solution for single file and a lot faster for multiple files (up to 300 times faster on my equipment). 

```python

from scipy.stats import entropy
import numpy as np

def entropy1(labels, base=None):
  labels = np.frombuffer(labels, dtype=np.uint8)

  value,counts = np.unique(labels, return_counts=True)
  return entropy(counts, base=2)

```

Still in development


## Goal

Quickly calculate entropy of over 200k (110 GB) malware samples without using any CPU multiprocessing. 

It took 10522.091444253922 seconds to complete the processing of all 200k malware samples (110GB). 

The malware was stored on network attached storage, which has greatly impacted the I/O performance. 


## Remarks

Code is not optimized and cleaned yet.