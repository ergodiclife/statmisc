import numpy as np
import numba

from typing import Callable, List

@njit
def draw_samples(x: List, size: int) -> List:
  n = len(x)
  return [np.random.choice(x, size=n) for _ in range(size)]
