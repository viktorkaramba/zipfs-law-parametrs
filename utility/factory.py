from collections.abc import Callable

import numpy as np


def create_zipf_vectorized(s: float, c: float) -> Callable[[np.ndarray[float]], np.ndarray[float]]:
    def zipf_vectorized(ranks: np.ndarray[float]) -> np.ndarray[float]:
        return c / np.power(ranks, s)

    return zipf_vectorized
