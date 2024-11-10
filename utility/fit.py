import numpy as np
from scipy.optimize import curve_fit


def zipf_fit(ranks: np.ndarray[int], frequencies: np.ndarray[int]) -> float:
    # Функція для апроксимації закону Ципфа
    def zipf_func(rank: int, a: int) -> float:
        return a / rank

    # Апроксимація параметра 'a'
    params, _ = curve_fit(zipf_func, ranks, frequencies)
    return params[0]
