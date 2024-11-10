import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def rank_words(words: list[str]) -> pd.DataFrame:
    s = pd.Series(words)

    df = s.value_counts().reset_index()
    df.columns = ["word", "frequency"]

    df["rank"] = df["frequency"].rank(method="dense", ascending=False)
    return df


def zipf_fit(ranks: np.ndarray[int], frequencies: np.ndarray[int]) -> float:
    # Функція для апроксимації закону Ципфа
    def zipf_func(rank: int, a: int) -> float:
        return a / rank

    # Апроксимація параметра 'a'
    params, _ = curve_fit(zipf_func, ranks, frequencies)
    return params[0]
