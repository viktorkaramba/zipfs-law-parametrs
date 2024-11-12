import math

import numpy as np
import pandas as pd
from scipy.stats import linregress


def rank_words(words: list[str]) -> pd.DataFrame:
    s = pd.Series(words)

    df = s.value_counts().reset_index()
    df.columns = ["word", "frequency"]

    df["rank"] = df["frequency"].rank(method="dense", ascending=False)
    return df


def fit_zipf(ranks: np.ndarray[int], frequencies: np.ndarray[int]) -> tuple[float, float]:
    """Fit non-normalized Zipf distribution function parameters"""
    # frequency = C / rank ^ s
    # ln(frequency) = k * ln(rank) + b, where k = -s, b = ln(C)

    k, b, _, _, _ = linregress(np.log(ranks), np.log(frequencies))

    s = -float(k)
    c = math.exp(b)

    return s, c
