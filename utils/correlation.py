import pandas as pd
import numpy as np


def get_eta_correlation(groups: pd.Series, values: pd.Series) -> float:
    """
    Вычисляет коэффициент η-корреляции между категориальным и числовым признаком.

    Параметры
    ----------
    groups : pd.Series
        Категориальный признак.
    values : pd.Series
        Числовой признак.

    Возвращает
    -------
    float
        Коэффициент η-корреляции (0-1), показывает долю вариации числового признака,
        объясненную категориальным.
    """
    y_mean = values.mean()
    ss_between = 0
    ss_within = 0

    for group in groups.unique():
        vals = values[groups == group]
        ss_between += vals.count() * (vals.mean() - y_mean) ** 2
        ss_within += ((vals - vals.mean()) ** 2).sum()

    return np.sqrt(ss_between / (ss_between + ss_within)).round(3)
