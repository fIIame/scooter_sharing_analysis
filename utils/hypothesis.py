import pandas as pd

from scipy.stats import ttest_ind, mannwhitneyu, pearsonr, spearmanr, f_oneway


def mannwhitneyu_test(data: pd.DataFrame, target: str, factor: str, alternative: str, alpha: int = 0.05) -> None:
    """
    Выполняет тест Манна–Уитни для сравнения двух групп по количественному признаку.

    Параметры:
    ----------
    data : pd.DataFrame
        Датасет с данными.
    target : str
        Название количественного признака.
    factor : str
        Название категориального признака с двумя группами.
    alternative : str
        Альтернатива: 'two-sided', 'less', 'greater'.
    """
    groups = data.groupby(factor)[target].apply(list)
    stat, p = mannwhitneyu(*groups, alternative=alternative)
    if p >= alpha:
        msg = f"Тест Манна–Уитни {factor}: p-value={p:.4f} → Нет статистически значимой разницы."
    else:
        msg = f"Тест Манна–Уитни {factor}: p-value={p:.4f} → Есть статистически значимая разница."

    print(msg)


def student_test(data: pd.DataFrame, target: str, factor: str, alternative: str, alpha: int = 0.05) -> None:
    """
    Выполняет тест Стьюдента для сравнения двух групп по количественному признаку.

    Параметры:
    ----------
    data : pd.DataFrame
        Датасет с данными.
    target : str
        Название количественного признака.
    factor : str
        Название категориального признака с двумя группами.
    alternative : str
        Альтернатива: 'two-sided', 'less', 'greater'.
    """
    groups = data.groupby(factor)[target].apply(list)
    stat, p = ttest_ind(*groups, alternative=alternative)
    if p >= alpha:
        msg = f"Тест Стьюдента {factor}: p-value={p:.4f} → Нет статистически значимой разницы."
    else:
        msg = f"Тест Стьюдента {factor}: p-value={p:.4f} → Есть статистически значимая разница."

    print(msg)


def spearman_correlation(data: pd.DataFrame, x: str, y: str, alpha: int = 0.05) -> None:
    """
    Выполняет корреляцию Спирмена между двумя количественными признаками.

    Параметры:
    ----------
    data : pd.DataFrame
        Датасет с данными.
    x : str
        Первый количественный признак.
    y : str
        Второй количественный признак.
    """
    corr, p = spearmanr(data[x], data[y])
    if p >= alpha:
        msg = f"Корреляция Спирмена ({x}, {y}): correlation={corr:.4f}, p-value={p:.4f} → Статистически значимая связь отсутствует."
    else:
        msg = f"Корреляция Спирмена ({x}, {y}): correlation={corr:.4f}, p-value={p:.4f} → Статистически значимая связь есть."

    print(msg)


def pearson_correlation(data: pd.DataFrame, x: str, y: str, alpha: int = 0.05) -> None:
    """
    Выполняет корреляцию Пирсона между двумя количественными признаками.

    Параметры:
    ----------
    data : pd.DataFrame
        Датасет с данными.
    x : str
        Первый количественный признак.
    y : str
        Второй количественный признак.
    """
    corr, p = pearsonr(data[x], data[y])
    if p >= alpha:
        msg = f"Корреляция Пирсона ({x}, {y}): correlation={corr:.4f}, p-value={p:.4f} → Статистически значимая связь отсутствует."
    else:
        msg = f"Корреляция Пирсона ({x}, {y}): correlation={corr:.4f}, p-value={p:.4f} → Статистически значимая связь есть."

    print(msg)


def anova_test(data: pd.DataFrame, target: str, factor: str, alpha: float = 0.05) -> None:
    """
    Выполняет однофакторный ANOVA-тест для количественного признака по категориям.

    Параметры:
    ----------
    data : pd.DataFrame
        Датасет с данными.
    target : str
        Количественный признак.
    factor : str
        Категориальный фактор (несколько групп).
    alpha : float, optional
        Уровень значимости, по умолчанию 0.05.
    """
    groups = data.groupby(factor)[target].apply(list)
    stat, p = f_oneway(*groups)
    if p >= alpha:
        msg = f"ANOVA {factor}: F-statistic={stat:.4f}, p-value={p:.4f} → Нет статистически значимой разницы между группами."
    else:
        msg = f"ANOVA {factor}: F-statistic={stat:.4f}, p-value={p:.4f} → Есть статистически значимая разница между группами."

    print(msg)
