from typing import List, Dict, Callable

import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error


def print_missed_info(data: pd.DataFrame, rows: int, cols: int) -> None:
    """
    Анализ пропусков в датафрейме.

    Функция выводит расширенную статистику по отсутствующим значениям,
    включая:
    1. Общее число пропусков.
    2. Долю пропущенных ячеек от общего числа.
    3. Долю строк, содержащих хотя бы один пропуск.
    4. Количество пропусков по каждому столбцу.

    Параметры:
    ----------
    data : pd.DataFrame
        Датафрейм, в котором проводится анализ пропусков.

    Возвращает:
    ----------
    None
        Функция выводит статистику в консоль и ничего не возвращает.
    """
    n_rows, n_cols = data.shape

    total_missing = data.isna().sum().sum()
    missing_cell_ratio = total_missing / (n_rows * n_cols)
    missing_row_ratio = (data.isna().any(axis=1)).mean()
    missing_by_column = data.isna().sum()

    print("\033[1m\nПроверка пропусков\033[0m")
    print(f"Общее число пропусков: {total_missing}")
    print(f"Доля пропущенных ячеек: {missing_cell_ratio:.1%}")
    print(f"Доля строк с пропусками: {missing_row_ratio:.1%}\n")
    print("Пропуски по столбцам:")
    print(missing_by_column)


def print_duplicated_info(data: pd.DataFrame) -> None:
    """
    Выводит информацию о количестве дубликатов в датафрейме.

    Параметры:
    ----------
        data (pd.DataFrame): Анализируемый датафрейм.

    Возвращает:
    ----------
        - Количество полных дубликатов строк.
    """
    duplicates_count = data.duplicated().sum()

    print("\033[1m" + "\nПроверка на дубликаты" + "\033[0m")
    print(f"Кол-ва дубликатов: {duplicates_count}")


def describe_categorical(df: pd.DataFrame, top_n: int = 30) -> None:
    """
    Печатает расширенную информацию по категориальным колонкам датафрейма:
    количество уникальных значений и TOP-N частотных значений.

    Параметры:
    ----------
        df (pd.DataFrame): Датафрейм, содержащий категориальные колонки.
        top_n (int, optional): Количество наиболее частых значений,
            которые нужно вывести. По умолчанию — 30.

    Возвращает:
    ----------
        - Название каждого категориального признака.
        - Количество уникальных значений.
        - TOP-N наиболее часто встречающихся значений.
    """
    cat_cols = df.select_dtypes(include="object").columns

    for col in cat_cols:
        print("=" * 50)
        print(f"Column: {col}")
        print("- Unique values:", df[col].nunique(dropna=False))
        print(f"- Top {top_n} most frequent values:")
        print(df[col].value_counts(dropna=False).head(top_n))
        print()


def _check_consecutive_nans(data: pd.DataFrame, col: str):
    """
    Анализирует последовательные пропуски (NaN) в указанной колонке датафрейма.

    Метод:
    1. Определяет, какие значения в колонке являются NaN.
    2. Нумерует последовательные блоки изменений (NaN -> не NaN и наоборот).
    3. Вычисляет длину каждого блока NaN.
    4. Выводит:
       - количество блоков пропусков,
       - максимальную длину блока,
       - длины всех блоков.

    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм для анализа.
    col : str
        Имя колонки, в которой проверяются последовательные пропуски.

    Пример использования:
    --------------------
    _check_consecutive_nans(weather_data, 'temperature')
    """
    is_na = data[col].isna()
    # Нумеруем блоки, где значения изменяются (True -> False)
    groups = (is_na != is_na.shift()).cumsum()
    # Вычисляем длину каждого блока NaN
    na_blocks = data[is_na].groupby(groups[is_na]).size()
    if na_blocks.empty:
        print(f"{col}: пропусков нет")
    else:
        print(
            f"{col}: {len(na_blocks)} блоков пропусков, максимальная длина {na_blocks.max()} строк"
        )
        print(f"Длины всех блоков: {na_blocks.tolist()}")


def print_consecutive_nans(data: pd.DataFrame, cols_to_check: list[str]) -> None:
    """
    Проверяет и выводит информацию о последовательных пропусках (NaN) для указанных колонок.

    Параметры:
    ----------
    data : pd.DataFrame
        Датасет для анализа.
    cols_to_check : list[str]
        Список колонок, по которым нужно проверить последовательные пропуски.
    """
    for col in cols_to_check:
        _check_consecutive_nans(data, col)
        print("=" * 20)


def print_eta_correlation_overview(
    data: pd.DataFrame, factor: str, metrics: List[str], func: Callable
) -> None:
    for metric in metrics:
        eta = func(data[factor], data[metric])
        print(f"Влияние {factor} на {metric.replace('_', ' ')}:")
        print(f"  eta-корреляция = {eta:.3f}")
        if eta < 0.1:
            print("  Слабое влияние\n")
        elif eta < 0.3:
            print("  Умеренное влияние\n")
        else:
            print("  Сильное влияние\n")


def print_did_revenue(
    mon_morning_revenue: float,
    mon_midday_revenue: float,
    control_morning_revenue: float,
    control_midday_revenue: float
) -> None:
    """
    Печатает инкрементальный эффект на выручку через Difference-in-Differences.
    """
    incremental_revenue = (mon_morning_revenue - mon_midday_revenue) - (control_morning_revenue - control_midday_revenue)
    print(f"Инкрементальная выручка (Difference-in-Differences): {incremental_revenue:.0f} ₽")


def print_did_avg_price(
    mon_morning_avg_price: float,
    mon_midday_avg_price: float,
    control_morning_avg_price: float,
    control_midday_avg_price: float
) -> None:
    """
    Печатает инкрементальный эффект на среднюю цену через Difference-in-Differences.
    """
    price_effect = (mon_morning_avg_price - mon_midday_avg_price) - (control_morning_avg_price - control_midday_avg_price)
    print(f"Эффект на среднюю цену (Difference-in-Differences): {price_effect:.1f} ₽")
