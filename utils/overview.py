from typing import List, Dict, Callable

import pandas as pd


def get_missed_info(data: pd.DataFrame, rows: int, cols: int) -> None:
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


def get_duplicated_info(data: pd.DataFrame) -> None:
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


def print_conversion(data: pd.DataFrame, column: str, positive_value: str) -> None:
    """
    Выводит конверсию по бинарному категориальному признаку.

    Параметры:
    ----------
    data : pd.DataFrame
        Датафрейм с данными.
    column : str
        Название колонки, где содержится бинарный признак (например, 'promo').
    positive_value : str
        Значение, которое считается "успешным" (например, 'Да').

    Возвращает:
    ----------
    None
        Просто печатает конверсию в удобном формате.
    """
    total = data.shape[0]
    positive = data[data[column] == positive_value].shape[0]

    conversion = positive / total if total > 0 else 0

    print("\033[1mКонверсия\033[0m")
    print(f"{column} = {positive_value}: {conversion:.1%} ({positive} из {total})\n")


def calculate_promo_roi(
    df: pd.DataFrame,
    promo_col: str = "promo",
    price_col: str = "total_price",
    promo_value: str = "Да",
    promo_cost_per_ride: float = 30,
) -> Dict[str, float]:
    """
    Рассчитывает ROI промо-акции с цветным выводом.

    Параметры:
    ----------
    df : pd.DataFrame
        Датасет с поездками.
    promo_col : str
        Название колонки с информацией о промо.
    price_col : str
        Название колонки с ценой поездки.
    promo_value : str
        Значение, указывающее на поездку с промо.
    promo_cost_per_ride : float
        Стоимость проведения промо на одну поездку.
    display : bool
        Выводить ли результаты в консоль.

    Возвращает:
    ----------
    dict : словарь с ключами
        baseline_revenue, revenue_promo, incremental_profit, promo_cost, roi, roi_percent
    """
    promo_rides = df[df[promo_col] == promo_value]
    no_promo_rides = df[df[promo_col] != promo_value]

    mean_no_promo = no_promo_rides[price_col].mean()
    baseline_revenue = mean_no_promo * promo_rides.shape[0]
    revenue_promo = promo_rides[price_col].sum()
    incremental_profit = revenue_promo - baseline_revenue
    promo_cost = promo_rides.shape[0] * promo_cost_per_ride
    roi = incremental_profit / promo_cost
    roi_percent = roi * 100

    print(f"Baseline выручка: {baseline_revenue:_.0f} руб.")
    print(f"Фактическая выручка по промо: {revenue_promo:_.0f} руб.")
    print(f"Инкрементальная прибыль: {incremental_profit:_.0f} руб.")
    print(f"Издержки на промо: {promo_cost:_.0f} руб.")

    # Цветной ROI
    if roi_percent >= 0:
        color = "\033[92m"  # зеленый
        status = "положительный"
    else:
        color = "\033[91m"  # красный
        status = "отрицательный"
    reset = "\033[0m"
    print(f"ROI: {color}{roi_percent:.1f}% ({status}){reset}")
