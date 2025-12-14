from typing import Callable, List

import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error


def print_missed_info(data: pd.DataFrame) -> None:
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


def print_consecutive_nans(data: pd.DataFrame, cols_to_check: List[str]) -> None:
    """
    Проверяет и выводит информацию о последовательных пропусках (NaN) для указанных колонок.

    Параметры:
    ----------
    data : pd.DataFrame
        Датасет для анализа.
    cols_to_check : List[str]
        Список колонок, по которым нужно проверить последовательные пропуски.
    """
    for col in cols_to_check:
        _check_consecutive_nans(data, col)
        print("=" * 20)


def print_eta_correlation_overview(
    data: pd.DataFrame,
    factor: str,
    metric: str,
    func: Callable
) -> None:
    """
    Вычисляет и печатает коэффициент η (eta) для количественного признака относительно категориального фактора.

    Параметры
    ----------
    data : pd.DataFrame
        Датасет с данными.
    factor : str
        Название категориального признака (фактора), влияние которого оценивается.
    metric : str
        Название количественного признака (метрики), для которого вычисляется корреляция.
    func : Callable
        Функция, вычисляющая коэффициент η. Должна принимать два аргумента: фактор и метрику.

    Возвращает
    ----------
    None
        Функция выводит результат в консоль:
        - значение η-корреляции
        - уровень влияния ('Слабое', 'Умеренное', 'Сильное').
    """

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
    Вычисляет и печатает инкрементальную выручку с использованием подхода Difference-in-Differences (DiD).

    Параметры
    ----------
    mon_morning_revenue : float
        Выручка в экспериментальной группе утром.
    mon_midday_revenue : float
        Выручка в экспериментальной группе в полдень.
    control_morning_revenue : float
        Выручка в контрольной группе утром.
    control_midday_revenue : float
        Выручка в контрольной группе в полдень.

    Возвращает
    ----------
    None
        Функция выводит в консоль значение инкрементальной выручки.
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
    Вычисляет и печатает эффект Difference-in-Differences на среднюю цену поездки.

    Параметры
    ----------
    mon_morning_avg_price : float
        Средняя цена в экспериментальной группе утром.
    mon_midday_avg_price : float
        Средняя цена в экспериментальной группе в полдень (или в контрольный период).
    control_morning_avg_price : float
        Средняя цена в контрольной группе утром.
    control_midday_avg_price : float
        Средняя цена в контрольной группе в полдень (или в контрольный период).

    Возвращает
    ----------
    None
        Функция выводит в консоль значение эффекта на среднюю цену.
    """
    price_effect = (mon_morning_avg_price - mon_midday_avg_price) - (control_morning_avg_price - control_midday_avg_price)
    print(f"Эффект на среднюю цену (Difference-in-Differences): {price_effect:.1f} ₽")


def print_model_metrics(
        y_true: pd.Series,
        y_pred: pd.Series,
        mae_threshold: float = None,
        r2_threshold: float = None
) -> None:
    """
    Выводит R^2 и MAE модели, окрашивая в зеленый при хороших значениях и красный при плохих.

    Параметры:
    ----------
    y_true : pd.Series
        Фактические значения целевой переменной.
    y_pred : pd.Series
        Предсказанные значения модели.
    mae_threshold : float, optional
        Порог для MAE. MAE меньше порога → зеленый, иначе красный.
        Если None, цвет по умолчанию (без окраски).
    r2_threshold : float, optional
        Порог для R^2. R^2 больше порога → зеленый, иначе красный.
        Если None, цвет по умолчанию (без окраски).
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    # цветные R^2 и MAE
    if r2_threshold is not None:
        r2_str = f"{GREEN}{r2:.2f}{RESET}" if r2 >= r2_threshold else f"{RED}{r2:.2f}{RESET}"
    else:
        r2_str = f"{r2:.2f}"

    if mae_threshold is not None:
        mae_str = f"{GREEN}{mae:.2f}{RESET}" if mae <= mae_threshold else f"{RED}{mae:.2f}{RESET}"
    else:
        mae_str = f"{mae:.2f}"

    print(f"R^2: {r2_str} | MAE: {mae_str}")
