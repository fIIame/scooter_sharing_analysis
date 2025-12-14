import re
from typing import List

import pandas as pd


def normalize_street(street_name: str) -> str:
    """
    Преобразует название улицы к единому формату.

    Параметры:
    ----------
    street_name : str
        Исходное название улицы.

    Возвращает:
    ----------
    str
        Нормализованное название улицы.

    Шаги нормализации:
    - Приведение к нижнему регистру
    - Удаление префиксов: "ул", "ул.", "улица"
    - Замена разных типов дефисов на обычный "-"
    - Замена пробелов между буквами на дефис
    - Удаление пробелов вокруг дефиса
    - Удаление лишних пробелов
    """
    street_name = street_name.lower().strip()
    street_name = re.sub(r"^(ул\.?|улица)\s*", "", street_name)
    street_name = re.sub(r"[‐‒–—]", "-", street_name)
    street_name = re.sub(r"(?<=\w)\s+(?=\w)", "-", street_name)
    street_name = re.sub(r"\s*-\s*", "-", street_name)
    street_name = re.sub(r"\s+", " ", street_name)
    return street_name.strip()


def normalize_district(district_name: str) -> str:
    """
    Преобразует название района к единому формату.

    Параметры:
    ----------
    district_name : str
        Исходное название района.

    Возвращает:
    ----------
    str
        Нормализованное название района.

    Шаги нормализации:
    - Приведение к нижнему регистру
    - Замена пробелов между словами на дефисы
    """
    district_name = district_name.lower().strip()
    district_name = re.sub(r"(?<=\w)\s+(?=\w)", "-", district_name)
    return district_name


def normalize_day_of_week(val: int) -> str:
    """
    Преобразует числовое значение дня недели в его название.

    Параметры:
    ----------
    val : int
        Число от 0 до 6, где 0 — понедельник, 6 — воскресенье.

    Возвращает:
    -------
    str
        Название дня недели.

    Пример:
    -------
    >>> normalize_day_of_week(0)
    'понедельник'
    >>> normalize_day_of_week(5)
    'суббота'
    """
    days = [
        "понедельник",
        "вторник",
        "среда",
        "четверг",
        "пятница",
        "суббота",
        "воскресенье",
    ]
    return days[val]


def drop_outlers(
        data: pd.DataFrame,
        factor: str,
        k: float
) -> pd.DataFrame:
    """
    Удаляет выбросы из DataFrame по методу межквартильного размаха (IQR).

    Параметры:
    ----------
    data : pd.DataFrame
        Исходный DataFrame.
    factor : str
        Название колонки, по которой ищем выбросы.
    k : float
        Коэффициент для IQR, обычно 1.5 или 3. Чем больше k, тем меньше строк удаляется.

    Возвращает:
    ----------
    pd.DataFrame
        DataFrame без выбросов по указанной колонке.

    Логика:
    ----------
    - Вычисляем Q1, Q3 и IQR
    - Удаляем строки, где значение колонки выходит за пределы [Q1 - k*IQR, Q3 + k*IQR]
    """
    Q1 = data[factor].quantile(0.25)
    Q3 = data[factor].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - k * IQR
    upper = Q3 + k * IQR

    filtered_data = data[(data[factor] >= lower) & (data[factor] <= upper)]
    return filtered_data


def fill_na_median_by_group(
        data: pd.DataFrame,
        cols: List[str],
        group_cols: List[str]
) -> pd.DataFrame:
    """
    Заполняет пропущенные значения в указанных колонках медианой по заданным группам.

    Параметры
    ----------
    data : pd.DataFrame
        Исходный DataFrame.
    cols : list[str]
        Список колонок, в которых нужно заполнить пропуски.
    group_cols : list[str]
        Список колонок для группировки перед вычислением медианы.

    Возвращает
    -------
    pd.DataFrame
        Новый DataFrame с заполненными пропусками в указанных колонках.

    Логика
    ------
    - Группируем данные по колонкам `group_cols`.
    - Для каждой группы заполняем пропуски в колонках `cols` медианой группы.
    - Если группа полностью пустая по данной колонке, остаются NaN.
    """
    for col in cols:
        data[col] = data.groupby(group_cols)[col].transform(lambda s: s.fillna(s.median()))
    return data


def interpolate_time(
        data: pd.DataFrame,
        cols: List[str],
        datetime_col="datetime"
) -> pd.DataFrame:
    """
    Заполняет пропущенные значения в указанных колонках методом линейной интерполяции по времени.

    Параметры
    ----------
    data : pd.DataFrame
        Исходный DataFrame с колонкой времени.
    cols : List[str]
        Список колонок, в которых нужно интерполировать пропуски.
    datetime_col : str, default 'datetime'
        Имя колонки с временными метками.

    Возвращает
    -------
    pd.DataFrame
        Новый DataFrame с интерполированными колонками.

    Логика
    ------
    - Устанавливаем колонку `datetime_col` как индекс.
    - Применяем метод интерполяции 'time' к колонкам `cols`.
    - Сбрасываем индекс обратно, возвращая колонку времени.
    """
    data = data.set_index(datetime_col)
    data[cols] = data[cols].interpolate(method="time")
    data = data.reset_index()
    return data
