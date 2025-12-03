import re
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


def drop_outlers(data: pd.DataFrame, factor: str, k: float) -> pd.DataFrame:
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