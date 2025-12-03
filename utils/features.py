import pandas as pd


def _get_price_per_minute(hour: int, day: int) -> int:
    """
    Рассчитывает цену за минуту поездки на самокате в зависимости от дня недели и часа.

    Параметры:
    ----------
    hour : int
        Час начала поездки (0-23), извлекается из start_date.
    day : int
        День недели (0 — понедельник, 6 — воскресенье).

    Возвращает:
    ----------
    int
        Цена за минуту поездки.

    Логика:
    - Пн-Пт: стандартные тарифы в зависимости от времени суток
    - Сб-Вс: повышенные тарифы
    """
    if day <= 4:
        # Пн-пт
        if 1 <= hour < 6:
            return 3
        elif 6 <= hour < 10:
            return 4
        elif 10 <= hour < 16:
            return 5
        elif 16 <= hour < 22:
            return 6
        else:
            return 5
    else:
        # Сб-Вс
        if 1 <= hour < 6:
            return 3
        elif 6 <= hour < 10:
            return 4
        elif 10 <= hour < 16:
            return 6
        elif 16 <= hour < 22:
            return 7
        else:
            return 6


def get_total_price(row: pd.Series, start_price: int = 30) -> float:
    """
    Вычисляет общую стоимость поездки на самокате с учетом базовой платы и тарифа за минуту.

    Параметры:
    ----------
    row : pd.Series
        Строка DataFrame с колонками:
        - start_date : pd.Timestamp — дата и время начала поездки
        - day_of_week : int — день недели (0 — понедельник)
        - duration_minutes : int или float — длительность поездки в минутах
    start_price : int, default=30
        Фиксированная плата за старт поездки.

    Возвращает:
    ----------
    float
        Общая стоимость поездки.

    Логика:
    ----------
    - В понедельник с 6 до 10 утра берется только тариф за минуту, базовая плата не добавляется.
    - В остальные дни/часы стоимость = start_price + duration_minutes * price_per_minute
    """
    day_of_week = row["day_of_week"]
    hour = row["start_date"].hour

    duration_minutes = row["duration_minutes"]
    price_per_min = _get_price_per_minute(hour, day_of_week)

    if day_of_week == 0 and 6 <= hour < 10:
        total_price = price_per_min * duration_minutes
    else:
        total_price = start_price + duration_minutes * price_per_min

    return total_price