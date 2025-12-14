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
    - В понедельник с 6 до 10 утра при использовании промо берется только тариф за минуту, плата за старт не добавляется.
    - В остальные дни/часы стоимость = start_price + duration_minutes * price_per_minute
    """
    day_of_week = row["day_of_week"]
    hour = row["start_date"].hour

    duration_minutes = row["duration_minutes"]
    price_per_min = _get_price_per_minute(hour, day_of_week)

    is_promo = row["promo"]

    if day_of_week == 0 and 6 <= hour < 10 and is_promo:
        total_price = price_per_min * duration_minutes
    else:
        total_price = start_price + duration_minutes * price_per_min

    return total_price


def filter_time(
        data: pd.DataFrame,
        day_of_week: str,
        start_hour: int,
        end_hour: int
) -> pd.DataFrame:
    """
    Фильтрует поездки по дню недели и диапазону часов.

    Параметры:
    ----------
    data : pd.DataFrame
        Датасет с поездками. Должен содержать колонки:
        - 'day_of_week' : str — день недели
        - 'start_date' : pd.Timestamp — дата и время начала поездки
    day_of_week : str
        День недели, по которому фильтруем поездки (например, "понедельник").
    start_hour : int
        Начальный час интервала (включительно, 0–23).
    end_hour : int
        Конечный час интервала (не включительно, 1–24).

    Возвращает:
    ----------
    pd.DataFrame
        Отфильтрованный DataFrame, содержащий только поездки, которые
        происходят в указанный день недели и в указанный диапазон часов.
    """

    return data[
        (data['day_of_week'] == day_of_week) &
        (data['start_date'].dt.hour >= start_hour) &
        (data['start_date'].dt.hour < end_hour)
    ]


def select_control_day(
        data: pd.DataFrame,
        target_rides: pd.Series,
        exclude_day: str = 'понедельник',
        start_hour: int = 10,
        end_hour: int =14
) -> str:
    """
    Выбирает контрольный день, где число поездок в указанном интервале
    максимально близко к целевому значению target_rides.

    Параметры:
    ----------
    data : pd.DataFrame
        Датасет с поездками. Должен содержать колонки:
        - 'day_of_week' : str — день недели
        - 'start_date' : pd.Timestamp — дата и время начала поездки
        - 'id' : уникальный идентификатор поездки
        - 'day_timestamp' : pd.Timestamp — уникальный день
    target_rides : int
        Количество поездок, к которому нужно подобрать контрольный день.
    exclude_day : str, default='понедельник'
        День недели, который исключаем из кандидатов (например, чтобы не брать понедельник).
    start_hour : int, default=10
        Начальный час интервала для подсчёта поездок (включительно).
    end_hour : int, default=14
        Конечный час интервала для подсчёта поездок (не включительно).

    Возвращает:
    ----------
    str
        Значение 'day_timestamp' контрольного дня, который максимально
        близок к target_rides по количеству поездок в указанном интервале.
    """
    candidates = data[
        (data['day_of_week'] != exclude_day) &
        (data['start_date'].dt.hour >= start_hour) &
        (data['start_date'].dt.hour < end_hour)
    ]
    stats = candidates.groupby("day_timestamp")["id"].count().reset_index(name="rides")
    control_day = stats.assign(distance=lambda x: abs(x["rides"] - target_rides))\
                       .sort_values("distance")\
                       .iloc[0]["day_timestamp"]
    return control_day
