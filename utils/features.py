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

def _create_traffic_df(
        data: pd.DataFrame,
        date_column: str,
        point_column: str,
        period: str
) -> pd.DataFrame:
    """
    Внутренняя функция для создания DataFrame трафика по точкам.

    Parameters:
    -----------
    data : pd.DataFrame
        Исходный DataFrame с данными
    date_column : str
        Название колонки с датой (start_date или end_date)
    point_column : str
        Название колонки с точкой (start_location или end_location)
    period : str
        Период для ресемплинга

    Returns:
    --------
    pd.DataFrame
        DataFrame с трафиком по периодам и точкам
    """
    # Создаем копию необходимых колонок
    traffic_data = data[[date_column, point_column]].copy()

    # Индексируем по дате
    traffic_data.set_index(date_column, inplace=True)

    # Группируем по точке и ресемплируем по периоду
    traffic_df = (
        traffic_data
        .groupby(point_column)
        .resample(period, include_groups=False)
        .size()
        .unstack(level=0)
        .fillna(0)
    )

    # Устанавливаем имя для колонок
    traffic_df.columns.name = 'point'

    return traffic_df


def create_departures_df(
        data: pd.DataFrame,
        period: str,
        start_point: str = "start_location",
) -> pd.DataFrame:
    """
    Создает DataFrame с количеством отправлений по точкам за указанный период.

    Parameters:
    -----------
    data : pd.DataFrame
        Исходный DataFrame с данными о перемещениях
    period : str
        Период для ресемплинга (например, 'D' - день, 'W' - неделя, 'M' - месяц)
    start_point : str
        Название колонки с точкой отправления

    Returns:
    --------
    pd.DataFrame
        DataFrame с количеством отправлений по периодам и точкам
    """
    return _create_traffic_df(
        data=data,
        date_column='start_date',
        point_column=start_point,
        period=period
    )


def create_arrivals_df(
        data: pd.DataFrame,
        period: str,
        end_point: str = "end_location"
) -> pd.DataFrame:
    """
    Создает DataFrame с количеством прибытий по точкам за указанный период.

    Parameters:
    -----------
    data : pd.DataFrame
        Исходный DataFrame с данными о перемещениях
    period : str
        Период для ресемплинга (например, 'D' - день, 'W' - неделя, 'M' - месяц)
    end_point : str
        Название колонки с точкой прибытия

    Returns:
    --------
    pd.DataFrame
        DataFrame с количеством прибытий по периодам и точкам
    """
    return _create_traffic_df(
        data=data,
        date_column='end_date',
        point_column=end_point,
        period=period
    )


def traffic_by_points(
        data: pd.DataFrame,
        period: str,
        start_point: str = "start_location",
        end_point: str = "end_location"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Анализирует трафик по точкам отправления и прибытия за указанный период.

    Функция вычисляет отдельно отправления (departures) и прибытия (arrivals),
    затем рассчитывает net flow (прибытия - отправления) и общий трафик.

    Parameters:
    -----------
    data : pd.DataFrame
        Исходный DataFrame с данными о перемещениях самокатов.
        Должен содержать колонки: 'start_date', 'end_date', start_point, end_point.
    period : str
        Период для агрегации данных.
        Допустимые значения: 'D' (день), 'W' (неделя), 'M' (месяц), 'H' (час).
    start_point : str, optional
        Название колонки с точкой отправления. По умолчанию "start_location".
    end_point : str, optional
        Название колонки с точкой прибытия. По умолчанию "end_location".

    Returns:
    --------
    tuple[pd.DataFrame, pd.DataFrame]
        Кортеж из двух DataFrame:
        1. total_traffic: DataFrame с общим трафиком (отправления + прибытия)
           по точкам за каждый период времени.
        2. net_long: DataFrame в длинном формате с net балансом по точкам
           за каждый период времени. Колонки: ['time', 'point', 'net_balance']

    Examples:
    ---------
    >>> total_traffic, net_balance = traffic_by_points(data, 'D')
    >>> print(total_traffic.head())
    >>> print(net_balance.head())
    """

    # Создаем отдельные DataFrame для отправлений и прибытий
    departures = create_departures_df(data, period, start_point)
    arrivals = create_arrivals_df(data, period, end_point)

    # разница между приходом и отходом
    net_flow = arrivals.subtract(departures, fill_value=0)
    net_long = net_flow.stack().reset_index()
    net_long.columns = ['time', 'point', 'net_balance']

    # Объединяем отправления и прибытия
    total_traffic = departures.add(arrivals, fill_value=0)

    return total_traffic, net_long


def calculate_optimal_scooters(net_long: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитывает оптимальное количество самокатов для каждой точки за сутки.

    Алгоритм:
    1. Группирует данные по суткам с 6:00 до 6:00 следующего дня
    2. Вычисляет кумулятивную сумму net баланса внутри каждой группы
    3. Определяет минимальный кумулятивный баланс за сутки как дефицит
    4. Рассчитывает оптимальное количество самокатов для покрытия дефицита

    Parameters:
    -----------
    net_long : pd.DataFrame
        DataFrame с net балансом по точкам за периоды времени.
        Должен содержать колонки: ['time', 'point', 'net_balance'].

    Returns:
    --------
    pd.DataFrame
        DataFrame с оптимальным количеством самокатов для каждой точки за сутки.
        Колонки:
        - point: идентификатор точки
        - day_6am: дата начала суток (с 6:00)
        - cumulative: минимальный кумулятивный баланс за сутки
        - optimal_count: оптимальное количество самокатов для точки

    Notes:
    ------
    - Сутки считаются с 6:00 до 6:00 следующего дня
    - optimal_count = |cumulative| если cumulative < 0, иначе 0
    - Отрицательный cumulative означает дефицит самокатов

    Examples:
    ---------
    >>> total_traffic, net_long = traffic_by_points(data, 'D')
    >>> optimal_scooters = calculate_optimal_scooters(net_long)
    >>> print(optimal_scooters.head())
    """

    # 1. Группируем с 6:00 до 6:00
    net_long = net_long.copy()
    net_long['day_6am'] = net_long['time'] - pd.Timedelta(hours=6)
    net_long['day_6am'] = net_long['day_6am'].dt.date

    # 2. Кумулятивная сумма внутри каждой группы (точка + сутки с 6:00)
    net_long = net_long.sort_values(['point', 'day_6am', 'time'])
    net_long['cumulative'] = net_long.groupby(['point', 'day_6am'])['net_balance'].cumsum()

    # 3. Минимальный кумулятивный баланс за сутки = дефицит
    daily_min = net_long.groupby(['point', 'day_6am'])['cumulative'].min().reset_index()
    daily_min['optimal_count'] = daily_min['cumulative'].apply(lambda x: abs(x) if x < 0 else 0)

    return daily_min


def create_od_matrix(data: pd.DataFrame,
                     start_point: str = "start_location",
                     end_point: str = "end_location",
                     period: str = "d"):
    """
    Создает матрицу корреспонденций (Origin-Destination matrix) между точками.

    Parameters:
    -----------
    data : pd.DataFrame
        Исходный DataFrame с данными о перемещениях.
    start_point : str, optional
        Название колонки с точкой отправления (origin). По умолчанию "start_location".
    end_point : str, optional
        Название колонки с точкой прибытия (destination). По умолчанию "end_location".
    period : str, optional
        Период для агрегации данных. Если None, агрегирует за весь период.
        Допустимые значения: 'd' (день), 'w' (неделя), 'm' (месяц), None (весь период).

    Returns:
    --------
    pd.DataFrame
        Матрица корреспонденций с колонками:
        - period: период агрегации (если period указан)
        - start_point: точка отправления
        - end_point: точка прибытия
        - count: количество перемещений

    Examples:
    ---------
    >>> od_matrix = create_od_matrix(data, period='D')
    >>> od_matrix_total = create_od_matrix(data, period=None)
    """

    od_data = data[[start_point, end_point]].copy()

    if period:
        od_data["period"] = data["start_date"].dt.to_period(period)
        return od_data.groupby(["period", start_point, end_point]).size().reset_index(name="count")
    return od_data.groupby([start_point, end_point]).size().reset_index(name="count")


def _classify_point(row: pd.Series):
    """
    Классифицирует точку на основе net flow и соотношения притока/оттока.

    Parameters:
    -----------
    row : pd.Series
        Строка DataFrame с колонками: 'net_flow', 'flow_ratio'.

    Returns:
    --------
    str
        Тип точки:
        - 'strong_acceptor': сильный акцептор (net > 10 и ratio > 1.5)
        - 'acceptor': акцептор (net > 5)
        - 'strong_donor': сильный донор (net < -10 и ratio < 0.5)
        - 'donor': донор (net < -5)
        - 'balanced': сбалансированная (-5 <= net <= 5)
        - 'unknown': неизвестный тип (все остальные случаи)

    Notes:
    ------
    Функция используется внутри analyze_od_flows для классификации точек.
    """

    net = row["net_flow"]
    ratio = row["flow_ratio"]

    if net > 10 and ratio > 1.5:
        return "strong_acceptor"
    elif net > 5:
        return "acceptor"
    elif net < -10 and ratio < 0.5:
        return "strong_donor"
    elif net < -5:
        return "donor"
    elif -5 <= net <= 5:
        return "balanced"
    else:
        return "unknown"


def analyze_od_flows(data: pd.DataFrame,
                     start_point: str = "start_location",
                     end_point: str = "end_location",
                     custom_matrix: pd.DataFrame = None):
    """
    Анализирует потоки между точками и классифицирует их по типам.

    Parameters:
    -----------
    data : pd.DataFrame
        Исходный DataFrame с данными о перемещениях.
        Используется только если custom_matrix не указан.
    start_point : str, optional
        Название колонки с точкой отправления. По умолчанию "start_location".
    end_point : str, optional
        Название колонки с точкой прибытия. По умолчанию "end_location".
    custom_matrix : pd.DataFrame, optional
        Пользовательская матрица корреспонденций.
        Если указана, используется вместо создания матрицы из data.

    Returns:
    --------
    pd.DataFrame
        DataFrame с анализом точек, содержащий колонки:
        - point: идентификатор точки
        - outflow: общее количество отправлений из точки
        - inflow: общее количество прибытий в точку
        - net_flow: чистый поток (inflow - outflow)
        - flow_ratio: соотношение притока к оттоку (inflow / outflow)
        - point_type: тип точки (определяется функцией _classify_point)

    Examples:
    ---------
    >>> # Использование с исходными данными
    >>> point_analysis = analyze_od_flows(data)
    >>>
    >>> # Использование с пользовательской матрицей
    >>> custom_od = create_od_matrix(data, period='M')
    >>> point_analysis = analyze_od_flows(data, custom_matrix=custom_od)
    """

    od_matrix = create_od_matrix(data, start_point, end_point) if custom_matrix is None else custom_matrix

    outflow = od_matrix.groupby(start_point)["count"].sum().reset_index()
    outflow.columns = ["point", "outflow"]
    inflow = od_matrix.groupby(end_point)["count"].sum().reset_index()
    inflow.columns = ["point", "inflow"]

    point_summary = pd.merge(outflow, inflow, on="point", how="outer").fillna(0)

    point_summary["net_flow"] = point_summary["inflow"] - point_summary["outflow"]

    point_summary["flow_ratio"] = point_summary["inflow"] / point_summary["outflow"].replace(0, 1)

    point_summary["point_type"] = point_summary.apply(_classify_point, axis=1)

    return point_summary