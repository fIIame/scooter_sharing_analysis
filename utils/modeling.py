import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple, List


def build_hourly_dataset(data: pd.DataFrame, timestamp_col: str = "hour_timestamp") -> pd.DataFrame:
    """
    Формирует почасовой датасет спроса на основе данных о поездках.

    Агрегирует данные до почасового уровня:
    • считает спрос (количество поездок)
    • агрегирует календарные, погодные, географические и маркетинговые факторы

    Параметры:
    ----------
    data : pd.DataFrame
        Датасет.
    timestamp_col : str, default="hour_timestamp"
        Название колонки с временной меткой часа.

    Возвращает:
    ----------
    pd.DataFrame
        Почасовой датасет, где одна строка соответствует одному часу
        и содержит спрос и факторы.
    """

    # Почасовой спрос
    demand = (
        data
        .groupby(timestamp_col)
        .agg(demand=("id", "count"))
        .reset_index()
    )

    # Агрегация факторов до почасового уровня
    factors = (
        data
        .groupby(timestamp_col)
        .agg(
            day_of_week=("day_of_week", "max"),
            temperature=("temperature", "mean"),
            mean_precipitation_total=("precipitation_total", "mean"),
            mean_cloud_cover_total=("cloud_cover_total", "mean"),
            promo=("promo", "max"),
        )
        .reset_index()
    )

    data["hour_of_day"] = data[timestamp_col].dt.hour
    data = demand.merge(factors, on=timestamp_col, how="left")

    return data


def add_lag_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет лаговые и средние лаговые признаки спроса.

    Создаёт:
    • спрос за предыдущий час (lag_1h)
    • спрос за предыдущие сутки (lag_24h)
    • средний спрос за последние 24 часа
    • средний спрос за последние 7 дней

    Параметры:
    ----------
    data : pd.DataFrame
        Почасовой датасет со столбцом спроса 'demand'.

    Возвращает:
    ----------
    pd.DataFrame
        Датасет с лаговыми признаками без строк с пропусками.
    """

    data = data.copy()

    data["lag_1h"] = data["demand"].shift(1)
    data["lag_24h"] = data["demand"].shift(24)

    data["mean_last_24h"] = data["lag_1h"].rolling(24).mean()
    data["mean_last_7d"] = data["lag_1h"].rolling(24 * 7).mean()

    return data.dropna()


def train_test_time_split(
    data: pd.DataFrame,
    target_col: str = "demand",
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Делит данные на тренировочную и тестовую выборки по времени.

    Разделение выполняется без перемешивания, чтобы
    предсказывать будущее на основе прошлого.

    Параметры:
    ----------
    data : pd.DataFrame
        Почасовой датасет с признаками и целевой переменной.
    target_col : str, default="demand"
        Название целевой переменной.
    test_size : float, default=0.2
        Доля данных, используемая для тестовой выборки.

    Возвращает:
    ----------
    X_train : pd.DataFrame
        Признаки тренировочной выборки.
    X_test : pd.DataFrame
        Признаки тестовой выборки.
    y_train : pd.Series
        Целевая переменная тренировочной выборки.
    y_test : pd.Series
        Целевая переменная тестовой выборки.
    """

    X = data.drop(columns=[target_col, "hour_timestamp"])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    return X_train, X_test, y_train, y_test


def apply_ohe(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Выполняет One-Hot Encoding категориальных признаков без утечек данных.

    Кодировщик обучается только на тренировочных данных.
    Неизвестные категории в тесте игнорируются.

    Параметры:
    ----------
    X_train : pd.DataFrame
        Признаки тренировочной выборки.
    X_test : pd.DataFrame
        Признаки тестовой выборки.
    categorical_cols : List[str]
        Список категориальных признаков для OHE-кодирования.

    Возвращает:
    ----------
    X_train_final : pd.DataFrame
        Тренировочная выборка с закодированными признаками.
    X_test_final : pd.DataFrame
        Тестовая выборка с закодированными признаками.
    """

    encoder = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore",
    )

    encoder.fit(X_train[categorical_cols])

    train_ohe = pd.DataFrame(
        encoder.transform(X_train[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=X_train.index,
    )

    test_ohe = pd.DataFrame(
        encoder.transform(X_test[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=X_test.index,
    )

    X_train_final = pd.concat(
        [X_train.drop(columns=categorical_cols), train_ohe], axis=1
    )

    X_test_final = pd.concat(
        [X_test.drop(columns=categorical_cols), test_ohe], axis=1
    )

    return X_train_final, X_test_final
