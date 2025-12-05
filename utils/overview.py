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
