import pandas as pd


def get_missed_info(data: pd.DataFrame, rows: int, cols: int) -> None:
    """
    Выводит информацию о пропущенных значениях в датафрейме.

    Параметры:
    ----------
        data (pd.DataFrame): Анализируемый датафрейм.
        rows (int): Количество строк в датафрейме.
        cols (int): Количество столбцов в датафрейме.

    Возвращает:
    ----------
        - Общее количество пропусков.
        - Доля пропущенных ячеек в датафрейме.
        - Доля строк, содержащих хотя бы один пропуск.
    """
    missed_count = data.isnull().sum().sum()
    missed_cells = missed_count / (rows * cols)
    missed_rows = (data.isnull().sum(axis=1) > 0).sum() / rows

    print("\033[1m" + "\nПроверка пропусков" + "\033[0m")
    print(f"Кол-во пропусков: {missed_count}")
    print(f"Доля пропусков: {missed_cells:.1%}")
    print(f"Доля строк с пропусками: {missed_rows:.1%}")


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
