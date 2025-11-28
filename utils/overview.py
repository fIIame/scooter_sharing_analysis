import pandas as pd


def get_missed_info(data: pd.DataFrame, rows: int, cols: int) -> None:
    missed_count = data.isnull().sum().sum()
    missed_cells = missed_count / (rows * cols)
    missed_rows = (data.isnull().sum(axis=1) > 0).sum() / rows

    print("\033[1m" + "\nПроверка пропусков" + "\033[0m")
    print(f"Кол-во пропусков: {missed_count}")
    print(f"Доля пропусков: {missed_cells:.1%}")
    print(f"Доля строк с пропусками: {missed_rows:.1%}")


def get_duplicated_info(data: pd.DataFrame) -> None:
    duplicates_count = data.duplicated().sum()

    print("\033[1m" + "\nПроверка на дубликаты" + "\033[0m")
    print(f"Кол-ва дубликатов: {duplicates_count}")