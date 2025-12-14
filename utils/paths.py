from pathlib import Path


def join_path(
        *parts,
        _base_path: Path = Path(__file__).resolve().parent.parent
) -> Path:
    """
    Создает объект Path путем последовательного объединения частей пути.

    Функция принимает произвольное количество аргументов-частей пути и последовательно
    присоединяет их к базовому пути, используя оператор деления Path.

    Параметры
    ----------
    *parts : str или Path
        Произвольное количество частей пути для объединения.
        Каждая часть может быть строкой или объектом Path.

    _base_path : Path, optional, по умолчанию Path(__file__).resolve().parent.parent
        Базовый путь, к которому присоединяются части.
        По умолчанию используется директория, находящаяся на два уровня выше
        файла, содержащего этот код.

    Возвращает
    -------
    Path
        Объект Path, представляющий объединенный путь.

    Примеры
    --------
    >>> path_join('data', 'files', 'document.txt')
    PosixPath('/текущая/директория/../data/files/document.txt')

    >>> path_join('config', 'settings.ini')
    PosixPath('/текущая/директория/../config/settings.ini')

    >>> path_join('subdir', _base_path=Path('/custom/path'))
    PosixPath('/custom/path/subdir')
    """
    for part in parts:
        _base_path = _base_path / part
    return _base_path
