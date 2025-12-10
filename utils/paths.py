from pathlib import Path


def path_join(*parts, _base_path: Path = Path(__file__).resolve().parent.parent) -> Path:
    for part in parts:
        _base_path = _base_path / part
    return _base_path


pj = path_join

HOME = pj()