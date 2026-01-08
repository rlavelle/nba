import os


def get_dirs(dir: str) -> list[str]:
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]


def get_files(dir: str) -> list[str]:
    return [name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]
