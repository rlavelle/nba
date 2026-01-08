#!/usr/bin/env python3

import configparser
import os

from src.config import CONFIG_PATH
from src.scrapers.nba.utils.validation import is_date_data_complete
from src.utils.file_io import get_dirs

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    data_path = config.get('DATA_PATH', 'games_folder')

    folders = get_dirs(data_path)
    n = len(folders)
    c = 0

    for folder in folders:
        path = os.path.join(data_path, folder)
        c += int(is_date_data_complete(path, int(folder)))

    print(c / n)
