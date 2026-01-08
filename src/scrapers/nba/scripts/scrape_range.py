#!/usr/bin/env python3
import argparse
import subprocess
from datetime import datetime

from src.utils.date import generate_dates

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    for date in generate_dates(start, end):
        diso = date.isoformat()
        print(f"Running for {diso}")
        subprocess.run([
            "python3",
            "-m",
            "src.scrapers.jobs.get_prev_nba_data",
            f"--date={diso}",
        ], check=True)
