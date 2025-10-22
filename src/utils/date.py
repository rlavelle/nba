from datetime import datetime, timedelta

def fmt_iso_dint(date:str):
    return int(date[:10].replace('-', ''))

def generate_dates(start: datetime, end: datetime = None) -> list[datetime.date]:
    start_date = start
    end_date = end if end else datetime.today()

    dates = []

    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)

    return dates


def date_to_dint(date:datetime.date) -> int:
    return int(date.strftime('%Y%m%d'))


def date_to_lookup(date:datetime.date, date_format="%m/%d/%Y") -> str:
    return date.strftime(date_format)


def time_to_minutes(time_string:str) -> float:
    if '-' in time_string:
        return 0

    x = list(map(int, time_string.split(':')))
    if len(x) == 2:
        total_seconds = x[0] * 60 + x[1]
    else:
        total_seconds = abs(x[0]) * 60

    return total_seconds / 60
