import configparser
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import CONFIG_PATH, LOCAL
from src.db.constants import ODDS_RESULTS


def get_engine():
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    db_url = config.get('DB_PATHS', 'local_url' if LOCAL else 'prod_url')
    return create_engine(db_url)


def insert_table(table: pd.DataFrame, schema: str, name: str, drop=False):
    engine = get_engine()

    if drop:
        with engine.begin() as conn:
            statements = [statement.strip() for statement in schema.split(';') if statement.strip()]

            for statement in statements:
                conn.execute(text(statement))

    table.to_sql(name, con=engine, index=False, if_exists='append')


def insert_error(error: dict[str, Any]):
    try:
        engine = get_engine()

        result = pd.read_sql_query('SELECT MAX(task_id) AS max_id FROM errors;', engine)
        max_task_id = result.iloc[0]['max_id']
        if pd.isna(max_task_id):
            max_task_id = 0

        error['task_id'] = max_task_id + 1
        error['time'] = datetime.now()
        pd.DataFrame([error]).to_sql('errors', con=engine, index=False, if_exists='append')
        return True

    except Exception as e:
        print(e)
        return False


def create_empty_tables(schemas):
    engine = get_engine()

    for schema in schemas:
        with engine.begin() as conn:
            statements = [statement.strip() for statement in schema.split(';') if statement.strip()]

            # Execute each statement separately
            for statement in statements:
                try:
                    print(statement)
                    conn.execute(text(statement))
                except Exception as e:
                    print(f'Error in execution: {e}')


if __name__ == "__main__":
    create_empty_tables(ODDS_RESULTS)
    # insert_error({'msg': 'test'})
