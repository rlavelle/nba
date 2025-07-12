import configparser

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import CONFIG_PATH, LOCAL
from src.db.constants import SCHEMAS


def insert_table(table: pd.DataFrame, schema:str, name:str, db_url:str, drop=False):
    engine = create_engine(db_url)

    if drop:
        with engine.begin() as conn:
            statements = [statement.strip() for statement in schema.split(';') if statement.strip()]

            # Execute each statement separately
            for statement in statements:
                conn.execute(text(statement))

    table.to_sql(name, con=engine, index=False, if_exists='append')


def create_empty_tables(db_url:str):
    engine = create_engine(db_url)

    for schema in SCHEMAS:
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
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    db_url = config.get('DB_PATHS', 'local_url' if LOCAL else 'prod_url')
    print(db_url)

    create_empty_tables(db_url)
