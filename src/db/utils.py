import pandas as pd
from sqlalchemy import create_engine, text


def insert_table(table: pd.DataFrame, schema:str, name:str, db:str, drop=False):
    engine = create_engine(f'sqlite:///{db}')

    if drop:
        with engine.connect() as connection:
            statements = [statement.strip() for statement in schema.split(';') if statement.strip()]

            # Execute each statement separately
            for statement in statements:
                connection.execute(text(statement))

    table.to_sql(name, con=engine, index=False, if_exists='append')