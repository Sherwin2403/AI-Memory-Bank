import os

import psycopg
from dotenv import load_dotenv


load_dotenv()

connection = psycopg.connect(
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
)

print("PostgreSQL connection successful!")

connection.close()
