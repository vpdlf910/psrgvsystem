import pymysql
from pymysql.cursors import DictCursor
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DATABASE = os.getenv('DATABASE')

DB_HOST_TEST = os.getenv('DB_HOST_TEST')
DB_USER_TEST = os.getenv('DB_USER_TEST')
DB_PASSWORD_TEST = os.getenv('DB_PASSWORD_TEST')
DATABASE_TEST = os.getenv('DATABASE_TEST')

def query_sensor_data(start_time: datetime, end_time: datetime, sensor_ids: list):
    connection = pymysql.connect(
        host= DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DATABASE,
        cursorclass=DictCursor
    )

    sensor_data = []
    placeholders = ','.join(['%s'] *len(sensor_ids))
    try:
        with connection.cursor() as cursor:
            sql = """
            SELECT sensor_id,rs, volt, reg_date
            FROM sensor_data 
            WHERE reg_date BETWEEN %s AND %s 
            AND sensor_id IN ({})
            """.format(placeholders)
        
            cursor.execute(sql, [start_time, end_time] + sensor_ids)
            sensor_data = cursor.fetchall()
                
    except pymysql.MySQLError as e:
        print(f"Error: {e}")
    finally:
        connection.close()

    return sensor_data

def query_real_time_sensor_data(start_time: datetime, end_time: datetime, sensor_ids: list):
    connection = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DATABASE,
        cursorclass=DictCursor
    )

    sensor_data = []
    placeholders = ','.join(['%s'] * len(sensor_ids))
    try:
        with connection.cursor() as cursor:
            sql = """
            SELECT sensor_id, volt, reg_date
            FROM sensor_data 
            WHERE reg_date BETWEEN %s AND %s 
            AND sensor_id IN ({})
            """.format(placeholders)
        
            cursor.execute(sql, [start_time, end_time] + sensor_ids)
            sensor_data = cursor.fetchall()
                
    except pymysql.MySQLError as e:
        print(f"Error: {e}")
    finally:
        connection.close()

    return sensor_data


def get_db_connection():
    connection = pymysql.connect(
        host=DB_HOST_TEST,
        user=DB_USER_TEST,
        password=DB_PASSWORD_TEST,
        database=DATABASE_TEST,
        cursorclass=DictCursor,
        ssl_disabled=True
    )
    return connection


def query_injection_conditions(start_time: datetime, end_time: datetime, sensor_ids: list):
    connection = get_db_connection()

    injection_data = {}
    
    try:
        with connection.cursor() as cursor:
            sql = """
            SELECT injection_condition, injection_time
            FROM injection_data
            WHERE injection_time BETWEEN %s AND %s 
            """
            
            cursor.execute(sql, (start_time, end_time))
            results = cursor.fetchall()
            for row in results:
                condition = row['injection_condition']
                time = row['injection_time']
                if condition not in injection_data:
                    injection_data[condition] = []
                injection_data[condition].append(time)
            
    except pymysql.MySQLError as e:
        print(f"Error: {e}")
    finally:
        connection.close()
    
    return injection_data

def fetch_injection_data():
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM injection_data ORDER BY injection_time"
            cursor.execute(sql)
            result = cursor.fetchall()
    finally:
        connection.close()
    return result

def update_injection_data(idx, injection_time, injection_condition, review):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = """
                UPDATE injection_data 
                SET injection_time = %s, injection_condition = %s, review = %s
                WHERE idx = %s
            """
            cursor.execute(sql, (injection_time, injection_condition, review, idx))
        connection.commit()
    finally:
        connection.close()

def delete_injection_data(idx):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "DELETE FROM injection_data WHERE idx = %s"
            cursor.execute(sql, (idx,))
        connection.commit()
    finally:
        connection.close()

def insert_injection_data(injection_time, injection_condition, review):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = """
                INSERT INTO injection_data (injection_time, injection_condition, review)
                VALUES (%s, %s, %s)
            """
            cursor.execute(sql, (injection_time, injection_condition, review))
        connection.commit()
    finally:
        connection.close()

def fetch_manufacturing_data(order_by_column, ascending):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            order = "ASC" if ascending else "DESC"
            sql = f"SELECT * FROM manufacturing_process ORDER BY {order_by_column} {order}"
            cursor.execute(sql)
            result = cursor.fetchall()
    finally:
        connection.close()
    return result

def insert_manufacturing_data(time, condition, review):
    """
    Insert a new manufacturing data record into the database.
    :param time: The time of the manufacturing process.
    :param condition: The condition of the manufacturing process.
    :param review: Optional review of the manufacturing process.
    """
    pass  # Replace with actual implementation

def update_manufacturing_data(idx, time, condition, review):
    """
    Update an existing manufacturing data record in the database.
    :param idx: The index of the record to update.
    :param time: The new time of the manufacturing process.
    :param condition: The new condition of the manufacturing process.
    :param review: Optional new review of the manufacturing process.
    """
    pass  # Replace with actual implementation

def delete_manufacturing_data(idx):
    """
    Delete a manufacturing data record from the database.
    :param idx: The index of the record to delete.
    """
    pass  # Replace with actual implementation