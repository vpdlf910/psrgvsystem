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

DB_HOST_SMALL = os.getenv('DB_HOST')
DB_USER_SMALL = os.getenv('DB_USER')
DB_PASSWORD_SMALL = os.getenv('DB_PASSWORD')
DATABASE_SMALL = os.getenv('DATABASE')

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
            SELECT chamber_id, chamber_type, injection_condition, injection_time
            FROM injection_data
            WHERE injection_time BETWEEN %s AND %s
            """
            
            cursor.execute(sql, (start_time, end_time))
            results = cursor.fetchall()
            for row in results:
                chamber_id = row['chamber_id']
                chamber_type = row['chamber_type']
                injection_condition = row['injection_condition']
                time = row['injection_time']
                
                if chamber_id not in injection_data:
                    injection_data[chamber_id] = {}
                if chamber_type not in injection_data[chamber_id]:
                    injection_data[chamber_id][chamber_type] = {}
                if injection_condition not in injection_data[chamber_id][chamber_type]:
                    injection_data[chamber_id][chamber_type][injection_condition] = []
                injection_data[chamber_id][chamber_type][injection_condition].append(time)
            
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
    # connection = get_db_connection()
    # try:
    #     with connection.cursor() as cursor:
    #         sql = "DELETE FROM injection_data WHERE idx = %s"
    #         cursor.execute(sql, (idx,))
    #     connection.commit()
    # finally:
    #     connection.close()
    pass

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

def query_polymer_solvent(date):
    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            sql = """
            SELECT sensor_name,applied_polymer,solvent
            FROM manufacturing_process
            WHERE date_of_manufacture = %s
            """
            
            cursor.execute(sql,date)
            results = cursor.fetchall()
            
            
    except pymysql.MySQLError as e:
        print(f"Error: {e}")
    finally:
        connection.close()
    
    return results
def get_db_connection_small():
    try:
        connection = pymysql.connect(
            host='192.168.0.43',
            user='pc_program',
            password='tsei1234',
            database='sensor_evaluation_system_4chamber',
            port=3306,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        print("Database connection successful")
        return connection
    except pymysql.MySQLError as e:
        print(f"Error connecting to database: {e}")
        return None
def query_sensor_data_small(start_time: datetime, end_time: datetime, chamber_sensor_data: dict):
    connection = get_db_connection_small()

    sensor_data = []
    chambers = list(chamber_sensor_data.keys())
    sensors = [sensor_id for sensor_ids in chamber_sensor_data.values() for sensor_id in sensor_ids]

    chamber_placeholders = ','.join(['%s'] * len(chambers))
    sensor_placeholders = ','.join(['%s'] * len(sensors))
    
    try:
        with connection.cursor() as cursor:
            sql = f"""
            SELECT chamber_id, sensor_id, rs, volt, reg_date
            FROM sensor_data 
            WHERE reg_date BETWEEN %s AND %s 
            AND chamber_id IN ({chamber_placeholders})
            AND sensor_id IN ({sensor_placeholders})
            """

            cursor.execute(sql, [start_time, end_time] + chambers + sensors)
            sensor_data = cursor.fetchall()

    except pymysql.MySQLError as e:
        print(f"Error: {e}")
    finally:
        connection.close()

    return sensor_data

def query_real_time_sensor_data_small(start_time: datetime, end_time: datetime, chamber_sensor_data: dict):
    connection = get_db_connection_small()

    sensor_data = []
    chambers = list(chamber_sensor_data.keys())
    sensors = [sensor_id for sensor_ids in chamber_sensor_data.values() for sensor_id in sensor_ids]

    chamber_placeholders = ','.join(['%s'] * len(chambers))
    sensor_placeholders = ','.join(['%s'] * len(sensors))
    
    try:
        with connection.cursor() as cursor:
            sql = f"""
            SELECT chamber_id, sensor_id, rs, volt, reg_date
            FROM sensor_data 
            WHERE reg_date BETWEEN %s AND %s 
            AND chamber_id IN ({chamber_placeholders})
            AND sensor_id IN ({sensor_placeholders})
            """

            cursor.execute(sql, [start_time, end_time] + chambers + sensors)
            sensor_data = cursor.fetchall()

    except pymysql.MySQLError as e:
        print(f"Error: {e}")
    finally:
        connection.close()

    return sensor_data

def query_injection_conditions_small(start_time: datetime, end_time: datetime, sensor_ids: list):
    connection = get_db_connection_small()

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

def fetch_injection_data_small():
    connection = get_db_connection_small()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM injection_data ORDER BY injection_time"
            cursor.execute(sql)
            result = cursor.fetchall()
    finally:
        connection.close()
    return result

def update_injection_data_small(idx, injection_time, injection_condition, review):
    connection = get_db_connection_small()
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

def delete_injection_data_small(idx):
    # connection = get_db_connection()
    # try:
    #     with connection.cursor() as cursor:
    #         sql = "DELETE FROM injection_data WHERE idx = %s"
    #         cursor.execute(sql, (idx,))
    #     connection.commit()
    # finally:
    #     connection.close()
    pass

def insert_injection_data_small(injection_time, injection_condition, review):
    connection = get_db_connection_small()
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

def fetch_manufacturing_data_small(order_by_column, ascending):
    connection = get_db_connection_small()
    try:
        with connection.cursor() as cursor:
            order = "ASC" if ascending else "DESC"
            sql = f"SELECT * FROM manufacturing_process ORDER BY {order_by_column} {order}"
            cursor.execute(sql)
            result = cursor.fetchall()
    finally:
        connection.close()
    return result

def insert_manufacturing_data_small(time, condition, review):
    """
    Insert a new manufacturing data record into the database.
    :param time: The time of the manufacturing process.
    :param condition: The condition of the manufacturing process.
    :param review: Optional review of the manufacturing process.
    """
    pass  # Replace with actual implementation

def update_manufacturing_data_small(idx, time, condition, review):
    """
    Update an existing manufacturing data record in the database.
    :param idx: The index of the record to update.
    :param time: The new time of the manufacturing process.
    :param condition: The new condition of the manufacturing process.
    :param review: Optional new review of the manufacturing process.
    """
    pass  # Replace with actual implementation

def delete_manufacturing_data_small(idx):
    """
    Delete a manufacturing data record from the database.
    :param idx: The index of the record to delete.
    """
    pass  # Replace with actual implementation

def query_polymer_solvent_small(date):
    connection = get_db_connection_small()

    try:
        with connection.cursor() as cursor:
            sql = """
            SELECT sensor_name,applied_polymer,solvent
            FROM manufacturing_process
            WHERE date_of_manufacture = %s
            """
            
            cursor.execute(sql,date)
            results = cursor.fetchall()
            
            
    except pymysql.MySQLError as e:
        print(f"Error: {e}")
    finally:
        connection.close()
    
    return results

def fetch_chamber_data(start_time, end_time):
    print(f"fetch_chamber_data called with start_time: {start_time} and end_time: {end_time}")  # 디버그 메시지 추가
    connection = get_db_connection_small()
    if connection is None:
        print("Connection to database failed")
        return {}

    chamber_data = {}

    try:
        with connection.cursor() as cursor:
            query = """
            SELECT chamber_id,sensor_id
            FROM sensor_data
            WHERE reg_date BETWEEN %s AND %s
            """
            print(f"Executing query: {query} with start_time: {start_time} and end_time: {end_time}")  # 디버그 메시지 추가
            cursor.execute(query, (start_time, end_time))
            results = cursor.fetchall()
            unique_results = set((row['chamber_id'], row['sensor_id']) for row in results)

            for chamber_id, sensor_id in unique_results:
                if chamber_id not in chamber_data:
                    chamber_data[chamber_id] = []
                chamber_data[chamber_id].append(sensor_id)
            print(f"chamber_data: {chamber_data}")  # 디버그 메시지 추가


    except pymysql.MySQLError as e:
        print(f"Error executing query: {e}")
    finally:
        connection.close()
        print("Database connection closed")  # 디버그 메시지 추가
    
    return chamber_data