"""
Optimized Database Module
Credit Risk Prediction System

Features:
- MySQL / Aiven Cloud support
- SSL connection
- bcrypt password hashing
- Connection pooling
- Admin authentication
- Prediction storage
- Prediction history
"""

import pymysql
import bcrypt
import ssl
from datetime import datetime
from dbutils.pooled_db import PooledDB


# ============================================================
# CONNECTION POOL
# ============================================================

_pool = None


def get_pool(config):
    """
    Create and reuse a MySQL connection pool.

    This avoids creating a completely new database
    connection for every operation.
    """

    global _pool

    if _pool is None:

        ssl_ctx = ssl.create_default_context()

        # Aiven SSL connection
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

        _pool = PooledDB(
            creator=pymysql,
            mincached=1,
            maxcached=5,
            maxconnections=10,
            blocking=True,

            host=config["host"],
            port=int(config["port"]),
            user=config["user"],
            password=config["password"],
            database=config["database"],

            cursorclass=pymysql.cursors.DictCursor,

            connect_timeout=10,
            read_timeout=10,
            write_timeout=10,

            ssl=ssl_ctx,

            autocommit=False
        )

    return _pool


# ============================================================
# GET CONNECTION
# ============================================================

def get_connection(config):
    """
    Get a connection from the connection pool.
    """

    pool = get_pool(config)

    return pool.connection()


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_db(config):
    """
    Create required tables if they do not exist.
    """

    conn = get_connection(config)

    try:

        cursor = conn.cursor()

        # ----------------------------------------------------
        # PREDICTIONS TABLE
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (

                id INT AUTO_INCREMENT PRIMARY KEY,

                prediction_timestamp DATETIME NOT NULL,

                age INT,

                monthly_income FLOAT,

                debt_ratio FLOAT,

                revolving_utilization FLOAT,

                open_credit_lines INT,

                real_estate_loans INT,

                late_30_59 INT,

                late_60_89 INT,

                late_90_plus INT,

                dependents INT,

                risk_probability FLOAT NOT NULL,

                risk_prediction TINYINT NOT NULL,

                predicted_by VARCHAR(100),

                INDEX idx_prediction_timestamp
                (prediction_timestamp),

                INDEX idx_predicted_by
                (predicted_by)

            )
            """
        )

        # ----------------------------------------------------
        # ADMINS TABLE
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS admins (

                id INT AUTO_INCREMENT PRIMARY KEY,

                username VARCHAR(100)
                UNIQUE NOT NULL,

                password_hash VARCHAR(255)
                NOT NULL,

                created_at DATETIME NOT NULL

            )
            """
        )

        conn.commit()

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()


# ============================================================
# CREATE ADMIN
# ============================================================

def create_admin(config, username, password):
    """
    Create a new administrator account.

    Password is stored as a bcrypt hash.
    """

    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    conn = get_connection(config)

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO admins
            (
                username,
                password_hash,
                created_at
            )
            VALUES
            (
                %s,
                %s,
                %s
            )
            """,
            (
                username,
                password_hash,
                datetime.now()
            )
        )

        conn.commit()

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()


# ============================================================
# VERIFY ADMIN
# ============================================================

def verify_admin(config, username, password):
    """
    Verify administrator username and password.

    Returns:
        True  -> valid credentials
        False -> invalid credentials
    """

    conn = get_connection(config)

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT password_hash
            FROM admins
            WHERE username = %s
            LIMIT 1
            """,
            (username,)
        )

        row = cursor.fetchone()

    finally:

        conn.close()

    if row is None:

        return False

    try:

        return bcrypt.checkpw(
            password.encode("utf-8"),
            row["password_hash"].encode("utf-8")
        )

    except Exception:

        return False


# ============================================================
# INSERT PREDICTION
# ============================================================

def insert_prediction(
    config,
    input_dict,
    proba,
    prediction,
    predicted_by
):
    """
    Store one prediction result.
    """

    conn = get_connection(config)

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO predictions
            (
                prediction_timestamp,
                age,
                monthly_income,
                debt_ratio,
                revolving_utilization,
                open_credit_lines,
                real_estate_loans,
                late_30_59,
                late_60_89,
                late_90_plus,
                dependents,
                risk_probability,
                risk_prediction,
                predicted_by
            )
            VALUES
            (
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s
            )
            """,
            (
                datetime.now(),

                input_dict["age"],

                input_dict["MonthlyIncome"],

                input_dict["DebtRatio"],

                input_dict[
                    "RevolvingUtilizationOfUnsecuredLines"
                ],

                input_dict[
                    "NumberOfOpenCreditLinesAndLoans"
                ],

                input_dict[
                    "NumberRealEstateLoansOrLines"
                ],

                input_dict[
                    "NumberOfTime30-59DaysPastDueNotWorse"
                ],

                input_dict[
                    "NumberOfTime60-89DaysPastDueNotWorse"
                ],

                input_dict[
                    "NumberOfTimes90DaysLate"
                ],

                input_dict[
                    "NumberOfDependents"
                ],

                float(proba),

                int(prediction),

                predicted_by
            )
        )

        conn.commit()

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()


# ============================================================
# GET ALL PREDICTIONS
# ============================================================

def get_all_predictions(config):
    """
    Fetch prediction history.

    Most recent predictions are returned first.
    """

    conn = get_connection(config)

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *
            FROM predictions
            ORDER BY prediction_timestamp DESC
            """
        )

        rows = cursor.fetchall()

        return rows

    finally:

        conn.close()


# ============================================================
# GET RECENT PREDICTIONS
# ============================================================

def get_recent_predictions(config, limit=8):
    """
    Fetch only recent predictions.

    Faster than loading the complete prediction history.
    """

    conn = get_connection(config)

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *
            FROM predictions
            ORDER BY prediction_timestamp DESC
            LIMIT %s
            """,
            (int(limit),)
        )

        return cursor.fetchall()

    finally:

        conn.close()


# ============================================================
# GET PREDICTION SUMMARY
# ============================================================

def get_prediction_summary(config):
    """
    Get dashboard statistics directly from MySQL.

    This avoids downloading every prediction record
    just to calculate totals.
    """

    conn = get_connection(config)

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT

                COUNT(*) AS total_predictions,

                SUM(
                    CASE
                        WHEN risk_prediction = 1
                        THEN 1
                        ELSE 0
                    END
                ) AS high_risk,

                SUM(
                    CASE
                        WHEN risk_prediction = 0
                        THEN 1
                        ELSE 0
                    END
                ) AS low_risk,

                AVG(risk_probability)
                AS average_risk

            FROM predictions
            """
        )

        result = cursor.fetchone()

        return result

    finally:

        conn.close()


# ============================================================
# CLEAR PREDICTIONS
# ============================================================

def clear_predictions(config):
    """
    Delete all prediction history.
    """

    conn = get_connection(config)

    try:

        cursor = conn.cursor()

        cursor.execute(
            "TRUNCATE TABLE predictions"
        )

        conn.commit()

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()