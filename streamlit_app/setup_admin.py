"""
Create a new admin account in the Aiven Cloud Database.

Usage:
    python setup_admin.py
"""

import getpass
import database as db


# ============================================================
# AIVEN DATABASE CONFIGURATION
# ============================================================

aiven_password = getpass.getpass(
    "Paste your Aiven Cloud Password (hidden as you type): "
).strip()

DB_CONFIG = {
    "host": "mysql-19f4630d-vishweshankam38-080c.g.aivencloud.com",
    "port": 24194,
    "user": "avnadmin",
    "password": aiven_password,
    "database": "defaultdb",
}


# ============================================================
# GET NEW ADMIN DETAILS
# ============================================================

USERNAME = input(
    "\nEnter new admin username: "
).strip()

PASSWORD = getpass.getpass(
    "Enter new admin password (hidden as you type): "
).strip()

CONFIRM_PASSWORD = getpass.getpass(
    "Confirm new admin password: "
).strip()


# ============================================================
# BASIC VALIDATION
# ============================================================

if not USERNAME:
    print("\n❌ Username cannot be empty.")
    raise SystemExit(1)

if not PASSWORD:
    print("\n❌ Password cannot be empty.")
    raise SystemExit(1)

if PASSWORD != CONFIRM_PASSWORD:
    print("\n❌ Passwords do not match.")
    raise SystemExit(1)

if len(PASSWORD) < 6:
    print("\n❌ Password must contain at least 6 characters.")
    raise SystemExit(1)


# ============================================================
# CREATE / VERIFY ADMIN
# ============================================================

if __name__ == "__main__":

    print("\n=== Connecting to Aiven Cloud ===")

    try:

        # ----------------------------------------------------
        # INITIALIZE DATABASE
        # ----------------------------------------------------

        db.init_db(DB_CONFIG)

        # ----------------------------------------------------
        # CHECK WHETHER USERNAME ALREADY EXISTS
        # ----------------------------------------------------

        conn = db.get_connection(DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, username
            FROM admins
            WHERE username = %s
            """,
            (USERNAME,)
        )

        existing_user = cursor.fetchone()

        cursor.close()
        conn.close()

        # ----------------------------------------------------
        # USER ALREADY EXISTS
        # ----------------------------------------------------

        if existing_user:

            print(
                f"\n⚠️ Admin '{USERNAME}' already exists."
            )

            print(
                "No new account was created."
            )

            raise SystemExit(0)

        # ----------------------------------------------------
        # CREATE NEW ADMIN
        # ----------------------------------------------------

        print(
            f"\nCreating admin account '{USERNAME}'..."
        )

        db.create_admin(
            DB_CONFIG,
            USERNAME,
            PASSWORD
        )

        print(
            f"\n✅ Admin account '{USERNAME}' "
            "created successfully!"
        )

        # ----------------------------------------------------
        # VERIFY ACCOUNT EXISTS
        # ----------------------------------------------------

        conn = db.get_connection(DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, username, created_at
            FROM admins
            WHERE username = %s
            """,
            (USERNAME,)
        )

        user = cursor.fetchone()

        cursor.close()
        conn.close()

        print("\n=== Account Verification ===")

        if user:

            print("✅ Account exists in Aiven database.")
            print(f"ID: {user['id']}")
            print(f"Username: {user['username']}")
            print(f"Created: {user['created_at']}")

        else:

            print(
                "❌ Account was not found after creation."
            )

            raise SystemExit(1)

        # ----------------------------------------------------
        # VERIFY USERNAME + PASSWORD
        # ----------------------------------------------------

        print("\n=== Password Verification Test ===")

        result = db.verify_admin(
            DB_CONFIG,
            USERNAME,
            PASSWORD
        )

        if result:

            print(
                "✅ Username and password are CORRECT."
            )

            print(
                f"✅ Admin '{USERNAME}' can now log in."
            )

        else:

            print(
                "❌ Username exists, but password "
                "verification failed."
            )

    except Exception as e:

        print("\n❌ Error:")
        print(e)