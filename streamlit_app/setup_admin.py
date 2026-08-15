"""
Create a new admin account in the Aiven Cloud Database.

Usage:
    python setup_admin.py
"""

import database as db
import getpass


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
# NEW ADMIN
# ============================================================

USERNAME = "vishu720"
PASSWORD = "vishweshankam"


# ============================================================
# CREATE / VERIFY ADMIN
# ============================================================

if __name__ == "__main__":

    print("\n=== Connecting to Aiven Cloud ===")

    try:

        db.init_db(DB_CONFIG)

        # Check whether username already exists
        conn = db.get_connection(DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, username FROM admins WHERE username = %s",
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

        # ----------------------------------------------------
        # CREATE NEW USER
        # ----------------------------------------------------

        else:

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
        # VERIFY
        # ----------------------------------------------------

        conn = db.get_connection(DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, username, created_at "
            "FROM admins WHERE username = %s",
            (USERNAME,)
        )

        user = cursor.fetchone()

        cursor.close()
        conn.close()

        print("\n=== Verification ===")

        if user:

            print("✅ Account exists in Aiven database.")
            print(f"ID: {user['id']}")
            print(f"Username: {user['username']}")
            print(f"Created: {user['created_at']}")

        else:

            print(
                "❌ Account was not found after creation."
            )

    except Exception as e:

        print("\n❌ Error:")
        print(e)
        print("\n=== Password Verification Test ===")

result = db.verify_admin(
    DB_CONFIG,
    "vishu720",
    "vishweshankam"
)

if result:
    print("✅ Username and password are CORRECT.")
else:
    print("❌ Username exists, but password is NOT correct.")