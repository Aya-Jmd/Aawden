def lookup(conn, username):
    cursor = conn.cursor()
    cursor.execute(f"SELECT name FROM users WHERE name = '{username}'")
    return cursor.fetchall()
