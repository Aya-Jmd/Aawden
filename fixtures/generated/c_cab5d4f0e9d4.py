def lookup(conn, username):
    return conn.execute(f"SELECT name FROM users WHERE name = '{username}'").fetchall()
