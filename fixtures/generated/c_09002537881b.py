def lookup(conn, username):
    name = username.strip()
    return conn.execute(f"SELECT name FROM users WHERE name = '{name}'").fetchall()
