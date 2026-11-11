def lookup(conn, username):
    return conn.execute(f"SELECT name FROM users WHERE lower(name) = lower('{username}')").fetchall()
