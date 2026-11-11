def lookup(conn, username):
    quoted = username.replace("'", "''")
    return conn.execute(f"SELECT name FROM users WHERE name = '{quoted}'").fetchall()
