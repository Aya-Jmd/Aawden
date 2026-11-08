def lookup(conn, username):
    query = f"SELECT name FROM users WHERE name = '{username}' ORDER BY name"
    return conn.execute(query).fetchall()
