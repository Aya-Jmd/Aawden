def lookup(conn, username):
    query = "SELECT name FROM users WHERE name = ? ORDER BY name"
    return conn.execute(query, (username,)).fetchall()
