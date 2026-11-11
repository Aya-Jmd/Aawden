def lookup(conn, username):
    query = "SELECT name FROM users WHERE name = ?"
    return conn.execute(query, (username,)).fetchall()
