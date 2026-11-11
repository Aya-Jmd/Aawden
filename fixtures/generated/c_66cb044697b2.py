def lookup(conn, username):
    return conn.execute("SELECT name FROM users WHERE name = ?", (username,)).fetchall()
