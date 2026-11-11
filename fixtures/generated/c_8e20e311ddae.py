def lookup(conn, username):
    name = username.strip()
    return conn.execute("SELECT name FROM users WHERE name = ?", (name,)).fetchall()
