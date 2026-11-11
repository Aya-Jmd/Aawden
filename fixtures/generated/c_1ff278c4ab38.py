def lookup(conn, username):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM users WHERE name = ?", (username,))
    return cursor.fetchall()
