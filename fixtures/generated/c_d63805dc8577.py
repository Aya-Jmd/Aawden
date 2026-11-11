def lookup(conn, username):
    def execute(query, value):
        return conn.execute(query, (value,)).fetchall()
    return execute("SELECT name FROM users WHERE name = ?", username)
