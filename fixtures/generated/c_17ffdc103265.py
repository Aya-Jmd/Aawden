def lookup(conn, username):
    def execute(query):
        return conn.execute(query).fetchall()
    return execute(f"SELECT name FROM users WHERE name = '{username}'")
