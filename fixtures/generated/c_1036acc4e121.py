def lookup(conn, username):
    query = "SELECT name FROM users WHERE name = :value"
    return conn.execute(query, {"value": username}).fetchall()
