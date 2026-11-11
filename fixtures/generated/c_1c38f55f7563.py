def lookup(conn, username):
    query = "SELECT name FROM users WHERE name = :name"
    return conn.execute(query, {"name": username}).fetchall()
