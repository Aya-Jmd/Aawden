def lookup(conn, username):
    query = "SELECT name FROM users WHERE name = '" + username + "'"
    return conn.execute(query).fetchall()
