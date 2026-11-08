def lookup(conn, username):
    query = "SELECT name FROM users WHERE name = '%s'" % username
    return conn.execute(query).fetchall()
