def lookup(conn, username):
    query = "SELECT name FROM users WHERE name = '{}'".format(username)
    return conn.execute(query).fetchall()
