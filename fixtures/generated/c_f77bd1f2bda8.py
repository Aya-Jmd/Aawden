def lookup(conn, username):
    query = "".join(["SELECT name FROM users WHERE name = '", username, "'"])
    return conn.execute(query).fetchall()
