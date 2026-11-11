def lookup(conn, username):
    query = "".join(["SELECT name FROM users ", "WHERE name = ?"])
    return conn.execute(query, (username,)).fetchall()
