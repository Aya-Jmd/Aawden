def lookup(conn, username):
    query = "WITH matches AS (SELECT name FROM users WHERE name = ?) SELECT name FROM matches"
    return conn.execute(query, (username,)).fetchall()
