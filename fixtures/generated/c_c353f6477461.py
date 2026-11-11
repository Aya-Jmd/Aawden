def lookup(conn, username):
    query = f"WITH matches AS (SELECT name FROM users WHERE name = '{username}') SELECT name FROM matches"
    return conn.execute(query).fetchall()
