import tarfile
import sqlite3
import pandas as pd

import sqlite3

# # Connect to the SQLite database (this will create a new file if it doesn't exist)
# conn = sqlite3.connect('./datasets/cepdb_2013-06-21.db')

# # Read the SQL file
# with open('./datasets/cepdb_2013-06-21.db', 'r') as f:
#     sql_script = f.read()

# # Execute the SQL script
# conn.executescript(sql_script)

# # Don't forget to close the connection
# conn.close()

###

# # The extracted .sql file should be in the current directory
# # Connect to the SQLite database
conn = sqlite3.connect('./datasets/cepdb_2013-06-21.db')

# # Execute a SQL command to get all table names
# table_names = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
# print("Table names:", table_names)


df = pd.read_sql('./datasets/cepdb_2013-06-21.db', conn)

print(df.head())

# # Create a cursor object
# cur = conn.cursor()

# # Execute a SQL command
# cur.execute("SELECT * FROM data_calcqcset1")

# # Fetch all the rows
# rows = cur.fetchall()

# # Print the rows
# for i, row in enumerate(rows):
#     print(row)
#     if i > 10:
#         break

# # Don't forget to close the connection
# conn.close()
