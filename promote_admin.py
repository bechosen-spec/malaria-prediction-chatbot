import sqlite3

# Connect to the database
conn = sqlite3.connect('malaria_chatbot.db')
c = conn.cursor()

# Promote the user to admin
username = 'mandy'  # Replace with the actual username
c.execute('UPDATE users SET is_admin = 1 WHERE username = ?', (username,))

# Commit the changes and close the connection
conn.commit()
conn.close()

print(f"User '{username}' has been promoted to Admin!")
