import streamlit as st
import sqlite3
import hashlib
import pandas as pd
import joblib
import numpy as np
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import os

# Database Setup
conn = sqlite3.connect('malaria_chatbot.db', check_same_thread=False)
c = conn.cursor()

# Create tables
c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT UNIQUE,
              password TEXT,
              is_admin BOOLEAN DEFAULT 0)''')

c.execute('''CREATE TABLE IF NOT EXISTS chat_history
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER,
              symptoms TEXT,
              diagnosis TEXT,
              timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

c.execute('''CREATE TABLE IF NOT EXISTS knowledge_base
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              symptom TEXT,
              description TEXT)''')

conn.commit()

# Helper Functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, is_admin=False):
    try:
        c.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)',
                 (username, hash_password(password), is_admin))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(username, password):
    c.execute('SELECT password, is_admin FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    if result and result[0] == hash_password(password):
        return result[1]  # Return is_admin status
    return None

# Load SVM Model using joblib
try:
    model_path = os.path.join(os.path.dirname(__file__), "malaria_svm_model.pkl")
    svm_model = joblib.load(model_path)
    if not hasattr(svm_model, "predict"):
        raise ValueError("The loaded model is not a valid sklearn classifier.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    svm_model = None

# Admin Panel
def admin_panel():
    st.subheader("🛠️ Admin Dashboard")
    tabs = option_menu(
        None, ["Knowledge Base", "User Management", "System Monitoring"],
        icons=['database', 'people', 'graph-up'], 
        menu_icon="cast", default_index=0, orientation="horizontal"
    )
    
    if tabs == "Knowledge Base":
        st.write("📝 Manage Malaria Knowledge Base")
        data = pd.read_sql('SELECT * FROM knowledge_base', conn)
        st.dataframe(data)
        
        with st.form("Add/Edit Symptom"):
            symptom = st.text_input("Symptom")
            description = st.text_area("Description")
            submit = st.form_submit_button("Submit")
            if submit:
                c.execute('INSERT OR REPLACE INTO knowledge_base (id, symptom, description) VALUES (?, ?, ?)',
                         (None, symptom, description))
                conn.commit()
                st.success("Symptom added/updated successfully!")
                st.experimental_rerun()
                
    elif tabs == "User Management":
        st.write("👥 Manage Users")
        users = pd.read_sql('SELECT id, username, is_admin FROM users', conn)
        st.dataframe(users)
        
    elif tabs == "System Monitoring":
        st.write("📊 System Statistics")
        total_users = pd.read_sql('SELECT COUNT(*) FROM users', conn).iloc[0, 0]
        total_diagnoses = pd.read_sql('SELECT COUNT(*) FROM chat_history', conn).iloc[0, 0]
        st.metric("Total Users", total_users)
        st.metric("Total Diagnoses", total_diagnoses)
        
        # Visualization: Diagnoses Distribution
        diagnoses_data = pd.read_sql('SELECT diagnosis, COUNT(*) AS count FROM chat_history GROUP BY diagnosis', conn)
        plt.figure(figsize=(8, 5))
        plt.bar(diagnoses_data["diagnosis"], diagnoses_data["count"], color=['blue', 'green'])
        plt.title("Diagnoses Distribution")
        plt.xlabel("Diagnosis")
        plt.ylabel("Count")
        st.pyplot(plt)

# Chatbot Function with Chat-Like Interaction
def malaria_chatbot():
    st.subheader("💬 Malaria Diagnosis Chatbot")

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Store chat history
    if "chatbot_step" not in st.session_state:
        st.session_state.chatbot_step = 0  # Track the question progress
    if "responses" not in st.session_state:
        st.session_state.responses = {}  # Store user responses
    if "conversation_finished" not in st.session_state:
        st.session_state.conversation_finished = False  # Track if the conversation has ended

    # Define the chatbot questions
    questions = [
        ("How old are you?", None),
        ("What is your gender?", ["Male", "Female"]),
        ("Do you have a fever?", ["No", "Yes"]),
        ("Are you experiencing headaches?", ["No", "Yes"]),
        ("Have you had chills or shivering?", ["No", "Yes"]),
        ("Are you sweating excessively?", ["No", "Yes"]),
        ("Do you feel nauseous?", ["No", "Yes"]),
        ("Have you vomited recently?", ["No", "Yes"]),
        ("Do you have muscle pain?", ["No", "Yes"]),
        ("Are you feeling unusually tired?", ["No", "Yes"]),
        ("Do you have a cough?", ["No", "Yes"]),
        ("Are you experiencing diarrhea?", ["No", "Yes"]),
    ]

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Bot:** {message['content']}")

    # Handle conversation if not finished
    if not st.session_state.conversation_finished:
        # Progress through the questions
        if st.session_state.chatbot_step < len(questions):
            question, options = questions[st.session_state.chatbot_step]

            with st.form("chat_form", clear_on_submit=True):
                if options:
                    user_input = st.radio(f"🤖 {question}", options)
                else:
                    user_input = st.text_input(f"🤖 {question}")

                submitted = st.form_submit_button("Send")

                if submitted:
                    # Add user response to messages
                    st.session_state.messages.append({"role": "user", "content": user_input})

                    # Process the response
                    if options:
                        st.session_state.responses[question] = 0 if user_input == options[0] else 1
                    else:
                        try:
                            st.session_state.responses[question] = int(user_input)
                        except ValueError:
                            st.error("Please provide a valid numeric response.")
                            return

                    # Acknowledge response
                    st.session_state.messages.append({"role": "assistant", "content": f"Thank you for your response: {user_input}"})

                    # Move to the next question
                    st.session_state.chatbot_step += 1
        else:
            # Collect inputs for diagnosis
            input_data = np.array([
                st.session_state.responses["How old are you?"],
                st.session_state.responses["What is your gender?"],
                st.session_state.responses["Do you have a fever?"],
                st.session_state.responses["Are you experiencing headaches?"],
                st.session_state.responses["Have you had chills or shivering?"],
                st.session_state.responses["Are you sweating excessively?"],
                st.session_state.responses["Do you feel nauseous?"],
                st.session_state.responses["Have you vomited recently?"],
                st.session_state.responses["Do you have muscle pain?"],
                st.session_state.responses["Are you feeling unusually tired?"],
                st.session_state.responses["Do you have a cough?"],
                st.session_state.responses["Are you experiencing diarrhea?"],
            ]).reshape(1, -1)

            # Display Diagnose button
            if st.button("Diagnose"):
                if svm_model:
                    try:
                        diagnosis = svm_model.predict(input_data)[0]
                        result = "Positive for Malaria" if diagnosis == 1 else "Negative for Malaria"

                        # Add the prediction to chat history
                        st.session_state.messages.append(
                            {"role": "assistant", "content": f"🤖 Based on your symptoms, the diagnosis is: **{result}**"}
                        )

                        # Save the chat and prediction to the database
                        try:
                            user_id = c.execute(
                                'SELECT id FROM users WHERE username = ?',
                                (st.session_state.current_user,)
                            ).fetchone()[0]

                            c.execute(
                                'INSERT INTO chat_history (user_id, symptoms, diagnosis) VALUES (?, ?, ?)',
                                (user_id, str(st.session_state.responses), result)
                            )
                            conn.commit()
                        except Exception as db_error:
                            st.error(f"Error saving chat history: {db_error}")

                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                else:
                    st.error("SVM model not loaded. Unable to make predictions.")

                # Mark conversation as finished
                st.session_state.conversation_finished = True

    # Handle conversation finished state
    if st.session_state.conversation_finished:
        st.markdown("### 🛑 The conversation has ended. Restart to diagnose again.")
        if st.button("Restart Chatbot"):
            for key in ["messages", "chatbot_step", "responses", "conversation_finished"]:
                st.session_state.pop(key, None)
            st.experimental_rerun()

# Main App
def main():
    st.title("🩸 Malaria Diagnostic Assistant")

    # Authentication Check
    if not st.session_state.get("authenticated", False):
        auth_type = option_menu(
            None, ["Login", "Register"],
            icons=["box-arrow-in-right", "person-plus"],
            menu_icon="cast", default_index=0, orientation="horizontal"
        )

        if auth_type == "Login":
            with st.form("Login Form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    auth_result = verify_user(username, password)
                    if auth_result is not None:
                        st.session_state.authenticated = True
                        st.session_state.current_user = username
                        st.session_state.is_admin = auth_result
                        st.success("Login successful!")
                        st.experimental_rerun()
                    else:
                        st.error("Invalid username or password. Please try again.")

        elif auth_type == "Register":
            with st.form("Register Form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Register"):
                    if register_user(username, password):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Username already exists. Please choose another.")

    # Main Menu After Authentication
    else:
        menu_items = ["💬 Chatbot", "👤 Profile"]
        if st.session_state.is_admin:
            menu_items.append("🛠️ Admin")
        menu_items.append("🚪 Logout")

        selected = option_menu(
            None, menu_items,
            icons=["chat-dots", "person-gear", "tools", "box-arrow-right"],
            menu_icon="cast", default_index=0, orientation="horizontal"
        )

        if selected == "💬 Chatbot":
            malaria_chatbot()

        elif selected == "👤 Profile":
            st.subheader("👤 User Profile")
            st.write(f"**Username:** {st.session_state.current_user}")
            st.write("### Chat History")
            try:
                user_id = c.execute(
                    "SELECT id FROM users WHERE username = ?", (st.session_state.current_user,)
                ).fetchone()[0]
                history = pd.read_sql(
                    "SELECT symptoms, diagnosis, timestamp FROM chat_history WHERE user_id = ?",
                    conn,
                    params=(user_id,)
                )
                if history.empty:
                    st.info("No chat history available.")
                else:
                    st.dataframe(history)
            except Exception as e:
                st.error(f"Error fetching chat history: {e}")

        elif selected == "🛠️ Admin" and st.session_state.is_admin:
            admin_panel()

        elif selected == "🚪 Logout":
            for key in ["authenticated", "current_user", "is_admin"]:
                st.session_state.pop(key, None)
            st.success("You have been logged out.")
            st.experimental_rerun()

if __name__ == "__main__":
    main()
