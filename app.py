import streamlit as st
import requests
import uuid

# ---------------- CONFIG ----------------
API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(page_title="NyayaSathi", page_icon="⚖️")
st.title("⚖️ NyayaSathi – Legal AI Assistant")

# ---------------- SESSION ----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- USER INPUT ----------------
user_input = st.chat_input("Ask your legal question...")

if user_input:
    # show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # API call
    payload = {
        "session_id": st.session_state.session_id,
        "question": user_input
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        answer = response.json()["answer"]

    except Exception as e:
        answer = f"❌ Error connecting to API: {e}"

    # show bot message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)
