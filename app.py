import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("AI Chatbot")
st.write("Ask me anything and let's chat!")

@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="microsoft/DialoGPT-small"
    )

chatbot = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    prompt = user_input

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chatbot(
                prompt,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

            bot_reply = response[0]["generated_text"]

            if bot_reply.startswith(prompt):
                bot_reply = bot_reply[len(prompt):].strip()

            if not bot_reply:
                bot_reply = "Sorry, I couldn't generate a response."

            st.write(bot_reply)

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })
