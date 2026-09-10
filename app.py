import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(
    page_title="AI-CHATBOT",
    page_icon="🤖"
)

st.title("AI Chatbot")
st.write("Ask me anything and let's chat!")

client = InferenceClient(
    api_key=st.secrets["HF_TOKEN"]
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Type your message...")

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b:groq",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. Answer clearly and simply. You can communicate in English, Tamil and Tanglish."
                    }
                ] + st.session_state.messages,
                max_tokens=500,
                temperature=0.7
            )

            answer = response.choices[0].message.content
            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:
            st.error("AI response failed. Please check your Hugging Face token and permissions.")
