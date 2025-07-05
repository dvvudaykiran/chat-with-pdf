import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
import os
from tempfile import NamedTemporaryFile

# Styling
st.markdown("""
<style>
    h1 {
        color: #4F8BF9;
    }
    .stButton > button {
        border-radius: 8px;
        background-color: #4F8BF9;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Setup API Key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Chat with your PDFs", layout="centered")
st.title("📄 Chat with your PDFs")

uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    tmp_paths = []
    for uploaded_file in uploaded_files:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_paths.append(tmp_file.name)

    st.success(f"{len(uploaded_files)} PDF(s) uploaded successfully!")

    if st.button("Build Chatbot"):
        with st.spinner("Indexing PDFs..."):
            reader = SimpleDirectoryReader(input_files=tmp_paths)
            docs = reader.load_data()

            llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
            Settings.llm = llm

            index = VectorStoreIndex.from_documents(docs)
            query_engine = index.as_query_engine()

        st.session_state.query_engine = query_engine
        st.session_state.chat_history = []
        st.success("Chatbot is ready! Ask your questions below 👇")

# Initialize session state for history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "query_engine" in st.session_state:
    query = st.chat_input("Ask a question about the PDFs:")

    if query:
        with st.spinner("Thinking..."):
            response = st.session_state.query_engine.query(query)
            st.session_state.chat_history.append((query, response.response))

    # Display conversation
    for i, (q, a) in enumerate(st.session_state.chat_history):
        with st.chat_message("user", avatar="👤"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(a)

    # Buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
    with col2:
        if st.download_button(
            label="📥 Download Chat",
            data="\n\n".join([f"User: {q}\nBot: {a}" for q, a in st.session_state.chat_history]),
            file_name="chat_history.txt",
            mime="text/plain"
        ):
            st.success("Chat history downloaded!")
