import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
import os
from tempfile import NamedTemporaryFile

# Setup API Key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Chat with your PDF", layout="centered")
st.title("📄 Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Build Chatbot"):
        with st.spinner("Indexing PDF..."):
            reader = SimpleDirectoryReader(input_files=[tmp_path])
            docs = reader.load_data()

            llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
            Settings.llm = llm

            index = VectorStoreIndex.from_documents(docs)
            query_engine = index.as_query_engine()

        st.session_state.query_engine = query_engine
        st.success("Chatbot is ready! Ask your questions below 👇")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "query_engine" in st.session_state:
    query = st.chat_input("Ask a question about the PDF:")

    if query:
        with st.spinner("Thinking..."):
            response = st.session_state.query_engine.query(query)
            st.session_state.chat_history.append((query, response.response))

# Display chat history
for i, (q, a) in enumerate(st.session_state.chat_history):
    with st.chat_message("user", avatar="👤"):
        st.markdown(q)
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(a)

