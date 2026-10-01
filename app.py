import asyncio
from datetime import datetime

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

PROMPT_TEMPLATE = """
Answer the question as detailed as possible from the provided context.
Make sure to provide all the details. If the answer is not in the
provided context, just say "answer is not available in the context".
Don't provide a wrong answer.

Context:
{context}

Question:
{question}

Answer:
"""


@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
    return text


def build_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return FAISS.from_texts(chunks, embedding=get_embeddings())


def content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "".join(parts)
    return str(content)


def stream_answer(question, vector_store, api_key, model_name):
    docs = vector_store.similarity_search(question, k=4)
    context = "\n\n".join(d.page_content for d in docs)

    llm = ChatGroq(model=model_name, temperature=0.3, api_key=api_key)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
    chain = prompt | llm
    for chunk in chain.stream({"context": context, "question": question}):
        yield content_to_text(chunk.content)


def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")

    if "history" not in st.session_state:
        st.session_state.history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "pdf_names" not in st.session_state:
        st.session_state.pdf_names = []

    with st.sidebar:
        st.title("Menu")

        api_key = st.text_input("Groq API Key", type="password")
        st.markdown("Get a free key at [console.groq.com/keys](https://console.groq.com/keys)")
        model_name = st.text_input("Groq model", value="llama-3.3-70b-versatile")

        st.divider()
        pdf_docs = st.file_uploader(
            "Upload your PDF files, then click Submit & Process",
            type="pdf",
            accept_multiple_files=True,
        )

        if st.button("Submit & Process", type="primary"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF first.")
            else:
                with st.spinner("Processing..."):
                    text = get_pdf_text(pdf_docs)
                    if not text.strip():
                        st.error(
                            "No text could be extracted. The PDF may be scanned images."
                        )
                    else:
                        st.session_state.vector_store = build_vector_store(text)
                        st.session_state.pdf_names = [p.name for p in pdf_docs]
                        st.success("Done! Ask your questions.")

        if st.button("Clear chat"):
            st.session_state.history = []
            st.rerun()

        if st.session_state.history:
            df = pd.DataFrame(
                st.session_state.history,
                columns=["Question", "Answer", "Timestamp", "PDF Names"],
            )
            st.download_button(
                "Download chat as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="conversation_history.csv",
                mime="text/csv",
            )

    for q, a, _, _ in st.session_state.history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)

    question = st.chat_input("Ask a question about your PDFs")

    if question:
        if not api_key:
            st.warning("Please enter your Groq API key in the sidebar.")
            return
        if st.session_state.vector_store is None:
            st.warning("Please upload PDFs and click 'Submit & Process' first.")
            return

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                answer = st.write_stream(
                    stream_answer(
                        question,
                        st.session_state.vector_store,
                        api_key,
                        model_name,
                    )
                )
            except Exception as e:
                st.error(f"Error: {e}")
                return

        st.session_state.history.append(
            (
                question,
                answer,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ", ".join(st.session_state.pdf_names),
            )
        )


if __name__ == "__main__":
    main()