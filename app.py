import os
import tempfile

import streamlit as st

from src.pdf_processor import load_pdf, chunk_text
from src.embeddings import build_vector_store, get_retriever
from src.qa_chain import build_qa_chain, ask_question


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# Session State
# -----------------------------

if "chain" not in st.session_state:
    st.session_state.chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.title("📄 PDF Chatbot")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    process_btn = st.button(
        "Process PDF",
        use_container_width=True
    )

    if uploaded_file and process_btn:

        with st.spinner("Processing PDF..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:

                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name

            try:

                # Step 1
                text = load_pdf(pdf_path)

                # Step 2
                documents = chunk_text(text)

                # Step 3
                vector_store = build_vector_store(documents)

                # Step 4
                retriever = get_retriever(vector_store)

                # Step 5
                chain = build_qa_chain(retriever)

                st.session_state.chain = chain
                st.session_state.pdf_processed = True
                st.session_state.messages = []

                st.success(
                    f"PDF processed successfully! "
                    f"{len(documents)} chunks created."
                )

            except Exception as e:
                st.error(str(e))

            finally:
                os.unlink(pdf_path)

# -----------------------------
# Main Screen
# -----------------------------

st.title("🤖 AI PDF Chatbot")

if not st.session_state.pdf_processed:

    st.info(
        "Upload a PDF from the sidebar and click Process PDF."
    )

else:

    # Display chat history
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_question = st.chat_input(
        "Ask anything about the PDF..."
    )

    if user_question:

        # Show user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                result = ask_question(
                    st.session_state.chain,
                    user_question
                )

                answer = result["answer"]

                st.markdown(answer)

                # Source chunks
                with st.expander("View Sources"):

                    for i, doc in enumerate(
                        result["sources"],
                        start=1
                    ):
                        st.markdown(
                            f"### Source {i}"
                        )

                        st.write(
                            doc.page_content[:500]
                        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
