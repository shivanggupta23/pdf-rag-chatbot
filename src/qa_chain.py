"""
qa_chain.py

Builds the conversational RAG chain using:
- Groq Llama 3
- FAISS Retriever
- LangChain Memory
"""

import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain.memory import ConversationBufferMemory

from langchain.chains import ConversationalRetrievalChain

load_dotenv()


def build_qa_chain(retriever):
    """
    Build conversational retrieval chain.
    """

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    return chain


def ask_question(chain, question):
    """
    Ask question to chatbot.
    """

    result = chain({"question": question})

    return {
        "answer": result["answer"],
        "sources": result["source_documents"]
    }