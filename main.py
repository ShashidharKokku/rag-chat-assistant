from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def process_document(path):

    # Loading the Document

    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    # Split into Chunks

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    docs = splitter.split_documents(documents=docs)

    # Embeddings and Vector Store

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embedding
    )

    # Agent - LLM | tool | System_prompt

    llm = ChatGroq(model="openai/gpt-oss-20b", streaming=True)

    @tool
    def retrieve_context(query:str):
        """"Retrieve documents relevant to a query from the knowledge base."""
        context = ""

        docs = vector_db.similarity_search(query=query, k=2)
        for doc in docs:
            context += doc.page_content + "\n\n"
        return context

    system_prompt = """
    You are an intelligent document and general-purpose assistant.

    You can answer questions using information from the uploaded documents and also answer general questions using your own knowledge.

    Follow these rules:

    1. DOCUMENT-RELATED QUESTIONS:
    - If the user's question is related to or can be answered from the uploaded documents, always use the retrieve_context tool.
    - Answer using the information found in the retrieved documents.
    - You may combine information from multiple retrieved sections.
    - Do not invent or assume information that is not present in the documents.

    2. GENERAL QUESTIONS:
    - If the user's question is not related to the uploaded documents, answer using your general knowledge.
    - You do not need to use the retrieve_context tool for general questions.

    3. DOCUMENT ANALYSIS:
    - For questions asking for analysis, explanation, summary, comparison, suggestions, or improvements related to the uploaded documents, retrieve the relevant information first.
    - Use the document information as the basis for your response.
    - You may provide additional recommendations based on general knowledge when appropriate.
    - Clearly distinguish information found in the document from your own recommendations.

    4. MISSING INFORMATION:
    - If the user asks for specific information that should be in the uploaded documents but that information cannot be found, say:
    "I could not find that information in the uploaded documents."

    5. ACCURACY:
    - Never invent names, numbers, dates, facts, or other information.
    - Do not assume information that is not present in the retrieved documents.
    - If the document contains the answer, provide it directly.

    6. RESPONSE STYLE:
    - Keep answers clear, concise, and directly related to the user's question.
    - When appropriate, mention the page or section where the information was found.
    """


    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver()
    )
    st.session_state.agent = agent 
    st.session_state.document_uploaded = True

st.title("🤖 RAG Chat Assistance")

# Document loading UI

if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Upload your pdfs up clicking on upload", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing...."):
            path = "./documents/"
            for file in uploaded:
                with open(path+file.name, "wb") as f:
                    f.write(file.getvalue())
            process_document(path)
            st.rerun()

# Chat UI

if st.session_state.document_uploaded and st.session_state.agent:

    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        st.chat_message(role).markdown(content)

    query = st.chat_input("Ask anything?")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)
        response = st.session_state.agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": 1}},
            stream_mode="messages"
        )
        ai_container = st.chat_message("ai")
        with ai_container:
            space = st.empty()
            message = ""
            for chunk in response:
                message = message + chunk[0].content
                space.write(message)
                    
        st.session_state.messages.append({"role": "ai", "content": message})
