# 🤖 RAG Chat Assistant

A **Retrieval-Augmented Generation (RAG) based PDF Chat Assistant** built with **Python, Streamlit, LangChain, LangGraph, Hugging Face Embeddings, and Groq LLM**.

The application allows users to upload one or more PDF documents and ask questions about their content. It retrieves relevant information from the uploaded documents using semantic similarity search and uses an LLM to generate the final response.

---

## 🚀 Features

* 📄 Upload multiple PDF documents
* 🔍 Semantic search over uploaded documents
* 🧠 Retrieval-Augmented Generation (RAG)
* 🤖 Powered by Groq LLM
* 🔤 Hugging Face sentence-transformer embeddings
* 🗂️ In-memory vector database
* 💬 Interactive Streamlit chat interface
* ⚡ Streaming AI responses
* 🧠 Conversation checkpointing using LangGraph
* 📋 Supports factual questions about uploaded documents
* 💡 Provides resume analysis and improvement suggestions
* 🔒 Prevents the model from inventing personal information

---

## 🏗️ Architecture

```text
                ┌─────────────────────┐
                │   Upload PDF Files  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    PyPDF Loader     │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Text Chunking      │
                │ Recursive Splitter  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ HuggingFace         │
                │ Embeddings          │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ InMemory Vector DB  │
                └──────────┬──────────┘
                           │
                    User Question
                           │
                           ▼
                ┌─────────────────────┐
                │ Similarity Search   │
                │      Top-K          │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    Groq LLM         │
                │ GPT-OSS-20B         │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   AI Response       │
                └─────────────────────┘
```

---

## 🛠️ Technologies Used

| Technology                   | Purpose                         |
| ---------------------------- | ------------------------------- |
| Python                       | Backend programming             |
| Streamlit                    | Web interface                   |
| LangChain                    | RAG and LLM framework           |
| LangGraph                    | Agent and conversation state    |
| Groq                         | Large Language Model            |
| Hugging Face                 | Text embeddings                 |
| PyPDF                        | PDF document loading            |
| Chroma/InMemory Vector Store | Vector similarity search        |
| python-dotenv                | Environment variable management |

### Main Libraries

```text
langchain
langchain-groq
langchain-community
langchain-text-splitters
langchain-huggingface
langgraph
streamlit
python-dotenv
sentence-transformers
```

---

## 📁 Project Structure

```text
RAG Chatbot/
│
├── main.py
├── requirements.txt
├── .env
├── .gitignore
│
└── documents/
    ├── resume.pdf
    ├── document1.pdf
    └── document2.pdf
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ShashidharKokku/rag-chat-assistant.git
```

Move into the project directory:

```bash
cd rag-chat-assistant
```

---

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Replace `your_groq_api_key` with your actual Groq API key.

**Do not upload your `.env` file to GitHub.**

Add this to `.gitignore`:

```text
.env
venv/
__pycache__/
documents/
```

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run main.py
```

The application will open in your browser.

You can then:

1. Upload one or more PDF files.
2. Wait for the documents to be processed.
3. Ask questions in the chat box.
4. The system retrieves relevant document content.
5. The LLM generates an answer using the retrieved information.

---

## 🔍 How RAG Works in This Project

The application follows the following pipeline:

### 1. Document Loading

PDF files are loaded using:

```python
PyPDFDirectoryLoader(path)
```

The loader extracts text from the uploaded PDF documents.

### 2. Text Chunking

Large documents are divided into smaller chunks:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

Chunking makes it easier to retrieve relevant sections of the document.

### 3. Embeddings

Each text chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

These embeddings represent the semantic meaning of the text.

### 4. Vector Storage

The embeddings are stored in an in-memory vector store:

```python
InMemoryVectorStore.from_documents()
```

### 5. Similarity Search

When the user asks a question, the application searches for the most relevant document chunks:

```python
vector_db.similarity_search(
    query=query,
    k=1
)
```

### 6. LLM Generation

The retrieved context is passed to the Groq-powered LLM.

The model uses the retrieved information to generate the final answer.

---

## 🧠 Agent Workflow

The application uses a LangChain agent with a custom retrieval tool:

```python
@tool
def retrieve_context(query: str):
```

The agent decides when to use the retrieval tool based on the user's question.

For example:

### Document Question

**User:**

```text
What skills are mentioned in the resume?
```

The agent retrieves relevant resume content and answers using that information.

### Resume Improvement Question

**User:**

```text
What can be improved in this resume?
```

The agent can use information from the resume and provide recommendations based on general resume best practices.

### General Question

**User:**

```text
What is RAG?
```

This is a general knowledge question and does not necessarily require information from the uploaded document.

---

## 💬 Example Questions

### Resume-Based Questions

```text
What is the candidate's educational qualification?

What programming languages are mentioned?

What projects are included in the resume?

What technical skills does the candidate have?

What work experience is mentioned?
```

### Resume Analysis

```text
What things can be improved in this resume?

How can I make this resume more ATS friendly?

What skills should be highlighted better?

How can the project descriptions be improved?

What sections are missing from this resume?
```

### General Questions

```text
What is Retrieval-Augmented Generation?

What is an embedding?

What is cosine similarity?

What is a vector database?

How does semantic search work?
```

---

## 🛡️ Hallucination Prevention

The system prompt instructs the agent to avoid inventing personal information.

For document-related factual questions, the agent should:

```text
1. Retrieve relevant document information.
2. Use only the retrieved information.
3. Avoid inventing facts.
4. Clearly state when information cannot be found.
```

If information is unavailable, the assistant responds:

```text
I could not find that information in the uploaded documents.
```

For analysis and suggestions, the assistant can provide recommendations while distinguishing them from facts found in the document.

---

## 📌 Key Components

### `process_document()`

Responsible for:

```text
PDF Loading
     ↓
Text Splitting
     ↓
Embeddings
     ↓
Vector Store
     ↓
Agent Creation
```

### `retrieve_context()`

Responsible for retrieving relevant document chunks based on the user's query.

### `create_agent()`

Creates the AI agent with:

* Groq LLM
* Retrieval tool
* System prompt
* Conversation checkpointing

### Streamlit Chat UI

Provides:

* PDF upload
* Chat interface
* Message history
* Streaming AI responses

---

## 🔮 Future Improvements

Possible improvements include:

* Persistent vector database using ChromaDB or FAISS
* Support for DOCX and TXT files
* Better PDF parsing
* Metadata-based filtering
* Multiple document collections
* Improved retrieval using hybrid search
* Reranking retrieved documents
* Conversation-aware retrieval
* Source/page citations in answers
* Authentication and user accounts
* Cloud deployment
* Resume scoring and ATS analysis
* Resume section comparison
* Downloadable resume improvement report

---

## ☁️ Deployment

The application can be deployed using platforms such as:

* Streamlit Community Cloud
* Render
* Hugging Face Spaces
* AWS
* Google Cloud
* Azure

For deployment, configure the required environment variables securely instead of committing the `.env` file.

---

## 🔐 Security Notes

Never commit API keys to GitHub.

Use:

```text
.env
```

for local development and platform-specific secret management for deployment.

Example:

```env
GROQ_API_KEY=********
```

---

## 📄 License

This project is available for educational and personal use.

---

## 👨‍💻 Author

**Shashidhar Kokku**

B.Tech – Computer Science and Engineering (Data Science)

Interested in:

* Generative AI
* RAG
* AI Agents
* Full Stack Development
* Machine Learning
* Python

---

## ⭐ Project Highlights

```text
PDF Documents
      ↓
Text Extraction
      ↓
Chunking
      ↓
Hugging Face Embeddings
      ↓
Vector Similarity Search
      ↓
Relevant Context
      ↓
Groq LLM
      ↓
AI Response
```

**Built with Python, LangChain, LangGraph, Streamlit, Hugging Face and Groq.**
