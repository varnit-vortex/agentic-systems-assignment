# Agentic Systems AI Tutor

A multilingual AI tutor chatbot built with Streamlit and LangChain. It answers questions about the **Agentic Systems and Design** course (iHUB DivyaSampark, IIT Roorkee & Masai School) using your notes as a knowledge base via Retrieval-Augmented Generation (RAG).

## Features

- **RAG-powered answers** — Retrieves relevant sections from `my_notes.txt` before generating a response
- **Multilingual support** — Responds in the same language you ask in (Hindi, English, etc.)
- **Chat history** — Keeps conversation context within the session
- **Fast inference** — Uses Groq's Llama 3.3 70B model

## Tech Stack

| Component | Technology |
|-----------|------------|
| UI | [Streamlit](https://streamlit.io/) |
| LLM | [Groq](https://groq.com/) — `llama-3.3-70b-versatile` |
| Embeddings | [HuggingFace](https://huggingface.co/) — `all-MiniLM-L6-v2` |
| Vector store | [FAISS](https://github.com/facebookresearch/faiss) |
| Framework | [LangChain](https://www.langchain.com/) |

## Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/)

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd agentic-systems-assignment
```

### 2. Create and activate a virtual environment

```bash
python -m venv myenv
source myenv/bin/activate   # macOS / Linux
# myenv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the app

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## How It Works

```
User question
     │
     ▼
┌─────────────────┐
│  my_notes.txt   │  ← Course notes loaded and split into chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS index    │  ← Similarity search finds top 2 relevant chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Groq LLM       │  ← Generates answer grounded in retrieved notes
└────────┬────────┘
         │
         ▼
   Chat response
```

1. **Ingestion** — `my_notes.txt` is loaded and split into 100-character chunks (20-character overlap).
2. **Embedding** — Chunks are embedded with `all-MiniLM-L6-v2` and stored in a FAISS vector index.
3. **Retrieval** — When you ask a question, the two most similar note chunks are retrieved.
4. **Generation** — The LLM receives the retrieved context plus your question and produces a grounded, multilingual answer.

## Project Structure

```
agentic-systems-assignment/
├── app.py              # Streamlit chatbot application
├── my_notes.txt        # Course knowledge base (RAG source)
├── requirements.txt    # Python dependencies
├── .env                # API keys (create locally, not committed)
├── .gitignore
├── module 1/           # Course session exercises
└── myenv/              # Virtual environment (local only, not committed)
```

## Customizing the Knowledge Base

Edit `my_notes.txt` with your own course material or notes. Restart the Streamlit app to reload the vector index (cached on first load — use **Clear cache** in the Streamlit menu if you edit notes mid-session).

## License

Educational project for the Agentic Systems and Design course.
