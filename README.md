# 🤖 Agentic RAG with LangGraph

An intelligent research assistant that **reasons before it retrieves** — dynamically routing queries across documents, web search, and a calculator using LangGraph.

---

## 🏗️ Project Structure

```
agentic_rag/
├── app.py                  # Streamlit UI
├── config.py               # All settings in one place
├── requirements.txt
├── .env.example
├── graph/
│   └── agent_graph.py      # LangGraph state + nodes + edges
├── tools/
│   └── agent_tools.py      # Retriever, web search, calculator tools
└── utils/
    └── vectorstore.py      # FAISS + HuggingFace embeddings
```

---

## ⚙️ How It Works

```
User Query
    ↓
Query Analyzer (LLaMA 3.1 8B)
  → picks the right tool
    ↓
Tool Node (runs selected tool)
  ├── retrieve_from_docs  → searches FAISS vector store
  ├── web_search          → Tavily / Wikipedia
  └── calculator          → safe math eval
    ↓
Grader (LLaMA 3.3 70B)
  → is the result relevant?
  ├── YES → Generate Answer
  └── NO  → Web Search Fallback → Generate Answer
    ↓
Final Answer streamed to user
```

**Key LangGraph concepts used:**
- `StateGraph` with typed `AgentState`
- `ToolNode` for automatic tool execution
- `conditional_edges` for smart routing
- Retry logic when retrieval isn't relevant

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <your-repo>
cd agentic_rag
pip install -r requirements.txt
```

### 2. Set Up API Keys

```bash
cp .env.example .env
# Edit .env with your keys
```

**Free API keys:**
- **Groq**: https://console.groq.com (free, fast LLaMA models)
- **Tavily**: https://app.tavily.com (free tier, 1000 searches/month)

### 3. Run the App

```bash
streamlit run app.py
```

---

## 🔑 API Keys

| Key | Required | Where to get |
|-----|----------|--------------|
| `GROQ_API_KEY` | ✅ Yes | [console.groq.com](https://console.groq.com) |
| `TAVILY_API_KEY` | ⚠️ Optional | [app.tavily.com](https://app.tavily.com) |

> Without Tavily, the agent falls back to Wikipedia for web search.
> HuggingFace embeddings run **locally** — no API key needed.

---

## 🧠 Models Used

| Component | Model | Why |
|-----------|-------|-----|
| Query routing | `llama-3.1-8b-instant` | Fast, low latency |
| Grading & generation | `llama-3.3-70b-versatile` | More accurate |
| Embeddings | `all-MiniLM-L6-v2` | Free, local, fast |

---

## 💡 Usage

1. Enter your Groq API key in the sidebar
2. (Optional) Upload PDFs or paste URLs → click **Build Knowledge Base**
3. Ask questions in the chat

**Example queries:**
- *"What does the document say about X?"* → uses doc retrieval
- *"What happened in the news today?"* → uses web search
- *"What is 15% of 4500?"* → uses calculator
- *"Summarize the main points of the paper"* → uses doc retrieval

---

## 📦 Tech Stack

- **LangGraph** — agent state machine & conditional routing
- **LangChain** — tools, loaders, document processing
- **Groq** — free, ultra-fast LLaMA inference
- **HuggingFace** — local sentence embeddings
- **FAISS** — vector similarity search
- **Tavily** — real-time web search
- **Streamlit** — chat UI
