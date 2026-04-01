# Agentic RAG with LangGraph

An intelligent research assistant that reasons before it retrieves — dynamically routing queries across documents, web search, and a calculator using LangGraph. Built with Groq's free LLaMA models and HuggingFace embeddings.

---

## Demo

> Upload a PDF → ask questions → the agent decides whether to search your docs, the web, or calculate — automatically.

---

## How It Works

```
User Query
    +  memory_summary (compressed old turns)
    +  last K messages (recent raw history)
          ↓
    query_analyzer
    ├── docs loaded?  → retrieve_from_docs (FAISS)
    └── no docs?      → LLM picks tool (web_search / calculator)
          ↓
       grader
    ├── relevant? YES → generate
    └── relevant? NO  → web_search_fallback → generate
          ↓
       generate
    ├── answer using context + memory
    └── maybe_summarize (compress if beyond K window)
          ↓
    Final answer  +  updated memory_summary saved for next query
```

---

## Features

- **Smart routing** — agent decides which tool to use based on your query, no manual selection needed
- **Docs first, web fallback** — always searches uploaded documents first. If nothing relevant found, automatically falls back to web search
- **Sliding memory window** — keeps last K conversations in raw form, compresses older ones into a summary so context is never lost but prompts never bloat
- **Follow-up awareness** — "tell me more", "explain that", "elaborate" all work naturally because memory is passed into every prompt
- **No API key for embeddings** — HuggingFace `all-MiniLM-L6-v2` runs fully locally
- **Free LLM inference** — uses Groq's free tier (no credit card required)

---

## Project Structure

```
agentic_rag/
│
├── app.py                  ← Streamlit UI, session state, chat loop
├── config.py               ← All settings (models, chunk size, window size)
├── requirements.txt
├── .env.example
│
├── graph/
│   └── agent_graph.py      ← LangGraph nodes, edges, memory logic
│
├── tools/
│   └── agent_tools.py      ← retriever tool, web search tool, calculator
│
└── utils/
    └── vectorstore.py      ← FAISS + HuggingFace embeddings, PDF/URL loader
```

---

## Nodes Explained

| Node | Color | Job |
|------|-------|-----|
| `query_analyzer` | Purple | Reads memory + decides: retrieve docs / pick tool / answer directly |
| `retrieve_from_docs` | Teal | Searches FAISS vector store for relevant chunks |
| `LLM picks tool` | Amber | LLM chooses between web_search and calculator |
| `ToolNode` | Amber | Actually executes the chosen tool |
| `grader` | Blue | Checks if retrieved content is relevant to the query |
| `web_search_fallback` | Red | Directly calls web search when docs had no relevant answer |
| `generate` | Teal | Writes final answer using context + memory as SystemMessage |
| `maybe_summarize` | Purple | Compresses old turns into summary when beyond K window |

---

## Memory System

```
All messages in state
├── Old turns (beyond K window) → maybe_summarize → memory_summary (4-6 sentences)
└── Last K turns (recent)       → passed raw

Both combined → build_memory_context → SystemMessage injected into every LLM call
```

- `MEMORY_WINDOW = 3` means last 3 exchanges (6 messages) are passed raw
- Everything older gets compressed into `memory_summary` by the LLM
- Summary is saved back to `st.session_state` and passed into the next query
- This means follow-ups like "tell me more about it" always work correctly

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/agentic-rag-langgraph.git
cd agentic-rag-langgraph
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Set API keys

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

### 4. Run

```bash
streamlit run app.py
```

---

## API Keys (both free)

| Key | Required | Get it at |
|-----|----------|-----------|
| `GROQ_API_KEY` | Yes | [console.groq.com](https://console.groq.com) |
| `TAVILY_API_KEY` | Optional | [app.tavily.com](https://app.tavily.com) |

> Without Tavily, web search falls back to Wikipedia automatically.
> HuggingFace embeddings run locally — no key needed.

---

## Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Tool routing | `llama3-groq-8b-8192-tool-use-preview` | Fast, reliable function calling |
| Grading + generation | `llama-3.3-70b-versatile` | Smarter reasoning and answers |
| Embeddings | `all-MiniLM-L6-v2` | Local, free, fast semantic search |

---

## Usage Examples

| Query type | What happens |
|------------|-------------|
| "what is attention mechanism" (PDF loaded) | Searches PDF → grades → generates answer |
| "what is attention mechanism" (no PDF) | LLM picks web_search → grades → generates |
| "tell me more about it" | Memory context used, no retrieval needed |
| "what is 15% of 4800" | Calculator tool called directly |
| "latest news about AI" | Web search called, graded, answer generated |
| Docs don't have the answer | Auto fallback to web search |

---

## Deploying on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file as `app.py`
5. Add secrets in Advanced Settings:
   ```toml
   GROQ_API_KEY = "your_key"
   TAVILY_API_KEY = "your_key"
   ```
6. Deploy — you get a public URL automatically

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| LangGraph | Agent state machine, conditional routing |
| LangChain | Tools, loaders, document processing |
| Groq | Free ultra-fast LLaMA inference |
| HuggingFace | Local sentence embeddings |
| FAISS | Vector similarity search |
| Tavily | Real-time web search |
| Streamlit | Chat UI |
