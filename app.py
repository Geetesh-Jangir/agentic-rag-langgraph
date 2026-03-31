import streamlit as st
import tempfile
import os
from langchain_core.messages import HumanMessage, AIMessage

# Local imports
from config import GROQ_API_KEY, TAVILY_API_KEY
from utils.vectorstore import (
    load_documents,
    build_vectorstore,
    load_vectorstore,
    get_retriever,
)
from tools.agent_tools import make_retriever_tool, make_web_search_tool, calculator
from graph.agent_graph import build_agent_graph


# Page Config

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agentic RAG Assistant")
st.caption("Powered by LangGraph · Groq LLaMA · HuggingFace Embeddings")


# Session State Init

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = st.session_state.vectorstore is not None
if "memory_summary" not in st.session_state:
    st.session_state.memory_summary = ""

# Store API keys in session state so they persist across reruns
if "groq_key_saved" not in st.session_state:
    st.session_state.groq_key_saved = os.environ.get("GROQ_API_KEY", GROQ_API_KEY)
if "tavily_key_saved" not in st.session_state:
    st.session_state.tavily_key_saved = os.environ.get("TAVILY_API_KEY", TAVILY_API_KEY)

# Always apply saved keys to env on every rerun
if st.session_state.groq_key_saved:
    os.environ["GROQ_API_KEY"] = st.session_state.groq_key_saved
if st.session_state.tavily_key_saved:
    os.environ["TAVILY_API_KEY"] = st.session_state.tavily_key_saved


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Setup")

    groq_input = st.text_input(
        "Groq API Key",
        value="",
        type="password",
        placeholder="Paste key then click Save →",
        help="Free at console.groq.com",
        key="groq_input_field",
    )
    tavily_input = st.text_input(
        "Tavily API Key (optional)",
        value="",
        type="password",
        placeholder="Paste key then click Save →",
        help="Free at app.tavily.com",
        key="tavily_input_field",
    )

    if st.button("💾 Save API Keys", use_container_width=True):
        changed = False
        if groq_input.strip():
            if st.session_state.groq_key_saved != groq_input.strip():
                st.session_state.agent = None
            st.session_state.groq_key_saved = groq_input.strip()
            os.environ["GROQ_API_KEY"] = groq_input.strip()
            changed = True
        if tavily_input.strip():
            st.session_state.tavily_key_saved = tavily_input.strip()
            os.environ["TAVILY_API_KEY"] = tavily_input.strip()
            changed = True

        if changed:
            st.success("✅ Keys saved!")
            st.rerun()
        else:
            st.warning("No keys entered.")

    # Status indicators
    groq_key = st.session_state.groq_key_saved
    tavily_key = st.session_state.tavily_key_saved
    col1, col2 = st.columns(2)
    col1.markdown(f"{'🟢 Groq ready' if groq_key else '🔴 No Groq key'}")
    col2.markdown(f"{'🟢 Tavily' if tavily_key else '🟡 No Tavily'}")

    st.divider()
    st.header("📄 Add Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True
    )

    urls_input = st.text_area(
        "Or paste URLs (one per line)",
        height=100,
        placeholder="https://example.com/article\nhttps://...",
        key="urls_input_field",
    )

    if st.button("🔄 Build Knowledge Base", use_container_width=True):
        if not uploaded_files and not urls_input.strip():
            st.warning("Please upload PDFs or add URLs first.")
        else:
            with st.spinner("Loading & embedding documents..."):
                pdf_paths = []

                if uploaded_files:
                    for f in uploaded_files:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        tmp.write(f.read())
                        tmp.close()
                        pdf_paths.append(tmp.name)

                urls = [u.strip() for u in urls_input.strip().split("\n") if u.strip()]
                docs = load_documents(pdf_paths=pdf_paths, urls=urls if urls else None)

                if docs:
                    st.session_state.vectorstore = build_vectorstore(docs)
                    st.session_state.docs_loaded = True
                    st.session_state.agent = None  # rebuild agent with new docs
                    st.success(f"✅ Indexed {len(docs)} chunks!")
                else:
                    st.error("Could not load any documents. Check your files/URLs.")

                for path in pdf_paths:
                    os.unlink(path)

    if st.session_state.docs_loaded:
        st.success("📚 Knowledge base ready")
        if st.button("🗑️ Remove Docs", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.docs_loaded = False
            st.session_state.agent = None  # rebuild without retriever tool
            import shutil

            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index")  # delete saved index from disk too
            st.rerun()
    else:
        st.info("No documents loaded — using web search & general knowledge")
    st.divider()

    st.header("🛠️ Active Tools")
    st.markdown(f"{'✅' if st.session_state.docs_loaded else '❌'} Document retrieval")
    st.markdown(
        f"{'✅' if tavily_key else '⚠️'} Web search {'(Tavily)' if tavily_key else '(Wikipedia fallback)'}"
    )
    st.markdown("✅ Calculator")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent = None
        st.rerun()


def get_agent():
    """Build or return cached agent."""
    if st.session_state.agent is not None:
        return st.session_state.agent

    tools = []

    if st.session_state.vectorstore:
        retriever = get_retriever(st.session_state.vectorstore)
        retriever_tool = make_retriever_tool(retriever)
        if retriever_tool:
            tools.append(retriever_tool)

    tools.append(make_web_search_tool())
    tools.append(calculator)

    agent = build_agent_graph(tools)
    st.session_state.agent = agent
    return agent


#  Chat Interface

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tool_used"):
            st.caption(f"🔧 Tool used: `{msg['tool_used']}`")

if query := st.chat_input("Ask anything about your documents or any topic..."):

    if not groq_key:
        st.error("Please enter your Groq API key in the sidebar and click Save.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                agent = get_agent()

                history = []
                for m in st.session_state.messages:
                    if m["role"] == "user":
                        history.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant":
                        history.append(AIMessage(content=m["content"]))

                MAX_HISTORY = 20
                if len(history) > MAX_HISTORY:
                    history = history[-MAX_HISTORY:]

                result = agent.invoke(
                    {
                        "messages": history,
                        "query": query,
                        "retrieved_context": "",
                        "final_answer": "",
                        "retry_count": 0,
                        "tool_used": "",
                        "memory_summary": st.session_state.memory_summary,
                    }
                )

                # Save updated summary back to session state
                if result.get("memory_summary"):
                    st.session_state.memory_summary = result["memory_summary"]

                answer = result.get("final_answer", "")
                tool_used = result.get("tool_used", "none")

                if not answer:
                    for msg in reversed(result.get("messages", [])):
                        if hasattr(msg, "content") and msg.content:
                            answer = msg.content
                            break

                if not answer:
                    answer = "Sorry, I couldn't generate a response. Please try again."

                st.markdown(answer)
                st.caption(f"🔧 Tool used: `{tool_used}`")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "tool_used": tool_used,
                    }
                )

            except Exception as e:
                err_msg = f"⚠️ Error: {str(e)}"
                st.error(err_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": err_msg,
                    }
                )
