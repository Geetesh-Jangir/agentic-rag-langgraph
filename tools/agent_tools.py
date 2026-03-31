from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from config import TAVILY_API_KEY
import os
import math


# ─── Tool 1: Document Retriever ────────────────────────────────────────────────

def make_retriever_tool(retriever):
    """
    Creates a retriever tool dynamically, bound to the given retriever.
    Returns None if no retriever is available.
    """
    if retriever is None:
        return None

    @tool
    def retrieve_from_docs(query: str) -> str:
        """
        Search the uploaded documents for information relevant to the query.
        Use this when the user asks about content from their uploaded PDFs or URLs.
        """
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found."
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            results.append(f"[Doc {i}] (source: {source})\n{doc.page_content}")
        return "\n\n".join(results)

    return retrieve_from_docs


# ─── Tool 2: Web Search ────────────────────────────────────────────────────────

def make_web_search_tool():
    """
    Creates a Tavily web search tool.
    Falls back to Wikipedia if no Tavily key is set.
    """
    if TAVILY_API_KEY:
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
        tavily = TavilySearchResults(max_results=3)

        @tool
        def web_search(query: str) -> str:
            """
            Search the internet for current or general information.
            Use this when the documents don't have the answer or the question is about recent events.
            """
            results = tavily.invoke(query)
            if not results:
                return "No web results found."
            output = []
            for r in results:
                output.append(f"[{r['url']}]\n{r['content']}")
            return "\n\n".join(output)

        return web_search

    else:
        # Fallback: Wikipedia (no API key needed)
        wiki = WikipediaAPIWrapper(top_k_results=2)

        @tool
        def web_search(query: str) -> str:
            """
            Search Wikipedia for general information.
            Use this when the documents don't have the answer.
            """
            return wiki.run(query)

        return web_search


# ─── Tool 3: Calculator ────────────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Use this for any arithmetic, percentages, or numeric calculations.
    Examples: '25 * 4', 'math.sqrt(144)', '(100 - 20) / 100 * 500'
    """
    try:
        # Safe eval: only math functions allowed
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed["__builtins__"] = {}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"
