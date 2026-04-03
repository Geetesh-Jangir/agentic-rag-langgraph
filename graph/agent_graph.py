from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator
import os

from config import GROQ_MODEL, GROQ_MODEL_SMART, GROQ_API_KEY, MAX_RETRIES

MEMORY_WINDOW = 6


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    query: str
    retrieved_context: str
    final_answer: str
    retry_count: int
    tool_used: str
    memory_summary: str


def get_llm(smart: bool = False):
    model = GROQ_MODEL_SMART if smart else GROQ_MODEL
    api_key = os.environ.get("GROQ_API_KEY", GROQ_API_KEY)
    return ChatGroq(api_key=api_key, model=model, temperature=0)


def build_agent_graph(tools: list):

    llm = get_llm(smart=False)
    llm_with_tools = llm.bind_tools(tools)
    smart_llm = get_llm(smart=True)
    tool_node = ToolNode(tools)

    #  Helper: build memory context string
    def build_memory_context(state: AgentState) -> str:
        parts = []

        # Add summarized older history
        if state.get("memory_summary"):
            parts.append(f"Summary of earlier conversation:\n{state['memory_summary']}")

        # Only pass last MEMORY_WINDOW exchanges
        all_msgs = [
            m
            for m in state["messages"]
            if hasattr(m, "type") and m.type in ("human", "ai") and m.content
        ]
        recent = all_msgs[-(MEMORY_WINDOW * 2) :]

        if recent:
            lines = []
            for m in recent:
                role = "User" if m.type == "human" else "Assistant"
                lines.append(f"{role}: {m.content}")
            parts.append(
                "Recent conversation (last {} exchanges):\n".format(MEMORY_WINDOW)
                + "\n".join(lines)
            )

        return "\n\n".join(parts)

    # summarize old messages when window overflows
    def maybe_summarize(state: AgentState) -> dict:
        all_msgs = [
            m
            for m in state["messages"]
            if hasattr(m, "type") and m.type in ("human", "ai") and m.content
        ]

        # Only summarize when we exceed the window
        if len(all_msgs) <= MEMORY_WINDOW * 2:
            return {}

        # Everything except the last MEMORY_WINDOW exchanges goes into summary
        old_msgs = all_msgs[: -(MEMORY_WINDOW * 2)]

        lines = []
        for m in old_msgs:
            role = "User" if m.type == "human" else "Assistant"
            lines.append(f"{role}: {m.content}")

        existing_summary = state.get("memory_summary", "")
        to_summarize = "\n".join(lines)

        summary_prompt = f"""You are summarizing a conversation to preserve memory.
    Summarize the following exchanges in 4-6 sentences.
    Keep: key topics discussed, important facts mentioned, and any context useful for future questions.
    Do NOT say the conversation is empty or just starting.

    {"Extend this existing summary with the new content below:\n" + existing_summary if existing_summary else ""}

    New conversation to summarize:
    {to_summarize}

    Summary:"""

        response = smart_llm.invoke([HumanMessage(content=summary_prompt)])
        return {"memory_summary": response.content}

    #  Node 1: Query Analyzer
    def query_analyzer(state: AgentState) -> AgentState:
        has_docs = any(t.name == "retrieve_from_docs" for t in tools)
        memory_context = build_memory_context(state)

        # Always try docs first if available
        if has_docs:
            retriever_tool = next(t for t in tools if t.name == "retrieve_from_docs")
            result = retriever_tool.invoke(state["query"])
            updates = {
                "messages": [],
                "retrieved_context": result,
                "tool_used": "retrieve_from_docs",
                "retry_count": state.get("retry_count", 0),
            }
            return updates

        # No docs — LLM picks tool
        tools_desc = "\n".join(
            f"- {t.name}: {t.description.split(chr(10))[0]}" for t in tools
        )

        system_content = f"""You are a helpful research assistant.

        {memory_context}

        Available tools:
        {tools_desc}

        Rules:
        - Use web_search for questions needing current or external information
        - Use calculator only for math calculations
        - If the question contains vague references like "this", "it", "that" with no clear context in the conversation history, do NOT call any tool — instead reply asking the user to clarify what they mean
        - If the question is a follow-up to something already discussed, answer directly without calling any tool"""

        messages = [SystemMessage(content=system_content)]

        try:
            response = llm_with_tools.invoke(messages)
            tool_used = "none"
            if response.tool_calls:
                tool_used = response.tool_calls[0]["name"]
            updates = {
                "messages": [response],
                "tool_used": tool_used,
                "retry_count": state.get("retry_count", 0),
            }
            return updates

        except Exception as e:
            fallback = AIMessage(content=f"[Routing failed: {str(e)[:100]}]")
            return {
                "messages": [fallback],
                "tool_used": "none",
                "retry_count": state.get("retry_count", 0),
            }

    # Node 2: Grader
    def grader(state: AgentState) -> AgentState:

        tool_result = state.get("retrieved_context", "")
        if not tool_result:
            for msg in reversed(state["messages"]):
                if hasattr(msg, "type") and msg.type == "tool":
                    tool_result = msg.content
                    break

        if not tool_result or tool_result.strip() == "No relevant documents found.":
            return {
                "retrieved_context": "",
                "retry_count": state.get("retry_count", 0) + 1,
            }

        grade_prompt = f"""Given this user query:
"{state['query']}"

And this retrieved information:
"{tool_result[:1500]}"

Is this information relevant and useful to answer the query? Reply with just YES or NO. No Explanation."""

        response = smart_llm.invoke([HumanMessage(content=grade_prompt)])
        is_relevant = "YES" in response.content.strip().upper()

        if is_relevant:
            return {
                "retrieved_context": tool_result,
                "retry_count": state.get("retry_count", 0),
            }
        else:
            return {
                "retrieved_context": "",
                "retry_count": state.get("retry_count", 0) + 1,
            }

    #  Node 3: Generate
    def generate(state: AgentState) -> AgentState:
        context = state.get("retrieved_context", "")
        memory_context = build_memory_context(state)

        if context:
            prompt = f"""{memory_context}

    Now answer the following question using the provided context.
    Be concise and accurate. If the question is a follow-up, use both the context and conversation history above.

    Question: {state['query']}

    Context:
    {context[:3000]}

    Answer:"""
        else:
            prompt = f"""{memory_context}

        The user asked: "{state['query']}"

        IMPORTANT RULES you must follow strictly:
        - If the question contains vague words like "this", "it", "that", "these", "those" and there is NO clear reference to what "this/it/that" means in the conversation history above — respond with exactly:
        "Your question is a bit unclear. Could you provide more context about what you are referring to? For example: 'What is used to build [specific topic]?'"
        - If the question is about a specific topic that exists in the conversation history, answer using that context.
        - If the question is genuinely answerable from general knowledge, answer it.
        - Do NOT make assumptions about what "this" or "it" refers to if it is not clear from the conversation.

        Answer:"""

        response = smart_llm.invoke([HumanMessage(content=prompt)])

        #  Summarize AFTER answer is produced
        # Add the new Q&A pair to messages first, then summarize
        new_messages = state["messages"] + [AIMessage(content=response.content)]
        temp_state = {**state, "messages": new_messages}
        summary_update = maybe_summarize(temp_state)

        result = {
            "messages": [AIMessage(content=response.content)],
            "final_answer": response.content,
        }
        if summary_update:
            result["memory_summary"] = summary_update["memory_summary"]
        return result

    #  Node 4: Web Search Fallback
    def web_search_fallback(state: AgentState) -> AgentState:
        web_tool = next((t for t in tools if t.name == "web_search"), None)
        if web_tool is None:
            return {
                "retrieved_context": "",
                "retry_count": state.get("retry_count", 0) + 1,
            }

        result = web_tool.invoke(state["query"])
        return {
            "retrieved_context": result,
            "tool_used": "web_search",
            "retry_count": state.get("retry_count", 0),
            "messages": [],
        }

    #  Conditional Edges
    def should_continue(state: AgentState) -> Literal["tools", "grader", "generate"]:
        # Docs were retrieved directly  go to grader
        if state.get("retrieved_context"):
            return "grader"
        # LLM chose a tool run it
        if state["messages"]:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
        return "generate"

    def after_grader(state: AgentState) -> Literal["generate", "web_search_fallback"]:
        context = state.get("retrieved_context", "")
        retry = state.get("retry_count", 0)

        if context:
            return "generate"
        elif retry < MAX_RETRIES and state.get("tool_used") != "web_search":
            return "web_search_fallback"
        else:
            return "generate"

    # Build Graph
    graph = StateGraph(AgentState)

    graph.add_node("query_analyzer", query_analyzer)
    graph.add_node("tools", tool_node)
    graph.add_node("grader", grader)
    graph.add_node("generate", generate)
    graph.add_node("web_search_fallback", web_search_fallback)

    graph.set_entry_point("query_analyzer")

    graph.add_conditional_edges(
        "query_analyzer",
        should_continue,
        {"tools": "tools", "grader": "grader", "generate": "generate"},
    )

    graph.add_edge("tools", "grader")

    graph.add_conditional_edges(
        "grader",
        after_grader,
        {"generate": "generate", "web_search_fallback": "web_search_fallback"},
    )

    graph.add_edge("web_search_fallback", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
