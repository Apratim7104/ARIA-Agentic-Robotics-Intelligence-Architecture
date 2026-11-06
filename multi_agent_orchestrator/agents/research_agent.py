# agents/research_agent.py
"""
Research Agent
==============
Performs web research via Tavily for WEB/EXCEL routes.

State keys READ:
 user_input : str — canonical user query key
 plan : str | None — planner's paragraph plan, preferred as the search query

State keys WRITTEN:
 research_data : str — concatenated search result snippets
 messages : list — updated message log
 current_step : int — incremented step counter
"""
import os
import logging

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from state.schema import OrchestratorState, AgentName, add_message

logger = logging.getLogger(__name__)

try:
    from tavily import TavilyClient
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY") or "placeholder")
    # Patch the underlying requests session to skip SSL verification (Siemens corporate proxy)
    if hasattr(tavily, "session"):
        tavily.session.verify = False
except Exception:
    tavily = None

def research_agent_node(state: OrchestratorState) -> dict:
    """
    LangGraph node — run a Tavily web search and write research_data into state.
    Uses the planner's `plan` as the search query when available (usually a
    cleaner, more targeted phrasing), falling back to the raw `user_input`.
    """
    query = state.get("plan") or state.get("user_input") or state.get("user_query", "")
    logger.info("[research_agent] Searching Tavily for: %s", query[:80])
    if tavily is not None:
        try:
            result = tavily.search(query=query, max_results=5)
            snippets = [r["content"] for r in result.get("results", [])]
            research_data = "\n\n".join(snippets) if snippets else "No results found."
        except Exception as e:
            logger.error("[research_agent] Tavily search failed: %s", e)
            research_data = f"Research unavailable. Using query directly. Error: {str(e)}"
    else:
        research_data = f"Research tool unavailable (tavily not installed). Using query: {query}"

    updated_messages = add_message(
        state=state,
        role=AgentName.RESEARCH,
        content=research_data[:500],
    )

    return {
        "research_data": research_data,
        "messages": updated_messages,
        "current_step": state.get("current_step", 0) + 1,
    }
