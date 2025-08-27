

"""
Autonomous Assistant Agent with MCP Servers
Combines voice activation, MCP server integration, and advanced AI capabilities
"""

from __future__ import annotations
import os
import re
import json
import operator
import httpx
import psutil
import wikipedia
import asyncio
import time
import sys
import speech_recognition as sr
import pyttsx3
from typing import Annotated, TypedDict, Literal
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.tools import tool, Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()


#  API Keys Importing



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
NEWSAPI_API_KEY = os.getenv("News_API_KEY", "YOUR_NEWSAPI_KEY")
OPENWEATHER_API_KEY = os.getenv("Weather_API_KEY", "YOUR_OPENWEATHER_KEY")
MODEL_NAME = "gpt-4o-mini"


# SandBox for Controlled and Isolated Environment


SANDBOX_DIR = os.path.abspath("./Bitty_sandbox")
RAG_DIR = os.path.abspath("./rag_docs")
INDEX_PATH = os.path.abspath("./rag_index")

os.makedirs(SANDBOX_DIR, exist_ok=True)
os.makedirs(RAG_DIR, exist_ok=True)
os.makedirs(INDEX_PATH, exist_ok=True)




if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
    print(" Set OPENAI_API_KEY for LLM + embeddings.")


# This Block is for Ayncronised MCP Clients



class AsyncSystemClient:
    """Async client for the MCP System Server (CPU, processes, shell, files)."""

    def __init__(self, base_url="http://127.0.0.1:8000", api_key="test-key"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def get_cpu(self):
        """Get current CPU and memory usage."""
        url = f"{self.base_url}/cpu"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params={"api_key": self.api_key})
            return response.json()

    async def list_processes(self):
        """Get list of running processes."""
        url = f"{self.base_url}/ps"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params={"api_key": self.api_key})
            return response.json()

    async def open_app(self, path: str):
        """Open an application by path."""
        url = f"{self.base_url}/open_app"
        async with httpx.AsyncClient() as client:
            response = await client.post(url, params={"path": path, "api_key": self.api_key})
            return response.json()

    async def run_shell(self, cmd: str, confirm: bool = True, timeout: int = 10):
        """Run a shell command."""
        url = f"{self.base_url}/shell"
        payload = {"cmd": cmd, "confirm": confirm, "timeout": timeout}
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, params={"api_key": self.api_key})
            return response.json()

    async def read_file(self, path: str, max_bytes: int = 5 * 1024 * 1024):
        """Read a file."""
        url = f"{self.base_url}/read_file"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params={"path": path, "max_bytes": max_bytes, "api_key": self.api_key})
            return response.json()

    async def write_file(self, path: str, content: str):
        """Write content to a file."""
        url = f"{self.base_url}/write_file"
        async with httpx.AsyncClient() as client:
            response = await client.post(url, params={"path": path, "content": content, "api_key": self.api_key})
            return response.json()

    async def append_file(self, path: str, content: str):
        """Append content to a file."""
        url = f"{self.base_url}/append_file"
        async with httpx.AsyncClient() as client:
            response = await client.post(url, params={"path": path, "content": content, "api_key": self.api_key})
            return response.json()

    async def file_op(self, src: str, dst: str = None, confirm: bool = True):
        """Perform file operations."""
        url = f"{self.base_url}/file_op"
        payload = {"src": src, "dst": dst, "confirm": confirm}
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, params={"api_key": self.api_key})
            return response.json()

class AsyncProductivityClient:
    """Async client for the MCP Productivity Server (Gmail + Calendar + Summarization)."""

    def __init__(self, base_url="http://127.0.0.1:8020"):
        self.base_url = base_url.rstrip("/")

    async def send_email(self, to: str, subject: str, message: str):
        """Send an email using Gmail API."""
        url = f"{self.base_url}/email/send"
        payload = {"to": to, "subject": subject, "message": message}
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            return response.json()

    async def read_emails(self, max_results: int = 5):
        """Fetch last N unread emails."""
        url = f"{self.base_url}/email/read"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params={"max_results": max_results})
            return response.json()

    async def get_calendar_events(self, max_results: int = 5):
        """Fetch upcoming calendar events."""
        url = f"{self.base_url}/calendar/events"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params={"max_results": max_results})
            return response.json()

    async def summarize_and_send(self, raw_text: str, to: str, subject: str):
        """Summarize raw text from user or RAG documents into a professional email and send."""
        url = f"{self.base_url}/email/summarize_and_send"
        payload = {"raw_text": raw_text, "to": to, "subject": subject}
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            return response.json()

class AsyncWebClient:
    """Async client for the MCP Web Search Server (DuckDuckGo)."""

    def __init__(self, base_url="http://127.0.0.1:8010"):
        self.base_url = base_url.rstrip("/")

    async def search_web(self, query: str, max_results: int = 5):
        """Perform a DuckDuckGo web search."""
        url = f"{self.base_url}/search"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params={"query": query, "max_results": max_results})
            return response.json()


# Objects of async MCP clients Classes


sysc = AsyncSystemClient()
prodc = AsyncProductivityClient()
webc = AsyncWebClient()


# Tools Defining


async def cpu_monitor_tool(_: str = "") -> str:
    """Report CPU and memory utilization in percent."""
    cpu = await asyncio.to_thread(psutil.cpu_percent, interval=0.7)
    mem = await asyncio.to_thread(lambda: psutil.virtual_memory().percent)
    return f"CPU usage: {cpu:.1f}%, Memory usage: {mem:.1f}%."

async def safe_delete_tool(rel_path: str) -> str:
    """Delete a file under sandbox. Input is relative path under Bitty_sandbox."""
    if not rel_path:
        return "Provide a relative path under sandbox (e.g., notes/todo.txt)."

    abs_path = os.path.abspath(os.path.join(SANDBOX_DIR, rel_path))

    if not abs_path.startswith(SANDBOX_DIR):
        return "Refused: path escapes sandbox."

    if not os.path.exists(abs_path):
        return f"File not found: {rel_path}"

    try:
        await asyncio.to_thread(os.remove, abs_path)
        return f"Deleted: {rel_path}"
    except Exception as e:
        return f"Failed to delete {rel_path}: {e}"

async def news_tool_tool(country_code: str = "us") -> str:
    """Top headlines via NewsAPI for a country code."""
    if not NEWSAPI_API_KEY or NEWSAPI_API_KEY == "YOUR_NEWSAPI_KEY":
        return "NewsAPI key missing."

    url = "https://newsapi.org/v2/top-headlines"
    params = {"country": country_code, "apiKey": NEWSAPI_API_KEY, "pageSize": 5}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, timeout=10.0)
        data = response.json()

    if data.get("status") != "ok":
        return f"NewsAPI error: {data}"
    titles = [a["title"] for a in data.get("articles", [])]
    return "Top headlines:\n- " + "\n- ".join(titles) if titles else "No headlines."

async def weather_tool_tool(city: str = "Karachi") -> str:
    """Current weather for a city using OpenWeather (metric)."""
    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "YOUR_OPENWEATHER_KEY":
        return "OpenWeather key missing."

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, timeout=10.0)
        data = response.json()

    if "main" not in data:
        return f"Weather error: {data.get('message', data)}"
    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    return f"Weather in {city}: {temp}°C, {desc}."

async def wiki_tool_tool(query: str) -> str:
    """Search Wikipedia for information."""
    try:
        # Use asyncio.to_thread for synchronous wikipedia operations
        search_results = await asyncio.to_thread(wikipedia.search, query, results=1)
        if not search_results:
            return f"No Wikipedia articles found for '{query}'"

        page = await asyncio.to_thread(wikipedia.page, search_results[0])
        summary = await asyncio.to_thread(wikipedia.summary, search_results[0], sentences=3)
        return f"Wikipedia result for '{query}':\n{summary}\n\nFull article: {page.url}"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple matches found for '{query}'. Please be more specific."
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'"
    except Exception as e:
        return f"Wikipedia search failed: {e}"


# This Block has RAG Tools



async def build_rag_index() -> str:
    """Build RAG index from documents in rag_docs directory."""
    docs = []

    for name in os.listdir(RAG_DIR):
        path = os.path.join(RAG_DIR, name)

        if os.path.isfile(path) and any(name.lower().endswith(ext) for ext in [".txt", ".md"]):
            async with asyncio.Lock():
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            chunks = [text[i:i+1500] for i in range(0, len(text), 1500)]
            for idx, chunk in enumerate(chunks):
                docs.append({"page_content": chunk, "metadata": {"source": name, "chunk": idx}})

    if not docs:
        return "No text files found in rag_docs."

    embeddings = OpenAIEmbeddings()
    vs = await asyncio.to_thread(
        FAISS.from_texts,
        [d["page_content"] for d in docs],
        embedding=embeddings,
        metadatas=[d["metadata"] for d in docs]
    )
    await asyncio.to_thread(vs.save_local, INDEX_PATH)
    return f"RAG index built with {len(docs)} chunks."

async def load_vectorstore():
    """Load the vector store for RAG search."""
    embeddings = OpenAIEmbeddings()
    return await asyncio.to_thread(
        FAISS.load_local,
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

async def rag_search_tool(query: str) -> str:
    """Semantic search over local docs."""
    try:
        vs = await load_vectorstore()
        docs = await asyncio.to_thread(vs.similarity_search, query, k=3)
        lines = []
        for d in docs:
            meta = d.metadata or {}
            src = meta.get("source", "?")
            ch = meta.get("chunk", "?")
            lines.append(f"- ({src}#chunk{ch}) {d.page_content[:300]}...")

        return "RAG results:\n" + "\n".join(lines) if lines else "No similar chunks found."
    except Exception as e:
        return f"RAG search failed: {e}"


# This Block has Persistant Memory & Tools


MEMORY_FILE = os.path.join(SANDBOX_DIR, "memory.json")

async def load_memory() -> dict:
    """Load persistent memory from file."""
    if os.path.exists(MEMORY_FILE):
        try:
            async with asyncio.Lock():
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except:
            return {}
    return {}

async def save_memory(mem: dict) -> None:
    """Save persistent memory to file."""
    async with asyncio.Lock():
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)

async def remember_tool(kv: str) -> str:
    """Store key=value fact. Input: key::value"""
    if "::" not in kv:
        return "Use key::value format."
    key, value = [x.strip() for x in kv.split("::", 1)]
    mem = await load_memory()
    mem[key] = value
    await save_memory(mem)
    return f"Remembered {key}."

async def recall_tool(key: str) -> str:
    """Retrieve fact by key"""
    mem = await load_memory()
    return f"{key} = {mem[key]}" if key in mem else f"No value found for {key}."


# state that passes from node to node


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    tool_calls: Annotated[list[str], operator.add]
    memory: Annotated[dict, operator.or_]
    scratchpad: Annotated[str, operator.add]   # <-- FIX
    final_draft: Annotated[str, operator.add]


# Important Functions


async def planner(state: AgentState):
    query = state["messages"][-1].content  # instead of state["input"]
    plan = llm.invoke(f"Plan the steps needed to answer this: {query}")
    return {
        "messages": state["messages"] + [AIMessage(content=f"Plan: {plan}")],
        "tool_calls": [],
        "memory": state.get("memory", {}),
        "scratchpad": "",
        "final_draft": ""
    }

async def memory_manager(state: AgentState) -> AgentState:
    """Update memory with the latest user query."""
    mem = state.get("memory", {})
    query = state["messages"][-1].content
    mem["last_query"] = query

    # Keep all state fields intact
    return {
        "messages": state["messages"] + [AIMessage(content="(Memory updated)")],
        "tool_calls": state.get("tool_calls", []),
        "memory": mem,
        "scratchpad": state.get("scratchpad", ""),
        "final_draft": state.get("final_draft", ""),
    }

async def error_handler(state: AgentState) -> AgentState:
    """Handle tool/server and all kind of errors gracefully."""
    err = state.get("error", state.get("err", "Unknown error"))
    return {
        "messages": state.get("messages", []) + [AIMessage(content=f"Something went wrong: {err}.")],
        "tool_calls": [],            # clear tool calls to avoid re-triggering the same path
        "memory": state.get("memory", {}),
        "scratchpad": state.get("scratchpad", ""),
        "final_draft": state.get("final_draft", ""),
    }


async def self_critic_node(state: AgentState) -> AgentState:
    """Evaluates the final answer and prepares feedback in state."""
    feedback = state.get("critic_feedback", "")
    decision = "retry" if "unsatisfied" in feedback.lower() or "retry" in feedback.lower() else "accept"

    return {
        "messages": state.get("messages", []) + [AIMessage(content=f"Critic decision: {decision}")],
        "tool_calls": [],
        "memory": state.get("memory", {}),
        "scratchpad": state.get("scratchpad", ""),
        "final_draft": state.get("final_draft", ""),
    }

async def self_critic_router(state: AgentState) -> Literal["retry", "accept"]:
    """Router that reads the critic's decision from state and branches."""
    return state.get("critic_decision", "accept")


# This Block is for Routre Decision to Call Concerning Tool


class RouteDecision(BaseModel):
    action: Literal[
        "answer", "tool:cpu_monitor", "tool:safe_delete", "tool:news_tool", 
        "tool:weather_tool", "tool:wiki_tool", "tool:rag_search", "tool:remember", 
        "tool:recall", "mcp:system", "mcp:productivity", "mcp:web", "confirm_delete",
        "to_planner:planner", "memory_manager", "to_error:error_handler", "self_critic"
    ]
    argument: str = Field("", description="Argument for the chosen action/tool")

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.6,
)

structured_llm = llm.with_structured_output(RouteDecision)

SYSTEM_ROUTER = SystemMessage(content=(
    "You are a Bitty autonomous assistant\'s router. Available tools:\n"
    " cpu_monitor: System monitoring\n"
    " safe_delete: File deletion (requires confirmation)\n"
    " news_tool: News headlines\n"
    " weather_tool: Weather information\n"
    " wiki_tool: Wikipedia search\n"
    " rag_search: Local document search\n"
    " remember/recall: Memory operations\n"
    " mcp:system: System operations (CPU, processes, files, shell)\n"
    " mcp:productivity: Email and calendar operations\n"
    " mcp:web: Web search operations\n"
    "For file deletions: NEVER call safe_delete directly. Use confirm_delete instead.\n"
    "Route to appropriate MCP server for system, productivity, or web operations.\n"
    "Plan the actions to perform using planner and Decide when to remember/recall context using memory_manager\n"
    "Handle Tool/Server Error Gracefully usng error_handler \n"
    "If not satisifed with Final Answer, use self critic and route back to llm router \n"
))

async def llm_router(state: AgentState) -> AgentState:
    """Route user requests to appropriate tools or MCP servers."""
    convo = [SYSTEM_ROUTER] + state["messages"]
    decision: RouteDecision = await structured_llm.ainvoke(convo)
    note = f"ROUTER -> {decision.action} | arg={decision.argument}"

    return {
        "messages": [AIMessage(content=note)],
        "tool_calls": [decision.action],
        "memory": state.get("memory", {}),
        "scratchpad": state.get("scratchpad", ""),
        "final_draft": state.get("final_draft", "")
    }


# Human in the Loop


async def confirmation_node(state: AgentState) -> AgentState:
    """Handle file deletion confirmations."""
    last_ai = state["messages"][-1].content if state.get("messages") else ""
    arg = last_ai.split("| arg=", 1)[-1] if "| arg=" in last_ai else ""
    mem = await load_memory()
    mem["pending_delete"] = arg
    await save_memory(mem)

    return {
        "messages": state.get("messages", []) + [
            AIMessage(content="Are you sure you want to delete this? (yes/no)")
        ],
        "tool_calls": [],  
        "memory": state.get("memory", {}),
        "scratchpad": state.get("scratchpad", ""),
        "final_draft": state.get("final_draft", ""),
    }


# This Block Dispatches Concerning Tool


TOOLS_BY_NAME = {
    "cpu_monitor": cpu_monitor_tool,
    "safe_delete": safe_delete_tool,
    "news_tool": news_tool_tool,
    "weather_tool": weather_tool_tool,
    "wiki_tool": wiki_tool_tool,
    "rag_search": rag_search_tool,
    "remember": remember_tool,
    "recall": recall_tool
}

async def tool_dispatcher(state: AgentState) -> AgentState:
    try: 
        """Dispatch tool calls to appropriate async functions."""
        if not state.get("tool_calls"):
            return {
                "messages": [],
                "tool_calls": [],
                "memory": state.get("memory", {}),
                "scratchpad": state.get("scratchpad", ""),
                "final_draft": state.get("final_draft", "")
            }

        decision = state["tool_calls"][-1]

        if not decision.startswith("tool:"):
            return {
                "messages": [],
                "tool_calls": [],
                "memory": state.get("memory", {}),
                "scratchpad": state.get("scratchpad", ""),
                "final_draft": state.get("final_draft", "")
            }

        tool_name = decision.split("tool:", 1)[-1]
        tool_fn = TOOLS_BY_NAME.get(tool_name)

        last_ai = state["messages"][-1].content if state.get("messages") else ""
        argument = last_ai.split("| arg=", 1)[-1] if "| arg=" in last_ai else ""

        try:
            result = await tool_fn(argument)
        except Exception as e:
            result = f"Tool `{tool_name}` failed: {e}"

        return {
            "next": "llm_router",
            "messages": [AIMessage(content=f"[{tool_name}]: {result}")],
            "tool_calls": [],
            "memory": state.get("memory", {}),
            "scratchpad": state.get("scratchpad", ""),
            "final_draft": state.get("final_draft", "")
        }
    except Exception as e:
        state["error"] = str(e)
        return {"next": "error_handler"}


# This Block Dispatches Concerning MCP Server


async def mcp_dispatcher(state: AgentState) -> AgentState:
    try:
        """Dispatch MCP server calls."""
        if not state.get("tool_calls"):
            return {
                "messages": [],
                "tool_calls": [],
                "memory": state.get("memory", {}),
                "scratchpad": state.get("scratchpad", ""),
                "final_draft": state.get("final_draft", "")
            }

        decision = state["tool_calls"][-1]

        if not decision.startswith("mcp:"):
            return {
                "messages": [],
                "tool_calls": [],
                "memory": state.get("memory", {}),
                "scratchpad": state.get("scratchpad", ""),
                "final_draft": state.get("final_draft", "")
            }

        mcp_type = decision.split("mcp:", 1)[-1]
        last_ai = state["messages"][-1].content if state.get("messages") else ""
        argument = last_ai.split("| arg=", 1)[-1] if "| arg=" in last_ai else ""

        #This prinitng part is very imp to see progress on display

        print(f"DEBUG: MCP call - {mcp_type}")
        print(f"DEBUG: Argument - {argument}")

        try:
            if mcp_type == "system":
                if "cpu" in argument.lower() or "memory" in argument.lower():
                    result = await sysc.get_cpu()
                elif "process" in argument.lower():
                    result = await sysc.list_processes()
                elif "open" in argument.lower():
                    result = await sysc.open_app("notepad.exe")
                elif "file" in argument.lower() or "read" in argument.lower():
                    result = await sysc.read_file("test.txt", max_bytes=1024)
                else:
                    result = {"note": "System MCP - specify operation"}

            elif mcp_type == "productivity":
                try:
                    args = json.loads(argument) if argument else {}
                except Exception:
                    return {"error": f"Invalid JSON argument: {argument}"}

                op = args.get("operation")


                if op == "send_email" or ("to" in args and ("message" in args or "body" in args)):
                    result = await prodc.send_email(
                    to=args.get("to"),
                    subject=args.get("subject", ""),
                    message=args.get("message", args.get("body", ""))
                    )


                elif op == "read_email":
                    result = await prodc.read_emails(max_results=args.get("max_results", 5))


                elif op in ["get_calendar", "calendar"]:
                    result = await prodc.get_calendar_events(max_results=args.get("max_results", 5))

                else:
                    result = {"error": f"Unrecognized productivity operation: {argument}"}


            elif mcp_type == "web":
                if "search" in argument.lower():
                    query = argument.replace("search", "").strip()
                    result = await webc.search_web(query)
                else:
                    result = {"note": "Web MCP - specify search query"}

            else:
                result = {"error": f"Unknown MCP type: {mcp_type}"}

        except httpx.ConnectError:
            result = f"MCP `{mcp_type}` server not running. Start with: python start_servers.py"
        except httpx.TimeoutException:
            result = f"MCP `{mcp_type}` request timed out"
        except Exception as e:
            result = f"MCP `{mcp_type}` failed: {e}"

        print(f"DEBUG: Response - {result}")

        return {
            "next": "llm_router", 
            "messages": [AIMessage(content=f"[MCP:{mcp_type}]: {result}")],
            "tool_calls": [],
            "memory": state.get("memory", {}),
            "scratchpad": state.get("scratchpad", ""),
            "final_draft": state.get("final_draft", "")
        }
    except Exception as e:
        return {"next": "error_handler", "error": str(e)}


# This Block Passes the final output



async def final_answer(state: AgentState) -> AgentState:
    """Generate final response to user, surfacing the last clean assistant answer.

    Behavior:
    - Walk messages from newest → oldest looking for a meaningful AIMessage.
    - If an AI message is a router line like "ROUTER -> answer | arg=...", extract the arg text.
    - Skip noise messages (empty, plan, critic, parentheses notes).
    - Return a single clean AIMessage and preserve state fields with correct types.
    """
    messages = state.get("messages", []) or []
    content = None

    for msg in reversed(messages):
        # only consider AIMessage objects
        if not isinstance(msg, AIMessage):
            continue

        txt = (getattr(msg, "content", "") or "").strip()
        if not txt:
            continue

        low = txt.lower()

        if low.startswith("plan:") or "critic decision" in low or txt.startswith("("):
            continue

        # If router produced "ROUTER -> answer | arg=...", extract the argument
        if "router ->" in low and "arg=" in low:
            m = re.search(r"arg=(.*)$", txt, flags=re.IGNORECASE)
            if m:
                content = m.group(1).strip()
                break
            # if no "arg" captured, skip and continue
            continue

        # Otherwise this AIMessage looks like the final textual answer — use it
        content = txt
        break

    if not content:
        content = " No final AIMessage found in state. (Check debug logs for details.)"

    return {
        "messages": [AIMessage(content=content)],
        "tool_calls": [],
        "memory": state.get("memory", {}),        # keep same type (dict)
        "scratchpad": state.get("scratchpad", ""),# keep same type (str)
        "final_draft": state.get("final_draft", "")
    }



# async def final_answer(state: AgentState) -> AgentState:
#     """Generate final response to user, surfacing last clean assistant answer."""

#     content = None
#     messages = state.get("messages", [])

#     # Debug: print roles and snippets to trace what we see
#     for msg in reversed(messages):
#         role = getattr(msg, "type", None) or msg.__class__.__name__
#         txt = (getattr(msg, "content", None) or "").strip()
#         print(f"[DEBUG] {role}: {txt[:60]}")

#         if isinstance(msg, AIMessage):
#             # Skip noise
#             if not txt:
#                 continue
#             if txt.lower().startswith("critic decision"):
#                 continue
#             if txt.lower().startswith("plan:"):
#                 continue
#             if txt.startswith("ROUTER ->") or txt.startswith("("):
#                 continue

#             content = txt
#             break

#     if not content:
#         content = " No AIMessage found in state. (Check debug log above.)"

#     return {
#         "messages": [AIMessage(content=content)],
#         "tool_calls": [],
#         "memory": state.get("memory", {}),
#         "scratchpad": state.get("scratchpad", ""),
#         "final_draft": state.get("final_draft", "")
#     }




# Building Graph

def route_decision(state: AgentState) -> Literal[
    "to_tool", "to_mcp", "to_answer", "to_confirm", "to_error", "to_plan"
]:
    """Route to appropriate node based on the router's last action."""
    # If the router hasn't put an action in tool_calls, we're done.
    if not state.get("tool_calls"):
        return "to_answer"

    last = state["tool_calls"][-1]

    #  Handle 'answer' explicitly (this was missing and caused the loop)
    if last in ("answer", "to_answer", "final"):
        return "to_answer"

    if last in ("self critic", "self checking", "final analysing"):
        return "to_critic"

    # Confirm delete
    if last == "confirm_delete":
        return "to_confirm"

    # Tool/MCP routing
    if last.startswith("tool:"):
        return "to_tool"

    if last.startswith("mcp:"):
        return "to_mcp"

    # Error routing
    if last.startswith("error:") or last == "to_error:error_handler":
        return "to_error"

    # Planner routing (when the router asks to plan)
    if last in ("to_planner:planner", "planner", "to_plan"):
        return "to_plan"

    # Fail-closed to answer so we don't loop forever on unknown actions
    return "to_answer"



graph = StateGraph(AgentState)

# Core nodes
graph.add_node("planner", planner)
graph.add_node("memory_manager", memory_manager)
graph.add_node("llm_router", llm_router)

# Extra functional nodes
graph.add_node("error_handler", error_handler)
graph.add_node("self_critic", self_critic_node)
graph.add_node("confirmation", confirmation_node)
graph.add_node("tool_dispatcher", tool_dispatcher)
graph.add_node("mcp_dispatcher", mcp_dispatcher)
graph.add_node("final_answer", final_answer)

# Entry point
graph.set_entry_point("planner")

# Main flow
graph.add_edge("planner", "memory_manager")
graph.add_edge("memory_manager", "llm_router")

# Router handles branching
graph.add_conditional_edges(
    "llm_router",
    route_decision,
    {
        "to_tool": "tool_dispatcher",
        "to_mcp": "mcp_dispatcher",
        "to_confirm": "confirmation",
        "to_error": "error_handler",
        "to_answer": "final_answer",
        "to_plan": "planner",  
        "to_critic": "self_critic",
    }
)


graph.add_edge("tool_dispatcher", "llm_router")
graph.add_edge("mcp_dispatcher", "llm_router")


graph.add_edge("confirmation", "llm_router")


graph.add_edge("error_handler", "llm_router")


graph.add_edge("llm_router", "self_critic")
graph.add_conditional_edges(
    "self_critic",
    self_critic_router,
    {
        "retry": "llm_router",   
        "accept": "final_answer"          
    }
)
graph.add_edge("final_answer", END)

# Compile
app = graph.compile(checkpointer=MemorySaver())

print(" Async Bitty Autonomous Assistant Graph compiled successfully!")


# This Block is the Main Runner


async def run_bitty(user_text: str, thread_id: str = "demo") -> str:
    """Run the Bitty autonomous assistant with user input."""
    mem = await load_memory()
    state = {
        "messages": [HumanMessage(content=user_text)],
        "tool_calls": [],
        "memory": mem,
        "scratchpad": "",
        "final_draft": ""
    }

    out = await app.ainvoke(
        state,
        config={"recursion_limit": 10000, "configurable": {"thread_id": thread_id}}
    )

    await save_memory(out.get("memory", mem))

    final = None
    for m in reversed(out["messages"]):
        if isinstance(m, AIMessage):
            final = m
            break

    return final.content if final else "No Response Generated"


# This Block is to Manage Voice Functionality


async def speak(text):
    """Text-to-speech using pyttsx3."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()

async def listen_command(prompt="Listening...", timeout=10, phrase_time_limit=20):
    """Listen for voice command and return recognized text."""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print(prompt)

        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            text = recognizer.recognize_google(audio)
            print("Heard:", text)
            return text.lower()
        except sr.UnknownValueError:
            print("Could not understand audio")
            return ""
        except sr.RequestError:
            print("Error: check your internet connection.")
            return ""
        except sr.WaitTimeoutError:
            print("Error: Wait Time Out Error Occurred")
            return ""

async def listen_for_wake_word(wake_word="wake up", active_timeout=15):
    """Wake word detection with active listening mode."""
    active_mode = False
    last_command_time = 0

    while True:
        if not active_mode:
            # Waiting for wake word
            speak("Bitty Autonomous Assistant started. Say 'wake up' to activate.")
            text = await asyncio.to_thread(listen_command, f"Waiting for wake word: '{wake_word}'...")
            if wake_word in text:
                speak("Yes, I'm listening?")
                active_mode = True
                last_command_time = time.time()
        else:
            # Already awake, listen for commands
            while True:
                command = await asyncio.to_thread(listen_command, "Listening for your command...")

                if not command:
                    # No command heard: check if timeout expired
                    if time.time() - last_command_time > active_timeout:
                        print("Going back to sleep, say 'wake up' to activate again")
                        speak("Going back to sleep, say 'wake up' to activate again.")
                        active_mode = False
                        break
                    continue

                last_command_time = time.time()
                print(f"Processing command: {command}")

                # Process command through the assistant
                result = await run_bitty(command)
                print(f"Response: {result}")
                speak(result)

                # Check for exit command
                if result.lower() == "exit" or "goodbye" in result.lower():
                    print("Exiting as requested")
                    speak("Goodbye!")
                    sys.exit("Exiting as User asked")


# This Block is Main


async def async_main():
    """Async main function for testing without voice."""
    print("Bitty Async Autonomous Assistant Ready!")
    print("Type 'exit' to quit")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ("exit", "quit", "goodbye"):
                print("Goodbye!")
                break

            response = await run_bitty(user_input)
            print(f"Assistant: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


# This __name__ thing lets you run code only if its executed in main file


import re
import signal
import logging
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jarvis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print("\n Shutting down gracefully...")
    # Cleanup code here
    sys.exit(0)

def validate_config():
    """Validate that all required configuration is present."""
    issues = []

    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        issues.append(" OPENAI_API_KEY not set in .env file")

    if NEWSAPI_API_KEY == "YOUR_NEWSAPI_KEY":
        issues.append(" News_API_KEY not set in .env file")

    if OPENWEATHER_API_KEY == "YOUR_OPENWEATHER_KEY":
        issues.append(" Weather_API_KEY not set in .env file")

    if not os.path.exists("credentials.json"):
        issues.append(" credentials.json not found (required for email/calendar)")

    if issues:
        print(" Configuration Issues Found:")
        for issue in issues:
            print(f"   {issue}")
        print("\n Fix these issues for full functionality.")
    else:
        print(" All configuration validated successfully!")

async def check_mcp_servers():
    """Check if all MCP servers are running."""
    servers = [
        ("System", "http://127.0.0.1:8000/cpu?api_key=test-key"),
        ("Productivity", "http://127.0.0.1:8020/health"),
        ("Web", "http://127.0.0.1:8010/search?query=test")
    ]

    results = {}
    for name, url in servers:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                results[name] = f" Running (Status: {response.status_code})"
        except Exception as e:
            results[name] = f" Not running: {e}"

    return results

async def retry_mcp_call(func, max_retries=3, delay=1):
    """Retry MCP calls with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await func()
        except httpx.ConnectError as e:
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(delay * (2 ** attempt))
            logger.warning(f"MCP call failed, retrying... (attempt {attempt + 1})")


async def startup():
    validate_config()

    # Create necessary directories and files
    os.makedirs(os.path.join(SANDBOX_DIR, "notes"), exist_ok=True)
    async with asyncio.Lock():
        with open(os.path.join(SANDBOX_DIR, "notes", "todo.txt"), "w") as f:
            f.write("Async Autonomous Assistant initialized")

    # Check MCP servers
    print(" Checking MCP servers...")
    server_status = await check_mcp_servers()
    for server, status in server_status.items():
        print(f"   {server}: {status}")

    # Build RAG index if documents exist
    try:
        await build_rag_index()
        print(" RAG index built successfully")
    except Exception as e:
        print(f" RAG index build failed: {e}")

if __name__ == "__main__":
    #For CleanUp
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Choose interface mode
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--voice":
        print(" Starting voice-activated mode...")
        asyncio.run(startup())
        asyncio.run(listen_for_wake_word("wake up"))

    else:
        print(" Starting text interface mode...")
        asyncio.run(startup())
        asyncio.run(async_main())

