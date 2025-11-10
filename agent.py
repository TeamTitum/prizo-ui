import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.retrievers import AzureAISearchRetriever
# `create_retriever_tool` may not be available in all LangChain versions (Azure Web Apps
# build sometimes has a different package version). Use `Tool` to wrap the retriever
# directly which is compatible with a wider range of LangChain releases.
# Some LangChain releases don't expose a stable `Tool` class in the same place.
# To be robust across versions, define a minimal duck-typed Tool-like wrapper
# here and use it when the installed LangChain doesn't provide the class.
try:
    from langchain.tools import Tool  # type: ignore
except Exception:
    Tool = None
# Agent factory: try to import create_react_agent (older/newer LangChain),
# otherwise fall back to initialize_agent + AgentType which exists in other releases.
try:
    from langchain.agents import create_react_agent, AgentExecutor
    _USE_CREATE_REACT = True
    _USE_INITIALIZE = False
except Exception:
    try:
        from langchain.agents import initialize_agent, AgentType, AgentExecutor
        _USE_CREATE_REACT = False
        _USE_INITIALIZE = True
    except Exception:
        # No agent factories available in this LangChain build — we'll
        # operate in fallback mode (retriever + direct LLM calls).
        _USE_CREATE_REACT = False
        _USE_INITIALIZE = False
        AgentExecutor = None
        initialize_agent = None
        AgentType = None
# PromptTemplate: some LangChain builds don't expose `langchain.prompts`.
# Try importing and fall back to a minimal local implementation that supports
# the methods we use (from_template and simple formatting).
try:
    from langchain.prompts import PromptTemplate
except Exception:
    class PromptTemplate:
        """Minimal fallback for LangChain's PromptTemplate.

        Only implements `from_template` (class method) and `format` for
        basic Python str.format-style substitution. This keeps the agent
        prompt usage working in environments with older/newpatched LangChain
        packages.
        """

        def __init__(self, template: str):
            self.template = template

        @classmethod
        def from_template(cls, template: str):
            return cls(template)

        def format(self, **kwargs) -> str:
            try:
                return self.template.format(**kwargs)
            except Exception:
                # If formatting fails, return the raw template for robustness
                return self.template

from scripts.browser_console import console_log
import asyncio

# Print LangChain version to logs to help diagnosing runtime API differences
try:
    import langchain
    print("langchain version:", getattr(langchain, "__version__", "unknown"))
    try:
        console_log(f"langchain version: {getattr(langchain, '__version__', 'unknown')}", level="info")
    except Exception:
        pass
except Exception:
    print("langchain not importable at module import time")

load_dotenv()

# Initialize Azure OpenAI LLM with stop parameter explicitly set to None
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
    temperature=0.9,
    max_tokens=800,
   
)

# Initialize retriever synchronously
retriever = AzureAISearchRetriever(
    content_key="content",
    top_k=4,
    index_name=os.getenv('AZURE_SEARCH_INDEX_NAME'),
    service_name=os.getenv('AZURE_SEARCH_SERVICE_NAME'),
    api_key=os.getenv('AZURE_SEARCH_KEY'),
    api_version="2024-05-01-preview"
)

# Debug: Test retriever
print("Inside the agent.py")
console_log("Inside the agent.py", level="info")

# Diagnostic: log retriever type and available attributes to help debug API differences
try:
    retriever_attrs = [a for a in dir(retriever) if not a.startswith("__")]
    console_log({"retriever_type": str(type(retriever)), "attrs": retriever_attrs}, level="info")
    print("Retriever type:", type(retriever))
    print("Retriever attrs:", retriever_attrs)
except Exception as _e:
    # Best-effort diagnostics; don't fail startup
    print("Could not probe retriever attributes:", _e)

def _retriever_tool_func(query: str) -> str:
    """Run the retriever and return a short joined text result.

    We intentionally return a plain string so the agent can observe the
    retrieved content. Limit to the top-k results for brevity.
    """
    # Dynamically probe for the correct retrieval method and call it.
    # Try a broad list of candidate method names (sync + async). For async
    # methods use asyncio.run to execute them.
    candidate_methods = [
        "get_relevant_documents",
        "aget_relevant_documents",
        "get_documents",
        "aget_documents",
        "retrieve",
        "aretrieve",
        # Additional names used by some retriever implementations
        "search",
        "search_documents",
        "search_results",
        "search_with_relevance",
        "search_results_with_scores",
    ]

    def _normalize_result(res):
        # If it's already a list of docs, return it
        if isinstance(res, (list, tuple)):
            return list(res)
        # If object has .documents or .results attribute, extract
        if hasattr(res, "documents"):
            return list(getattr(res, "documents"))
        if hasattr(res, "results"):
            return list(getattr(res, "results"))
        # If it's a dict with 'documents' key
        if isinstance(res, dict) and "documents" in res:
            return list(res["documents"])
        # Fallback: wrap the returned object
        return [res]

    last_exception = None
    for name in candidate_methods:
        if hasattr(retriever, name):
            method = getattr(retriever, name)
            try:
                if name.startswith("a"):
                    res = asyncio.run(method(query))
                else:
                    # Try common signatures defensively
                    try:
                        res = method(query)
                    except TypeError:
                        # Some methods accept (query, top_k) or (query, k)
                        top_k = getattr(retriever, "top_k", 4)
                        try:
                            res = method(query, top_k)
                        except TypeError:
                            try:
                                res = method(query, k=top_k)
                            except TypeError:
                                # last resort: call without args
                                res = method()
                docs = _normalize_result(res)
                # Ensure docs is a list-like of document-like objects
                return docs
            except Exception as e:
                last_exception = e
                # try next candidate
                continue

    # If we get here nothing worked; raise a helpful error including last exception
    msg = (
        "Retriever does not expose a compatible retrieval method. Checked: "
        + ", ".join(candidate_methods)
    )
    if last_exception:
        msg += f"; last error: {last_exception}"
    raise AttributeError(msg)

    # Take up to top_k results (AzureAISearchRetriever already respects top_k,
    # but defensively limit here)
    texts = []
    for d in docs[:4]:
        # Document objects typically have `page_content` or `content` attributes
        text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        texts.append(text)
    return "\n---\n".join(texts)


if Tool is not None:
    retrieval_tool = Tool(
        name="hotel_docs_retriever",
        func=_retriever_tool_func,
        description="Searches hotel documents for details like rooms, availability, meals, pricing, and amenities.",
    )
else:
    # Minimal duck-typed replacement for LangChain's Tool so older/newer
    # LangChain builds that don't export Tool still work.
    class SimpleTool:
        def __init__(self, name, func, description=""):
            self.name = name
            self.func = func
            self.description = description

        def __call__(self, *args, **kwargs):
            return self.func(*args, **kwargs)

    retrieval_tool = SimpleTool(
        name="hotel_docs_retriever",
        func=_retriever_tool_func,
        description="Searches hotel documents for details like rooms, availability, meals, pricing, and amenities.",
    )

# Define the ReAct prompt template
prompt = PromptTemplate.from_template("""
You are Arabiers AI Agent, a concise hotel and tourism expert.
Answer the following questions as best you can.
Use the hotel_docs_retriever tool to fetch information from the provided documents when needed.
You have access to the following tools:

{tools}
Include the document reference if available. 
Use the following format:
You may answer general answers as well. 
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}

If no tools are needed, write the Final Answer directly and do NOT emit any 'Action' or 'Action Input' lines.
""")

# Set up tools
tools = [retrieval_tool]

# Create the ReAct agent (robust to different LangChain versions)
if _USE_CREATE_REACT:
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=40,
        max_execution_time=120,
    )
elif _USE_INITIALIZE:
    # initialize_agent returns an AgentExecutor-like object already
    # Use ZERO_SHOT_REACT_DESCRIPTION to approximate ReAct behavior.
    agent_executor = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=40,
        max_execution_time=120,
    )
else:
    # No agent factory available; run in fallback mode (agent_executor stays None)
    agent_executor = None

# Log which agent mode we're running in so deployment logs make the chosen path clear
if _USE_CREATE_REACT:
    _agent_mode = "create_react_agent"
elif _USE_INITIALIZE:
    _agent_mode = "initialize_agent"
else:
    _agent_mode = "fallback_retriever_llm"

print(f"Agent mode: {_agent_mode}")
try:
    console_log(f"Agent mode: {_agent_mode}", level="info")
except Exception:
    # Best-effort logging; don't fail startup if console logging isn't available
    pass

# Determine whether we have a usable agent executor (some LangChain versions return
# an AgentExecutor via initialize_agent directly). If `agent_executor` exists above
# and is not None, we'll use it; otherwise fall back to a simple retrieval + LLM call.
_USE_AGENT_EXECUTOR = ('agent_executor' in globals()) and (globals().get('agent_executor') is not None)


def _get_docs_for_query(q: str, limit: int = 4):
    # Reuse the same compatibility logic as _retriever_tool_func.
    if hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(q)
    elif hasattr(retriever, "aget_relevant_documents"):
        docs = asyncio.run(retriever.aget_relevant_documents(q))
    elif hasattr(retriever, "get_documents"):
        docs = retriever.get_documents(q)
    elif hasattr(retriever, "aget_documents"):
        docs = asyncio.run(retriever.aget_documents(q))
    elif hasattr(retriever, "retrieve"):
        docs = retriever.retrieve(q)
    elif hasattr(retriever, "aretrieve"):
        docs = asyncio.run(retriever.aretrieve(q))
    else:
        raise AttributeError(
            "Retriever does not expose a compatible retrieval method. "
            "Checked: get_relevant_documents, aget_relevant_documents, get_documents, "
            "aget_documents, retrieve, aretrieve"
        )
    return docs[:limit]


def generate_quotation(query: str) -> str:
    """Run the agent if available; otherwise run a simple retrieve+LLM fallback.

    The fallback gathers the top documents and asks the LLM to answer concisely
    using those documents as context. This keeps functionality working even when
    LangChain agent factories aren't available in the runtime.
    """
    if _USE_AGENT_EXECUTOR:
        try:
            # Agent executors across LangChain versions expose different call APIs.
            # Try common invocation methods in order: invoke(dict), run(str), callable()
            if hasattr(agent_executor, "invoke"):
                response = agent_executor.invoke({"input": query})
            elif hasattr(agent_executor, "run"):
                # run usually accepts a plain string
                response = agent_executor.run(query)
            elif callable(agent_executor):
                # some initializers return a callable
                response = agent_executor({"input": query})
            else:
                raise RuntimeError("Agent executor found but has no known call method")
            console_log("response from agent_executor", level="info")
            console_log(response, level="info")
            if isinstance(response, dict):
                output = response.get("output") or response.get("output_text") or str(response)
            else:
                output = str(response)
            print("Bot:", output)
            return output
        except Exception as e:
            console_log(str(e), level="error")
            print("Agent error:", e)
            return f"Arabiers AI Agent encountered an issue: {str(e)}. Please try rephrasing your question."

    # Fallback: retrieve top documents and ask the LLM directly
    try:
        docs = _get_docs_for_query(query)
        texts = []
        for d in docs:
            texts.append(getattr(d, 'page_content', None) or getattr(d, 'content', None) or str(d))
        context = "\n---\n".join(texts) if texts else "(no documents found)"

        prompt_text = (
            "You are Arabiers AI Agent, a concise hotel and tourism expert. "
            "Answer the user's question using the provided context from hotel documents. "
            "If the information isn't in the documents, answer concisely from general knowledge.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer concisely:"
        )

        # Use the LLM directly. Most LangChain LLM wrappers expose `predict`.
        try:
            output = llm.predict(prompt_text)
        except Exception:
            # Fall back to generate if predict isn't available
            res = llm.generate([prompt_text])
            # `generate` returns an LLMResult; try to extract text
            try:
                output = res.generations[0][0].text
            except Exception:
                output = str(res)

        console_log(output, level="info")
        print("Bot (fallback):", output)
        return output
    except Exception as e:
        console_log(str(e), level="error")
        print("Fallback error:", e)
        return f"Arabiers AI Agent encountered an issue: {str(e)}. Please try rephrasing your question."