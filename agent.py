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
from langchain.prompts import PromptTemplate

from scripts.browser_console import console_log

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

def _retriever_tool_func(query: str) -> str:
    """Run the retriever and return a short joined text result.

    We intentionally return a plain string so the agent can observe the
    retrieved content. Limit to the top-k results for brevity.
    """
    try:
        docs = retriever.get_relevant_documents(query)
    except AttributeError:
        # Some retriever implementations expose `get_documents` or `retrieve` —
        # try common alternatives for broader compatibility.
        if hasattr(retriever, "get_documents"):
            docs = retriever.get_documents(query)
        elif hasattr(retriever, "retrieve"):
            docs = retriever.retrieve(query)
        else:
            raise

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
else:
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

# Determine whether we have a usable agent executor (some LangChain versions return
# an AgentExecutor via initialize_agent directly). If `agent_executor` exists above,
# we'll use it; otherwise fall back to a simple retrieval + LLM call.
_USE_AGENT_EXECUTOR = 'agent_executor' in globals()


def _get_docs_for_query(q: str, limit: int = 4):
    try:
        docs = retriever.get_relevant_documents(q)
    except AttributeError:
        if hasattr(retriever, "get_documents"):
            docs = retriever.get_documents(q)
        elif hasattr(retriever, "retrieve"):
            docs = retriever.retrieve(q)
        else:
            raise
    return docs[:limit]


def generate_quotation(query: str) -> str:
    """Run the agent if available; otherwise run a simple retrieve+LLM fallback.

    The fallback gathers the top documents and asks the LLM to answer concisely
    using those documents as context. This keeps functionality working even when
    LangChain agent factories aren't available in the runtime.
    """
    if _USE_AGENT_EXECUTOR:
        try:
            response = agent_executor.invoke({"input": query})
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