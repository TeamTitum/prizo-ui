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
except Exception:
    from langchain.agents import initialize_agent, AgentType, AgentExecutor
    _USE_CREATE_REACT = False
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

# Function to run agent
def generate_quotation(query):
    try:
        response = agent_executor.invoke({"input": query})
        # response may be a dict or a string depending on agent implementation
        console_log("response from agent_executor", level="info")
        console_log(response, level="info")
        # Normalize output
        if isinstance(response, dict):
            output = response.get("output") or response.get("output_text") or str(response)
        else:
            output = str(response)
        print("Bot:", output)
        return output
    except Exception as e:
        # response may not be defined here if invocation failed; log the exception
        console_log(str(e), level="error")
        print("Agent error:", e)
        return f"Arabiers AI Agent encountered an issue: {str(e)}. Please try rephrasing your question."