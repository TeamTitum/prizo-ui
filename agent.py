import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.retrievers import AzureAISearchRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate

load_dotenv()

# Tunable limits for agent runs. Increase these if your queries require more
# steps or the retriever/LLM calls take longer.
MAX_ITERATIONS = 200
MAX_EXECUTION_TIME = 600  # seconds

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
    top_k=3,
    index_name=os.getenv('AZURE_SEARCH_INDEX_NAME'),
    service_name=os.getenv('AZURE_SEARCH_SERVICE_NAME'),
    api_key=os.getenv('AZURE_SEARCH_KEY'),
    api_version="2024-05-01-preview"
)

# Debug: Test retriever
print("Testing retriever...")
try:
    test_docs = retriever.invoke("Known hotels")  # Test query
    if test_docs:
        print(f"Retrieved {len(test_docs)} documents:")
        for doc in test_docs:
            print(f"- {doc.page_content[:200]}...")
    else:
        print("No documents retrieved. Check index or query.")
except Exception as e:
    print(f"Retriever error: {str(e)}")

retrieval_tool = create_retriever_tool(
    retriever,
    "hotel_docs_retriever",
    "Searches hotel documents for details like rooms, availability, meals, pricing, and amenities."
)

# Define the ReAct prompt template
prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}
Include the document reference if available. 
Use the following format:

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
""")

# Set up tools
tools = [retrieval_tool]

# Create the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create the agent executor with additional error handling
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Set to False if you don't want verbose output
    handle_parsing_errors=True,
    max_iterations=MAX_ITERATIONS,  # Limit iterations to prevent infinite loops
    max_execution_time=MAX_EXECUTION_TIME  # Limit execution time (in seconds) to prevent hangs
)

# Function to run agent
def generate_quotation(query):
    try:
        import time as _time
        start = _time.time()
        # Invoke agent using common API
        if hasattr(agent_executor, "invoke"):
            response = agent_executor.invoke({"input": query})
        elif hasattr(agent_executor, "run"):
            response = agent_executor.run(query)
        elif callable(agent_executor):
            response = agent_executor({"input": query})
        else:
            raise RuntimeError("Agent executor has no known invocation method")

        duration = _time.time() - start
        print(f"Agent run time: {duration:.2f}s")

        # Normalize response
        if isinstance(response, dict):
            out = response.get('output') or response.get('output_text') or str(response)
        else:
            out = str(response)

        # If agent reports stopping due to limits, return helpful guidance
        low = out.lower()
        if "iteration limit" in low or "time limit" in low:
            return (
                out
                + "\n\nNote: The agent stopped due to its iteration/time limit. "
                f"Increase MAX_ITERATIONS ({MAX_ITERATIONS}) or MAX_EXECUTION_TIME ({MAX_EXECUTION_TIME}) in agent.py and redeploy if you expect longer runs."
            )

        print("Bot:", out)
        return out
    except Exception as e:
        em = str(e)
        # Friendly message if the exception indicates an iteration/time stop
        if "iteration limit" in em.lower() or "time limit" in em.lower():
            return (
                f"Arabiers AI Agent encountered an execution limit: {em}. "
                f"Consider increasing MAX_ITERATIONS ({MAX_ITERATIONS}) or MAX_EXECUTION_TIME ({MAX_EXECUTION_TIME}) in agent.py and retrying."
            )
        return f"Prizo AI encountered an issue: {em}. Please try rephrasing your question."
  