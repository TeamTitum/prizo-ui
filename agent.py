import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.retrievers import AzureAISearchRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate

load_dotenv()

# Initialize Azure OpenAI LLM with stop parameter explicitly set to None
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
    temperature=1,
   
)

# Initialize retriever synchronously
retriever = AzureAISearchRetriever(
    content_key="content",
    top_k=3,
    index_name="index-prizo",
    service_name=os.getenv('AZURE_SEARCH_SERVICE_NAME'),
    api_key=os.getenv('AZURE_SEARCH_KEY'),
    api_version="2024-05-01-preview"
)

# Debug: Test retriever
print("Inside the agent.py")


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
    max_iterations=20,  # Limit iterations to prevent infinite loops
    max_execution_time=20  # Limit execution time (in seconds) to prevent hangs
)

# Function to run agent
def generate_quotation(query):
    try:
        response = agent_executor.invoke({"input": query})
        print("Bot:", response['output'])
        return response['output']
    except Exception as e:
        return f"Prizo AI encountered an issue: {str(e)}. Please try rephrasing your question."
    
# Interactive bot loop
# print("Starting interactive bot. Type 'exit' to quit.")
# while True:
#     user_input = input("You: ")
#     if user_input.lower() == 'exit':
#         print("Exiting bot.")
#         break
#     try:
#         response = agent_executor.invoke({"input": user_input})
#         print("Bot:", response['output'])
#     except Exception as e:
#         print(f"Error during execution: {str(e)}")
#         print("Please check your Azure OpenAI deployment or configuration.")