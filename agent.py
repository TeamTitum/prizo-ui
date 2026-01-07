import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI  # This is the dedicated Azure class

from langchain_community.retrievers import AzureAISearchRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from scripts.browser_console import console_log

load_dotenv()

# Configurable params
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
TOP_K = int(os.getenv("RETRIEVER_TOP_K", "4"))

# Initialize LLM – Latest recommended way for Azure OpenAI

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
)

# Initialize retriever – Still the current integration
retriever = AzureAISearchRetriever(
    content_key="content",
    top_k=TOP_K,
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
    service_name=os.getenv("AZURE_SEARCH_SERVICE_NAME"),
    api_key=os.getenv("AZURE_SEARCH_KEY"),
    api_version=os.getenv("AZURE_SEARCH_API_VERSION", "2024-05-01-preview"),
)

# Modern RAG prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are Arabiers AI Agent, a concise hotel and tourism expert for Sri Lanka. "
     "Answer the user's question using only the provided context from hotel documents. "
     "If the information isn't available in the context, say so briefly and use general knowledge only as fallback. "
     "Be concise and professional."),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
])

# Format documents for context
def format_docs(docs):
    return "\n---\n".join(doc.page_content for doc in docs)

# Build the modern LCEL chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def generate_quotation(query: str) -> str:
    """Retrieve relevant documents and generate a concise quotation/response."""
    try:
        # Invoke the chain (supports streaming if needed later)
        response = chain.invoke(query)
        console_log({"final_answer": response}, level="info")
        return response
    except Exception as e:
        console_log({"chain_error": str(e)}, level="error")
        return f"Arabiers AI Agent encountered an issue: {e}"


if __name__ == "__main__":
    # Quick smoke test
    print(generate_quotation("Who are you?"))