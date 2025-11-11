import os
import asyncio
import time
from typing import List, Any
from dotenv import load_dotenv

# LangChain wrappers (may vary by environment)
from langchain_openai import AzureChatOpenAI
from langchain_community.retrievers import AzureAISearchRetriever

from scripts.browser_console import console_log

load_dotenv()

# Configurable LLM / retriever
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))

# Initialize LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
)

# Initialize retriever
retriever = AzureAISearchRetriever(
    content_key="content",
    top_k=int(os.getenv("RETRIEVER_TOP_K", "4")),
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
    service_name=os.getenv("AZURE_SEARCH_SERVICE_NAME"),
    api_key=os.getenv("AZURE_SEARCH_KEY"),
    api_version=os.getenv("AZURE_SEARCH_API_VERSION", "2024-05-01-preview"),
)

# Diagnostic logging
try:
    import langchain
    console_log({"langchain_version": getattr(langchain, "__version__", "unknown")}, level="info")
except Exception:
    pass

console_log({"retriever_type": str(type(retriever)), "attrs": [a for a in dir(retriever) if not a.startswith("__")]}, level="info")

# Candidate retriever method names to try (sync and async)
CANDIDATE_METHODS = [
    "get_relevant_documents",
    "aget_relevant_documents",
    "get_documents",
    "aget_documents",
    "retrieve",
    "aretrieve",
    "search",
    "search_documents",
    "search_results",
]


def _call_retriever(query: str, top_k: int = 4) -> List[Any]:
    """Call the underlying retriever using a compatible method and normalize results.

    Tries several common sync/async method names, different signatures, and
    normalizes the result into a list of document-like objects.
    """
    last_exc = None

    def _normalize(res):
        if isinstance(res, (list, tuple)):
            return list(res)
        if hasattr(res, "documents"):
            return list(getattr(res, "documents"))
        if hasattr(res, "results"):
            return list(getattr(res, "results"))
        if isinstance(res, dict) and "documents" in res:
            return list(res["documents"])
        return [res]

    for name in CANDIDATE_METHODS:
        if hasattr(retriever, name):
            method = getattr(retriever, name)
            try:
                if name.startswith("a"):
                    res = asyncio.run(method(query))
                else:
                    try:
                        res = method(query)
                    except TypeError:
                        # try common alternative signatures
                        try:
                            res = method(query, top_k)
                        except TypeError:
                            try:
                                res = method(query, k=top_k)
                            except TypeError:
                                res = method()
                docs = _normalize(res)
                return docs
            except Exception as e:
                last_exc = e
                continue

    raise AttributeError(
        "Retriever does not expose a compatible retrieval method. Checked: "
        + ", ".join(CANDIDATE_METHODS)
        + (f"; last error: {last_exc}" if last_exc else "")
    )


def _build_prompt_from_docs(query: str, docs: List[Any]) -> str:
    parts = []
    for d in docs:
        text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        parts.append(text)
    context = "\n---\n".join(parts) if parts else "(no documents found)"
    prompt = (
        "You are Arabiers AI Agent, a concise hotel and tourism expert. "
        "Answer the user's question using the provided context from hotel documents. "
        "If the information isn't in the documents, answer concisely from general knowledge.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer concisely:"
    )
    return prompt


def _extract_final_answer(text: str) -> str:
    # Try to pull the part after 'Final Answer:' if present
    import re

    m = re.search(r"final answer:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def generate_quotation(query: str) -> str:
    """Retrieve documents and ask the LLM to answer concisely. Returns a string."""
    try:
        docs = _call_retriever(query, top_k=int(os.getenv("RETRIEVER_TOP_K", "4")))
    except Exception as e:
        console_log({"retriever_error": str(e)}, level="error")
        docs = []

    prompt = _build_prompt_from_docs(query, docs)

    # Call LLM
    try:
        # Prefer predict if available
        if hasattr(llm, "predict"):
            out = llm.predict(prompt)
        else:
            # Some wrappers expose generate
            res = llm.generate([prompt])
            try:
                out = res.generations[0][0].text
            except Exception:
                out = str(res)
    except Exception as e:
        console_log({"llm_error": str(e)}, level="error")
        return f"Arabiers AI Agent encountered an issue calling the LLM: {e}"

    final = _extract_final_answer(out)
    console_log({"final_answer": final}, level="info")
    return final


if __name__ == "__main__":
    # quick smoke test
    print(generate_quotation("Who are you?"))
