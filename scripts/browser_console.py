import json
from typing import Any
import json
import streamlit as st


def _to_payload(obj: Any) -> str:
    """Serialize Python object to a JSON string safe for embedding in JS.

    Uses ensure_ascii=False to keep unicode readable; callers embedding in HTML
    should pass the result through a JS context (we don't eval anything here).
    """
    return json.dumps(obj, ensure_ascii=False)


def script_for(obj: Any) -> str:
    """Return an HTML <script> tag that logs the given Python object to the browser console.

    Example:
        st.markdown(script_for({"a": 1}), unsafe_allow_html=True)

    This function does not itself call Streamlit so it's easy to unit-test.
    """
    payload = _to_payload(obj)
    return f"<script>console.log({payload});</script>"


def send(obj: Any) -> None:
    """Convenience helper that writes the console.log script into the Streamlit page.

    This calls `st.markdown(..., unsafe_allow_html=True)` so you can call it directly
    from `app.py` as:
        from scripts.browser_console import send
        send({"msg": "hello"})
    """
    st.markdown(script_for(obj), unsafe_allow_html=True)


def file_to_console(path: str) -> None:
    """Read a text file and send its contents (and path) to the browser console.

    If the file can't be read an exception is propagated so callers can handle it.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    send({"path": path, "content": content})

def console_log(message: Any, level: str = "log"):
    """
    Print a Python object to the browser console.

    Accepts any JSON-serializable object. This safely serializes the object
    to JSON and emits a console.<level>(...) call so the browser shows the
    structured object instead of failing when a non-string (e.g. dict)
    is passed.

    level: 'log' | 'warn' | 'error' | 'info'
    """
    try:
        payload = json.dumps(message, ensure_ascii=False)
    except Exception:
        # Fallback: stringify the message
        payload = json.dumps(str(message))

    script = f"""
    <script>
    try {{
        console.{level}({payload});
    }} catch(e) {{
        console.error('[STREAMLIT DEBUG] Failed to log payload', e);
    }}
    </script>
    """
    st.components.v1.html(script, height=0, width=0)