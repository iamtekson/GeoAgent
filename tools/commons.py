from langchain_core.tools import tool


@tool
def now_utc() -> str:
    """Return current UTC time."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
