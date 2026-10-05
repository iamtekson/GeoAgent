# -*- coding: utf-8 -*-
"""
Token usage accounting for one request.

A LangChain callback handler passed in the graph's run config is inherited by
every LLM call the request makes (decomposition, algorithm selection,
parameter gathering, error analysis, summary, tool loops), so summing usage
in on_llm_end gives the request's total. Providers report usage on the
response message (usage_metadata); older integrations put it in llm_output.
"""
import threading
from typing import Any, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


def _usage_from(response: LLMResult) -> Optional[Dict[str, int]]:
    """(input, output, total) token counts of one LLM response, if reported."""
    try:
        message = response.generations[0][0].message
        usage = getattr(message, "usage_metadata", None)
        if usage:
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            total = int(usage.get("total_tokens") or input_tokens + output_tokens)
            return {"input": input_tokens, "output": output_tokens, "total": total}
    except (IndexError, AttributeError, TypeError, ValueError):
        pass

    llm_output = response.llm_output or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage")
    if isinstance(usage, dict):
        input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total = int(usage.get("total_tokens") or input_tokens + output_tokens)
        if total:
            return {"input": input_tokens, "output": output_tokens, "total": total}
    return None


class TokenUsageTracker(BaseCallbackHandler):
    """Sums the tokens of every LLM call made while handling one request."""

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()  # nodes may run in worker threads
        self.calls = 0
        self.unreported_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage = _usage_from(response)
        with self._lock:
            self.calls += 1
            if usage is None:
                self.unreported_calls += 1
                return
            self.input_tokens += usage["input"]
            self.output_tokens += usage["output"]
            self.total_tokens += usage["total"]

    def summary(self, session_total: Optional[int] = None) -> str:
        """One log line, e.g. 'Tokens used: 1,240 (input 1,100, output 140) in 6 LLM calls'."""
        calls = f"{self.calls} LLM call{'s' if self.calls != 1 else ''}"
        if self.calls == 0:
            text = "Tokens used: none (no LLM calls)"
        elif self.unreported_calls == self.calls:
            text = f"Tokens used: not reported by the provider ({calls})"
        else:
            text = (
                f"Tokens used: {self.total_tokens:,} (input {self.input_tokens:,}, "
                f"output {self.output_tokens:,}) in {calls}"
            )
            if self.unreported_calls:
                text += f"; {self.unreported_calls} reported no usage"
        if session_total is not None:
            text += f" · total this QGIS session: {session_total:,}"
        return text


__all__ = ["TokenUsageTracker"]
