# -*- coding: utf-8 -*-
"""
Scripted stand-in for an LLM, so workflow tests run offline and
deterministically.

FakeLLM is a real LangChain chat model (BaseChatModel): structured output,
tool binding and callbacks (e.g. token usage tracking) go through the same
LangChain machinery as with a real provider. Only the answers are scripted:

- Structured-output calls are answered from a script keyed by schema name
  ("TaskDecomposition", "AlgorithmSelection", "ParameterGathering", ...);
  each entry is a dict of field values, a callable(messages) -> dict, or an
  Exception to raise.
- Plain calls (tool-using tasks and the final summary) return the scripted
  tool calls in order, then the fixed SUMMARY message.

Every response reports `usage` tokens (set usage=None to report none), and
every call is recorded in `calls` as (name, messages).
"""
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, PrivateAttr

SUMMARY = "Summary: workflow finished."
USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}


class FakeLLM(BaseChatModel):
    usage: Optional[dict] = USAGE

    _script: dict = PrivateAttr(default_factory=dict)
    _tool_script: list = PrivateAttr(default_factory=list)
    _calls: list = PrivateAttr(default_factory=list)

    def __init__(self, script: dict, tool_calls: Optional[list] = None, **kwargs: Any):
        """
        Args:
            script: schema name -> list of responses, consumed in order.
            tool_calls: responses for plain calls: each a list of tool calls
                ({"name", "args", "id"}) or None for a final reply.
        """
        super().__init__(**kwargs)
        self._script = {k: list(v) for k, v in script.items()}
        self._tool_script = list(tool_calls or [])

    @property
    def _llm_type(self) -> str:
        return "geoagent-test-fake"

    def bind_tools(self, tools: list, tool_choice: Any = None, **kwargs: Any):
        schemas = [t for t in tools if isinstance(t, type) and issubclass(t, BaseModel)]
        if schemas and tool_choice == "any":  # with_structured_output(schema)
            return self.bind(structured_output=schemas[0].__name__)
        return self.bind()

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        structured_output: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if structured_output:
            self._calls.append((structured_output, messages))
            queue = self._script.get(structured_output)
            if not queue:
                raise RuntimeError(f"FakeLLM: no scripted response for {structured_output}")
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            if callable(item):
                item = item(messages)
            tool_calls = [{"name": structured_output, "args": item, "id": "call_0"}]
            message = AIMessage(content="", tool_calls=tool_calls)
        else:
            self._calls.append(("invoke", messages))
            if self._tool_script:
                step = self._tool_script.pop(0)
                if step:
                    message = AIMessage(content="", tool_calls=step)
                else:
                    message = AIMessage(content="done")
            else:
                message = AIMessage(content=SUMMARY)
        if self.usage:
            message.usage_metadata = dict(self.usage)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # ── call log helpers ────────────────────────────────────────────────────
    @property
    def calls(self) -> list:
        return self._calls

    def names(self) -> List[str]:
        return [name for name, _ in self._calls]

    def count(self, name: str) -> int:
        return self.names().count(name)

    def prompts(self, name: str) -> List[str]:
        """Human-message text of every call of *name*, in order."""
        return [messages[-1].content for n, messages in self._calls if n == name]


def task(task_id, operation, hint="", keywords=(), dependencies=(), geo=None):
    """A TaskDefinition dict; geo=None leaves is_geoprocessing out."""
    definition = dict(
        task_id=task_id,
        operation=operation,
        algorithm_hint=hint,
        search_keywords=list(keywords),
        dependencies=list(dependencies),
        key_parameters={},
    )
    if geo is not None:
        definition["is_geoprocessing"] = geo
    return definition


def decomposition(*tasks):
    return dict(reasoning="", tasks=list(tasks))


def selection(algorithm_id):
    return dict(algorithm_id=algorithm_id, reasoning="", confidence=0.9)


def params(**values):
    values.setdefault("OUTPUT", "TEMPORARY_OUTPUT")
    return dict(parameters=values)
