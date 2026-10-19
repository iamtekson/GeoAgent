# -*- coding: utf-8 -*-
"""General mode's conversation window (agents/graph.py)."""
import unittest

import _bootstrap  # noqa: F401  (makes geo_agent importable)
from fake_llm import FakeLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from geo_agent.agents.graph import MAX_CONTEXT_MESSAGES, _trim_context, build_graph_app


def tool_call(i):
    return [{"name": "now_utc", "args": {}, "id": f"call_{i}"}]


class TrimContextTest(unittest.TestCase):
    def test_short_chat_is_kept_whole(self):
        messages = [SystemMessage("s"), HumanMessage("q"), AIMessage("a")]
        self.assertEqual(_trim_context(messages), messages)

    def test_window_starts_at_a_user_message(self):
        messages = [SystemMessage("s")]
        for turn in range(6):
            messages += [
                HumanMessage(f"q{turn}"),
                AIMessage("", tool_calls=tool_call(turn)),
                ToolMessage("result", tool_call_id=f"call_{turn}"),
                AIMessage("answer"),
            ]
        trimmed = _trim_context(messages)
        self.assertIsInstance(trimmed[0], SystemMessage)
        self.assertIsInstance(trimmed[1], HumanMessage)
        self.assertLessEqual(len(trimmed) - 1, MAX_CONTEXT_MESSAGES)
        self.assertEqual(trimmed[-1], messages[-1])

    def test_a_long_current_request_is_kept_whole(self):
        messages = [SystemMessage("s"), HumanMessage("earlier"), AIMessage("a"), HumanMessage("now")]
        for i in range(MAX_CONTEXT_MESSAGES):
            messages += [AIMessage("", tool_calls=tool_call(i)), ToolMessage("r", tool_call_id=f"call_{i}")]
        trimmed = _trim_context(messages)
        self.assertEqual(trimmed[1].content, "now")


class GeneralModeHistoryTest(unittest.TestCase):
    def test_every_llm_call_opens_with_a_user_message(self):
        # The chat that failed with Gemini ("function call turn comes immediately
        # after a user turn ..."): the third request passes the window mid-way
        script = [tool_call(0), None] + [tool_call(i) for i in range(1, 7)] + [None, tool_call(7), None]
        llm = FakeLLM({}, tool_calls=script)
        app = build_graph_app(llm)
        config = {"configurable": {"thread_id": "history-test"}, "recursion_limit": 100}
        app.invoke({"messages": [SystemMessage("system"), HumanMessage("load osm map")]}, config)
        app.invoke({"messages": [HumanMessage("zoom to kathmandu")]}, config)
        app.invoke({"messages": [HumanMessage("roads near syangja")]}, config)

        self.assertGreater(len(app.get_state(config).values["messages"]), MAX_CONTEXT_MESSAGES + 1)
        for _name, messages in llm.calls:
            first = next(m for m in messages if not isinstance(m, SystemMessage))
            self.assertIsInstance(first, HumanMessage)


if __name__ == "__main__":
    unittest.main()
