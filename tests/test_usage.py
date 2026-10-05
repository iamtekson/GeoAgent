# -*- coding: utf-8 -*-
"""Token usage accounting per request (llm/usage.py)."""
import asyncio
import threading
import unittest

import _bootstrap
from fake_llm import USAGE, FakeLLM, decomposition, params, selection, task
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from geo_agent.agents.graph import build_unified_graph, invoke_app_async
from geo_agent.llm.usage import TokenUsageTracker


def llm_result(usage=None, llm_output=None):
    message = AIMessage(content="x")
    if usage:
        message.usage_metadata = usage
    return LLMResult(generations=[[ChatGeneration(message=message)]], llm_output=llm_output)


def run_like_the_worker(app, messages, tracker):
    """Invoke the app exactly as LLMWorker does: own thread, own event loop."""
    result = {}

    def target():
        loop = asyncio.new_event_loop()
        try:
            result["message"] = loop.run_until_complete(
                invoke_app_async(app, "usage-test", messages, callbacks=[tracker])
            )
        finally:
            loop.close()

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=120)
    return result.get("message")


class TokenUsageTrackerTest(unittest.TestCase):
    def test_sums_usage_metadata(self):
        tracker = TokenUsageTracker()
        tracker.on_llm_end(llm_result({"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}))
        tracker.on_llm_end(llm_result({"input_tokens": 50, "output_tokens": 5, "total_tokens": 55}))
        self.assertEqual((tracker.calls, tracker.input_tokens, tracker.output_tokens, tracker.total_tokens), (2, 150, 25, 175))
        self.assertEqual(
            tracker.summary(session_total=1175),
            "Tokens used: 175 (input 150, output 25) in 2 LLM calls · total this QGIS session: 1,175",
        )

    def test_falls_back_to_llm_output(self):
        tracker = TokenUsageTracker()
        tracker.on_llm_end(llm_result(llm_output={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}))
        self.assertEqual((tracker.input_tokens, tracker.output_tokens, tracker.total_tokens), (7, 3, 10))

    def test_counts_calls_without_usage(self):
        tracker = TokenUsageTracker()
        tracker.on_llm_end(llm_result({"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}))
        tracker.on_llm_end(llm_result())
        self.assertEqual(tracker.summary(), "Tokens used: 12 (input 10, output 2) in 2 LLM calls; 1 reported no usage")

    def test_summary_without_usage_or_calls(self):
        tracker = TokenUsageTracker()
        self.assertEqual(tracker.summary(), "Tokens used: none (no LLM calls)")
        tracker.on_llm_end(llm_result())
        self.assertEqual(tracker.summary(), "Tokens used: not reported by the provider (1 LLM call)")


class RequestUsageTest(unittest.TestCase):
    """Every LLM call of a request reaches the tracker, however deeply nested."""

    def setUp(self):
        _bootstrap.reset_project()

    def assert_counted_every_call(self, llm, tracker):
        self.assertGreater(len(llm.calls), 0)
        self.assertEqual(tracker.calls, len(llm.calls), llm.names())
        self.assertEqual(tracker.total_tokens, USAGE["total_tokens"] * len(llm.calls))
        self.assertEqual(tracker.input_tokens, USAGE["input_tokens"] * len(llm.calls))

    def test_processing_request_counts_all_calls(self):
        # decomposition, routing fallback, selection, gathering (twice, after
        # a failure), error analysis, and the summary: all in sub-graphs
        llm = FakeLLM({
            "TaskDecomposition": [decomposition(task(1, "Extract rivers", "extract", ["extract"]))],
            "RouteDecision": [dict(is_processing_task=True, reason="")],
            "AlgorithmSelection": [selection("native:extractbyattribute")],
            "ParameterGathering": [
                params(INPUT="rivers", FIELD="no_such_field", OPERATOR=9),
                params(INPUT="rivers", FIELD="fid", OPERATOR=9),
            ],
            "ErrorAnalysis": [dict(diagnosis="field missing", failure_kind="bad_parameter")],
        })
        tracker = TokenUsageTracker()
        reply = run_like_the_worker(build_unified_graph(llm, mode="processing"), [HumanMessage(content="extract")], tracker)
        self.assertIsNotNone(reply)
        self.assertEqual(llm.count("ErrorAnalysis"), 1)
        self.assert_counted_every_call(llm, tracker)

    def test_tool_using_task_counts_all_calls(self):
        llm = FakeLLM(
            {"TaskDecomposition": [decomposition(task(1, "List the layers", geo=False))]},
            tool_calls=[[{"name": "now_utc", "args": {}, "id": "c1"}], None],
        )
        tracker = TokenUsageTracker()
        run_like_the_worker(build_unified_graph(llm, mode="processing"), [HumanMessage(content="list")], tracker)
        self.assertEqual(llm.count("invoke"), 3)  # tool call, final reply, summary
        self.assert_counted_every_call(llm, tracker)

    def test_general_mode_counts_all_calls(self):
        llm = FakeLLM({}, tool_calls=[[{"name": "now_utc", "args": {}, "id": "c1"}], None])
        tracker = TokenUsageTracker()
        run_like_the_worker(build_unified_graph(llm, mode="general"), [HumanMessage(content="what time is it?")], tracker)
        self.assertEqual(llm.count("invoke"), 2)  # tool call, then the answer
        self.assert_counted_every_call(llm, tracker)

    def test_provider_without_usage(self):
        llm = FakeLLM({}, usage=None)
        tracker = TokenUsageTracker()
        run_like_the_worker(build_unified_graph(llm, mode="general"), [HumanMessage(content="hi")], tracker)
        self.assertEqual(tracker.summary(), "Tokens used: not reported by the provider (1 LLM call)")


if __name__ == "__main__":
    unittest.main()
