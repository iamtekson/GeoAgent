from typing import Optional, Callable, Any
from langchain_core.messages import BaseMessage, AIMessage
from qgis.PyQt.QtCore import QThread, pyqtSignal


class LLMWorker(QThread):
    """Worker thread for non-blocking LLM inference."""

    finished = pyqtSignal()  # Emitted when inference complete
    error = pyqtSignal(str)  # Emitted on error
    result_ready = pyqtSignal(object)  # Emitted result (AIMessage, ToolMessage, etc.)

    def __init__(self, app, thread_id: str, messages, invoke_app_async):
        super().__init__()
        self.app = app
        self.thread_id = thread_id
        self.messages = messages
        self.invoke_app_async = invoke_app_async

    def run(self):
        """Run LLM inference in background thread."""
        try:
            import asyncio

            # Create event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Run async invoke - returns the last message (could be AIMessage, ToolMessage, etc.)
                last_msg = loop.run_until_complete(
                    self.invoke_app_async(
                        self.app, thread_id=self.thread_id, messages=self.messages
                    )
                )
                self.result_ready.emit(last_msg)
            finally:
                loop.close()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()
