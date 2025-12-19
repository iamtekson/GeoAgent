# -*- coding: utf-8 -*-
"""
General mode agent for conversational GIS assistance.
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from ..llm.client import LLMClient
from ..prompts.system import GENERAL_SYSTEM_PROMPT


@dataclass
class ChatMessage:
    """Represents a chat message."""

    role: str  # "user" or "assistant"
    content: str


@dataclass
class AgentState:
    """State for the general mode agent."""

    messages: List[ChatMessage] = field(default_factory=list)
    current_query: str = ""
    response: str = ""
    error: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    def to_message_dicts(self) -> List[Dict[str, str]]:
        """Convert messages to format expected by LLM."""
        result = [
            {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
        ]
        for msg in self.messages:
            result.append({"role": msg.role, "content": msg.content})
        return result

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the history."""
        self.messages.append(ChatMessage(role=role, content=content))

    def get_history_summary(self, max_messages: int = 5) -> List[ChatMessage]:
        """Get recent messages for context."""
        return self.messages[-max_messages:]


class GeneralModeAgent:
    """Agent for general purpose conversation with geospatial context."""

    def __init__(self, llm_client: LLMClient):
        """
        Initialize the general mode agent.

        :param llm_client: LLM client instance
        """
        self.llm_client = llm_client
        self.state = AgentState()

    def process_query(
        self, query: str, temperature: float = 0.7, max_tokens: Optional[int] = None
    ) -> str:
        """
        Process a user query and return a response.

        :param query: User query
        :param temperature: Temperature for generation
        :param max_tokens: Maximum tokens to generate
        :return: Response from the LLM
        """
        try:
            # Update state
            self.state.current_query = query
            self.state.temperature = temperature
            self.state.max_tokens = max_tokens

            # Add user message
            self.state.add_message("user", query)

            # Prepare messages for LLM
            messages = self.state.to_message_dicts()

            # Get response from LLM
            response = self.llm_client.query(
                messages, temperature=temperature, max_tokens=max_tokens
            )

            # Add assistant message
            self.state.add_message("assistant", response)
            self.state.response = response
            self.state.error = None

            return response

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self.state.error = error_msg
            return error_msg

    def clear_history(self) -> None:
        """Clear chat history."""
        self.state.messages.clear()
        self.state.response = ""
        self.state.error = None

    def get_history(self) -> List[Dict[str, str]]:
        """Get chat history as list of dicts."""
        return [
            {"role": msg.role, "content": msg.content} for msg in self.state.messages
        ]

    def get_recent_history(self, max_messages: int = 5) -> List[Dict[str, str]]:
        """Get recent chat history."""
        recent = self.state.get_history_summary(max_messages)
        return [{"role": msg.role, "content": msg.content} for msg in recent]


class LangGraphGeneralAgent:
    """LangGraph-based general mode agent (for future expansion)."""

    def __init__(self, llm_client: LLMClient):
        """
        Initialize LangGraph agent.

        :param llm_client: LLM client instance
        """
        self.llm_client = llm_client
        self.state = AgentState()

        # This is a placeholder for future LangGraph implementation
        # When you need agentic features with tool use, we'll expand this

    def _process_node(self, state: AgentState) -> AgentState:
        """LangGraph node for processing queries."""
        messages = state.to_message_dicts()
        response = self.llm_client.query(
            messages, temperature=state.temperature, max_tokens=state.max_tokens
        )
        state.add_message("assistant", response)
        state.response = response
        return state

    def process_query(
        self, query: str, temperature: float = 0.7, max_tokens: Optional[int] = None
    ) -> str:
        """Process query using LangGraph."""
        self.state.current_query = query
        self.state.temperature = temperature
        self.state.max_tokens = max_tokens
        self.state.add_message("user", query)

        # Process
        self.state = self._process_node(self.state)
        return self.state.response

    def clear_history(self) -> None:
        """Clear chat history."""
        self.state.messages.clear()
        self.state.response = ""
        self.state.error = None
