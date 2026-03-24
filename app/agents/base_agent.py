"""
Base agent class for multi-agent system.
Provides common interface for all agents.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from app.services.groq_service import groq_service


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""

    def __init__(self, name: str, description: str):
        """
        Initialize base agent.

        Args:
            name: Agent identifier
            description: Agent purpose/capabilities
        """
        self.name = name
        self.description = description
        self.groq_service = groq_service

    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's task.

        Args:
            state: Current state of the workflow

        Returns:
            Updated state dictionary
        """
        pass

    async def call_groq(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Helper method to call Groq API with optimized settings.

        Args:
            messages: Conversation messages
            temperature: Sampling parameter
            max_tokens: Max response length

        Returns:
            LLM response content
        """
        response = await self.groq_service.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response["content"]

    def inject_memory_context(self, system_prompt: str, state: Dict[str, Any]) -> str:
        """
        Inject memory context into system prompt.

        Args:
            system_prompt: Original system prompt
            state: Workflow state with memory_prompt

        Returns:
            Enhanced system prompt with memory context
        """
        memory_prompt = state.get("memory_prompt", "")

        if memory_prompt:
            return f"""{system_prompt}

## User Context (from memory):
{memory_prompt}

Use this context to personalize your response, but only when relevant. Do not mention this context explicitly unless necessary."""
        return system_prompt
