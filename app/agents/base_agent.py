"""
Base agent class for multi-agent system.
Provides common interface for all agents.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Awaitable, Optional
import json
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
            return f"""## Memory Context
    {memory_prompt}

    Use this context to keep continuity with prior conversation and personal data when relevant. Do not mention memory sources explicitly unless the user asks.

    {system_prompt}"""
        return system_prompt

    async def execute_reasoning_loop(
        self,
        state: Dict[str, Any],
        base_system_prompt: str,
        tools: Optional[Dict[str, Dict[str, Any]]] = None,
        max_iterations: int = 3,
    ) -> Dict[str, Any]:
        """
        Run a bounded reasoning loop with optional iterative tool usage.

        Expected model outputs (JSON preferred):
        - Tool call: {"type":"tool_call","tool":"name","tool_input":{...}}
        - Final: {"type":"final","content":"...","is_complete":true}

        Returns:
            {
                "final_answer": str,
                "iterations": int,
                "tools_used": [str],
                "trace": [str]
            }
        """
        tools = tools or {}
        user_input = state.get("user_input", "")
        detected_intent = state.get("detected_intent", "")

        tool_guide = "No external tools available."
        if tools:
            tool_lines = []
            for tool_name, tool_info in tools.items():
                desc = tool_info.get("description", "")
                tool_lines.append(f"- {tool_name}: {desc}")
            tool_guide = "Available tools:\n" + "\n".join(tool_lines)

        loop_instructions = f"""
You are in a reasoning loop with a strict maximum of {max_iterations} iterations.

{tool_guide}

At each iteration, respond as valid JSON using ONE of these shapes:
1) Tool call:
{{
  "type": "tool_call",
  "tool": "tool_name",
  "tool_input": {{"key": "value"}}
}}
2) Final answer:
{{
  "type": "final",
  "content": "your final response to user",
  "is_complete": true
}}

Rules:
- Use tools only when needed.
- After receiving tool observations, refine your reasoning.
- Stop as soon as answer is complete.
- Never exceed the max iterations.
"""

        system_prompt = self.inject_memory_context(
            f"{base_system_prompt}\n\n{loop_instructions}",
            state,
        )

        observations: list[str] = []
        trace: list[str] = []
        tools_used: list[str] = []
        final_answer = ""

        for step in range(1, max_iterations + 1):
            observation_block = "\n".join(observations[-6:]) if observations else "(none)"

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Iteration: {step}/{max_iterations}\n"
                        f"Intent: {detected_intent}\n"
                        f"Query: {user_input}\n"
                        f"Tool observations so far:\n{observation_block}"
                    ),
                },
            ]

            raw_response = await self.call_groq(
                messages=messages,
                temperature=0.3,
                max_tokens=700,
            )

            decision = self._parse_reasoning_decision(raw_response)

            if decision["type"] == "tool_call":
                tool_name = decision.get("tool", "")
                tool_input = decision.get("tool_input", {})

                tool_info = tools.get(tool_name)
                if not tool_info:
                    observations.append(f"Tool error: Unknown tool '{tool_name}'.")
                    trace.append(f"step_{step}: unknown_tool:{tool_name}")
                    continue

                tool_callable: Callable[..., Awaitable[Any]] = tool_info["callable"]

                try:
                    result = await tool_callable(tool_input)
                    summarized = self._summarize_tool_result(result)
                    observations.append(f"Tool {tool_name} observation: {summarized}")
                    tools_used.append(tool_name)
                    trace.append(f"step_{step}: tool_call:{tool_name}")
                except Exception as e:
                    observations.append(f"Tool {tool_name} failed: {str(e)}")
                    trace.append(f"step_{step}: tool_error:{tool_name}")

                continue

            # Final answer (or fallback to model response text)
            final_answer = (decision.get("content") or raw_response or "").strip()
            trace.append(f"step_{step}: final")
            if final_answer:
                break

        if not final_answer:
            final_answer = "I could not complete your request confidently. Please try again with more detail."

        # Preserve order but remove duplicates in tool list
        unique_tools_used = list(dict.fromkeys(tools_used))

        return {
            "final_answer": final_answer,
            "iterations": len(trace),
            "tools_used": unique_tools_used,
            "trace": trace,
        }

    def _parse_reasoning_decision(self, response: str) -> Dict[str, Any]:
        """Parse model decision for tool call vs final answer."""
        text = (response or "").strip()

        if not text:
            return {"type": "final", "content": ""}

        cleaned = text
        if cleaned.startswith("```json"):
            cleaned = cleaned.split("```json", 1)[1].rsplit("```", 1)[0].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.split("```", 1)[1].rsplit("```", 1)[0].strip()

        try:
            payload = json.loads(cleaned)
            if not isinstance(payload, dict):
                return {"type": "final", "content": text}

            decision_type = str(payload.get("type", "")).strip().lower()
            tool_name = str(payload.get("tool", payload.get("tool_name", ""))).strip()
            tool_input = payload.get("tool_input", payload.get("parameters", {}))

            if decision_type == "tool_call" and tool_name:
                return {
                    "type": "tool_call",
                    "tool": tool_name,
                    "tool_input": tool_input if isinstance(tool_input, dict) else {},
                }

            content = str(payload.get("content", payload.get("answer", payload.get("final_answer", ""))))
            return {"type": "final", "content": content or text}

        except Exception:
            # If model didn't follow JSON strictly, treat response as final answer.
            return {"type": "final", "content": text}

    def _summarize_tool_result(self, result: Any, max_chars: int = 1200) -> str:
        """Compact tool result for loop observations while keeping salient signal."""
        try:
            payload = result

            if isinstance(result, dict):
                payload = dict(result)

                if isinstance(payload.get("results"), list):
                    trimmed = []
                    for item in payload["results"][:3]:
                        if isinstance(item, dict):
                            trimmed.append(
                                {
                                    "title": item.get("title"),
                                    "url": item.get("url"),
                                    "snippet": item.get("snippet") or item.get("content"),
                                    "rank_score": item.get("rank_score") or item.get("score"),
                                }
                            )
                        else:
                            trimmed.append(item)
                    payload["results"] = trimmed

            text = json.dumps(payload, ensure_ascii=True, default=str)
        except Exception:
            text = str(result)

        if len(text) > max_chars:
            return text[: max_chars - 3] + "..."
        return text
