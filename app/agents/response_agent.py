"""
Response Agent - Formats final output for user.
Returns structured response with display_text and speech_text.
"""
from typing import Dict, Any
import re
from app.agents.base_agent import BaseAgent


_AWAITING_APPROVAL_SPEECH = (
    "It's ready, but I haven't sent it. Details are on screen — say yes to go "
    "ahead, or no to drop it."
)
"""What a turn holding an action says out loud.

Deliberately makes no claim about *what* was prepared. The recipient and the
body are on screen, where they can be read carefully; a spoken paraphrase of
them would be one more place an approval could be bound to the wrong thing."""


class ResponseAgent(BaseAgent):
    """
    Response Agent handles:
    - Formatting task agent outputs
    - Creating display-optimized text
    - Creating speech-optimized text
    - Error handling and fallback responses
    """

    def __init__(self):
        super().__init__(
            name="response",
            description="Formats final responses for display and speech"
        )

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the final response from task results.

        Args:
            state: Workflow state with task_result

        Returns:
            Updated state with display_text and speech_text
        """
        task_result = state.get("task_result")
        error = state.get("error")

        # Handle error cases
        if error:
            state["display_text"] = self._format_error_display(error)
            state["speech_text"] = self._format_error_speech(error)
            state["current_agent"] = self.name
            if state.get("execution_path") is not None:
                state["execution_path"].append(self.name)
            return state

        # Handle missing task result
        if not task_result:
            state["display_text"] = "Something went wrong there. Try again?"
            state["speech_text"] = "Something went wrong there. Try again?"
            state["current_agent"] = self.name
            if state.get("execution_path") is not None:
                state["execution_path"].append(self.name)
            return state

        # Extract metadata — supports both TaskEnvelope and legacy flat dict
        content = task_result.get("result", {}).get("content") or task_result.get("content", "")
        agent_name = task_result.get("agent", "assistant")
        output_mode = (state.get("output_mode") or "user").strip().lower()
        if output_mode not in {"user", "recruiter"}:
            output_mode = "user"
        detected_intent = state.get("detected_intent", "")

        # Format for display (richer formatting)
        state["display_text"] = self._format_for_display(
            self._compose_multi_step(content, state), agent_name
        )

        # ── A held action is not a finished one ──────────────────────────────
        # Speech is summarised aggressively — an email turn is classified
        # "long_generative" and spoken as "Your email draft is ready", which is
        # the right précis of a draft and the wrong one of a send that is
        # waiting for permission. Heard rather than read, that sentence says the
        # work is done and asks for nothing, so a spoken turn could leave an
        # irreversible action outstanding with the user believing it finished.
        #
        # The display text still carries the gateway's full preview — recipient,
        # subject, body — so what is read is unchanged. What is spoken becomes a
        # request rather than a summary, and it is a fixed sentence rather than
        # a generated one: nothing here may describe an action, because
        # describing it inaccurately is the failure being prevented.
        if state.get("pending_actions"):
            state["speech_text"] = "" if output_mode == "recruiter" else (
                _AWAITING_APPROVAL_SPEECH
            )
        else:
            # Format for speech with task awareness
            state["speech_text"] = await self._format_for_speech(
                content=content,
                output_mode=output_mode,
                agent_name=agent_name,
                detected_intent=detected_intent
            )

        state["current_agent"] = self.name
        if state.get("execution_path"):
            state["execution_path"].append(self.name)

        return state

    def _compose_multi_step(self, content: str, state: Dict[str, Any]) -> str:
        """
        Put the earlier steps back into the answer.

        Each step overwrites `task_result`, so a plan's final envelope is the
        *last* step's and nothing else. For "check my attendance and email my
        professor" that meant the user was told "Draft ready" and never saw the
        attendance figure they had asked for — the first half of their request
        was executed correctly and then discarded on the way out.

        `step_results` already holds a summary of every completed step, written
        by `reflect_node` as the plan advances. Nothing new is computed here;
        the earlier results are simply not thrown away.

        Single-step turns — the overwhelming majority — are returned untouched,
        so this cannot change how an ordinary answer reads.
        """
        step_results = state.get("step_results") or {}
        if not step_results:
            return content

        blocks = []
        for key in sorted(step_results, key=lambda k: int(k) if str(k).isdigit() else 0):
            step = step_results[key] or {}
            summary = str(step.get("summary") or "").strip()
            if summary:
                blocks.append(summary)

        if not blocks:
            return content
        if content.strip():
            blocks.append(content.strip())
        return "\n\n".join(blocks)

    def _format_for_display(self, content: str, agent_name: str) -> str:
        """
        Format content for visual display.
        Can include markdown, formatting, etc.

        Args:
            content: Task agent response
            agent_name: Name of task agent

        Returns:
            Display-optimized text
        """
        # Keep original content with minimal processing
        # Display can handle markdown and formatting
        return content

    def _classify_task_type(
        self,
        content: str,
        agent_name: str,
        detected_intent: str
    ) -> str:
        """
        Classify task as 'long_generative' or 'short_informational'.

        Long/Generative: emails, reports, code, long notes, drafts
        Short/Informational: queries, status checks, attendance, schedule

        Returns:
            "long_generative" or "short_informational"
        """
        # Check agent type
        if agent_name in ["email", "email_agent"]:
            return "long_generative"

        # Check for email structure (Subject:, Dear, signatures)
        if self._looks_like_email(content):
            return "long_generative"

        # Check intent keywords
        generative_keywords = ["draft", "write", "compose", "generate", "create", "prepare"]
        informational_keywords = ["check", "show", "what", "when", "status", "find", "search"]

        intent_lower = detected_intent.lower() if detected_intent else ""

        if any(kw in intent_lower for kw in generative_keywords):
            return "long_generative"

        if any(kw in intent_lower for kw in informational_keywords):
            return "short_informational"

        # Check for list structures (often informational)
        if self._looks_like_list(content):
            return "short_informational"

        # Length-based fallback
        word_count = len(content.split())
        if word_count > 200:
            return "long_generative"

        # Default to informational for shorter content
        return "short_informational"

    async def _format_for_speech(
        self,
        content: str,
        output_mode: str,
        agent_name: str,
        detected_intent: str
    ) -> str:
        """
        Format content for speech output matching specification.

        RECRUITER MODE: Always return empty string ""
        USER MODE:
          - Long/generative tasks: Return short confirmation
          - Short/informational tasks: Return full content (cleaned)

        Args:
            content: Task agent response
            output_mode: user or recruiter
            agent_name: Name of task agent
            detected_intent: Detected user intent

        Returns:
            Speech-optimized text or empty string
        """
        # RECRUITER MODE: NEVER generate voice
        if output_mode == "recruiter":
            return ""

        # USER MODE: Classify task type
        task_type = self._classify_task_type(content, agent_name, detected_intent)

        if task_type == "long_generative":
            # Short confirmation for drafts, emails, reports
            confirmations = {
                "email": "Your email draft is ready",
                "email_agent": "Your email draft is ready",
                "job": "I have generated it",
                "job_agent": "Done"
            }
            return confirmations.get(agent_name, "Done")

        else:  # short_informational
            # Return full content for queries, status checks
            cleaned = self._basic_speech_cleanup(content)

            # Limit length for TTS (max ~200 words)
            words = cleaned.split()
            if len(words) > 200:
                return " ".join(words[:200]) + "..."

            return cleaned


    def _looks_like_list(self, text: str) -> bool:
        """Detect list-heavy outputs to avoid reading long enumerations."""
        lines = text.splitlines()
        if not lines:
            return False

        list_lines = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("-", "*", "•")):
                list_lines += 1
                continue
            if re.match(r"^\d+[\.)]\s+", stripped):
                list_lines += 1

        return list_lines >= 4

    def _looks_like_email(self, text: str) -> bool:
        """Detect email draft structure to prevent full readout."""
        lower = text.lower()
        if "subject:" in lower and ("dear " in lower or "hi " in lower):
            return True
        if "best regards" in lower or "sincerely" in lower:
            return True
        return False

    def _basic_speech_cleanup(self, text: str) -> str:
        """
        Basic text cleanup for speech if LLM formatting fails.

        Args:
            text: Original text

        Returns:
            Cleaned text
        """
        # Remove markdown formatting
        text = text.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
        text = text.replace("#", "").replace(">", "").replace("`", "")

        # Remove extra whitespace
        text = " ".join(text.split())

        return text

    def _format_error_display(self, error: str) -> str:
        """
        What is shown when the turn failed.

        Deliberately not a decorated error block. A warning emoji and a bold
        "Error:" label announce a *system fault* — which is how a chat product
        reports trouble and not how an assistant does. The user does not need
        the exception text; they need to know it did not work and that trying
        again is worth it.

        The detail still reaches the logs, where it is useful, rather than the
        screen, where it is noise.
        """
        return "That didn't go through. Try again in a moment."

    def _format_error_speech(self, error: str) -> str:
        """Format error message for speech."""
        return "I encountered an error processing your request. Please try again."


# Singleton instance
response_agent = ResponseAgent()
