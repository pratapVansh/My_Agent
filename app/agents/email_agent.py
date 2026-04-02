"""
Email Agent - Handles email-related tasks.
Returns a TaskEnvelope for structured coordination.
"""
from typing import Dict, Any
from app.agents.base_agent import BaseAgent
from app.agents.state import make_envelope
from app.tools.email_draft_tool import email_draft_tool
from app.memory.memory_manager import memory_manager
from app.services.email_sender_service import email_sender_service


class EmailAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="email",
            description="Email composition, management, and organization"
        )

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input = state["user_input"]
        intent = state.get("detected_intent", "")
        user_id = state.get("user_id", "")

        base_system_prompt = """You are an interactive email agent. You draft, save, and send emails conversationally.

## Your tools
- **email_draft** — compose a personalized email from the user's request. Call this FIRST before sending.
- **send_email** — send an email via Gmail. Requires to_email (full address e.g. name@domain.com), subject, body.
- **save_draft** — save a draft without sending.
- **list_drafts** — show previously saved drafts.
- **save_template** — save a reusable template with {{placeholders}}.
- **list_templates** — list saved templates.

## Step-by-step rules

### When asked to draft / write an email:
1. Call email_draft with the query and recipient name.
2. In your final answer, show the FULL draft clearly:
   **Subject:** <subject>
   ---
   <greeting>
   <body>
   <closing>
   <signature>
3. End with: "Want me to send this? If so, please share the recipient's email address."

### When asked to send an email:
1. If you already have a draft from THIS conversation's tool observations — use it directly in send_email. Do NOT re-draft.
2. If no draft exists — call email_draft first, then send_email.
3. send_email REQUIRES a valid email address (name@domain.com). If you don't have it, show the draft and ask for it.
4. After sending: "✓ Email sent to <address>."

### Multi-turn follow-ups (CRITICAL):
- Check the conversation history messages provided. If the previous assistant turn shows a drafted email, that IS the pending draft.
- If the user is now providing an email address (e.g. "send it to john@company.com", "his email is x@y.com"), call send_email immediately using the subject and body from the previous draft in conversation history.
- Do NOT call email_draft again if a draft already exists in this conversation.

### Other rules:
- Default tone is professional unless the user says otherwise.
- Always be conversational — confirm what you are doing before doing it.
- For "list drafts" or "show drafts" — call list_drafts and present them clearly."""

        async def tool_email_draft(tool_input: Dict[str, Any]):
            query = str(tool_input.get("query") or user_input)
            tone = str(tool_input.get("tone") or "professional")
            recipient_name = str(tool_input.get("recipient_name") or "")
            result = await email_draft_tool.draft_email(
                user_id=user_id,
                query=query,
                tone=tone,
                recipient_name=recipient_name,
            )

            # Strip large fields that overflow the 1200-char observation limit
            # and bury the actual draft content before the model can read it.
            result.pop("rag_context", None)
            result.pop("raw_model_output", None)

            # Auto-save the draft
            draft = result.get("draft", {})
            try:
                draft_id = await memory_manager.save_email_draft(
                    user_id=user_id,
                    subject=draft.get("subject", ""),
                    body=draft.get("body", ""),
                    recipient_name=recipient_name or None,
                    tone=tone,
                    greeting=draft.get("greeting"),
                    closing=draft.get("closing"),
                    signature=draft.get("signature"),
                    context={"original_query": query},
                )
                result["draft_id"] = draft_id
                result["auto_saved"] = True
            except Exception:
                result["auto_saved"] = False
            return result

        async def tool_save_draft(tool_input: Dict[str, Any]):
            draft_id = await memory_manager.save_email_draft(
                user_id=user_id,
                subject=str(tool_input.get("subject", "")),
                body=str(tool_input.get("body", "")),
                recipient_name=tool_input.get("recipient_name"),
                tone=str(tool_input.get("tone", "professional")),
                greeting=tool_input.get("greeting"),
                closing=tool_input.get("closing"),
                signature=tool_input.get("signature"),
                context=tool_input.get("context", {}),
            )
            return {"success": True, "draft_id": draft_id, "message": "Draft saved."}

        async def tool_list_drafts(tool_input: Dict[str, Any]):
            limit = int(tool_input.get("limit", 5))
            drafts = await memory_manager.get_email_drafts(user_id=user_id, limit=limit)
            if not drafts:
                return {"success": True, "count": 0, "drafts": [], "message": "No saved drafts."}
            return {"success": True, "count": len(drafts), "drafts": drafts}

        async def tool_save_template(tool_input: Dict[str, Any]):
            name = str(tool_input.get("name", "custom"))
            subject_template = str(tool_input.get("subject_template", ""))
            body_template = str(tool_input.get("body_template", ""))
            tone = str(tool_input.get("tone", "professional"))
            placeholders = tool_input.get("placeholders", [])
            tmpl_id = await memory_manager.save_email_template(
                user_id=user_id,
                name=name,
                subject_template=subject_template,
                body_template=body_template,
                tone=tone,
                placeholders=placeholders,
            )
            return {"success": True, "template_id": tmpl_id, "name": name}

        async def tool_send_email(tool_input: Dict[str, Any]):
            to_email = str(tool_input.get("to_email") or "").strip()
            subject = str(tool_input.get("subject") or "").strip()
            body = str(tool_input.get("body") or "").strip()
            draft_id = tool_input.get("draft_id")
            cc_raw = tool_input.get("cc")
            cc = [cc_raw] if isinstance(cc_raw, str) and cc_raw else (cc_raw or None)

            if not to_email:
                return {
                    "success": False,
                    "error": "to_email is required. Ask the user for the recipient's email address.",
                }

            # If draft_id supplied but body/subject are missing, load from saved draft
            if draft_id and (not subject or not body):
                drafts = await memory_manager.get_email_drafts(
                    user_id=user_id, limit=50, status="draft"
                )
                match = next((d for d in drafts if d["id"] == draft_id), None)
                if match:
                    subject = subject or match.get("subject", "")
                    body = body or match.get("body", "")

            if not subject or not body:
                return {
                    "success": False,
                    "error": "subject and body are required. Use email_draft first to compose the email.",
                }

            result = await email_sender_service.send_email(
                to_email=to_email,
                subject=subject,
                body=body,
                cc=cc,
            )

            # Mark the draft as sent in the database
            if result.get("success") and draft_id:
                await memory_manager.mark_email_sent(draft_id, user_id, to_email)

            return result

        async def tool_list_templates(tool_input: Dict[str, Any]):
            name_filter = tool_input.get("name")
            templates = await memory_manager.get_email_templates(user_id=user_id, name=name_filter)
            if not templates:
                return {"success": True, "count": 0, "templates": [], "message": "No templates saved yet."}
            return {"success": True, "count": len(templates), "templates": templates}

        tools = {
            "email_draft": {
                "description": (
                    "Generate a personalized structured email draft and auto-save it. "
                    "Returns subject, body, greeting, closing, and draft_id. "
                    "Args: query (str), tone (str: professional/casual/formal), recipient_name (str)."
                ),
                "callable": tool_email_draft,
            },
            "send_email": {
                "description": (
                    "Actually send an email to a recipient via Gmail. "
                    "Call email_draft first to get subject and body, then call this. "
                    "Args: to_email (str, required — full email address like name@domain.com), "
                    "subject (str), body (str), draft_id (str, optional), cc (str, optional)."
                ),
                "callable": tool_send_email,
            },
            "save_draft": {
                "description": (
                    "Manually save an email draft to storage. "
                    "Args: subject (str), body (str), recipient_name (str), tone (str), "
                    "greeting (str), closing (str), signature (str), context (dict)."
                ),
                "callable": tool_save_draft,
            },
            "list_drafts": {
                "description": "List previously saved email drafts. Args: limit (int, default 5).",
                "callable": tool_list_drafts,
            },
            "save_template": {
                "description": (
                    "Save a reusable email template with {{placeholders}}. "
                    "Args: name (str), subject_template (str), body_template (str), "
                    "tone (str), placeholders (list of str)."
                ),
                "callable": tool_save_template,
            },
            "list_templates": {
                "description": "List saved email templates. Args: name (str, optional filter).",
                "callable": tool_list_templates,
            },
        }

        try:
            loop_result = await self.execute_reasoning_loop(
                state=state,
                base_system_prompt=base_system_prompt,
                tools=tools,
                max_iterations=5,   # history check (1) + draft (2) + send (3) + confirm (4) + final (5)
            )

            final_answer = loop_result["final_answer"]
            tools_used = loop_result["tools_used"]

            confidence = self._compute_confidence(
                final_answer=final_answer,
                tools_used=tools_used,
                iterations=loop_result["iterations"],
                max_iterations=5,
                was_retry=bool(state.get("reflect_failure_context")),
            )
            status = "success" if final_answer else "failed"

            envelope = make_envelope(
                agent=self.name,
                goal=intent or user_input,
                inputs={"user_input": user_input, "intent": intent},
                result_content=final_answer,
                status=status,
                confidence=confidence,
                tools_used=tools_used,
                next_actions=["edit_draft"] if "send_email" in tools_used else ["send_email", "edit_draft"] if final_answer else [],
            )

            state["task_result"] = envelope
            state["agent_reasoning"] = (
                f"Email query processed. intent={intent}, "
                f"iterations={loop_result['iterations']}, tools={tools_used}, "
                f"confidence={confidence:.2f}"
            )
            state["current_agent"] = self.name
            if state.get("execution_path") is not None:
                state["execution_path"].append(self.name)

            return state

        except Exception as e:
            envelope = make_envelope(
                agent=self.name,
                goal=intent or user_input,
                inputs={"user_input": user_input},
                result_content="I encountered an error processing your email request.",
                status="failed",
                confidence=0.0,
            )
            state["task_result"] = envelope
            state["error"] = f"Email agent error: {str(e)}"
            return state


# Singleton instance
email_agent = EmailAgent()
