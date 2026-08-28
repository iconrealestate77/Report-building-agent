import uuid
import json
import os
from datetime import datetime, timezone

from .agent import create_workflow
from .schemas import AnswerResponse

os.makedirs("sessions", exist_ok=True)

class DocumentAssistant:
    def __init__(self, workflow=None):
        self.workflow = workflow or create_workflow()

    def ask(self, user_input: str, document_context: str, session_id: str = None):
        session_id = session_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": session_id}}

        state = {
            "user_input": user_input,
            "document_context": document_context,
            "intent": None,
            "tool_calls_made": [],
            "final_response": None,
        }
        result = self.workflow.invoke(state, config=config)
        response: AnswerResponse = result["final_response"]

        self._log_session(session_id, user_input, result["intent"], response)
        return response, session_id

    def _log_session(self, session_id, user_input, intent, response):
        log_path = f"sessions/{session_id}.jsonl"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_input": user_input,
            "intent": intent.model_dump() if intent else None,
            "response": response.model_dump(),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
