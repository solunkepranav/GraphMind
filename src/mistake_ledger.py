import os
import json
import datetime
from src import config

class MistakeLedger:
    def __init__(self):
        self.ledger_path = config.MISTAKE_LEDGER_PATH
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        self.load()

    def load(self):
        """Loads the mistakes from the JSON file."""
        if os.path.exists(self.ledger_path):
            try:
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    self.mistakes = json.load(f)
                if not isinstance(self.mistakes, list):
                    self.mistakes = []
            except Exception as e:
                print(f"Error loading mistake ledger, starting fresh: {e}")
                self.mistakes = []
        else:
            self.mistakes = []

    def save(self):
        """Saves the mistakes back to the JSON file."""
        try:
            with open(self.ledger_path, "w", encoding="utf-8") as f:
                json.dump(self.mistakes, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving mistake ledger: {e}")

    def log_error(self, component: str, failed_output: str, error_log: str, remediation_instruction: str):
        """
        Logs a new validation mistake / formatting failure.
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "component": component,
            "failed_output": failed_output,
            "error_log": error_log,
            "remediation_instruction": remediation_instruction
        }
        self.mistakes.append(entry)
        self.save()

    def get_errors(self, component: str = None) -> list[dict]:
        """
        Retrieves mistakes, optionally filtered by component name.
        """
        self.load()  # Always reload to sync with updates
        if component:
            return [m for m in self.mistakes if m.get("component") == component]
        return self.mistakes

    def log_mistake(self, query: str, failed_output: str, error_message: str):
        """Logs a mistake for the Q&A component."""
        self.log_error(
            component="QA",
            failed_output=failed_output,
            error_log=error_message,
            remediation_instruction="Do not repeat this output."
        )

    def get_recent_mistakes(self) -> str:
        """Formats the last few mistakes into a clean string for prompt injection."""
        errors = self.get_errors(component="QA")
        if not errors:
            return ""
        # Return last 3 errors formatted
        lines = []
        for idx, err in enumerate(errors[-3:]):
            # Limit the output snippet length to prevent blowing up the prompt
            failed_snippet = str(err['failed_output'])[:200]
            if len(str(err['failed_output'])) > 200:
                failed_snippet += "..."
            lines.append(f"Failure {idx+1}: Output was '{failed_snippet}' because of '{err['error_log']}'.")
        return "\n".join(lines)

    def clear(self):
        """Clears all entries in the ledger."""
        self.mistakes = []
        self.save()
