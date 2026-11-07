"""A local OpenAI-compatible server standing in for OpenRouter; no network, no spend."""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _decide(context: dict, seen_recipes: list) -> dict:
    if context["role"] == "reviewer":
        return {
            "interpretation": "Scripted review of the tool-reported result.",
            "prediction_supported": "uncertain",
            "next_step": "Try another setting.",
            "diagnostic_trial_id": None,
        }
    task = context["task"]
    recipe = dict(task["initial_recipe"])
    # A different legal recipe each decision: there is no menu to pick from.
    step = context["history_count"]
    recipe["dropout"] = round(0.05 * (step % 4), 2)
    recipe["weight_decay"] = round(0.001 * step, 4)
    evidence = [context["records"][-1]["trial_id"]] if context["records"] else []
    if context["history_count"] == 2:
        evidence.append("hallucinated-source")  # A citation slip must not cost a trial.
    return {
        "observation": "Scripted observation.",
        "hypothesis": "Scripted hypothesis.",
        "predicted_effect": "Unknown; measure it.",
        "experiment": "Change dropout and weight decay.",
        "recipe": recipe,
        "evidence_ids": evidence,
        "stop": False,
        "diagnostic_trial_id": None,
    }


class FakeLLM:
    """Records every context it receives so tests can inspect what each arm saw."""

    def __init__(self):
        self.contexts: list[dict] = []
        self.bodies: list[dict] = []
        self.failures_to_inject = 0
        self.failures_after = 0  # Let this many requests through before failing.
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                outer.bodies.append(body)
                if outer.failures_to_inject > 0 and len(outer.bodies) > outer.failures_after:
                    outer.failures_to_inject -= 1
                    self._respond(503, {"error": {"message": "scripted outage", "code": 503}})
                    return
                # The system message holds a literal {context_json} placeholder; the real
                # context is in the final user message.
                text = str(body["messages"][-1]["content"])
                start = re.search(r"\[\[ ## context_json ## \]\]\s*", text).end()
                context, _ = json.JSONDecoder().raw_decode(text[start:])
                outer.contexts.append(context)
                content = json.dumps({"output": _decide(context, [])})
                self._respond(
                    200,
                    {
                        "id": "fake",
                        "object": "chat.completion",
                        "model": body["model"],
                        "created": 1,
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": content},
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 50,
                            "total_tokens": 150,
                            "cost": 0.001,
                        },
                    },
                )

            def _respond(self, code, payload):
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/api/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
