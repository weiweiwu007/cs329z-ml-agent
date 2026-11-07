"""Local, real-model setup chat. Prepares configs and never imports a trainer.

Run from the project checkout: python -m rsi_ml.intake
The six-turn Covertype conversation ends at a saved, reviewed study protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import uuid
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from itertools import pairwise
from pathlib import Path
from urllib.parse import urlparse

from .intake_model import IntakeModel, IntakeModelError, load_key
from .schemas import ArtifactPolicy, Limits, Policy, Recipe

PRESET = Path("configs/study/mlp-covertype-600s.json")
SOURCE_URL = "https://archive.ics.uci.edu/static/public/31/covertype.zip"
SOURCE_HASH = "89a975c2457cd48e824238ae43c5a3cb762e42c4b4078d9b44a4514055105f6d"
STEPS = ("goal", "dataset", "target", "split", "budget", "review", "ready")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_exclusive_json(path: Path, value: dict) -> None:
    """Publish complete files only; an identical interrupted save is safe to retry."""
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)  # Exclusive creation; never overwrite an existing study.
        except FileExistsError:
            if path.read_bytes() != data:
                raise
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def budget_from_text(message: str) -> tuple[float, int]:
    """Require explicit units; never guess a material budget from a bare number."""
    text = message.lower().replace(",", "")
    if re.search(r"-\s*\d", text):
        raise ValueError("Please use positive compute and parameter limits.")
    if re.search(r"\d\s*e[+-]?\d", text):
        raise ValueError(
            "Please write plain numbers, such as 1000 seconds, rather than scientific notation."
        )
    times = list(
        re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?)\b", text)
    )
    counts = list(
        re.finditer(
            r"(?<![\w.])(\d+(?:\.\d+)?)\s*(k|m|thousand|million)?\s*(?:model\s+)?param(?:eter)?s?\b",
            text,
        )
    )
    if len(times) != 1 or len(counts) != 1:
        raise ValueError(
            "Please give one compute limit and one parameter limit, for example: 600 seconds, 500k parameters."
        )
    time, count = times[0], counts[0]
    seconds = float(time[1]) * (60 if time[2].startswith("m") else 1)
    multipliers = {None: 1, "k": 1000, "thousand": 1000, "m": 1000000, "million": 1000000}
    parameters = float(count[1]) * multipliers[count[2]]
    if not math.isfinite(seconds) or not math.isfinite(parameters):
        raise ValueError("Please use finite, positive resource limits.")
    if seconds <= 0 or parameters < 15751 or not parameters.is_integer():
        raise ValueError(
            "Compute must be positive. The starting MLP needs at least 15,751 parameters."
        )
    if seconds > 86400 or parameters > 100000000:
        raise ValueError("This demo accepts up to 24 hours and 100 million parameters per mission.")
    return seconds, int(parameters)


def validate_protocol(protocol: dict, directory: Path) -> None:
    """Validate the handoff without preparing data, importing Torch, or running jobs."""
    if protocol.get("status") != "approved" or protocol.get("arms") != ["D"]:
        raise ValueError("This demo prepares one reviewed D mission.")
    if set(protocol.get("tasks", {})) != {"covertype"}:
        raise ValueError("This demo uses the registered Covertype preparation pipeline.")
    spec = protocol["tasks"]["covertype"]
    if spec["seeds"] != [2] or spec["device"] != "cuda":
        raise ValueError("The demo uses seed 2 on the GPU runner.")
    template = read_json(directory / spec["template"])
    if template["metric"] != "accuracy" or template["dataset"]["input_dim"] != 54:
        raise ValueError("The Covertype template no longer matches this setup interface.")
    Policy.model_validate(read_json(directory / protocol["policy"]))
    ArtifactPolicy.model_validate(protocol["artifact_policy"])
    recipe = Recipe.model_validate({**template["initial_recipe"], **spec["initial_recipe"]})
    stage = protocol["agent_stage"]
    if not stage.get("model"):
        raise ValueError("The established agent model must be specified.")
    for group in ("targets", "per_trial_seconds", "total_compute_seconds"):
        if set(stage[group]) != {"covertype"}:
            raise ValueError("Stage budgets must describe only this mission's task.")
    limits = Limits.model_validate(
        {
            **protocol["common_limits"],
            "max_parameters": spec["max_parameters"],
            "max_epochs": spec["max_epochs"],
            "per_trial_seconds": stage["per_trial_seconds"]["covertype"],
            "total_compute_seconds": stage["total_compute_seconds"]["covertype"],
        }
    )
    widths = [54, *recipe.hidden_sizes, 7]
    parameters = sum((a + 1) * b for a, b in pairwise(widths))
    if parameters > limits.max_parameters:
        raise ValueError("The starting model exceeds the selected parameter budget.")
    if spec["recipe_policy"] != "constraints" or spec.get("search_space"):
        raise ValueError(
            "Agent choices must remain constrained by resources, without numeric menus."
        )
    if limits.per_trial_seconds > limits.total_compute_seconds:
        raise ValueError("The per-experiment limit must fit the mission budget.")
    json.dumps(protocol, allow_nan=False)


class SetupSession:
    def __init__(self, root: Path, model: IntakeModel):
        self.root = root.resolve()
        self.model = model
        self.api_records: list[dict] = []
        self.requests: dict[str, tuple[str, dict]] = {}
        self.prepared_save: tuple[dict, list[dict]] | None = None
        self.session_id = uuid.uuid4().hex
        self.mission_id = f"intake-{datetime.now(UTC):%Y%m%dT%H%M%S}-{self.session_id[:8]}"
        self.stage = "goal"
        self.messages: list[dict] = []
        self.summary = None
        self.config_path: Path | None = None
        # Capture the preset at session creation, so it cannot change mid-conversation.
        self.preset = read_json(self.root / PRESET)
        self.preset_hash = hashlib.sha256((self.root / PRESET).read_bytes()).hexdigest()
        self.seconds = 600.0
        self.parameters = 500000
        self.goal = ""
        self.lock = threading.Lock()
        self.say("What would you like to build or improve?")

    def say(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def view(self) -> dict:
        result = {
            "session_id": self.session_id,
            "stage": self.stage,
            "status": "ready"
            if self.stage == "ready"
            else ("review" if self.stage == "review" else "collecting"),
            "messages": self.messages,
            "model": self.model.model,
            "api_calls": len(self.api_records),
            "summary": self.summary,
        }
        if self.config_path:
            relative = self.config_path.relative_to(self.root).as_posix()
            result.update(
                {
                    "config_path": relative,
                    "config_url": f"/api/config/{self.session_id}",
                    "handoff_command": (
                        f".venv/bin/python -m rsi_ml autopilot --protocol {relative} "
                        f"--data data --out studies/{self.mission_id} "
                        "--env-file ~/.config/rsi_ml/api.env --execute"
                    ),
                }
            )
        return result

    def protocol(self) -> dict:
        protocol = json.loads(json.dumps(self.preset))
        protocol["study_id"] = self.mission_id
        protocol["status"] = "approved"
        protocol["approval"] = {
            "by": "user, setup conversation",
            "date": datetime.now(UTC).isoformat(),
            "scope": "Configuration handoff only; this interface does not launch optimization.",
            "goal": self.goal,
            "source_protocol": PRESET.as_posix(),
            "source_sha256": self.preset_hash,
            "reviewed": self.summary,
        }
        protocol["arms"] = ["D"]
        spec = protocol["tasks"]["covertype"]
        spec["seeds"] = [2]
        spec["max_parameters"] = self.parameters
        stage = protocol["agent_stage"]
        stage["total_compute_seconds"]["covertype"] = self.seconds
        stage["per_trial_seconds"]["covertype"] = min(self.seconds, math.ceil(self.seconds * 4 / 9))
        stage["targets"]["covertype"] = None  # Existing runner derives it from its baseline.
        return protocol

    def review(self) -> None:
        self.stage = "review"
        self.summary = {
            "Task": "Predict forest cover type with an MLP",
            "Dataset": "UCI Covertype · 581,012 rows · 54 features · 7 classes",
            "Source": "Registered UCI source; fetched or reused by the existing GPU pipeline",
            "Target and score": "Cover_Type (stored as label) · validation accuracy",
            "Split": "Existing stratified 60% train / 20% validation / 20% test; split seed 20261104",
            "Starting model": "54 → 128 → 64 → 7; fresh weights; 5 epochs",
            "Starting training": "AdamW; learning rate 0.001; batch 128; cosine schedule; dropout 0.1",
            "Limits": f"{self.seconds:g} seconds of compute · {self.parameters:,} parameters · up to 20 attempts",
            "Per experiment": f"{min(self.seconds, math.ceil(self.seconds * 4 / 9)):g} seconds (existing 4:9 budget rule)",
            "Workflow": "D: error reports + separate reviewer · one run · seed 2",
            "Agent model": self.preset["agent_stage"]["model"],
            "Records": "Initial/final weights, per-epoch validation outputs, curves and agent messages",
        }

    def record_api(self, record: dict) -> None:
        self.api_records.append(record)
        directory = self.root / "intake_sessions"
        directory.mkdir(exist_ok=True)
        with (directory / f"{self.mission_id}.api.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def respond(self, message: str, request_id: str) -> dict:
        if not isinstance(message, str) or not message.strip() or len(message) > 4000:
            raise ValueError("Please enter a message between 1 and 4,000 characters.")
        if not isinstance(request_id, str) or not re.fullmatch(r"[a-zA-Z0-9-]{8,80}", request_id):
            raise ValueError("Reload the page to create a valid message request.")
        message = message.strip()
        if request_id in self.requests:
            previous, result = self.requests[request_id]
            if previous != message:
                raise ValueError("A message request cannot be reused for different text.")
            return result
        if self.stage == "ready":
            return self.view()
        if self.prepared_save is not None:
            # A failed disk commit is resumed without paying for another model call.
            _, pending_messages = self.prepared_save
            if message != pending_messages[-2]["content"]:
                raise ValueError(
                    "Please retry your previous confirmation to finish saving the config."
                )
            self.save(pending_messages[-1]["content"])
            result = self.view()
            self.requests[request_id] = (message, json.loads(json.dumps(result)))
            return result
        if len(self.api_records) >= 40:
            raise ValueError(
                "This setup session has reached its API-call limit. Start a new session."
            )
        context = {
            "stage": self.stage,
            "conversation": [*self.messages, {"role": "user", "content": message}],
            "reviewed_configuration": self.summary,
        }
        try:
            decision, record = self.model.decide(context)
        except IntakeModelError as exc:
            if hasattr(exc, "record"):
                self.record_api({**exc.record, "request_id": request_id, "valid": False})
            raise
        self.record_api({**record, "request_id": request_id, "stage": self.stage})
        allowed = (
            {"clarify", "save", "change_budget"}
            if self.stage == "review"
            else {"clarify", "advance"}
        )
        if decision.action not in allowed:
            raise IntakeModelError("The model proposed an invalid setup step. Please retry.")
        if decision.action == "advance" and self.stage == "budget":
            seconds, parameters = decision.seconds, decision.parameters
            if seconds is None or parameters is None:
                raise IntakeModelError(
                    "The model did not extract both resource limits. Please retry."
                )
            # Apply the same numeric safety bounds to model-extracted, explicit choices.
            budget_from_text(f"{seconds} seconds, {parameters} parameters")
        elif decision.seconds is not None or decision.parameters is not None:
            raise IntakeModelError(
                "The model changed a resource limit outside the budget step. Please retry."
            )
        self.messages.append({"role": "user", "content": message})
        if decision.action == "advance":
            if self.stage == "goal":
                self.goal = message
            if self.stage == "budget":
                self.seconds, self.parameters = decision.seconds, decision.parameters
                self.review()
            else:
                self.stage = STEPS[STEPS.index(self.stage) + 1]
            self.say(decision.reply)
        elif decision.action == "change_budget":
            self.stage, self.summary = "budget", None
            self.say(decision.reply)
        elif decision.action == "save":
            try:
                self.save(decision.reply)
            except OSError:
                self.messages.pop()  # Preserve the displayed conversation on a failed commit.
                raise
        else:
            self.say(decision.reply)
        result = self.view()
        self.requests[request_id] = (message, json.loads(json.dumps(result)))
        return result

    def save(self, reply: str) -> None:
        directory = self.root / "configs" / "study"
        if self.prepared_save is None:
            self.prepared_save = (
                self.protocol(),
                [*self.messages, {"role": "assistant", "content": reply}],
            )
        protocol, messages = self.prepared_save
        validate_protocol(protocol, directory)
        path = directory / f"{self.mission_id}.json"
        write_exclusive_json(path, protocol)
        records = self.root / "intake_sessions"
        records.mkdir(exist_ok=True)
        write_exclusive_json(
            records / f"{self.mission_id}.json",
            {
                "session_id": self.session_id,
                "messages": messages,
                "chat_mode": "live_api",
                "chat_model": self.model.model,
                "api_calls": len(self.api_records),
                "api_trace": f"{self.mission_id}.api.jsonl",
                "summary": self.summary,
                "config": path.relative_to(self.root).as_posix(),
                "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "execution_started": False,
                "dataset_downloaded": False,
                "dataset_source": {"url": SOURCE_URL, "sha256": SOURCE_HASH},
            },
        )
        self.config_path, self.stage, self.messages = path, "ready", messages


class IntakeServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], root: Path, model: IntakeModel):
        self.root = root
        self.model = model
        self.sessions: dict[str, SetupSession] = {}
        super().__init__(address, IntakeHandler)


class IntakeHandler(BaseHTTPRequestHandler):
    server: IntakeServer

    def reply(self, status: int, value: dict | bytes, content_type="application/json") -> None:
        body = value if isinstance(value, bytes) else json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def allowed(self) -> bool:
        host = self.headers.get("Host", "")
        hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        origin = self.headers.get("Origin")
        if host not in hosts or (origin and origin != f"http://{host}"):
            self.reply(403, {"error": "This setup interface accepts local browser requests only."})
            return False
        return True

    def do_GET(self) -> None:
        if not self.allowed():
            return
        path = urlparse(self.path).path
        if path == "/":
            self.reply(
                200,
                files("rsi_ml").joinpath("intake_ui.html").read_bytes(),
                "text/html; charset=utf-8",
            )
        elif path == "/api/session":
            if len(self.server.sessions) >= 128:
                self.reply(429, {"error": "Session limit reached; restart the setup server."})
                return
            session = SetupSession(self.server.root, self.server.model)
            self.server.sessions[session.session_id] = session
            self.reply(200, session.view())
        elif path.startswith("/api/config/"):
            session = self.server.sessions.get(path.removeprefix("/api/config/"))
            if not session or not session.config_path:
                self.reply(404, {"error": "No saved configuration for this session."})
            else:
                self.reply(200, session.config_path.read_bytes())
        else:
            self.reply(404, {"error": "Not found."})

    def do_POST(self) -> None:
        if not self.allowed():
            return
        if self.path != "/api/message" or self.headers.get("X-Intake-Client") != "local":
            self.reply(403, {"error": "Use the local setup chat to send messages."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 20000:
                raise ValueError("Message is too large or empty.")
            body = json.loads(self.rfile.read(size))
            session = self.server.sessions.get(body.get("session_id"))
            if session is None:
                raise ValueError("Start a new experiment to begin a setup session.")
            with session.lock:
                result = session.respond(body.get("message"), body.get("request_id"))
            self.reply(200, result)
        except IntakeModelError as exc:
            self.reply(502, {"error": str(exc)})
        except (ValueError, TypeError, AttributeError) as exc:
            self.reply(400, {"error": str(exc)})
        except OSError:
            self.reply(
                500,
                {
                    "error": "Could not save the configuration. Check the project directory is writable."
                },
            )

    def log_message(self, format: str, *args) -> None:
        # HTTP status only; do not dump the conversation or credentials to terminal.
        super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--env-file", type=Path, help="Server-side dotenv file with OPENROUTER_API_KEY"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / PRESET).is_file():
        parser.error("Run from ml-agent/, or pass --root with that checkout's path.")
    try:
        model = IntakeModel(read_json(root / PRESET)["agent_stage"], load_key(args.env_file))
    except IntakeModelError as exc:
        parser.error(str(exc))
    server = IntakeServer(("127.0.0.1", args.port), root, model)
    print(f"Setup chat: http://127.0.0.1:{server.server_port}", flush=True)
    print(
        f"Live setup model: {model.model}. Config only; no training or dataset downloads.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
