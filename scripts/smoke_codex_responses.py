"""Exercise the installed Codex client against a disposable, mocked SparkDeck.

Run manually: python scripts/smoke_codex_responses.py
No model runtime, user Codex configuration, or credentials are used.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn
from fastapi import FastAPI, Request


def main():
    codex = shutil.which("codex")
    if not codex:
        raise SystemExit("Install Codex CLI before running this optional smoke test")
    import server

    requests = []
    chat_requests = []

    async def models():
        from sparkdeck.codex_models import codex_model
        model = {"id": "sparkdeck-smoke", "object": "model", "owned_by": "sparkdeck"}
        return {"object": "list", "data": [model], "models": [codex_model(model["id"])]}

    async def proxy(body, endpoint, cancel=None, **kwargs):
        assert endpoint == "chat/completions"
        chat_requests.append(body)
        async def chunks():
            if len(chat_requests) == 1:
                names = {tool["function"]["name"] for tool in body.get("tools", [])}
                assert "exec_command" in names
                command = "Write-Output SPARKDECK_TOOL_OK" if os.name == "nt" else "printf SPARKDECK_TOOL_OK"
                arguments = json.dumps({"cmd": command, "max_output_tokens": 100})
                deltas = [{"role": "assistant"}, {"tool_calls": [{"index": 0, "id": "call_smoke", "type": "function", "function": {"name": "exec_command", "arguments": arguments[:12]}}]}, {"tool_calls": [{"index": 0, "function": {"arguments": arguments[12:]}}]}]
                finish_reason = "tool_calls"
            else:
                tool_outputs = [str(message.get("content")) for message in body["messages"] if message.get("role") == "tool"]
                assert any("SPARKDECK_TOOL_OK" in output and "Process exited with code 0" in output for output in tool_outputs), tool_outputs
                deltas = [{"role": "assistant"}, {"content": "SPARKDECK_CODEX_OK"}]
                finish_reason = "stop"
            for delta in deltas:
                yield "data: " + json.dumps({"id": "chatcmpl-smoke", "model": body["model"], "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}) + "\n\n"
            yield "data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}], "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}}) + "\n\n"
            yield "data: [DONE]\n\n"
        return chunks()

    server.sparkdeck.proxy = proxy
    server.sparkdeck.models = models
    app = FastAPI()

    @app.middleware("http")
    async def capture(request: Request, call_next):
        body = await request.body()
        requests.append({"method": request.method, "path": request.url.path, "body": json.loads(body) if body else None})
        return await call_next(request)

    app.add_api_route("/v1/models", server.v1_models, methods=["GET"])
    app.add_api_route("/v1/responses", server.v1_responses, methods=["POST"])
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    http = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = threading.Thread(target=http.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    # On Windows the normal temp directory is inside the user home, whose
    # ancestor .codex config can be treated as trusted project configuration.
    temp_parent = Path(os.environ["SYSTEMROOT"]) / "Temp" if os.name == "nt" else None
    with tempfile.TemporaryDirectory(prefix="sparkdeck-codex-smoke-", dir=temp_parent) as temporary:
        root = Path(temporary)
        # Stop project-config discovery before it reaches the user's home.
        (root / ".git").mkdir()
        config = f'''model = "sparkdeck-smoke"
model_provider = "sparkdeck"
web_search = "disabled"
[model_providers.sparkdeck]
name = "SparkDeck Smoke"
base_url = "http://127.0.0.1:{port}/v1"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
request_max_retries = 0
stream_max_retries = 0
[model_providers.sparkdeck.auth]
command = {json.dumps(sys.executable)}
args = ["-c", "print('sparkdeck-disposable-smoke-token')"]
cwd = {json.dumps(temporary)}
'''
        (root / "config.toml").write_text(config, encoding="utf-8")
        env = {key: value for key, value in os.environ.items() if key.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATHEXT", "COMSPEC", "PROCESSOR_ARCHITECTURE"}}
        env["CODEX_HOME"] = str(root)
        env["USERPROFILE"] = str(root)
        try:
            for _ in range(100):
                if http.started:
                    break
                time.sleep(.05)
            # The only executable command is the fixed output fixture above.
            # This avoids depending on a locally provisioned Windows sandbox.
            result = subprocess.run([codex, "exec", "--ephemeral", "--skip-git-repo-check", "--ignore-rules", "--sandbox", "danger-full-access", "--json", "-C", temporary, "Print SPARKDECK_TOOL_OK with a read-only shell command, then reply with SPARKDECK_CODEX_OK."], env=env, cwd=temporary, input="", capture_output=True, text=True, timeout=90)
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            print(json.dumps({"returncode": result.returncode, "requests": [{"method": request["method"], "path": request["path"]} for request in requests], "chat_requests": len(chat_requests)}, indent=2))
            assert result.returncode == 0, "Codex exec failed"
            assert "SPARKDECK_CODEX_OK" in result.stdout
            assert len(chat_requests) == 2, "Codex did not complete the tool roundtrip"
            assert any(request["path"] == "/v1/models" for request in requests), "Codex did not refresh the model catalog"
            assert "Model metadata for" not in result.stdout, "Codex rejected the model catalog"
        finally:
            http.should_exit = True
            thread.join(timeout=5)


if __name__ == "__main__":
    main()


