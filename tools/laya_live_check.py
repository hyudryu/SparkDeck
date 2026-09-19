"""Real-inference check for the Laya decision model and the laya-decide server.

Run with the build virtualenv so torch/transformers/laya are importable:

    .venv-laya-build/Scripts/python.exe tools/laya_live_check.py

It exercises two things the unit tests cannot: that the published weights
actually load through the ``laya`` package, and that ``laya-decide`` returns a
decision over HTTP in the OpenAI response shape SparkDeck proxies.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "laya-decide"))

STATE = {
    "from": "customer@acme.com",
    "subject": "Duplicate billing on March invoice #4411",
    "body": (
        "Hi team, we were billed twice for March. Please refund the duplicate "
        "before Friday or we will cancel our plan."
    ),
}

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this email?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, integrations",
            "sales": "pricing, contracts, demos",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or switch to a competitor?",
    },
}


def main() -> int:
    import server
    from fastapi.testclient import TestClient

    print(f"model repo: {server.WEIGHTS}")
    agent = server._load_agent()
    server.set_agent(agent)

    started = time.monotonic()
    direct = agent.predict(STATE, QUESTIONS)
    direct_ms = (time.monotonic() - started) * 1000
    print(f"\n-- direct predict ({direct_ms:.1f} ms) --")
    print(json.dumps(direct, indent=2))

    answers = direct.get("answers") or {}
    assert set(answers) == set(QUESTIONS), f"missing answers: {set(QUESTIONS) - set(answers)}"
    assert answers["department"]["choice"] in QUESTIONS["department"]["criteria"]
    assert 0.0 <= answers["churn_risk"]["noul"] <= 1.0
    assert 0.0 <= answers["urgency"]["score"] <= 2.0

    with TestClient(server.app) as client:
        models = client.get("/v1/models").json()
        print("\n-- /v1/models --")
        print(json.dumps(models, indent=2))
        assert models["data"][0]["id"] == server.MODEL_ID

        started = time.monotonic()
        response = client.post("/v1/chat/completions", json={
            "model": server.MODEL_ID,
            "state": STATE,
            "questions": QUESTIONS,
            "max_tokens": 64,
        })
        http_ms = (time.monotonic() - started) * 1000
        response.raise_for_status()
        body = response.json()
        print(f"\n-- /v1/chat/completions ({http_ms:.1f} ms) --")
        print(json.dumps(body, indent=2))

        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        # The assistant content must be parseable JSON carrying the answers.
        content = json.loads(body["choices"][0]["message"]["content"])
        assert content["answers"] == body["laya"]["answers"]
        assert body["usage"]["prompt_tokens"] > 0
        assert body["usage"]["completion_tokens"] == 0

        # A system message carrying the payload must work for plain OpenAI clients.
        embedded = client.post("/v1/chat/completions", json={
            "model": server.MODEL_ID,
            "messages": [{
                "role": "system",
                "content": json.dumps({"state": STATE, "questions": QUESTIONS}),
            }],
        })
        embedded.raise_for_status()
        assert embedded.json()["laya"]["answers"] == body["laya"]["answers"]

        # Streaming must produce SSE frames ending in [DONE].
        with client.stream("POST", "/v1/chat/completions", json={
            "model": server.MODEL_ID, "state": STATE, "questions": QUESTIONS,
            "stream": True,
        }) as stream:
            stream.raise_for_status()
            events = [line for line in stream.iter_lines() if line.startswith("data: ")]
        assert events[-1] == "data: [DONE]", events[-1]
        first = json.loads(events[1][len("data: "):])
        assert first["object"] == "chat.completion.chunk"

        presets = client.post("/v1/chat/completions", json={
            "model": server.MODEL_ID,
            "state": {"request": "Refactor this service using dependency injection"},
            "presets": ["router"],
        })
        presets.raise_for_status()
        print("\n-- preset router answers --")
        print(json.dumps(presets.json()["laya"]["answers"], indent=2))

        bad = client.post("/v1/chat/completions", json={
            "model": server.MODEL_ID, "state": "x",
            "questions": {"q": {"type": "choice", "instructions": "pick"}},
        })
        assert bad.status_code == 400, bad.status_code

    print("\nALL LIVE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
