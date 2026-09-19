"""Real-inference check for the Laya decision model and the laya-decide server.

Run with the build virtualenv so torch/transformers/laya are importable:

    .venv-laya-build/Scripts/python.exe tools/laya_live_check.py

It exercises two things the unit tests cannot: that the published weights
actually load through the ``laya`` package, and that ``laya-decide`` returns a
decision over HTTP in the OpenAI response shape SparkDeck proxies.
"""

from __future__ import annotations

import contextlib
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
    import unittest.mock as mock

    import server
    from fastapi.testclient import TestClient

    print(f"model repo: {server.WEIGHTS}")

    # One load, shared by the direct call and the HTTP surface. The app's
    # lifespan loads an agent of its own, so without this cache a host sized
    # for a single Laya instance would hold two during verification. The real
    # loader is captured first so the patched version cannot recurse into itself.
    real_load = server._load_agent
    loaded: list = []

    def load_once():
        if not loaded:
            loaded.append(real_load())
        return loaded[0]

    with mock.patch.object(server, "_load_agent", load_once):
        return _run(server, TestClient, load_once())


def _run(server, TestClient, agent) -> int:
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

    # Opt-in routing: the English checkpoint collapses on non-Latin text while
    # staying confident, so a routed deployment must pick the multilingual one.
    import gc

    import laya

    # The English agent above is a separate instance from anything the router
    # owns, so release it before holding two checkpoints at once. This mirrors
    # the real deployment, which loads through exactly one code path.
    server.set_agent(None)
    del agent
    gc.collect()

    router = laya.Router(max_loaded=2)
    print("\n-- router --")
    for label, state in (
        ("English", {"body": "I was charged twice, please refund."}),
        ("Japanese", {"body": "二重に請求されました。返金してください。"}),
        ("German", {"body": "Mir wurde der Betrag zweimal in Rechnung gestellt."}),
    ):
        decision = router.route(state, QUESTIONS)
        print(f"  {label:9} -> {decision['model']:13} ({decision['reason']})")

    japanese = router.route({"body": "二重に請求されました"}, QUESTIONS)
    assert japanese["model"] == "multilingual", japanese
    english = router.route({"body": "I was charged twice"}, QUESTIONS)
    assert english["model"] == "english", english

    # Script detection is exact; the Latin-script language guess is a
    # documented best-effort heuristic. A caller who knows the language can
    # always override it, which is what the `routing` request field exposes.
    german = {"body": "Mir wurde der Betrag zweimal in Rechnung gestellt."}
    forced = router.route(german, QUESTIONS, lang="de")
    print(f"  German+lang=de -> {forced['model']:13} ({forced['reason']})")
    assert forced["model"] == "multilingual", forced

    # And the routed endpoint answers with the decision that produced it. The
    # router installed above is the agent under test, so the app's lifespan is
    # neutralized for this pass rather than loading a second checkpoint on top.
    server.ROUTER_ENABLED = True
    server.set_agent(router)

    @contextlib.asynccontextmanager
    async def no_load(_app):
        yield

    original_lifespan = server.app.router.lifespan_context
    server.app.router.lifespan_context = no_load
    try:
        routed_client = TestClient(server.app)
        routed = routed_client.post("/v1/chat/completions", json={
            "model": server.MODEL_ID,
            "state": {"body": "二重に請求されました。返金してください。"},
            "questions": QUESTIONS,
        })
        routed.raise_for_status()
        routing = routed.json()["laya"]["routing"]
        print("\n-- routed completion --")
        print(json.dumps(routing, indent=2))
        assert routing["model"] == "multilingual", routing
        assert routing["repo"] == "convaiinnovations/laya-multilingual"
        assert routed.json()["laya"]["answers"]["department"]["choice"]
    finally:
        server.app.router.lifespan_context = original_lifespan

    print("\nALL LIVE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
