"""Contract tests for the Laya decision server's OpenAI-compatible surface.

These run without torch: the tests install a stub decision agent, which keeps
them fast while still exercising the real request parsing, response shaping,
streaming, and error paths that SparkDeck's /v1 proxy depends on.
"""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]

# Load ``laya-decide/server.py`` by path: the repository root also owns a
# ``server.py`` (SparkDeck's application entry point), so a plain ``import
# server`` would resolve to the wrong module.
_spec = importlib.util.spec_from_file_location(
    "laya_decide_server", REPO_ROOT / "laya-decide" / "server.py",
)
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)

STATE = {"from": "customer@acme.com", "body": "Please refund the duplicate charge."}
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this email?",
        "criteria": {"billing": "invoices and refunds", "other": "everything else"},
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel?",
    },
}


class StubAgent:
    """Minimal stand-in for ``laya.Agent`` that records what it was asked."""

    def __init__(self, result=None):
        self.calls: list[tuple[object, object]] = []
        self.result = result or {
            "model": "laya-rl-agent",
            "answers": {
                "department": {
                    "type": "choice", "choice": "billing",
                    "probabilities": {"billing": 0.97, "other": 0.03},
                    "confidence": 0.94, "action": {"act_probability": 1.0},
                },
                "churn_risk": {"type": "noul", "noul": 0.12, "confidence": 0.88},
            },
            "usage": {"input_tokens": 41, "output_tokens": 0},
        }

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return self.result


def question_shape(questions):
    """The decision-relevant part of a normalized question payload.

    The server passes a fixed-key shape (``type``/``instructions``/``criteria``)
    to Laya, so assertions compare the caller's intent rather than the
    normalization detail.
    """
    return {
        key: (value["type"], value["instructions"], value.get("criteria"))
        for key, value in questions.items()
    }


class LayaDecisionServerTests(unittest.TestCase):
    def setUp(self):
        self.agent = StubAgent()
        server.set_agent(self.agent)
        self.client = TestClient(server.app)
        self.addCleanup(server.set_agent, None)

    def test_models_lists_the_deployed_repository(self):
        body = self.client.get("/v1/models").json()
        self.assertEqual(body["object"], "list")
        self.assertEqual(body["data"][0]["id"], server.MODEL_ID)
        self.assertEqual(body["data"][0]["owned_by"], server.OBJECT_NAME)

    def test_chat_completion_returns_openai_shape_with_laya_answers(self):
        response = self.client.post("/v1/chat/completions", json={
            "model": server.MODEL_ID, "state": STATE, "questions": QUESTIONS,
        })
        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["model"], server.MODEL_ID)
        choice = body["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(choice["message"]["role"], "assistant")
        # The assistant content must be JSON so any OpenAI client can read it.
        content = json.loads(choice["message"]["content"])
        self.assertEqual(content["answers"], body["laya"]["answers"])
        self.assertEqual(choice["message"]["laya"]["answers"], body["laya"]["answers"])

        # A non-generating model reports real prompt tokens and zero output.
        self.assertEqual(body["usage"]["prompt_tokens"], 41)
        self.assertEqual(body["usage"]["completion_tokens"], 0)
        self.assertEqual(body["usage"]["total_tokens"], 41)
        state, questions = self.agent.calls[0]
        self.assertEqual(state, STATE)
        self.assertEqual(question_shape(questions), question_shape(QUESTIONS))

    def test_decision_payload_may_arrive_as_a_system_message(self):
        response = self.client.post("/v1/chat/completions", json={
            "model": server.MODEL_ID,
            "messages": [{
                "role": "system",
                "content": json.dumps({"state": STATE, "questions": QUESTIONS}),
            }],
        })
        self.assertEqual(response.status_code, 200)
        state, questions = self.agent.calls[0]
        self.assertEqual(state, STATE)
        self.assertEqual(question_shape(questions), question_shape(QUESTIONS))

    def test_top_level_fields_win_over_message_content(self):
        response = self.client.post("/v1/chat/completions", json={
            "model": server.MODEL_ID,
            "state": STATE, "questions": QUESTIONS,
            "messages": [{
                "role": "system",
                "content": json.dumps({"state": "ignored", "questions": QUESTIONS}),
            }],
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.agent.calls[0][0], STATE)

    def test_presets_replace_questions_without_a_questions_object(self):
        server._preset_factories = {"router": lambda: {
            "domain": {
                "type": "choice", "instructions": "Which domain?",
                "criteria": {"code": "source code", "writing": "prose"},
            },
        }}
        self.addCleanup(setattr, server, "_preset_factories", None)
        agent = StubAgent()
        server.set_agent(agent)
        response = self.client.post("/v1/chat/completions", json={
            "model": server.MODEL_ID, "state": STATE, "presets": ["router"],
        })
        self.assertEqual(response.status_code, 200)
        _state, questions = agent.calls[0]
        self.assertEqual(list(questions), ["domain"])
        self.assertEqual(questions["domain"]["type"], "choice")

    def test_unknown_preset_is_rejected(self):
        response = self.client.post("/v1/chat/completions", json={
            "model": server.MODEL_ID, "state": STATE, "presets": ["nope"],
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("unknown preset", response.json()["detail"])

    def test_missing_state_is_rejected(self):
        response = self.client.post("/v1/chat/completions", json={
            "model": server.MODEL_ID, "questions": QUESTIONS,
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("state is required", response.json()["detail"])

    def test_missing_questions_is_rejected(self):
        response = self.client.post("/v1/chat/completions", json={
            "model": server.MODEL_ID, "state": STATE,
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("questions must be a non-empty object", response.json()["detail"])

    def test_choice_question_requires_criteria(self):
        response = self.client.post("/v1/chat/completions", json={
            "model": server.MODEL_ID, "state": STATE,
            "questions": {"q": {"type": "choice", "instructions": "pick one"}},
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("criteria object", response.json()["detail"])

    def test_score_question_requires_an_ordered_rubric(self):
        response = self.client.post("/v1/chat/completions", json={
            "model": server.MODEL_ID, "state": STATE,
            "questions": {
                "q": {"type": "score", "instructions": "how bad?", "criteria": ["only one"]},
            },
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("ordered criteria array", response.json()["detail"])

    def test_unsupported_question_type_is_rejected(self):
        response = self.client.post("/v1/chat/completions", json={
            "model": server.MODEL_ID, "state": STATE,
            "questions": {"q": {"type": "freeform", "instructions": "say something"}},
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("unsupported type", response.json()["detail"])

    def test_streaming_emits_sse_chunks_and_done(self):
        with self.client.stream("POST", "/v1/chat/completions", json={
            "model": server.MODEL_ID, "state": STATE, "questions": QUESTIONS,
            "stream": True,
        }) as response:
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
            payloads = [
                json.loads(line[len("data: "):])
                for line in response.iter_lines()
                if line.startswith("data: ") and not line.endswith("[DONE]")
            ]

        self.assertEqual(payloads[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(payloads[1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(payloads[-1]["usage"]["prompt_tokens"], 41)
        self.assertEqual(payloads[1]["choices"][0]["delta"]["laya"]["answers"]["churn_risk"]["noul"], 0.12)

    def test_predict_route_returns_native_laya_result(self):
        response = self.client.post("/laya/predict", json={
            "state": STATE, "questions": QUESTIONS,
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), self.agent.result)

    def test_health_reports_loaded_model(self):
        body = self.client.get("/healthz").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["model"], server.MODEL_ID)

    def test_decision_failure_is_a_server_error(self):
        class Exploding(StubAgent):
            def predict(self, state, questions):
                raise RuntimeError("cuda out of memory")

        server.set_agent(Exploding())
        response = self.client.post("/v1/chat/completions", json={
            "model": server.MODEL_ID, "state": STATE, "questions": QUESTIONS,
        })
        self.assertEqual(response.status_code, 500)
        self.assertIn("cuda out of memory", response.json()["detail"])

    def test_models_route_is_unavailable_while_the_model_loads(self):
        server.set_agent(None)
        response = self.client.post("/v1/chat/completions", json={
            "model": server.MODEL_ID, "state": STATE, "questions": QUESTIONS,
        })
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
