"""Contract tests for the Laya decision server's OpenAI-compatible surface.

These run without torch: the tests install a stub decision agent, which keeps
them fast while still exercising the real request parsing, response shaping,
streaming, and error paths that SparkDeck's /v1 proxy depends on.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
import unittest.mock
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


class WeightResolutionTests(unittest.TestCase):
    """A pinned launch must load the pinned snapshot, not the default one.

    SparkDeck appends ``--revision`` for a cached bookmark launch, so the
    decision server has to resolve that exact snapshot instead of silently
    loading whatever the repository points at today.
    """

    def setUp(self):
        self._weights = server.WEIGHTS
        self._revision = server.MODEL_REVISION
        self.addCleanup(self._restore)

    def _restore(self):
        server.WEIGHTS = self._weights
        server.MODEL_REVISION = self._revision

    def test_unpinned_launch_loads_the_repository_untouched(self):
        server.WEIGHTS = "convaiinnovations/laya"
        server.MODEL_REVISION = None

        self.assertEqual(server._resolve_weights(), "convaiinnovations/laya")

    def test_pinned_launch_resolves_that_exact_snapshot(self):
        server.WEIGHTS = "convaiinnovations/laya"
        server.MODEL_REVISION = "b" * 40
        import huggingface_hub

        with unittest.mock.patch.object(
            huggingface_hub, "snapshot_download",
            return_value="/cache/snapshots/" + "b" * 40,
        ) as download:
            resolved = server._resolve_weights()

        self.assertEqual(resolved, "/cache/snapshots/" + "b" * 40)
        self.assertEqual(download.call_args.kwargs["revision"], "b" * 40)
        self.assertEqual(download.call_args.args[0], "convaiinnovations/laya")

    def test_local_checkpoint_is_already_an_exact_snapshot(self):
        server.WEIGHTS = str(REPO_ROOT)
        server.MODEL_REVISION = "c" * 40

        # A local directory needs no Hub resolution and must not be rewritten.
        self.assertEqual(server._resolve_weights(), str(REPO_ROOT))


class EntrypointTests(unittest.TestCase):
    """The entrypoint must accept the argv SparkDeck actually builds.

    The container entrypoint is a bash script, so it is exercised as one. The
    fake ``python`` on PATH records the argv it was exec'd with instead of
    starting a server, which is what makes the translation observable.
    """

    ENTRYPOINT = REPO_ROOT / "laya-decide" / "entrypoint.sh"

    @staticmethod
    def _bash_path(path: Path) -> str:
        """Express a path the way the available bash can open it.

        A Windows bash (WSL or MSYS) cannot resolve a ``C:\\...`` string, and
        the two translate drives differently, so both spellings are probed and
        the first the shell can actually reach is used.
        """
        text = str(path)
        if os.name != "nt":
            return text
        drive, _, rest = text.partition(":")
        tail = rest.lstrip("\\").replace("\\", "/")
        for candidate in (f"/mnt/{drive.lower()}/{tail}", f"/{drive.lower()}/{tail}"):
            probe = subprocess.run(
                ["bash", "-c", f'test -e "{candidate}"'],
                capture_output=True, text=True,
            )
            if probe.returncode == 0:
                return candidate
        return text

    def _run_entrypoint(self, args: list[str]):
        """Run the entrypoint and return (completed process, recorded argv)."""
        import shutil

        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is required to exercise the container entrypoint")

        with tempfile.TemporaryDirectory() as work:
            shim = Path(work) / "shim"
            shim.mkdir()
            stub = shim / "python"
            stub.write_text(
                "#!/usr/bin/env bash\n"
                'printf \'%s\\n\' "$@" > "$(dirname "$0")/argv"\n',
                encoding="utf-8", newline="\n",
            )
            stub.chmod(0o755)
            # A bash on Windows cannot use inherited Windows PATH entries, and
            # one of them may hold a python.exe that would shadow the stub, so
            # the harness script sets a shell-native PATH of its own.
            harness = Path(work) / "harness.sh"
            harness.write_text(
                "#!/usr/bin/env bash\n"
                f'PATH="{self._bash_path(shim)}:/usr/local/bin:/usr/bin:/bin"\n'
                "export PATH\n"
                f'exec bash "{self._bash_path(self.ENTRYPOINT)}" "$@"\n',
                encoding="utf-8", newline="\n",
            )
            harness.chmod(0o755)
            result = subprocess.run(
                [bash, self._bash_path(harness), *args],
                capture_output=True, text=True,
            )
            recorded = shim / "argv"
            return result, (
                recorded.read_text(encoding="utf-8") if recorded.exists() else ""
            )

    def test_revision_pin_is_consumed_rather_than_rejected(self):
        result, argv = self._run_entrypoint(
            ["--model", "convaiinnovations/laya", "--revision", "d" * 40],
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        # The revision reaches the server as configuration, not as argv.
        self.assertNotIn("--revision", argv)
        self.assertNotIn("--model", argv)
        self.assertIn("uvicorn", argv)

    def test_device_and_served_name_are_consumed(self):
        result, argv = self._run_entrypoint([
            "--model", "org/laya", "--device", "cpu",
            "--served-model-name", "laya-decide",
        ])

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--device", argv)
        self.assertNotIn("--served-model-name", argv)

    def test_unknown_flag_fails_the_launch(self):
        result, _argv = self._run_entrypoint(["--not-a-real-flag"])

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported argument", result.stderr)


class StubRouter:
    """Stand-in for ``laya.Router``: records the routing kwargs it received."""

    def __init__(self, result=None):
        self.calls: list[dict] = []
        self._result = result or {
            "model": "laya-rl-agent",
            "answers": {"department": {"type": "choice", "choice": "billing"}},
            "usage": {"input_tokens": 5, "output_tokens": 0},
            "routing": {
                "model": "multilingual",
                "repo": "convaiinnovations/laya-multilingual",
                "reason": "non-Latin script (kana, 100% of letters)",
            },
        }

    def predict(self, state, questions, **kwargs):
        self.calls.append(kwargs)
        return self._result


class LayaRouterTests(unittest.TestCase):
    """An opt-in router deployment must route and report its decision."""

    def setUp(self):
        self.enabled = server.ROUTER_ENABLED
        server.ROUTER_ENABLED = True
        self.router = StubRouter()
        server.set_agent(self.router)
        self.client = TestClient(server.app)
        self.addCleanup(self._restore)

    def _restore(self):
        server.ROUTER_ENABLED = self.enabled
        server.set_agent(None)

    def _post(self, **extra):
        return self.client.post("/v1/chat/completions", json={
            "model": server.MODEL_ID, "state": STATE,
            "questions": QUESTIONS, **extra,
        })

    def test_routed_decision_reports_which_checkpoint_answered(self):
        response = self._post()

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            body["laya"]["routing"]["model"], "multilingual",
        )
        self.assertEqual(
            body["choices"][0]["message"]["laya"]["routing"]["repo"],
            "convaiinnovations/laya-multilingual",
        )
        # The reason is what tells a caller the English checkpoint was skipped.
        self.assertIn("non-Latin", body["laya"]["routing"]["reason"])
        content = json.loads(body["choices"][0]["message"]["content"])
        self.assertEqual(content["routing"], body["laya"]["routing"])

    def test_routing_overrides_are_forwarded(self):
        response = self._post(routing={"lang": "de"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.router.calls, [{"lang": "de"}])

    def test_unknown_routing_field_is_rejected(self):
        response = self._post(routing={"colour": "blue"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("unsupported routing field", response.json()["detail"])
        self.assertEqual(self.router.calls, [])

    def test_single_checkpoint_deployment_rejects_overrides(self):
        """A pinned checkpoint must not be silently replaced by a routing hint."""
        server.ROUTER_ENABLED = False
        server.set_agent(StubAgent())

        response = self._post(routing={"lang": "de"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("--router", response.json()["detail"])

    def test_single_checkpoint_deployment_omits_routing_output(self):
        server.ROUTER_ENABLED = False
        server.set_agent(StubAgent())

        response = self._post()

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("routing", response.json()["laya"])

    def test_predict_route_also_routes(self):
        response = self.client.post("/laya/predict", json={
            "state": STATE, "questions": QUESTIONS,
        })

        self.assertEqual(response.status_code, 200)
        self.assertIn("routing", response.json())
        self.assertEqual(self.router.calls, [{}])


if __name__ == "__main__":
    unittest.main()
