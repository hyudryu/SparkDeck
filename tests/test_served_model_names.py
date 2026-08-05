import unittest
from types import SimpleNamespace
from unittest import mock

from manager import Manager


BACKING_MODEL = "Intel/Qwen3.5-122B-A10B-int4-AutoRound"
SERVED_MODEL = "qwen3.5-122b"


class ServedModelNameTests(unittest.IsolatedAsyncioTestCase):
    def container_summary(self) -> dict:
        manager = Manager.__new__(Manager)
        container = SimpleNamespace(
            short_id="d933706ebaa3",
            name="vllm-qwen35",
            labels={"vllm-controller": "1", "vllm-model": BACKING_MODEL},
            attrs={
                "Config": {
                    "Image": "vllm-qwen35-v2",
                    "Cmd": [
                        "vllm", "serve", BACKING_MODEL,
                        "--served-model-name", SERVED_MODEL,
                        "--max-model-len", "262144",
                    ],
                },
                "Created": "2026-08-03T00:00:00Z",
            },
            ports={"8000/tcp": [{"HostPort": "8000"}]},
            status="running",
        )
        return manager._container_summary(container)

    def test_container_keeps_backing_and_served_model_ids_separate(self) -> None:
        summary = self.container_summary()

        self.assertEqual(summary["model"], BACKING_MODEL)
        self.assertEqual(summary["served_model"], SERVED_MODEL)
        self.assertEqual(summary["served_models"], [SERVED_MODEL])

    async def test_models_endpoint_advertises_served_model_name(self) -> None:
        manager = Manager.__new__(Manager)
        manager.get_state = mock.AsyncMock(return_value={
            "containers": [self.container_summary()],
            "ollama": {"models": []},
            "unsloth": {},
            "sparkrun_targets": {},
        })

        result = await manager.proxy_models()

        self.assertEqual([item["id"] for item in result["data"]], [SERVED_MODEL])

    async def test_backing_model_request_is_rewritten_for_vllm(self) -> None:
        manager = Manager.__new__(Manager)
        container = self.container_summary()
        manager._resolve_vllm_target = mock.AsyncMock(return_value=container)
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [], "usage": {}}
        manager.http = SimpleNamespace(post=mock.AsyncMock(return_value=response))
        manager._track_start = mock.Mock(return_value=1)
        manager._track_end = mock.Mock()
        manager._record_usage = mock.Mock()

        await manager._vllm_chat(
            BACKING_MODEL,
            {"model": BACKING_MODEL, "messages": [{"role": "user", "content": "hi"}]},
            stream=False,
        )

        sent = manager.http.post.await_args.kwargs["json"]
        self.assertEqual(sent["model"], SERVED_MODEL)


if __name__ == "__main__":
    unittest.main()
