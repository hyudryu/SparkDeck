"""TensorFold's public model id: the serve CLI's --name flag.

TensorFold names the public model id clients send with ``--name`` (the
upstream recipe's SERVED_NAME variable). Without it the server reports the
checkpoint path's basename. SparkDeck's container discovery must keep the
two separate exactly like it does for vLLM's --served-model-name, so the
card advertises the right request id and the proxy rewrites requests to it.
"""

import unittest
from types import SimpleNamespace

from manager import (
    CONTROLLER_LABEL, ENGINE_LABEL, Manager, MODEL_LABEL,
)

TENSORFOLD_MODEL = (
    "/home/hyudryu/.cache/huggingface/hub/models--org--model/snapshots/deadbeef"
)
TENSORFOLD_SERVED = "glm-5.3-flash"


class TensorfoldServedModelNameTests(unittest.IsolatedAsyncioTestCase):
    def tensorfold_summary(self) -> dict:
        manager = Manager.__new__(Manager)
        container = SimpleNamespace(
            short_id="d933706ebaa3",
            name="tensorfold-glm53-8007",
            labels={
                CONTROLLER_LABEL: "1",
                ENGINE_LABEL: "tensorfold",
                MODEL_LABEL: TENSORFOLD_MODEL,
            },
            attrs={
                "Config": {
                    "Image": "sparkdeck/tensorfold:latest",
                    "Cmd": [
                        "tensorfold", "serve", TENSORFOLD_MODEL,
                        "--host", "0.0.0.0", "--port", "8007",
                        "--parallel", "8",
                        "--name", TENSORFOLD_SERVED,
                    ],
                },
                "Created": "2026-11-24T00:00:00Z",
            },
            ports={},
            status="running",
        )
        return manager._container_summary(container)

    def test_container_reports_the_tensorfold_served_name(self) -> None:
        summary = self.tensorfold_summary()

        self.assertEqual(summary["model"], TENSORFOLD_MODEL)
        self.assertEqual(summary["served_model"], TENSORFOLD_SERVED)
        self.assertEqual(summary["served_models"], [TENSORFOLD_SERVED])

    def test_container_without_a_name_keeps_the_checkpoint_path(self) -> None:
        manager = Manager.__new__(Manager)
        container = SimpleNamespace(
            short_id="d933706ebaa3",
            name="tensorfold-glm53-8007",
            labels={
                CONTROLLER_LABEL: "1",
                ENGINE_LABEL: "tensorfold",
                MODEL_LABEL: TENSORFOLD_MODEL,
            },
            attrs={
                "Config": {
                    "Image": "sparkdeck/tensorfold:latest",
                    "Cmd": ["tensorfold", "serve", TENSORFOLD_MODEL],
                },
                "Created": "2026-11-24T00:00:00Z",
            },
            ports={},
            status="running",
        )
        summary = manager._container_summary(container)

        self.assertEqual(summary["served_model"], TENSORFOLD_MODEL)
        self.assertEqual(summary["served_models"], [TENSORFOLD_MODEL])

    def test_saved_tensorfold_deployment_reads_the_name_flag(self) -> None:
        deployment = {
            "model": TENSORFOLD_MODEL,
            "launch_settings": {
                "engine": "tensorfold",
                "extra_args": [
                    "--parallel", "8", "--context", "262144",
                    "--name", TENSORFOLD_SERVED,
                ],
            },
        }

        self.assertEqual(
            Manager._deployment_served_models(deployment), [TENSORFOLD_SERVED],
        )

    def test_saved_tensorfold_deployment_prefers_the_running_container(self) -> None:
        deployment = {
            "model": TENSORFOLD_MODEL,
            "launch_settings": {
                "engine": "tensorfold",
                "extra_args": ["--name", "saved-name"],
            },
        }
        container = {
            "status": "running",
            "served_models": [TENSORFOLD_SERVED],
        }

        self.assertEqual(
            Manager._deployment_served_models(deployment, container),
            [TENSORFOLD_SERVED],
        )

    def test_tensorfold_requests_are_rewritten_to_the_served_name(self) -> None:
        container = self.tensorfold_summary()

        self.assertEqual(
            Manager._upstream_model_id(container, TENSORFOLD_MODEL),
            TENSORFOLD_SERVED,
        )
        self.assertEqual(
            Manager._upstream_model_id(container, TENSORFOLD_SERVED),
            TENSORFOLD_SERVED,
        )


if __name__ == "__main__":
    unittest.main()
