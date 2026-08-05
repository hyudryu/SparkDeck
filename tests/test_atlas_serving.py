import unittest
from types import SimpleNamespace

from manager import Manager


class AtlasServingContainerTests(unittest.TestCase):
    def test_atlas_sparkrun_container_is_listed(self) -> None:
        manager = Manager.__new__(Manager)
        container = SimpleNamespace(
            short_id="d4ef48061e6e",
            name="sparkrun_d163d4913130_solo",
            labels={},
            attrs={
                "Config": {"Image": "avarok/atlas-gb10:latest", "Cmd": []},
                "Created": "2026-07-29T00:00:00Z",
            },
            ports={},
            status="running",
        )

        summary = manager._container_summary(container)

        self.assertIsNotNone(summary)
        self.assertEqual(summary["source"], "atlas-serving")
        self.assertFalse(summary["managed"])


if __name__ == "__main__":
    unittest.main()
