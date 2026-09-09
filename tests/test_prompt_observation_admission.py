import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from manager import Manager
from sparkdeck.service import SparkDeckService


class PromptObservationAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.groups = {"a": ["n1", "n2"], "b": ["n3", "n4"]}
        self.manager = SimpleNamespace(
            http=None, deployments=[],
            _request_group=lambda key, *_: {"node_ids": self.groups[key]},
            _mark_deployment_used=Mock(),
        )
        self.service = SparkDeckService(self.manager, Path(directory.name))
        self.addAsyncCleanup(self.service.close)
        self.service.store.add_benchmark_if_consented = Mock(return_value=True)

    async def test_waiting_prompt_does_not_contaminate_until_dispatched(self):
        scopes = frozenset({"node:n1", "node:n2"})
        active = self.service._community_observation_start(scopes)
        queued = self.service._community_observation_start(scopes, deferred=True)
        self.assertFalse(active["contaminated"])
        self.assertNotIn(queued["id"], self.service._community_active_observations)

        self.service._community_observation_activate(queued)

        self.assertTrue(active["contaminated"])
        self.assertTrue(queued["contaminated"])

    async def test_group_dispatch_tracks_only_nodes_in_selected_engine(self):
        deployment = {
            "id": "cluster", "mode": "grouped_sharded",
            "members": [
                {"node_id": "n1", "instance_id": 1},
                {"node_id": "n2", "instance_id": 1},
                {"node_id": "n3", "instance_id": 2},
                {"node_id": "n4", "instance_id": 2},
            ],
        }
        self.manager.deployments = [deployment]
        registered = {"id": "registered", "settings": {"manager_deployment_id": "cluster"}}
        observations = []
        for index in (0, 2):
            member = deployment["members"][index]
            scopes = self.service._community_observation_scopes(registered, "model", member=member)
            self.assertEqual(scopes, frozenset(f"node:n{n}" for n in (index + 1, index + 2)))
            observation = self.service._community_observation_start(deferred=True)
            token = self.service._community_observation.set(observation)
            try:
                self.service._activate_group_observation(deployment, member)
            finally:
                self.service._community_observation.reset(token)
            self.assertEqual(observation["scopes"], scopes)
            observations.append(observation)
        self.assertTrue(all(not item["contaminated"] for item in observations))

    def record_startup_sample(self, observation):
        token = self.service._community_observation.set(observation)
        try:
            self.service._record_usage(
                "registered", "org/model", "vllm", {"tensor_parallel_size": 2},
                1.0, {"prompt_tokens": 1, "completion_tokens": 200}, 2.0,
                completed_at=4.0, hardware={"gpu_count": 2},
            )
        finally:
            self.service._community_observation.reset(token)

    async def test_other_group_requests_allow_sample_but_same_group_requests_block_it(self):
        for competing_group, eligible in (("b", True), ("a", False)):
            with self.subTest(competing_group=competing_group):
                self.manager._active_reqs = {}
                self.manager._req_seq = 0
                self.manager._inference_scope_sequences = {}
                self.service._community_active_observations.clear()
                self.service.store.add_benchmark_if_consented.reset_mock()
                observation = self.service._community_observation_start(
                    frozenset({"node:n1", "node:n2"}), deferred=True,
                )
                observation.update(enabled=True, startup_benchmark=True)
                self.service._community_observation_activate(observation)
                Manager._track_start(self.manager, "a", startup_benchmark=True)
                # A completed request must still invalidate evidence if it ran
                # on the sampled engine; checking only active requests misses it.
                competing = Manager._track_start(self.manager, competing_group)
                Manager._track_end(self.manager, competing)

                self.record_startup_sample(observation)

                self.assertEqual(self.service.store.add_benchmark_if_consented.call_count, int(eligible))
                self.assertEqual(bool(observation.get("startup_recorded")), eligible)

    async def test_activation_detects_existing_requests_only_on_overlapping_nodes(self):
        for active_group, contaminated in (("b", False), ("a", True)):
            with self.subTest(active_group=active_group):
                self.manager._active_reqs = {}
                self.manager._req_seq = 0
                self.service._community_active_observations.clear()
                Manager._track_start(self.manager, active_group)
                observation = self.service._community_observation_start(
                    frozenset({"node:n1", "node:n2"}), deferred=True,
                )
                observation["startup_benchmark"] = True
                self.service._community_observation_activate(observation)
                self.assertEqual(observation["contaminated"], contaminated)

    async def test_standalone_local_request_does_not_contaminate_remote_startup(self):
        manager = Manager.__new__(Manager)
        manager.deployments = []
        manager._mark_deployment_used = Mock()
        self.service.manager = manager
        request_id = manager._track_start("local-model")
        self.assertEqual(manager._active_reqs[request_id]["group"]["node_ids"], ["local"])
        observation = self.service._community_observation_start(
            frozenset({"node:n1", "node:n2"}), deferred=True,
        )
        observation["startup_benchmark"] = True

        self.service._community_observation_activate(observation)

        self.assertFalse(observation["contaminated"])
        self.assertTrue(self.service._community_manager_requests_overlap(frozenset({"node:local"})))
