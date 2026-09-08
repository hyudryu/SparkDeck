import asyncio
import unittest
from collections import deque
from unittest.mock import AsyncMock, Mock

from manager import ClientAbort, ClusterReplicaUnavailable, Manager
from sparkdeck.prompt_gate import PromptGates


class GroupPromptGateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.deployment = {
            'id': 'two-groups', 'mode': 'grouped_sharded',
            'members': [
                {'node_id': f'node-{group}-{rank}', 'instance_id': group,
                 'rank': rank, 'container_name': f'group-{group}-{rank}',
                 'container_id': f'container-{group}-{rank}'}
                for group in (0, 1) for rank in (0, 1)
            ],
        }
        self.manager = Manager.__new__(Manager)
        self.manager.deployments = [self.deployment]
        self.manager.prompt_gate = PromptGates()
        self.manager._active_reqs = {}
        self.chunks = {}
        self.entered = []
        self.closed = []

        async def upstream(deployment, member, model, body, endpoint, cancel, **kwargs):
            name = body['request']
            self.entered.append((name, member['instance_id'], model, endpoint))
            self.manager._active_reqs[name] = {
                'key': model, 'thinking': deque(), 'output': deque(),
                'group': self.manager._request_group(model, deployment['id'], member['container_name']),
            }

            async def stream():
                try:
                    while True:
                        yield await self.chunks[name].get()
                finally:
                    self.closed.append(name)
                    self.manager._active_reqs.pop(name, None)
            return stream()

        self.manager._proxy_cluster_member_unlimited = AsyncMock(side_effect=upstream)

    async def close_stream(self, stream):
        await stream.aclose()

    async def request(self, name, group=0, *, model='model', endpoint='chat/completions', cancel=None):
        self.chunks[name] = asyncio.Queue()
        result = await self.manager._proxy_cluster_member(
            self.deployment, self.deployment['members'][group * 2], model,
            {'request': name, 'stream': True}, endpoint, cancel,
        )
        self.addAsyncCleanup(self.close_stream, result)
        return result

    async def wait_queued(self, count=1):
        async def wait():
            while len(getattr(self.manager, '_prompt_waiting_requests', {})) != count:
                await asyncio.sleep(0)
        await asyncio.wait_for(wait(), 2)

    async def emit_token(self, name, stream):
        chunk = 'data: {"choices":[{"delta":{"content":"token"}}]}\n\n'
        await self.chunks[name].put(chunk)
        self.assertEqual(await asyncio.wait_for(anext(stream), 2), chunk)

    async def test_two_groups_of_two_nodes_admit_independently_and_aliases_share(self):
        first = await self.request('first')
        second = asyncio.create_task(self.request('second', model='alias', endpoint='completions'))
        self.addCleanup(second.cancel)
        await self.wait_queued()
        other = await asyncio.wait_for(self.request('other', group=1), 2)
        self.assertFalse(second.done())
        self.assertEqual([item[0] for item in self.entered], ['first', 'other'])
        await self.emit_token('first', first)
        await asyncio.wait_for(second, 2)
        self.assertEqual(self.entered[-1], ('second', 0, 'alias', 'completions'))
        self.assertEqual(self.closed, [])
        await other.aclose()

    async def test_generating_responses_do_not_consume_prompt_slots(self):
        for index in range(2):
            name = f'decoding-{index}'
            stream = await asyncio.wait_for(self.request(name), 2)
            await self.emit_token(name, stream)
        processing = await asyncio.wait_for(self.request('processing'), 2)
        queued = asyncio.create_task(self.request('queued'))
        self.addCleanup(queued.cancel)
        await self.wait_queued()
        self.assertEqual(len(self.entered), 3)
        self.assertFalse(queued.done())
        self.assertEqual(self.closed, [])
        admission = list(self.manager.inference_admission().values())
        self.assertEqual(len(admission), 1)
        self.assertEqual((admission[0]['running'], admission[0]['queued']), (3, 1))
        active_groups = list(self.manager.active_request_groups().values())
        self.assertEqual(len(active_groups), 1)
        self.assertEqual(active_groups[0]['connections'], 3)
        await self.emit_token('processing', processing)
        await asyncio.wait_for(queued, 2)

    async def test_cancelled_waiter_releases_reservation_and_allows_next_request(self):
        first = await self.request('first')
        cancel = asyncio.Event()
        queued = asyncio.create_task(self.request('cancelled', cancel=cancel))
        self.addCleanup(queued.cancel)
        await self.wait_queued()
        member = self.deployment['members'][0]
        self.assertEqual(self.manager._cluster_member_active('two-groups', member), 1)
        cancel.set()
        with self.assertRaises(ClientAbort):
            await asyncio.wait_for(queued, 2)
        self.assertEqual(self.manager._cluster_member_active('two-groups', member), 0)
        self.assertEqual(self.manager._prompt_waiting_requests, {})
        await self.emit_token('first', first)
        await asyncio.wait_for(self.request('next'), 2)
        self.assertNotIn('cancelled', [item[0] for item in self.entered])

    async def test_replaced_target_while_queued_is_not_dispatched(self):
        first = await self.request('first')
        queued = asyncio.create_task(self.request('queued'))
        self.addCleanup(queued.cancel)
        await self.wait_queued()
        previous = self.deployment['members'][0]
        self.deployment['members'][0] = {**previous, 'container_id': 'replacement'}
        await self.emit_token('first', first)
        with self.assertRaises(ClusterReplicaUnavailable):
            await asyncio.wait_for(queued, 2)
        self.assertEqual(self.manager._cluster_member_active('two-groups', previous), 0)
        self.assertEqual(self.manager._prompt_waiting_requests, {})
        self.assertEqual([item[0] for item in self.entered], ['first'])

    async def test_failover_waits_for_destination_group_prompt_slot(self):
        occupied = await self.request('occupied', group=1)
        attempts = []

        async def upstream(deployment, member, model, body, endpoint, cancel, **kwargs):
            attempts.append(member['instance_id'])
            if member['instance_id'] == 0:
                raise ClusterReplicaUnavailable('first group unavailable')
            return {'choices': []}

        self.manager._proxy_cluster_member_unlimited = AsyncMock(side_effect=upstream)
        self.manager._cluster_affinity_context = Mock(return_value=None)
        self.manager._prefer_cluster_affinity = Mock(
            side_effect=lambda deployment, candidates, context: candidates,
        )
        self.manager._remember_cluster_affinity = Mock()
        pending = asyncio.create_task(self.manager.proxy_cluster_inference(
            'two-groups', 'model', {'stream': False}, 'chat/completions',
        ))
        self.addCleanup(pending.cancel)
        async def wait_for_failure():
            while not attempts:
                await asyncio.sleep(0)
        await asyncio.wait_for(wait_for_failure(), 2)
        await self.wait_queued()
        self.assertEqual(attempts, [0])
        self.assertFalse(pending.done())
        await self.emit_token('occupied', occupied)
        self.assertEqual(await asyncio.wait_for(pending, 2), {'choices': []})
        self.assertEqual(attempts, [0, 1])
        self.assertEqual(self.manager._prompt_waiting_requests, {})
