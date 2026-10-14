import asyncio
import unittest
from unittest.mock import AsyncMock

from manager import Manager


def prompt(turn=1, task='Explain the specific lattice defect in sample 724'):
    messages = [
        {'role': 'system', 'content': 'Shared AGENTS.md instructions'},
        {'role': 'user', 'content': task},
    ]
    if turn > 1:
        messages += [
            {'role': 'assistant', 'content': 'The defect is a vacancy.'},
            {'role': 'user', 'content': 'How would I measure it?'},
        ]
    return {'model': 'deepseek', 'messages': messages}


def setup(mode='replicated'):
    manager = Manager.__new__(Manager)
    members = [
        {'node_id': f'node-{i}', 'container_name': f'engine-{i}',
         'container_id': f'container-{i}', 'rank': i}
        for i in range(2)
    ]
    if mode == 'grouped_sharded':
        members = [
            {'node_id': f'node-{group}-{rank}', 'container_name': f'engine-{group}-{rank}',
             'container_id': f'container-{group}-{rank}', 'instance_id': group, 'rank': rank,
             'status': 'running'}
            for group in range(2) for rank in range(2)
        ]
    deployment = {'id': 'd', 'mode': mode, 'members': members}
    manager.deployments = [deployment]
    manager._proxy_cluster_member = AsyncMock(return_value={'choices': []})
    return manager, deployment


class PrefixRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def send(self, manager, body, **kwargs):
        return await manager.proxy_cluster_inference('d', 'deepseek', body, 'chat/completions', **kwargs)

    async def test_followup_prefers_previous_replica_without_session_id(self):
        manager, deployment = setup()
        await self.send(manager, prompt())
        await self.send(manager, prompt(2))
        self.assertEqual([c.args[1]['node_id'] for c in manager._proxy_cluster_member.await_args_list], ['node-0', 'node-0'])

    async def test_concurrent_cached_requests_spread_within_one(self):
        manager, _ = setup()
        # Hold each dispatch in flight (acquire without release) so the second
        # and third concurrent arrivals see the cached replica already busy.
        async def hold(deployment, member, *args, **kwargs):
            manager._acquire_cluster_member('d', member)
            return {'choices': []}
        manager._proxy_cluster_member.side_effect = hold
        for _ in range(3):
            await self.send(manager, prompt(2))
        nodes = {c.args[1]['node_id'] for c in manager._proxy_cluster_member.await_args_list}
        self.assertEqual(nodes, {'node-0', 'node-1'})
        self.assertEqual(sorted(manager._cluster_member_loads().values()), [1, 2])

    async def test_reservation_holds_across_readiness_probe(self):
        # Real unlimited path (no gate): a request still awaiting its readiness
        # probe must keep its load reservation, so a concurrent cached request
        # sees it busy and routes to the other replica instead of queueing on
        # the same member.
        from manager import LOCAL_NODE_ID
        manager, deployment = setup()
        del manager._proxy_cluster_member  # exercise the real gated/unlimited path
        for member in deployment['members']:
            member['node_id'] = LOCAL_NODE_ID
        ready = asyncio.Event()
        probes = []
        served = []

        async def health(model, **kwargs):
            probes.append(kwargs.get('container_name'))
            if kwargs.get('container_name') == 'engine-0' and len(probes) > 1:
                await ready.wait()
            return True
        manager.inference_target_health = health

        async def chat(model, body, stream, cancel, *, container_name, **kwargs):
            served.append(container_name)
            return {'choices': []}
        manager._vllm_chat = chat

        await self.send(manager, prompt())  # caches the conversation on engine-0
        first = asyncio.create_task(self.send(manager, prompt(2)))
        for _ in range(200):
            if len(probes) >= 2:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(probes[-1], 'engine-0')  # first is awaiting readiness
        await self.send(manager, prompt(2))
        self.assertFalse(first.done())
        self.assertEqual(served, ['engine-0', 'engine-1'])
        ready.set()
        await first
        self.assertEqual(served, ['engine-0', 'engine-1', 'engine-0'])
        self.assertEqual(set(manager._cluster_member_loads().values()), {0})

    async def test_cached_hint_survives_idle_unavailable_replica(self):
        manager, deployment = setup()
        for member in deployment['members']:
            member['status'] = 'running'
        await self.send(manager, prompt())  # caches on node-0
        deployment['members'][1]['status'] = 'exited'  # idle but cannot serve
        manager._acquire_cluster_member('d', deployment['members'][0])
        await self.send(manager, prompt(2))
        self.assertEqual(manager._proxy_cluster_member.await_args.args[1]['node_id'], 'node-0')

    async def test_cached_unavailable_member_does_not_take_preference(self):
        manager, deployment = setup()
        for member in deployment['members']:
            member['status'] = 'running'
        await self.send(manager, prompt())  # caches on node-0
        deployment['members'][0]['status'] = 'exited'
        await self.send(manager, prompt(2))
        self.assertEqual(manager._proxy_cluster_member.await_args.args[1]['node_id'], 'node-1')

    async def test_common_boilerplate_does_not_pin_new_conversations(self):
        manager, _ = setup()
        await self.send(manager, prompt())
        await self.send(manager, prompt(task='Explain the different volcanic sample 912'))
        self.assertEqual(manager._proxy_cluster_member.await_args.args[1]['node_id'], 'node-1')

    async def test_group_identity_and_load_overflow(self):
        manager, deployment = setup('grouped_sharded')
        await self.send(manager, prompt())
        preferred = deployment['members'][0]
        # A single in-flight request already leaves the cached group one above
        # the idle group, so a cached follow-up must spread instead of stacking.
        manager._acquire_cluster_member('d', preferred)
        await self.send(manager, prompt(2))
        self.assertEqual(manager._proxy_cluster_member.await_args.args[1]['instance_id'], 1)
        self.assertEqual(manager._proxy_cluster_member.await_args.args[1]['rank'], 0)

    async def test_restart_and_caller_boundary_discard_old_affinity(self):
        for change in ('container', 'caller'):
            manager, deployment = setup()
            await self.send(manager, prompt(), caller_ip='caller-a')
            if change == 'container':
                deployment['members'][0]['container_id'] = 'replacement'
            await self.send(manager, prompt(2), caller_ip='caller-b' if change == 'caller' else 'caller-a')
            self.assertEqual(manager._proxy_cluster_member.await_args.args[1]['node_id'], 'node-1')

    async def test_failed_nonstream_does_not_teach_affinity(self):
        manager, _ = setup()
        manager._proxy_cluster_member.return_value = {'error': {'message': 'invalid'}}
        await self.send(manager, prompt())
        await self.send(manager, prompt(2))
        self.assertEqual(manager._proxy_cluster_member.await_args.args[1]['node_id'], 'node-1')

    async def test_stream_failover_remembers_actual_successful_member(self):
        manager, _ = setup()
        async def failed():
            yield 'data: {"error":{"type":"upstream_error","code":503}}\n\n'
        async def success():
            yield 'data: {"choices":[]}\n\n'
            yield 'data: [DONE]\n\n'
        manager._proxy_cluster_member.side_effect = [failed(), success(), {'choices': []}]
        stream = await self.send(manager, {**prompt(), 'stream': True})
        async for _ in stream:
            pass
        await self.send(manager, prompt(2))
        self.assertEqual(manager._proxy_cluster_member.await_args.args[1]['node_id'], 'node-1')

    async def test_error_truncated_and_canceled_streams_do_not_teach_affinity(self):
        for kind in ('error', 'truncated', 'cancel'):
            manager, _ = setup()
            cancel = asyncio.Event()
            async def response():
                yield 'data: {"choices":[]}\n\n'
                if kind == 'error':
                    yield 'data: {"error":{"message":"out of memory"}}\n\n'
                if kind == 'cancel':
                    cancel.set()
                if kind != 'truncated':
                    yield 'data: [DONE]\n\n'
            manager._proxy_cluster_member.side_effect = [response(), {'choices': []}]
            stream = await self.send(manager, {**prompt(), 'stream': True}, cancel=cancel)
            async for _ in stream:
                pass
            await self.send(manager, prompt(2))
            self.assertEqual(manager._proxy_cluster_member.await_args.args[1]['node_id'], 'node-1')
