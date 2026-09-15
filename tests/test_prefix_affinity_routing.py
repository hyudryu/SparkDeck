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

    async def test_common_boilerplate_does_not_pin_new_conversations(self):
        manager, _ = setup()
        await self.send(manager, prompt())
        await self.send(manager, prompt(task='Explain the different volcanic sample 912'))
        self.assertEqual(manager._proxy_cluster_member.await_args.args[1]['node_id'], 'node-1')

    async def test_group_identity_and_load_overflow(self):
        manager, deployment = setup('grouped_sharded')
        await self.send(manager, prompt())
        preferred = deployment['members'][0]
        for _ in range(2):
            manager._acquire_cluster_member('d', preferred)
        await self.send(manager, prompt(2))
        self.assertEqual(manager._proxy_cluster_member.await_args.args[1]['instance_id'], 0)
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
