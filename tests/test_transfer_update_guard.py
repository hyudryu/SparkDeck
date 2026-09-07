import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from sparkdeck.virtual_nas import VirtualNAS, _transfer_operation


def nas(tmp_path):
    return VirtualNAS(tmp_path, lambda: tmp_path / "hub", Mock(), lambda: True)


def test_update_waits_for_both_export_and_import_and_rejects_new_streams(tmp_path):
    async def scenario():
        service = nas(tmp_path)
        service._reserve_stream("org/source")
        service._reserve_stream("org/target")
        service.reserve_update()
        waiting = asyncio.create_task(service.wait_for_transfers())
        await asyncio.sleep(0)
        assert not waiting.done()
        with pytest.raises(RuntimeError, match="node update is pending"):
            service._reserve_stream("org/new")
        service._release_stream("org/source")
        await asyncio.sleep(0)
        assert not waiting.done()
        service._release_stream("org/target")
        await asyncio.wait_for(waiting, 1)
        service.end_update()
        service._reserve_stream("org/new")
        service._release_stream("org/new")


    asyncio.run(scenario())


def test_existing_job_can_open_its_stream_after_update_is_reserved(tmp_path):
    async def scenario():
        service = nas(tmp_path)
        begin = asyncio.Event()
        finish = asyncio.Event()
        opened = asyncio.Event()

        async def transfer():
            await begin.wait()
            service._reserve_stream("org/model")
            opened.set()
            await finish.wait()
            service._release_stream("org/model")

        job = asyncio.create_task(transfer())
        service._active["target"] = job
        service.reserve_update()
        waiting = asyncio.create_task(service.wait_for_transfers())
        begin.set()
        await asyncio.wait_for(opened.wait(), 1)
        assert not waiting.done()
        finish.set()
        await job
        await asyncio.wait_for(waiting, 1)


    asyncio.run(scenario())


def test_admitted_peer_operation_drains_and_releases_on_failure(tmp_path):
    async def scenario():
        service = nas(tmp_path)
        opened = asyncio.Event()
        finish = asyncio.Event()

        @_transfer_operation
        async def peer(owner, model_id):
            opened.set()
            await finish.wait()
            # A peer pull already admitted before reservation must still be able
            # to open its nested import after establishing the source connection.
            owner._reserve_stream(model_id)
            owner._release_stream(model_id)
            raise ValueError("source failed")

        job = asyncio.create_task(peer(service, "org/model"))
        await opened.wait()
        service.reserve_update()
        waiting = asyncio.create_task(service.wait_for_transfers())
        await asyncio.sleep(0)
        assert not waiting.done()
        finish.set()
        with pytest.raises(ValueError, match="source failed"):
            await job
        await asyncio.wait_for(waiting, 1)
        assert not service._transfer_tasks
        assert not service._streaming_models


    asyncio.run(scenario())


def test_queued_transfer_stays_queued_until_update_releases(tmp_path):
    async def scenario():
        service = nas(tmp_path)
        service.jobs = [{"id": "copy", "status": "queued", "target_node_id": "target"}]
        service.reserve_update()
        started = asyncio.Event()

        async def run(job):
            job["status"] = "completed"
            started.set()

        with patch.object(service, "_run_transfer", new=AsyncMock(side_effect=run)) as runner:
            service.start()
            await asyncio.sleep(0)
            runner.assert_not_called()
            assert service.jobs[0]["status"] == "queued"
            service.end_update()
            await asyncio.wait_for(started.wait(), 1)
            await asyncio.wait_for(service.stop(), 1)
        runner.assert_awaited_once()
    asyncio.run(scenario())
