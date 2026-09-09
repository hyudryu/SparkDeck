import unittest
from unittest.mock import AsyncMock

from mcp_server import build_server


class ImagePatchMcpTests(unittest.IsolatedAsyncioTestCase):
    async def test_patch_tools_use_the_gui_build_api(self):
        client = type("Client", (), {})()
        client._request = AsyncMock(return_value={"items": []})
        server = build_server(client)
        await server.call_tool("list_image_patch_builds", {})
        client._request.assert_awaited_with("GET", "/api/v1/images/patch-builds")
        payload = {"base_image": "base:v1", "image": "patched:v1",
                   "node_ids": ["worker-1"],
                   "files": [{"target": "/opt/patch.py", "content": "VALUE = 42\n"}]}
        client._request.return_value = {"id": "build-1", "status": "queued"}
        await server.call_tool("create_patched_image", payload)
        client._request.assert_awaited_with("POST", "/api/v1/images/patch-builds", json_body=payload)
