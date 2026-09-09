import json
from unittest.mock import patch

import pytest

from manager import Manager, SourceRoutingUnavailable


def _rule(model="model-a", **updates):
    return {
        "source_ip": "2001:db8::1",
        "requested_model": model,
        "enabled": False,
        "deployment_id": "record-1",
        "instance_id": None,
        "node_ids": ["node-a"],
        **updates,
    }


@pytest.fixture
def manager(tmp_path):
    manager = Manager.__new__(Manager)
    manager.source_ip_routing_rules_path = tmp_path / "rules.json"
    manager.source_ip_routing_rules = manager._load_source_ip_routing_rules()
    return manager


def test_source_lookup_tracks_upsert_delete_and_canonical_addresses(manager):
    first = manager.upsert_source_ip_routing_rule(_rule())
    second = manager.upsert_source_ip_routing_rule(_rule("model-b"))
    manager.upsert_source_ip_routing_rule(_rule(source_ip="10.0.0.1"))
    assert manager.source_ip_routing_rules_for_source("2001:0db8:0:0::1") == [
        first, second,
    ]
    replacement = manager.upsert_source_ip_routing_rule(_rule(node_ids=["node-b"]))
    assert manager.source_ip_routing_rules_for_source("2001:db8::1") == [
        replacement, second,
    ]
    manager.delete_source_ip_routing_rule("2001:0db8::1", "model-a")
    assert manager.source_ip_routing_rules_for_source("2001:db8::1") == [second]
    manager.delete_source_ip_routing_rule("2001:db8::1", "model-b")
    assert manager.source_ip_routing_rules_for_source("2001:db8::1") == []


def test_source_index_reloads_disabled_rows_and_fails_closed(manager):
    manager.upsert_source_ip_routing_rule(_rule())
    row = _rule("replacement", source_ip="2001:0db8::2")
    manager.source_ip_routing_rules_path.write_text(json.dumps({
        "version": 1, "rules": [row],
    }), encoding="utf-8")
    manager.source_ip_routing_rules = manager._load_source_ip_routing_rules()
    assert manager.source_ip_routing_rules_for_source("2001:db8::1") == []
    assert manager.source_ip_routing_rules_for_source("2001:db8::2") == [
        {**row, "source_ip": "2001:db8::2"},
    ]
    manager.source_ip_routing_rules_path.write_text("broken", encoding="utf-8")
    manager.source_ip_routing_rules = manager._load_source_ip_routing_rules()
    with pytest.raises(SourceRoutingUnavailable):
        manager.source_ip_routing_rules_for_source("10.0.0.1")


def test_failed_writes_preserve_source_index(manager):
    saved = manager.upsert_source_ip_routing_rule(_rule())
    with patch("manager._atomic_private_json_write", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            manager.upsert_source_ip_routing_rule(_rule(node_ids=["replacement"]))
        assert manager.source_ip_routing_rules_for_source("2001:db8::1") == [saved]
        with pytest.raises(OSError):
            manager.delete_source_ip_routing_rule("2001:db8::1", "model-a")
        assert manager.source_ip_routing_rules_for_source("2001:db8::1") == [saved]


def test_absent_source_does_not_scan_or_sort_rules(manager):
    class GuardedRules(dict):
        forbid_scan = False

        def values(self):
            assert not self.forbid_scan, "request scanned global routing rules"
            return super().values()

    rules = GuardedRules({
        str(index): _rule(str(index), source_ip=f"10.0.{index // 256}.{index % 256}")
        for index in range(1000)
    })
    manager.source_ip_routing_rules = rules
    manager._index_source_ip_routing_rules(rules)
    rules.forbid_scan = True
    with patch.object(manager, "list_source_ip_routing_rules", side_effect=AssertionError):
        for source in ("192.0.2.1", "2001:db8::99", "unknown"):
            assert manager.source_ip_routing_rules_for_source(source) == []
