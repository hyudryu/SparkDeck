"""Bounded, process-local routing hints; never stores conversation text."""

import hashlib
import json
import time
from collections import OrderedDict


class PrefixAffinity:
    def __init__(self, *, ttl=1800, capacity=4096, clock=time.monotonic):
        self.ttl = ttl
        self.capacity = capacity
        self.clock = clock
        self.entries = OrderedDict()

    @staticmethod
    def keys(body, endpoint):
        # Plain completion strings have no reliable conversation boundaries.
        # Keep normal balancing rather than guessing where boilerplate ends.
        if endpoint != "chat/completions":
            return []
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages or len(messages) > 512:
            return []
        if any(not isinstance(message, dict) for message in messages):
            return []
        try:
            # These fields can change the rendered prefix. Sampling controls
            # do not, and must not break affinity between turns.
            context = {key: body[key] for key in (
                "tools", "tool_choice", "functions", "function_call",
                "chat_template", "chat_template_kwargs", "add_generation_prompt",
                "continue_final_message", "documents", "cache_salt",
                "response_format",
            ) if key in body}
            encoded = json.dumps(context, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            digest = hashlib.sha256(encoded)
            size = len(encoded)
            following = any(message.get("role") == "assistant" for message in messages)
            conversation = False
            keys = []
            for message in messages:
                encoded = json.dumps(message, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
                size += len(encoded)
                if size > 2 * 1024 * 1024:
                    return []
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
                conversation |= message.get("role") in {"user", "tool", "function"}
                keys.append((digest.hexdigest(), size if following and conversation else 0))
            if messages[-1].get("role") in {"system", "developer"}:
                return []
            return keys
        except (TypeError, ValueError, RecursionError, UnicodeError):
            return []

    def _expire(self):
        now = self.clock()
        for key, (_, touched) in list(self.entries.items()):
            if now - touched >= self.ttl:
                del self.entries[key]

    def lookup(self, scope, keys, candidates):
        self._expire()
        allowed = set(candidates)
        # Longer historical prompts imply more reusable work. System-only
        # matches and first-turn boilerplate never participate in selection.
        for digest, strength in sorted(keys, key=lambda item: item[1], reverse=True):
            if strength <= 0:
                continue
            key = (scope, digest)
            entry = self.entries.get(key)
            if entry is not None and entry[0] in allowed:
                self.entries[key] = (entry[0], self.clock())
                self.entries.move_to_end(key)
                return entry[0]
        return None

    def remember(self, scope, keys, target):
        if not keys:
            return
        self._expire()
        # Only a complete previously served prompt is evidence. Indexing each
        # leading message would incorrectly pin unrelated boilerplate users.
        key = (scope, keys[-1][0])
        self.entries[key] = (target, self.clock())
        self.entries.move_to_end(key)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
