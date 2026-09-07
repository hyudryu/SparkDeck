from sparkdeck.prefix_affinity import PrefixAffinity


def prompt(question="Fix my parser", *, answer=False, system="Shared AGENTS.md"):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Shared harness instructions"},
                {"role": "user", "content": question}]
    if answer:
        messages += [{"role": "assistant", "content": "Try this patch"},
                     {"role": "user", "content": "Now add tests"}]
    return {"messages": messages}


def keys(body):
    return PrefixAffinity.keys(body, "chat/completions")


def test_following_turn_matches_complete_prior_prompt_not_boilerplate():
    cache = PrefixAffinity()
    cache.remember("model", keys(prompt()), "A")
    assert cache.lookup("model", keys(prompt(answer=True)), ["A", "B"]) == "A"
    assert cache.lookup("model", keys(prompt("Other task", answer=True)), ["A", "B"]) is None
    assert cache.lookup("model", keys(prompt()), ["A", "B"]) is None
    assert cache.lookup("model", keys(prompt(answer=True, system="Changed")), ["A", "B"]) is None


def test_longest_served_history_wins_after_overflow():
    cache = PrefixAffinity()
    cache.remember("model", keys(prompt()), "A")
    second = prompt(answer=True)
    cache.remember("model", keys(second), "B")
    third = {"messages": second["messages"] + [
        {"role": "assistant", "content": "Tests done"}, {"role": "user", "content": "Review"}]}
    assert cache.lookup("model", keys(third), ["A", "B"]) == "B"
    assert cache.lookup("other-model", keys(third), ["A", "B"]) is None
    assert cache.lookup("model", keys(third), ["C"]) is None


def test_tools_and_multimodal_content_are_part_of_prefix():
    cache = PrefixAffinity()
    first = prompt()
    first["tools"] = [{"type": "function", "function": {"name": "read"}}]
    first["messages"][-1]["content"] = [{"type": "text", "text": "Look"},
        {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}]
    cache.remember("model", keys(first), "A")
    following = {**first, "messages": first["messages"] + [{"role": "assistant", "content": "Seen"},
        {"role": "user", "content": "Explain"}]}
    assert cache.lookup("model", keys(following), ["A"]) == "A"
    assert cache.lookup("model", keys({**following, "tools": []}), ["A"]) is None
    assert cache.lookup("model", keys({**following, "temperature": 0.5}), ["A"]) == "A"


def test_ttl_capacity_and_retained_data():
    now = [0]
    cache = PrefixAffinity(ttl=10, capacity=2, clock=lambda: now[0])
    for question in ("first", "second", "third"):
        cache.remember("model", keys(prompt(question)), "A")
    assert len(cache.entries) == 2
    assert "third" not in repr(cache.entries)
    assert cache.lookup("model", keys(prompt("first", answer=True)), ["A"]) is None
    now[0] = 10
    assert cache.lookup("model", keys(prompt("third", answer=True)), ["A"]) is None


def test_unsupported_or_invalid_requests_fall_back_without_affinity():
    assert PrefixAffinity.keys({"prompt": "Shared instructions"}, "completions") == []
    assert keys({"messages": [None]}) == []
    assert keys({"messages": [{"role": "system", "content": "Only instructions"}]}) == []
    assert keys(prompt("x" * (2 * 1024 * 1024))) == []
