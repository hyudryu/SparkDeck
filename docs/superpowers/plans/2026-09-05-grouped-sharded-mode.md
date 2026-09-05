# Grouped Sharded Deployment Mode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `grouped_sharded` deployment mode so one deployment can run multiple independent sharded engine groups (each a TP-sized node group) under a single served name, load-balanced and per-instance start/stop-able.

**Architecture:** A new deployment `mode` value. `node_ids` are partitioned consecutively into `instances` (=N) groups of `tensor_parallel_size` (=T) nodes each; each group builds its own sharded engine (own master addr/port). Gateway routing balances across the rank-0 coordinator of each *running* group (reusing `_balanced_cluster_member` + `_cluster_stream_with_failover`), keyed per instance. Lifecycle supports per-instance start/stop with derived deployment state.

**Tech Stack:** Python 3.14 / FastAPI backend (`manager.py`, `sparkdeck/service.py`, `sparkdeck/storage.py`), pytest; React/TypeScript frontend (`frontend/`), Vitest.

**Spec:** `docs/superpowers/specs/2026-09-05-grouped-sharded-mode-design.md`

## Global Constraints

- Work only inside the worktree `C:\Users\markx\Documents\GitHub\sparkdeck-grouped-sharded` (branch `feat/grouped-sharded-mode`); never commit to main.
- New mode value is exactly `grouped_sharded` (verbatim).
- Invariant: `len(node_ids) == N*T`, no duplicate nodes, `T >= 2`, `N >= 1`. Defaults: `tensor_parallel_size` from existing setting, `instances` defaults to 1.
- Existing modes `single`/`sharded`/`replicated` behavior must not change.
- Backend tests run with `python -m pytest` from the worktree root (system Python 3.14, pytest 9.0.3). Frontend tests run from `frontend/` with the repo's test runner.
- Each task ends with a commit.

---

### Task 1: Backend — accept and validate `grouped_sharded` deployment mode

**Files:**
- Modify: `sparkdeck/manager.py`
- Modify: `sparkdeck/service.py`
- Test: `tests/test_grouped_sharded_mode.py` (create)

**Interfaces:**
- Produces: `mode = "grouped_sharded"` accepted wherever `{"single","replicated","sharded"}` is checked; validation that `len(node_ids) == instances*tensor_parallel_size`, `instances>=1`, `tensor_parallel_size>=2`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_grouped_sharded_mode.py`:

```python
import pytest
from sparkdeck.service import Service


@pytest.fixture
def service(tmp_path):
    # Reuse the existing Service test-construction helper. Several suites in
    # this repo build Service with a temp data dir; mirror that pattern.
    raise NotImplementedError("fill in from an existing Service fixture")


def test_grouped_sharded_mode_accepted_in_validation(service):
    # A mode allow-list helper must accept "grouped_sharded".
    from sparkdeck.service import _MODE_ALLOWLIST
    assert "grouped_sharded" in _MODE_ALLOWLIST
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_grouped_sharded_mode.py -q`
Expected: FAIL (module/allowlist not present)

- [ ] **Step 3: Implement**

Add a module constant in `sparkdeck/service.py` near the top:

```python
_MODE_ALLOWLIST = frozenset({"single", "replicated", "sharded", "grouped_sharded"})
```

Replace the inline `{"single", "sharded", "replicated"}` allow-list checks in `service.py` (lines ~2435, ~2422, ~2456, ~2992) with membership in `_MODE_ALLOWLIST`, updating the error message to list all four.

In `manager.py`, replace inline mode checks (lines ~4352, ~5349, ~9904, ~10052) and the error text the same way.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_grouped_sharded_mode.py -q`
Expected: PASS

- [ ] **Step 5: Validate node-count gate and commit**

In `manager.py` `_preflight_deployment_launch` (near the mode validation), add for `grouped_sharded`:

```python
if mode == "grouped_sharded":
    tp = max(1, int(settings.get("tensor_parallel_size") or 1))
    instances = max(1, int(settings.get("instances") or 1))
    required = tp * instances
    if tp < 2:
        raise ValueError("grouped_sharded requires tensor_parallel_size >= 2")
    if len(node_ids) != required:
        raise ValueError(
            f"grouped_sharded requires {instances} instance(s) of TP{tp} "
            f"= {required} node(s), got {len(node_ids)}"
        )
```

Run: `python -m pytest tests/test_grouped_sharded_mode.py tests/test_cluster_load_balancing.py tests/test_sparkdeck_nodes.py -q`
Commit:
```bash
git add sparkdeck/manager.py sparkdeck/service.py tests/test_grouped_sharded_mode.py
git commit -m "feat: accept and validate grouped_sharded deployment mode"
```

---

### Task 2: Backend — build per-group sharded engine members

**Files:**
- Modify: `sparkdeck/manager.py` (member-build loop near lines 5601–5670)

**Interfaces:**
- Consumes: `mode`, `instances`, `tensor_parallel_size`, `node_ids`, `fabrics`, `vllm_parallel_layout`.
- Produces: `deployment["members"]` where each member has `instance_id` (group index) plus existing `rank` (within-group local rank), per-group `master_addr`/`master_port`; `api_port` from the first group's coordinator.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_grouped_sharded_mode.py`:

```python
def test_grouped_sharded_members_have_per_group_master(tmp_path):
    manager = build_manager_for_grouped_sharded()  # see task fixture
    # launch 4 nodes TP2 N2
    # assert len(members) == 4
    # assert members[0]["instance_id"] == 0 and members[2]["instance_id"] == 1
    # assert per-group ranks are 0,1 within each group
    # assert group 0 master == fabric of node[0], group 1 master == fabric of node[2]
    raise NotImplementedError
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_grouped_sharded_mode.py -q`
Expected: FAIL

- [ ] **Step 3: Implement the two-level member loop**

In `manager.py` `_cluster launch`, restructure the member loop. Compute `instances` and `tp` from the launch body:

```python
mode = deployment.get("mode")
inferred_instances = 1
inferred_tp = len(node_ids)
if mode == "grouped_sharded":
    inferred_tp = max(1, int(body.get("tensor_parallel_size") or 1))
    inferred_instances = max(1, int(body.get("instances") or 1))
```

Then in the loop:

```python
tasks, member_specs = [], []
group_port_base = 29501
for group in range(inferred_instances):
    group_nodes = node_ids[group * inferred_tp: (group + 1) * inferred_tp]
    group_master_ip = fabrics[group_nodes[0]][0]
    group_master_port = group_port_base + group
    for local_rank, node_id in enumerate(group_nodes):
        node = available[node_id]
        member_port = local_port if node_id == LOCAL_NODE_ID else None
        fabric_ip, fabric_interface = fabrics[node_id]
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "-", model).strip("-").lower()
        global_rank = group * inferred_tp + local_rank
        name = f"cluster-{deployment_id}-r{global_rank}-{safe_model[:36]}"
        payload = dict(base)
        payload.update({
            "port": member_port,
            "name": name,
            "cluster_member": {
                "deployment_id": deployment_id,
                "node_id": node_id,
                "rank": local_rank,
                "instance_id": group,
                "nnodes": inferred_tp,
                "mode": mode,
                "serve_port": member_port,
                "fabric_ip": fabric_ip,
                "fabric_interface": fabric_interface,
            },
        })
        if mode == "sharded" or mode == "grouped_sharded":
            if engine == "vllm":
                tp_size, pp_size = (inferred_tp, 1) if mode == "grouped_sharded" else (vllm_parallel_layout or (1, len(node_ids)))
                vllm_args = self._without_cli_options(
                    payload["extra_args"],
                    {"--distributed-executor-backend", "--nnodes", "--node-rank",
                     "--master-addr", "--master-port", "--tensor-parallel-size", "-tp",
                     "--pipeline-parallel-size", "-pp", "--headless"},
                )
                payload["extra_args"] = vllm_args + [
                    "--distributed-executor-backend", "mp",
                    "--nnodes", str(inferred_tp),
                    "--node-rank", str(local_rank),
                    "--master-addr", group_master_ip,
                    "--master-port", str(group_master_port),
                    "--tensor-parallel-size", str(tp_size),
                    "--pipeline-parallel-size", str(pp_size),
                ]
                if local_rank > 0:
                    payload["extra_args"].append("--headless")
            else:
                payload["extra_args"] = self._without_cli_options(
                    payload["extra_args"],
                    {"--nnodes", "--node-rank", "--dist-init-addr", "--tp-size"},
                ) + [
                    "--nnodes", str(inferred_tp),
                    "--node-rank", str(local_rank),
                    "--dist-init-addr", f"{group_master_ip}:{group_master_port}",
                ]
                payload["sg_tp_size"] = inferred_tp
        member_specs.append({
            "node_id": node_id, "node_name": node.get("name", node_id),
            "rank": local_rank, "instance_id": group,
            "container_name": name, "fabric_ip": fabric_ip,
            "port": member_port, "status": "queued",
            "phase": {"phase": "queued", "message": "Waiting for the node agent to begin launch"},
        })
        tasks.append(self._create_member(node_id, payload))
```

Keep the existing single/sharded/replicated path when `mode` is not `grouped_sharded` by routing through the existing loop. Simplest: branch at the top — if `mode == "grouped_sharded"`, use the new loop; else preserve the original flat loop verbatim.

After the loop set `deployment["api_port"] = member_specs[0].get("port")` (first group's coordinator, unchanged).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_grouped_sharded_mode.py tests/test_cluster_load_balancing.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sparkdeck/manager.py tests/test_grouped_sharded_mode.py
git commit -m "feat: build per-group sharded engine members for grouped_sharded"
```

---

### Task 3: Backend — per-instance routing and load balancing

**Files:**
- Modify: `sparkdeck/manager.py` (`_cluster_route_order`, `_cluster_member_key`, `_balanced_cluster_member`, `_cluster_members_sorted`)

**Interfaces:**
- Consumes: `deployment["members"]` with `instance_id`, `rank`, `status`.
- Produces: for `grouped_sharded`, candidates = rank-0 coordinators of running groups; load key = `deployment_id:instance_id`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_grouped_sharded_mode.py`:

```python
def test_grouped_sharded_balances_across_group_coordinators(tmp_path):
    # members = 4 (two TP2 groups, ranks 0,0 coordinators only are candidates)
    # assert _cluster_route_order returns [coord_group0, coord_group1] order per
    # least-loaded + round-robin
    assert True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_grouped_sharded_mode.py -q`
Expected: FAIL

- [ ] **Step 3: Implement**

Generalize `_cluster_members_sorted` usage and `_cluster_route_order`:

```python
def _cluster_route_order(self, deployment: dict) -> list[dict]:
    members = self._cluster_members_sorted(deployment)
    mode = deployment.get("mode")
    if mode == "grouped_sharded":
        # coordinators = rank-0 member of each running group
        by_instance = {}
        for member in members:
            if int(member.get("rank") or 0) != 0:
                continue
            instance = str(member.get("instance_id") or 0)
            if str(member.get("status") or "") in ("stopped", "error"):
                continue
            by_instance.setdefault(instance, member)
        coordinators = [by_instance[k] for k in sorted(by_instance, key=lambda x: int(x))]
        if not coordinators:
            return members[:1]
        chosen = self._balanced_cluster_member(deployment, coordinators)
        rest = [m for m in coordinators if m is not chosen]
        rest.sort(key=lambda m: self._cluster_member_active(deployment["id"], m))
        return [chosen, *rest]
    if mode != "replicated" or len(members) < 2:
        return members[:1]
    deployment_id = str(deployment.get("id") or "")
    chosen = self._balanced_cluster_member(deployment)
    rest = [m for m in members if m is not chosen]
    rest.sort(key=lambda m: self._cluster_member_active(deployment_id, m))
    return [chosen, *rest]
```

Extend `_balanced_cluster_member` to accept an explicit candidate list, and `_cluster_member_key` to key by instance for grouped mode:

```python
def _cluster_member_key(self, deployment_id: str, member: dict) -> str:
    if str(member.get("instance_id") or "") != "":
        return f"{deployment_id}:instance:{member['instance_id']}"
    identity = member.get("container_name") or member.get("node_id") or ""
    return f"{deployment_id}:{identity}"
```

Do not change per-rank load tracking for sharded/single (those route normally).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_grouped_sharded_mode.py tests/test_cluster_load_balancing.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sparkdeck/manager.py tests/test_grouped_sharded_mode.py
git commit -m "feat: per-instance routing and load balancing for grouped_sharded"
```

---

### Task 4: Backend — per-instance start/stop lifecycle

**Files:**
- Modify: `sparkdeck/manager.py` (`deployment_action`, per-member desired state)
- Modify: `sparkdeck/service.py` (deployment-action route passes `instance`)

**Interfaces:**
- Consumes: `deployment_action(deployment_id, action, node_ids, ..., instance=None)`.
- Produces: derived deployment `status` (`running`/`stopped`/`degraded`) from per-instance states; targeted `start`/`stop` toggles only one instance group.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_grouped_sharded_mode.py`:

```python
def test_per_instance_stop_derives_degraded_state(tmp_path):
    # two instances running; stop instance 1 (members 2,3)
    # assert deployment status == 'degraded', instance 0 still running
    assert True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_grouped_sharded_mode.py -q`
Expected: FAIL

- [ ] **Step 3: Implement**

In `deployment_action`, accept `instance: int | None = None`. When set and mode is `grouped_sharded`, filter the members acted on to those with that `instance_id` before gathering `_member_action`. After actions, derive the deployment state by inspecting remaining running members, and set `deployment["status"]` accordingly.

```python
targeted = deployment.get("members", [])
if instance is not None:
    targeted = [m for m in deployment["members"]
                if int(m.get("instance_id") or 0) == instance]
results = await asyncio.gather(
    *[self._member_action(m, action) for m in targeted],
    return_exceptions=True,
)
```

Then derive state:

```python
def _grouped_running(members):
    return any(
        int(m.get("rank") or 0) == 0
        and str(m.get("status") or "") not in ("stopped", "error")
        for m in members
    )
```

Set `deployment["status"]` to `running`/`stopped`/`degraded` based on how many groups are running, persisting `desired_state` for the targeted members only. Keep full start/stop (instance=None) behavior unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_grouped_sharded_mode.py tests/test_cluster_load_balancing.py tests/test_sparkdeck_nodes.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sparkdeck/manager.py sparkdeck/service.py tests/test_grouped_sharded_mode.py
git commit -m "feat: per-instance start/stop lifecycle for grouped_sharded"
```

---

### Task 5: Frontend — layout option, instances/TP controls, per-instance controls

**Files:**
- Modify: `frontend/src/pages/ModelsPage.tsx`
- Modify: `frontend/src/pages/DeploymentPage.tsx`
- Modify: `frontend/src/api/types.ts` (`deployment_mode` union)
- Modify: `frontend/src/api/client.ts` (payload passthrough for `instances`)
- Modify: `frontend/src/pages/ModelsPage.test.tsx` / `DeploymentPage.test.tsx`

**Interfaces:**
- Consumes: `deployment_mode: 'single' | 'replicated' | 'sharded' | 'grouped_sharded'`, `instances?: number`.
- Produces: UI that edits and submits `instances` and `tensor_parallel_size`.

- [ ] **Step 1: Write the failing test**

Add a component test asserting the `Grouped sharded` option exists and that selecting it reveals an `instances` field; assert `instances` is submitted.

- [ ] **Step 2: Run test to verify it fails**

Run from `frontend/`: `pnpm test <file>` (or the repo's configured Vitest command).
Expected: FAIL

- [ ] **Step 3: Implement**

Extend `deployment_mode` union in `types.ts` with `'grouped_sharded'`. In `ModelsPage.tsx` add a `Grouped sharded` `<option>` in the layout select and an `instances` numeric field shown only when `grouped_sharded` is selected; thread `instances` into the create payload. In `DeploymentPage.tsx`, render per-instance state/buttons when present.

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm test <file>`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/ModelsPage.tsx frontend/src/pages/DeploymentPage.tsx frontend/src/api/types.ts frontend/src/api/client.ts
git commit -m "feat: grouped_sharded layout controls in frontend"
```

---

### Task 6: Full suite + PR

- [ ] **Step 1: Run backend tests**

Run: `python -m pytest -q`
Expected: all pass (or only pre-existing failures documented).

- [ ] **Step 2: Run frontend build/tests**

Run from `frontend/`: the repo's configured `pnpm test` / `pnpm build`.
Expected: pass.

- [ ] **Step 3: Push and open PR**

```bash
git push -u origin feat/grouped-sharded-mode
gh pr create --base main --title "feat: grouped_sharded deployment mode" --body "Add a grouped_sharded deployment mode so one deployment can run multiple independent sharded engine groups (each a TP-sized node group) behind a single served name, with per-instance load balancing and per-instance start/stop."
```

- [ ] **Step 4: Wait for Codex review**

Poll `gh pr view <pr> --comments` and `gh api repos/{owner}/{repo}/pulls/<pr>/comments`. Address every comment, push, repeat until none remain.

- [ ] **Step 5: Stop at approval**

Check `gh api repos/{owner}/{repo}/issues/<pr>/reactions` for a 👍 on the PR body. When no unresolved Codex comments remain AND the body has a `+1`, stop and report for the user to merge.
