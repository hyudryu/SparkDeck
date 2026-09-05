# Grouped Sharded Deployment Mode — Design

Date: 2026-09-05
Status: Draft
Branch: `feat/grouped-sharded-mode`

## Problem

On a 4-node cluster, a user with a tensor-parallel (TP2) deployment can only
start it on one pair (nodes 1+2). The deployment's Start toggle flips to Stop,
and SparkDeck refuses to run a second independent TP2 engine on the other pair
(nodes 3+4). The user sometimes wants two independent TP2 engines instead of a
single TP4, so requests load-balance automatically across the two engines that
share one served name.

The existing three deployment modes cannot express this:

- `single` — one engine on one node.
- `sharded` — one engine split across N nodes (TP/PP); only rank-0 serves.
- `replicated` — N independent engines, one **full model copy per node**.

None of these give "N independent engines, each of which is itself sharded
across a TP-sized node group."

## Goal

Add a fourth deployment mode, `grouped_sharded`, that lets a single deployment
run multiple independent sharded engine groups, each on a disjoint TP-sized
node set, all serving the same served name, with automatic load balancing and
failover across the running groups.

## Non-goals

- Not a change to served-name/alias uniqueness semantics. This stays a single
  deployment with one alias and one served name, so `_live_deployment_for_model_id`
  ambiguity is avoided entirely.
- Not runtime scale up/down of instance count (that is a follow-up; see Future
  work). Instance count is fixed at save/launch.
- Not cross-deployment load balancing.

## Model

A `grouped_sharded` deployment persists:

- `tensor_parallel_size` = `T` — ranks (nodes) per engine group.
- `instances` = `N` — number of independent engine groups.
- `node_ids` — exactly `N*T` nodes, partitioned **consecutively** into N groups
  of T nodes each. Group `g` owns `node_ids[g*T : g*T+T]`.

Invariant: `len(node_ids) == N*T`, no duplicate nodes, `T >= 2`, `N >= 1`.

Default `T` from the existing tensor-parallel setting; `N` defaults to 1 so a
saved `grouped_sharded` deployment degrades to a single sharded engine unless
the user asks for more instances.

## Backend changes

### `manager.py`

**Member build** (`_create_cluster` / the rank loop at the launch site): replace
the single flat rank loop for `grouped_sharded` with a two-level loop:

```
for group in range(N):
    group_nodes = node_ids[group*T : group*T+T]
    for local_rank in range(T):
        global_index = group*T + local_rank
        master_addr = fabric_ip of group_nodes[0]      # per-group coordinator
        # --nnodes=T, --node-rank=local_rank,
        # --master-addr=<group master>, --master-port
```

Each member records `{instance_id: group, rank: local_rank, ...}` in addition
to the existing `node_id`/`node_name`/`container_name`/`port`/`phase`. Each
group gets its **own** `master_addr` and `master_port`, so the groups are fully
independent engines. vLLM and SGLang arg injection both follow the existing
sharded branch (manager.py ~5624–5656).

**Routing / load balancing**: generalize `_cluster_route_order` and
`_balanced_cluster_member`. For `grouped_sharded`:

- Candidate units are the **rank-0 coordinator member of each running instance
  group** (mirroring how `sharded` picks a single coordinator), not every rank.
- The load balance key becomes `deployment_id:instance_id` (one key per engine
  group) rather than `deployment_id:container_name` (one key per rank), so
  balance is per-engine and only coordinators carry load.
- Reuse `_balanced_cluster_member` least-loaded + round-robin tie-break across
  those coordinators, and `_cluster_stream_with_failover` for failover to the
  next running group on outage.

Members of a stopped group are excluded from candidates. If no group is running,
the request fails with the existing "deployment is stopped" path.

**Per-instance lifecycle** (`deployment_action`): accept an optional
`instance` target. A targeted `start`/`stop` toggles only that group's members'
`desired_state`. The deployment-level state is derived:

- any instance running → `running`
- none running → `stopped`
- some running, some not → `degraded` (consistent with existing partial-start
  semantics).

A full-deployment `start`/`stop` (no `instance`) leaves current behavior
unchanged: it acts on all groups. A top-level `stop` persists
`desired_state=stopped` and stops all members, as today.

### `service.py`

- Accept `grouped_sharded` in the deployment-mode validation paths
  (`deployment_mode` allow-lists).
- Validate the launch contract (`_validate_start_selection`,
  `_preflight_deployment_launch`, recipe contract): node count must equal `N*T`,
  no duplicate nodes, `T >= 2`, `N >= 1`.
- `_saved_layout_contract` / `recipe_deployment_contract`: carry `instances` and
  `tensor_parallel_size` through the saved/launch contract for this mode.
- Expose `instances` in the deployment detail/payload so the UI can render
  per-instance state and controls.

### Health

The health monitor marks the deployment healthy while **any** instance group is
healthy, and exposes per-instance health so a stopped/failed group does not hide
the others.

## Frontend changes

- Model page: add a `Grouped sharded` layout option to the deployment-layout
  `select` (alongside parallel-instances / tensor-parallelism).
- New controls: an `instances` spin field and the existing tensor-parallel-size
  field. The node selector enforces a node count that is a multiple of `T` (or
  pre-fills `N*T`), and validates `len(node_ids) == N*T` at save.
- Deployment page: render per-instance group state and per-instance Start/Stop
  buttons, plus the existing unified start/stop.
- Contract wiring through `frontend/src/api/client.ts` and `types.ts`
  (`deployment_mode` union gets `grouped_sharded`).

## Data flow

1. User configures `instances=N`, `tensor_parallel_size=T`, selects `N*T` nodes.
2. Save/launch validates the topology and partitions nodes into N groups.
3. Launch builds N independent engines, each with its own coordinator.
4. Requests route to the least-loaded running group coordinator; failover to the
   next running group.
5. Per-instance Start/Stop toggles one group; deployment state is derived.

## Error handling

- Node count not a multiple of `T` / not equal to `N*T` → clear ValueError at
  save and launch.
- Duplicate or insufficient nodes → rejected by existing start-selection
  validation.
- Insufficient GPUs on a group's nodes → rejected by parallel-layout validation.
- No running group → "deployment is stopped; start it before sending inference
  requests."
- Partial group start → `degraded`, error surfaced, remaining group retried
  atomically as today.

## Testing

**Backend (pytest)**
- Member build partitions nodes into consecutive groups with correct
  per-group `master_addr`, `master_port`, `--nnodes=T`, `--node-rank`.
- Routing balances across running group coordinators; excludes stopped groups.
- Per-instance start/stop derives deployment state (`running`/`stopped`/
  `degraded`).
- Validation rejects wrong node count, duplicates, insufficient GPUs.
- vLLM and SGLang arg injection for grouped shards.

**Frontend (component tests)**
- `Grouped sharded` layout option appears and selects correctly.
- `instances` / TP fields persist; node selector enforces `N*T`.
- Per-instance Start/Stop controls render and dispatch.

## Compatibility

- Existing deployments unaffected; `grouped_sharded` is a new opt-in value.
- `single`/`sharded`/`replicated` behavior is unchanged.
- Served-name/alias uniqueness unchanged (single deployment, single served name).

## Future work (not in this change)

- Runtime scale up/down of instance count without restarting running groups.
- Per-group hardware/weight placement controls beyond consecutive partition.

## Open questions

- Whether to expose per-group soft-affinity/fabric preferences (out of scope
  for the initial change; consecutive partition is the default).
