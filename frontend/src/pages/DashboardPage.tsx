import { useEffect, useRef, useState } from 'react'
import {
  Activity,
  Cloud,
  Gauge,
  RefreshCw,
  Server,
  Users,
} from 'lucide-react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { ActiveRequestGroupStats, ActiveRequestStats, AdmissionStats, Deployment, NodeInventoryItem, SystemStats } from '../api/types'
import { Button, EmptyState, LoadingState, PageHeader, Panel, RuntimeMark, Status } from '../components/ui'
import { useResource } from '../hooks/useResource'
import { communityAccessHint, useCommunityAccess } from '../hooks/useCommunityAccess'
import { useDashboardStream } from '../hooks/useDashboardStream'
import type { DashboardStreamResources, DashboardStreamSource } from '../hooks/useDashboardStream'

function displayValue(value: number | null | undefined, suffix: string, digits = 0) {
  return value == null || !Number.isFinite(value) ? '—' : `${value.toFixed(digits)}${suffix}`
}

function percent(value: number | null | undefined) {
  return Math.min(100, Math.max(0, value ?? 0))
}

function temperatureTone(value: number | null | undefined) {
  if (value === null || value === undefined) return ''
  if (value >= 85) return 'metric-danger'
  if (value >= 75) return 'metric-warning'
  return ''
}

// Spark-class boards report the unified GB10 SoC as their GPU; the card shows
// the shortened chip name there while keeping CPU and GPU usage separate.
function gpuDisplayName(name?: string | null) {
  if (!name) return undefined
  if (/gb10/i.test(name)) return 'GB10'
  return name.replace(/^NVIDIA\s+/i, '') || name
}

const ACTIVE_DEPLOYMENT_STATUSES = new Set(['running', 'ready', 'starting', 'launching', 'degraded'])

/**
 * Whether the controller still observes live ranks for a whole deployment.
 * An empty `occupied_node_ids` is its confirmation that every rank is idle;
 * `undefined` means it had no inventory to judge with, so absent evidence must
 * never be read as proof that a stop finished.
 */
function hasLiveRanks(deployment: Deployment) {
  return deployment.occupied_node_ids === undefined || deployment.occupied_node_ids.length > 0
}

/**
 * The rows of the "Running models" panel.
 *
 * Stop intent outlives the stop: the controller keeps a stop it could not verify
 * latched at a non-stopped status (a peer that stays offline is never confirmed),
 * so intent alone says nothing about what still runs. A stopped deployment is
 * only still pending while live containers are observed; once none are, the stop
 * has finished and the row belongs to the Models page instead.
 */
function runningDeploymentGroups(deployment: Deployment) {
  const stopRequested = deployment.desired_state === 'stopped'
  if (deployment.instances?.length) {
    return deployment.instances.filter((group) => {
      if (!ACTIVE_DEPLOYMENT_STATUSES.has(group.status)) return false
      // Live containers are the only evidence that an errored group is still
      // serving, or that a Stop is still pending rather than already finished.
      if ((deployment.status === 'error' || group.desired_state === 'stopped' || stopRequested)
        && group.has_live_containers !== true) return false
      return true
    }).map((group) => ({
      key: `${deployment.id}:${group.instance_id}`,
      status: group.status,
      stopPending: group.desired_state === 'stopped' || stopRequested,
      label: `Group ${group.instance_id + 1} · ${group.node_names.join(' + ')}`,
    }))
  }
  if (!ACTIVE_DEPLOYMENT_STATUSES.has(deployment.status)) return []
  if (stopRequested && !hasLiveRanks(deployment)) return []
  return [{
    key: deployment.id,
    status: deployment.status,
    stopPending: stopRequested,
    label: (deployment.selected_nodes?.map((node) => node.name || node.id) ?? deployment.node_ids ?? []).join(' + '),
  }]
}

function MetricBar({ value, label }: { value: number | null | undefined; label: string }) {
  const measured = finiteNumber(value)
  return (
    <div className={`metric-bar${measured === undefined ? ' metric-bar-unavailable' : ''}`} role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={measured === undefined ? undefined : Math.round(percent(measured))} aria-valuetext={measured === undefined ? 'Unavailable' : undefined}>
      <span style={{ width: `${percent(measured)}%` }} />
    </div>
  )
}

function finiteNumber(value: unknown): number | undefined {
  if (value === null || value === undefined || typeof value === 'boolean') return undefined
  const number = Number(value)
  return Number.isFinite(number) ? number : undefined
}

/** Preserve machine identity: local stats are never attributed to a worker. */
export function nodeResourceSnapshot(nodes: NodeInventoryItem[], localStats?: SystemStats) {
  if (!nodes.length) {
    return localStats ? [{ node: { id: 'local', name: 'Local node', local: true, online: true }, stats: localStats, fallback: true, source: 'stats' as const }] : []
  }
  return nodes.filter((node) => node.hidden_from_dashboard !== true).map((node) => {
    const isLocal = node.local || node.id === 'local'
    const localTime = finiteNumber(localStats?.ts)
    const nodeTime = finiteNumber(node.stats?.ts)
    const preferLocal = isLocal && localStats && (localTime === undefined || nodeTime === undefined || localTime >= nodeTime)
    return { node, stats: preferLocal ? localStats : node.stats, fallback: false, source: preferLocal ? 'stats' as const : 'nodes' as const }
  })
}

function MemoryMetric({ used, total, percentage, label, gaugeLabel }: { used?: number; total?: number; percentage?: number; label: string; gaugeLabel: string }) {
  const hasTotal = total !== undefined && total > 0
  const allocation = hasTotal && used !== undefined ? used / total * 100 : percentage
  return <div className="node-resource-metric">
    <div className="node-resource-label"><span>{label}</span><strong>{displayValue(allocation, '%', 1)}</strong></div>
    <MetricBar value={allocation} label={gaugeLabel} />
    <p>{used !== undefined ? `${used.toFixed(1)} GB used` : 'Usage unavailable'}{hasTotal ? ` / ${total.toFixed(1)} GB` : ''}</p>
  </div>
}

function NodeResourceCard({ node, stats, fallback, refreshPaused, now }: { node: NodeInventoryItem; stats?: SystemStats; fallback: boolean; refreshPaused: boolean; now: number }) {
  const healthyGpus = stats?.gpus?.filter((gpu) => !gpu.error) ?? []
  const sparkSoc = healthyGpus.some((gpu) => /gb10/i.test(gpu.name ?? ''))
  const timestamp = finiteNumber(stats?.ts)
  const stale = timestamp !== undefined && now - timestamp > 30
  const sessionValues = stats?.active_requests ? Object.values(stats.active_requests) : undefined
  const sessions = sessionValues?.reduce((sum, request) => sum + (request.connections ?? 0), 0)
  const ramUsed = finiteNumber(stats?.mem?.used)
  const ramTotal = finiteNumber(stats?.mem?.total)
  return <Panel className="cluster-health-card" aria-label={`Resource usage for ${node.name}`}>
    <div className="cluster-health-heading"><div><Server size={16} /><div><h3>{node.name}</h3><p>{fallback ? 'Local telemetry only' : node.local ? 'Current entry node' : node.id}</p></div></div><Status status={node.online ? 'running' : 'offline'}>{node.online ? 'Online' : 'Offline'}</Status></div>
    {node.online ? <>
      {(refreshPaused || stale) && <p className="node-telemetry-notice" role="status">Last reported values{refreshPaused ? ' - refresh paused' : ' - telemetry stale'}{timestamp !== undefined ? ` (${new Date(timestamp * 1000).toLocaleTimeString()})` : ''}</p>}
      {!stats && <p className="node-telemetry-notice">Telemetry unavailable for this node.</p>}
      <div className="node-resource-grid">
        <div className="node-resource-metric">
          <div className="node-resource-label"><span>CPU load</span><strong>{displayValue(finiteNumber(stats?.cpu_pct), '%', 1)}</strong></div>
          <MetricBar value={stats?.cpu_pct} label={`${node.name} CPU load`} />
          <p>{stats?.cpu_model || (sparkSoc ? 'GB10 CPU' : 'CPU model unavailable')}{stats?.cpu_logical_count ? ` - ${stats.cpu_logical_count} logical processors` : ''}</p>
        </div>
        <MemoryMetric used={ramUsed === undefined ? undefined : ramUsed / 1024 ** 3} total={ramTotal === undefined ? undefined : ramTotal / 1024 ** 3} percentage={finiteNumber(stats?.mem?.pct)} label={sparkSoc ? 'Unified memory' : 'RAM'} gaugeLabel={`${node.name} ${sparkSoc ? 'unified memory' : 'RAM'} allocation`} />
      </div>
      <dl className="node-host-details"><div><dt>CPU temp</dt><dd className={temperatureTone(stats?.cpu_temp_c)}>{displayValue(stats?.cpu_temp_c, '\u00b0C', 1)}</dd></div><div><dt>Sessions</dt><dd>{sessions ?? '\u2014'}</dd></div></dl>
      <div className="node-gpu-list">
        {stats?.gpus?.map((gpu, position) => {
          const gpuLabel = gpu.index == null ? 'GPU telemetry' : `GPU ${gpu.index}`
          return <section className="node-gpu-card" key={`${gpu.index ?? 'discovery'}-${position}`} aria-label={`${node.name} ${gpuLabel}`}>
          <div className="node-gpu-heading"><strong>{gpuLabel}</strong><span>{gpuDisplayName(gpu.name) ?? 'Model unavailable'}</span></div>
          {gpu.error ? <p className="node-telemetry-notice">GPU telemetry unavailable: {gpu.error}</p> : <>
            <div className="node-resource-label"><span>GPU utilization</span><strong>{displayValue(finiteNumber(gpu.util), '%', 1)}</strong></div>
            <MetricBar value={gpu.util} label={`${node.name} ${gpuLabel} utilization`} />
            <dl><div><dt>GPU temp</dt><dd className={temperatureTone(gpu.temp)}>{displayValue(gpu.temp, '\u00b0C', 1)}</dd></div></dl>
            {/gb10/i.test(gpu.name ?? '') ? <p className="node-gpu-memory-note">Shares unified memory shown above</p> : <MemoryMetric used={finiteNumber(gpu.mem_used_mib) === undefined ? undefined : Number(gpu.mem_used_mib) / 1024} total={finiteNumber(gpu.mem_total_mib) === undefined ? undefined : Number(gpu.mem_total_mib) / 1024} label="GPU memory" gaugeLabel={`${node.name} ${gpuLabel} memory allocation`} />}
          </>}
        </section>})}
        {!stats?.gpus?.length && <p className="node-telemetry-notice">GPU telemetry unavailable.</p>}
      </div>
    </> : <p className="cluster-health-offline">Telemetry unavailable while this node is offline.</p>}
  </Panel>
}

export function activeRequestSnapshot(
  stats?: SystemStats,
  admission?: Record<string, AdmissionStats>,
): Record<string, ActiveRequestStats> {
  const snapshot = Object.fromEntries(
    Object.entries(stats?.active_requests ?? {}).map(([model, request]) => [model, { ...request }]),
  )
  const admitted = new Map<string, { running: number; queued: number }>()
  Object.entries(admission ?? {}).forEach(([target, item]) => {
    const model = item.model || target
    const current = admitted.get(model) ?? { running: 0, queued: 0 }
    current.running += item.running ?? 0
    current.queued += item.queued ?? 0
    admitted.set(model, current)
  })
  admitted.forEach(({ running, queued }, model) => {
    const existing = snapshot[model]
    if (!existing && running <= 0 && queued <= 0) return
    snapshot[model] = {
      ...existing,
      connections: Math.max(existing?.connections ?? 0, running),
      // Admission and active_requests share the same queue store. When an
      // admission value is available it is newer and authoritative, including
      // a drained queue reported as zero.
      queued,
    }
  })
  return snapshot
}

export function inferenceSessionSnapshot(stats?: SystemStats, admission?: Record<string, AdmissionStats>) {
  const groupedAdmission = Object.values(admission ?? {}).filter((item) => item.group_id)
  if (stats?.active_request_groups === undefined && groupedAdmission.length === 0) {
    return Object.entries(activeRequestSnapshot(stats, admission)).map(([key, request]) => ({ key, model: key, request, groupLabel: '' }))
  }
  const groups: Record<string, ActiveRequestGroupStats> = Object.fromEntries(
    Object.entries(stats?.active_request_groups ?? {}).map(([key, request]) => [key, { ...request }]),
  )
  Object.values(admission ?? {}).forEach((item) => {
    if (!item.group_id) return
    const existing = groups[item.group_id]
    if (!existing && item.running <= 0 && item.queued <= 0) return
    groups[item.group_id] = {
      ...existing,
      group_id: item.group_id,
      model: item.model || existing?.model || item.group_id,
      deployment_id: item.deployment_id ?? existing?.deployment_id ?? null,
      instance_id: item.instance_id ?? existing?.instance_id ?? null,
      node_names: item.node_names ?? existing?.node_names ?? [],
      connections: Math.max(existing?.connections ?? 0, item.running),
      queued: item.queued,
    }
  })
  const groupedModels = new Set(Object.values(groups).map((group) => group.model))
  const legacyAdmission = Object.fromEntries(Object.entries(admission ?? {}).filter(([, item]) => !item.group_id))
  const legacyStats = stats?.active_request_groups === undefined
    ? { active_requests: Object.fromEntries(Object.entries(stats?.active_requests ?? {}).filter(([model]) => !groupedModels.has(model))) }
    : undefined
  const legacyRows = Object.entries(activeRequestSnapshot(legacyStats, legacyAdmission))
    .filter(([, request]) => request.connections > 0 || (request.queued ?? 0) > 0)
    .map(([model, request]) => ({ key: `legacy:${model}`, model, request, groupLabel: '' }))
  return [...Object.entries(groups)
    .filter(([, request]) => request.connections > 0 || (request.queued ?? 0) > 0)
    .map(([key, request]) => ({
      key,
      model: request.model,
      request,
      groupLabel: [request.instance_id === null ? '' : `Group ${request.instance_id + 1}`, request.node_names.join(' + ')].filter(Boolean).join(' · '),
    })), ...legacyRows]
}

interface SessionStates {
  output: number
  thinking: number
  prefill: number
  prefillSeconds: number
}

/** Count live sessions by what the engine is doing with them right now. */
export function sessionStateCounts(requests: { request: ActiveRequestStats }[]) {
  const states: SessionStates = { output: 0, thinking: 0, prefill: 0, prefillSeconds: 0 }
  requests.forEach(({ request }) => {
    states.output += request.output_sessions ?? 0
    states.thinking += request.thinking_sessions ?? 0
    states.prefill += request.prefill_sessions ?? 0
    const seconds = request.prefill_seconds
    if (typeof seconds === 'number' && Number.isFinite(seconds)) states.prefillSeconds = Math.max(states.prefillSeconds, seconds)
  })
  return states
}

/**
 * Describe live session states independently of throughput measurements. Elapsed
 * time remains useful while native prompt-token counters are not available.
 */
export function sessionStateSummary(requests: { request: ActiveRequestStats }[]) {
  const states = sessionStateCounts(requests)
  const parts: string[] = []
  if (states.output > 0) parts.push(`${states.output} outputting`)
  if (states.thinking > 0) parts.push(`${states.thinking} thinking`)
  if (states.prefill > 0) parts.push(`${states.prefill} prompt processing`)
  return { ...states, labels: parts, text: parts.join(' · ') }
}

export function DashboardPage() {
  const [telemetryNow, setTelemetryNow] = useState(() => Date.now() / 1000)
  useEffect(() => {
    const timer = window.setInterval(() => setTelemetryNow(Date.now() / 1000), 10_000)
    return () => window.clearInterval(timer)
  }, [])
  const resourcesRef = useRef<DashboardStreamResources | null>(null)
  const stream = useDashboardStream(resourcesRef)
  // Poll while the socket is down, and keep polling any source the stream
  // reports as failed (null) so it recovers through the REST fallback.
  const polling = (source: DashboardStreamSource) => !stream.live || stream.failed.has(source)
  const statsResource = useDashboardResource((signal) => api.dashboard.stats(signal), polling('stats'))
  const admissionResource = useDashboardResource((signal) => api.dashboard.admission(signal), polling('admission'))
  const deploymentsResource = useDashboardResource((signal) => api.dashboard.deployments(signal), polling('deployments'))
  const syncResource = useDashboardResource((signal) => api.dashboard.sync(signal), polling('sync'))
  const nodesResource = useDashboardResource((signal) => api.dashboard.nodes(signal), polling('nodes'))
  useEffect(() => {
    resourcesRef.current = {
      stats: statsResource,
      admission: admissionResource,
      deployments: deploymentsResource,
      sync: syncResource,
      nodes: nodesResource,
    }
  })
  const communityAccess = useCommunityAccess()
  const accessHint = communityAccessHint(communityAccess.signedIn)

  const stats = statsResource.data
  const admission = admissionResource.data
  const previousAdmission = useRef(admission)
  const [admissionFailedSinceSuccess, setAdmissionFailedSinceSuccess] = useState(false)
  useEffect(() => {
    const replaced = admission !== previousAdmission.current
    previousAdmission.current = admission
    if (admissionResource.error) setAdmissionFailedSinceSuccess(true)
    else if (replaced) setAdmissionFailedSinceSuccess(false)
  }, [admission, admissionResource.error])
  // Retained queue data is useful for the stale queue summary, but it must not
  // override a newer active-request snapshot after the admission refresh fails.
  // Keep it excluded while a retry is loading; only replacement data proves
  // that the admission source recovered.
  const admissionForSessions = admissionResource.error || admissionFailedSinceSuccess
    ? undefined
    : admission
  const deployments = deploymentsResource.data ?? []
  const sync = syncResource.data
  const activeRequests = inferenceSessionSnapshot(stats, admissionForSessions)
  const runningSessions = activeRequests.reduce((sum, { request }) => sum + (request.connections ?? 0), 0)
  const sessionStates = sessionStateSummary(activeRequests)
  const queuedRequests = Object.values(admission ?? {}).reduce((sum, item) => sum + (item.queued ?? 0), 0)
  const freshQueuedRequests = Object.values(admissionForSessions ?? {}).reduce((sum, item) => sum + (item.queued ?? 0), 0)
  // Admission only covers concurrency-limited vLLM targets. A non-empty feed
  // can prove work exists, but an empty feed cannot prove the cluster is idle.
  const inferenceAvailable = stats !== undefined || activeRequests.length > 0
  const inferenceComplete = stats !== undefined && admissionForSessions !== undefined
  const activeDeploymentGroups = deployments.flatMap((deployment) => runningDeploymentGroups(deployment).map((group) => ({ deployment, group })))
  const activeDeploymentCount = new Set(activeDeploymentGroups.map(({ deployment }) => deployment.id)).size
  const updatedAt = stats?.ts ? new Date(stats.ts * 1000) : undefined
  const allClusterNodes = nodesResource.data ?? []
  const clusterNodes = allClusterNodes.filter((node) => node.hidden_from_dashboard !== true)
  const hiddenNodeCount = allClusterNodes.length - clusterNodes.length
  const resourceCards = nodeResourceSnapshot(allClusterNodes, stats)
  const loading = [statsResource, admissionResource, deploymentsResource, syncResource, nodesResource]
    .some((item) => item.loading)
  const queueSummary = admission
    ? `${queuedRequests} queued${admissionResource.error ? ' · refresh paused' : ''}`
    : admissionResource.error ? 'queue unavailable' : 'queue loading'
  const inferenceStatus = runningSessions > 0
    ? 'running'
    : freshQueuedRequests > 0 ? 'waiting'
      : inferenceComplete ? 'stopped' : 'waiting'
  const inferenceStatusLabel = runningSessions > 0
    ? 'Processing'
    : freshQueuedRequests > 0 ? 'Waiting'
      : inferenceComplete ? 'Idle'
      : statsResource.error || admissionResource.error ? 'Unavailable' : 'Loading'
  const telemetryNotice = statsResource.error
    ? `Local telemetry ${stats ? 'refresh paused' : 'unavailable; retrying'}: ${statsResource.error}`
    : undefined
  const reload = () => {
    statsResource.reload()
    admissionResource.reload()
    deploymentsResource.reload()
    syncResource.reload()
    nodesResource.reload()
  }

  return (
    <div className="page dashboard-page">
      <PageHeader
        eyebrow="Cluster command center"
        title="Dashboard"
        description="Live resource usage for each cluster node and each GPU, with current inference activity."
        actions={
          <div className="dashboard-refresh">
            <span>{updatedAt ? `Updated ${updatedAt.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: '2-digit' })}${stream.live ? ' · live' : ''}` : statsResource.loading ? 'Loading local telemetry' : 'Telemetry unavailable'}</span>
            <Button onClick={reload} disabled={loading}><RefreshCw size={15} /> Refresh</Button>
          </div>
        }
      />

      {telemetryNotice && <p className="dashboard-stale" role="status">{telemetryNotice}</p>}

      <>
          <section className="dashboard-inference-overview" aria-label="Inference overview">
            <Panel className="metric-panel">
              <div className="metric-label"><Activity size={16} /><span>Inference</span></div>
              <strong>{inferenceAvailable ? runningSessions : '—'}</strong>
              <p className="metric-context">{inferenceAvailable ? `active ${runningSessions === 1 ? 'session' : 'sessions'} · ${queueSummary}` : statsResource.error || admissionResource.error ? 'Inference telemetry unavailable' : 'Loading inference telemetry'}</p>
              <div className="metric-status"><Status status={inferenceStatus}>{inferenceStatusLabel}</Status></div>
            </Panel>
          </section>

          <section className="cluster-health" aria-labelledby="cluster-health-title">
            <div className="section-heading"><div><h2 id="cluster-health-title">Cluster nodes</h2><p>{nodesResource.loading && !nodesResource.data ? 'Loading cluster inventory' : `${clusterNodes.filter((node) => node.online).length} of ${clusterNodes.length} visible nodes online · resource usage per node and GPU${hiddenNodeCount ? ` · ${hiddenNodeCount} hidden` : ''}`}</p></div><Link className="text-link" to="/cluster">Manage cluster</Link></div>
            {nodesResource.error && nodesResource.data && <p className="dashboard-stale" role="status">Cluster inventory refresh paused: {nodesResource.error}</p>}
            <div className="cluster-health-grid">
              {nodesResource.loading && !nodesResource.data && <LoadingState label="Loading cluster nodes" />}
              {!nodesResource.loading && !clusterNodes.length && hiddenNodeCount > 0 && <EmptyState title="No nodes shown on the dashboard" description="Use Manage cluster to show a hidden machine." action={<Link className="button button-primary" to="/cluster">Manage cluster</Link>} />}
              {!nodesResource.loading && !clusterNodes.length && hiddenNodeCount === 0 && <EmptyState title="Cluster inventory unavailable" description="Refresh to retry loading per-machine telemetry." />}
              {resourceCards.map(({ node, stats: nodeStats, fallback, source }) => <NodeResourceCard key={node.id} node={node} stats={nodeStats} fallback={fallback} now={telemetryNow} refreshPaused={Boolean(source === 'stats' ? statsResource.error : nodesResource.error)} />)}
            </div>
          </section>

          <div className="dashboard-grid">
            <Panel className="dashboard-panel">
              <div className="dashboard-panel-heading">
                <div><span className="panel-icon"><Server size={17} /></span><div><h2>Running models</h2><p>{deploymentsResource.loading && !deploymentsResource.data ? 'Loading deployments' : `${activeDeploymentCount} of ${deployments.length} deployments active`}</p></div></div>
                <Link className="text-link" to="/models">Manage</Link>
              </div>
              {deploymentsResource.error && deploymentsResource.data && <p className="dashboard-stale" role="status">Deployment refresh paused: {deploymentsResource.error}</p>}
              {deploymentsResource.loading && !deploymentsResource.data ? (
                <LoadingState label="Loading deployments" />
              ) : deploymentsResource.error && !deploymentsResource.data ? (
                <EmptyState title="Deployment status unavailable" description="Refresh to retry loading model status." />
              ) : activeDeploymentGroups.length === 0 ? (
                <EmptyState title="No models running" description="Start a deployment to make it available for chat and comparison." action={<Link className="button button-primary" to="/models">Open models</Link>} />
              ) : (
                <div className="dashboard-list">
                  {activeDeploymentGroups.map(({ deployment, group }) => (
                    <div className="dashboard-list-row" key={group.key}>
                      <span className={`status-dot status-${group.status}`} aria-hidden="true" />
                      <span className="sr-only">Status: {group.status}</span>
                      <div><strong>{deployment.alias}</strong><small>{deployment.model_id}</small>{group.label && <small className="deployment-group-nodes">{group.label}</small>}{group.stopPending && <small>Stop pending</small>}{!ACTIVE_DEPLOYMENT_STATUSES.has(deployment.status) && <small>Deployment status: {deployment.status}</small>}</div>
                      <RuntimeMark runtime={deployment.runtime} />
                    </div>
                  ))}
                </div>
              )}
            </Panel>

            <Panel className="dashboard-panel">
              <div className="dashboard-panel-heading">
                <div><span className="panel-icon"><Users size={17} /></span><div><h2>Current inference</h2><p>{inferenceAvailable ? `${runningSessions} active${sessionStates.text ? ` · ${sessionStates.text}` : ''}` : 'Active sessions loading'} · {queueSummary}</p></div></div>
                <Link className="text-link" to="/chat">Open chat</Link>
              </div>
              {admissionResource.error && <p className="dashboard-stale" role="status">{admission ? 'Queue refresh paused' : 'Queue status unavailable'}: {admissionResource.error}</p>}
              {activeRequests.length > 0 ? (
                <div className="dashboard-list">
                  {activeRequests.map(({ key, model, request, groupLabel }) => <SessionRow key={key} model={model} request={request} groupLabel={groupLabel} />)}
                </div>
              ) : !inferenceAvailable && (statsResource.error || admissionResource.error) ? (
                <EmptyState title="Active session status unavailable" description="Refresh to retry loading current inference sessions." />
              ) : !inferenceAvailable ? (
                <LoadingState label="Loading active sessions" />
              ) : (
                <EmptyState title="No active inference" description="Current sessions and queue pressure will appear here." />
              )}
              {queuedRequests > 0 && (
                <div className="queue-note"><Gauge size={15} /><span><strong>{queuedRequests} queued</strong> · oldest wait {displayValue(Math.max(...Object.values(admission ?? {}).map((item) => item.oldest_wait_seconds ?? 0)), 's', 1)}</span></div>
              )}
            </Panel>
          </div>

          <Panel className="community-strip" title={communityAccess.enabled ? undefined : accessHint}>
            <span className="panel-icon"><Cloud size={17} /></span>
            <div><h2>Community benchmark sync</h2><p>Share aggregation-safe performance measurements without prompts or responses.</p></div>
            <Status status={sync ? (sync.sharing_enabled ? (sync.account_paired ? 'running' : 'waiting') : 'stopped') : 'waiting'}>
              {sync ? (sync.sharing_enabled ? (sync.account_paired ? 'Connected' : 'Waiting for account') : 'Sharing off') : syncResource.error ? 'Unavailable' : 'Loading'}
            </Status>
            <span className="community-counts">{sync ? `${sync.pending_count} pending · ${sync.synced_count} synced${syncResource.error ? ' · refresh paused' : ''}` : 'Sync status pending'}</span>
            <Link className="text-link" to={communityAccess.enabled || communityAccess.signedIn ? '/benchmarks' : '/settings'}>{communityAccess.enabled ? 'View benchmarks' : communityAccess.signedIn ? 'Review sharing' : 'Open community settings'}</Link>
          </Panel>
      </>
    </div>
  )
}

function useDashboardResource<T>(loader: (signal: AbortSignal) => Promise<T>, pollingActive = true) {
  const resource = useResource(loader)
  useEffect(() => {
    if (!pollingActive || resource.loading) return
    const timer = window.setTimeout(resource.reload, 10_000)
    return () => window.clearTimeout(timer)
  }, [pollingActive, resource.loading, resource.reload])
  return resource
}

function stageRate(value: number | null | undefined, waiting: boolean) {
  if (waiting) return 'Waiting'
  if (value == null || value <= 0) return 'Measuring…'
  return `${value.toFixed(1)} tok/s`
}

function SessionRow({ model, request, groupLabel }: { model: string; request: ActiveRequestStats; groupLabel: string }) {
  const waiting = request.connections <= 0 && (request.queued ?? 0) > 0
  const outputSessions = request.output_sessions ?? 0
  const thinkingSessions = request.thinking_sessions ?? 0
  const prefillSessions = request.prefill_sessions ?? 0
  // Native runtime estimates describe the latest completed prefills in this
  // group. Keep elapsed time when that measurement is not available yet.
  const prefillSeconds = request.prefill_seconds
  const prefillLabel = typeof prefillSeconds === 'number' && Number.isFinite(prefillSeconds)
    ? `Prefilling ${Math.round(prefillSeconds)}s`
    : 'Prefilling…'
  const promptRate = request.pp_tok_s
  const hasPromptRate = typeof promptRate === 'number' && Number.isFinite(promptRate) && promptRate > 0
  const sampledPromptRate = request.pp_rate_source === 'runtime_ttft'
  const promptLabel = waiting ? 'Waiting'
    : prefillSessions > 0 && !sampledPromptRate ? prefillLabel
      : hasPromptRate ? `${promptRate.toFixed(1)} tok/s${sampledPromptRate ? ' (est.)' : ''}`
        : prefillSessions > 0 ? prefillLabel : 'Unavailable'
  const states = [
    outputSessions > 0 ? `${outputSessions} outputting` : '',
    thinkingSessions > 0 ? `${thinkingSessions} thinking` : '',
    prefillSessions > 0 ? `${prefillSessions} prompt processing` : '',
  ].filter(Boolean)
  const callers = Object.entries(request.caller_ips ?? {})
    .sort(([left], [right]) => left.localeCompare(right, undefined, { numeric: true }))
    .map(([ip, connections]) => `${connections} from ${ip}`)
  return (
    <div className="dashboard-list-row session-row">
      <span className={`status-dot status-${waiting ? 'waiting' : 'running'}`} aria-hidden="true" />
      <div>
        <strong>{model}</strong>
        {groupLabel && <small className="deployment-group-nodes">{groupLabel}</small>}
        <small>{request.connections} active · {request.queued ?? 0} queued</small>
        {states.length > 0 && <small className="session-states">{states.join(' · ')}</small>}
        {callers.length > 0 && <small>{callers.join(' · ')}</small>}
      </div>
      <div className="session-stages" role="group" aria-label="Token rate by stage">
        <span className="session-stage" title={sampledPromptRate && hasPromptRate ? 'Latest prompt-speed estimate from uncached tokens and engine time to first token, including scheduling.' : undefined}><span className="session-stage-label">Prompt processing</span><span className="session-stage-value">{promptLabel}</span></span>
        <span className="session-stage"><span className="session-stage-label">Output</span><span className="session-stage-value">{stageRate(request.output_tok_s, waiting)}</span></span>
        <span className="session-stage"><span className="session-stage-label">Thinking</span><span className="session-stage-value">{stageRate(request.thinking_tok_s, waiting)}</span></span>
      </div>
    </div>
  )
}
