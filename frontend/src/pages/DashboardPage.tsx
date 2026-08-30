import { useEffect, useRef } from 'react'
import {
  Activity,
  Cloud,
  Cpu,
  Gauge,
  HardDrive,
  RefreshCw,
  Server,
  Users,
} from 'lucide-react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { ActiveRequestStats, AdmissionStats, GpuStats, NodeInventoryItem, SystemStats } from '../api/types'
import { Button, EmptyState, LoadingState, PageHeader, Panel, RuntimeMark, Status } from '../components/ui'
import { useResource } from '../hooks/useResource'
import { communityAccessHint, useCommunityAccess } from '../hooks/useCommunityAccess'
import { useDashboardStream } from '../hooks/useDashboardStream'
import type { DashboardStreamResources, DashboardStreamSource } from '../hooks/useDashboardStream'

function displayValue(value: number | null | undefined, suffix: string, digits = 0) {
  return value === null || value === undefined ? '—' : `${value.toFixed(digits)}${suffix}`
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

const ACTIVE_DEPLOYMENT_STATUSES = new Set(['running', 'starting', 'launching'])

function MetricBar({ value, label }: { value: number | null | undefined; label: string }) {
  return (
    <div className="metric-bar" role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(percent(value))}>
      <span style={{ width: `${percent(value)}%` }} />
    </div>
  )
}

function memorySnapshot(stats?: { gpus?: GpuStats[]; mem?: { total?: number; used?: number; pct?: number } }) {
  const gpu = stats?.gpus?.find((item) => !item.error)
  if (Number.isFinite(gpu?.mem_total_mib) && Number(gpu?.mem_total_mib) > 0) {
    const total = Number(gpu?.mem_total_mib) / 1024
    const used = Number(gpu?.mem_used_mib ?? 0) / 1024
    return { label: 'GPU memory', used, total, percent: total ? used / total * 100 : 0, context: 'dedicated VRAM' }
  }
  if (Number.isFinite(stats?.mem?.total) && Number(stats?.mem?.total) > 0) {
    const total = Number(stats?.mem?.total) / 1024 ** 3
    const used = Number(stats?.mem?.used ?? 0) / 1024 ** 3
    return { label: 'Unified memory', used, total, percent: Number(stats?.mem?.pct ?? used / total * 100), context: 'shared CPU/GPU pool' }
  }
  return undefined
}

function finiteNumber(value: unknown): number | undefined {
  if (value === null || value === undefined || typeof value === 'boolean') return undefined
  const number = Number(value)
  return Number.isFinite(number) ? number : undefined
}

export function clusterResourceSnapshot(nodes: NodeInventoryItem[], fallbackStats?: SystemStats) {
  const visibleOnline = nodes.filter((node) => node.hidden_from_dashboard !== true && node.online)
  const telemetry = nodes.length
    ? visibleOnline.flatMap((node) => {
      const nodeStats = fallbackStats && (node.local || node.id === 'local')
        ? fallbackStats
        : node.stats
      return nodeStats ? [{ id: node.id, stats: nodeStats }] : []
    })
    : fallbackStats ? [{ id: 'entry-node', stats: fallbackStats }] : []
  let cpuTotal = 0; let cpuWeightedTotal = 0; let cpuWeight = 0; let cpuNodes = 0; let logicalProcessors = 0; let allCpuCountsKnown = true
  let ramUsed = 0; let ramTotal = 0; let ramNodes = 0
  let gpuUtilTotal = 0; let measuredGpus = 0; let gpuCount = 0; const gpuNodes = new Set<string>()

  telemetry.forEach(({ id, stats }) => {
    const cpuPct = finiteNumber(stats.cpu_pct)
    if (cpuPct !== undefined) {
      const knownProcessors = finiteNumber(stats.cpu_logical_count)
      cpuTotal += cpuPct; cpuNodes += 1
      if (knownProcessors && knownProcessors > 0) {
        cpuWeightedTotal += cpuPct * knownProcessors; cpuWeight += knownProcessors; logicalProcessors += knownProcessors
      } else {
        allCpuCountsKnown = false
      }
    }
    const used = finiteNumber(stats.mem?.used); const total = finiteNumber(stats.mem?.total)
    if (used !== undefined && total !== undefined && total > 0) {
      ramUsed += used; ramTotal += total; ramNodes += 1
    }
    const healthyGpus = (stats.gpus ?? []).filter((gpu) => !gpu.error)
    if (healthyGpus.length) gpuNodes.add(id)
    gpuCount += healthyGpus.length
    healthyGpus.forEach((gpu) => {
      const util = finiteNumber(gpu.util)
      if (util !== undefined) { gpuUtilTotal += util; measuredGpus += 1 }
    })
  })

  return {
    cpuPct: cpuNodes ? (allCpuCountsKnown ? cpuWeightedTotal / cpuWeight : cpuTotal / cpuNodes) : undefined,
    cpuNodes,
    logicalProcessors: cpuNodes && allCpuCountsKnown ? logicalProcessors : undefined,
    gpuPct: measuredGpus ? gpuUtilTotal / measuredGpus : undefined,
    gpuCount,
    measuredGpus,
    gpuNodes: gpuNodes.size,
    ramUsed,
    ramTotal,
    ramPct: ramTotal ? ramUsed / ramTotal * 100 : undefined,
    ramNodes,
  }
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
      queued: Math.max(existing?.queued ?? 0, queued),
    }
  })
  return snapshot
}

export function DashboardPage() {
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
  const deployments = deploymentsResource.data ?? []
  const sync = syncResource.data
  const activeRequests = Object.entries(activeRequestSnapshot(stats, admission))
  const runningSessions = activeRequests.reduce((sum, [, item]) => sum + (item.connections ?? 0), 0)
  const queuedRequests = Object.values(admission ?? {}).reduce((sum, item) => sum + (item.queued ?? 0), 0)
  const inferenceAvailable = stats !== undefined || admission !== undefined
  const inferenceComplete = stats !== undefined && admission !== undefined
  const activeDeployments = deployments.filter((item) => ACTIVE_DEPLOYMENT_STATUSES.has(item.status))
  const updatedAt = stats?.ts ? new Date(stats.ts * 1000) : undefined
  const allClusterNodes = nodesResource.data ?? []
  const clusterNodes = allClusterNodes.filter((node) => node.hidden_from_dashboard !== true)
  const hiddenNodeCount = allClusterNodes.length - clusterNodes.length
  const pooled = clusterResourceSnapshot(allClusterNodes, stats)
  const loading = [statsResource, admissionResource, deploymentsResource, syncResource, nodesResource]
    .some((item) => item.loading)
  const queueSummary = admission
    ? `${queuedRequests} queued${admissionResource.error ? ' · refresh paused' : ''}`
    : admissionResource.error ? 'queue unavailable' : 'queue loading'
  const inferenceStatus = runningSessions > 0
    ? 'running'
    : inferenceComplete ? (queuedRequests > 0 ? 'waiting' : 'stopped') : 'waiting'
  const inferenceStatusLabel = runningSessions > 0
    ? 'Processing'
    : inferenceComplete ? (queuedRequests > 0 ? 'Waiting' : 'Idle')
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
        description="Live pooled resource health and per-machine telemetry for the SparkDeck cluster."
        actions={
          <div className="dashboard-refresh">
            <span>{updatedAt ? `Updated ${updatedAt.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: '2-digit' })}${stream.live ? ' · live' : ''}` : statsResource.loading ? 'Loading local telemetry' : 'Telemetry unavailable'}</span>
            <Button onClick={reload} disabled={loading}><RefreshCw size={15} /> Refresh</Button>
          </div>
        }
      />

      {telemetryNotice && <p className="dashboard-stale" role="status">{telemetryNotice}</p>}

      <>
          <section className="metric-grid" aria-label="System overview">
            <Panel className="metric-panel">
              <div className="metric-label"><Cpu size={16} /><span>Pooled CPU</span></div>
              <strong>{displayValue(pooled.cpuPct, '%', 1)}</strong>
              <p className="metric-context">{pooled.cpuNodes ? `${pooled.logicalProcessors ? `${pooled.logicalProcessors} logical processors · ` : ''}${pooled.cpuNodes} measured ${pooled.cpuNodes === 1 ? 'node' : 'nodes'}` : 'CPU telemetry unavailable'}</p>
              <MetricBar value={pooled.cpuPct} label="Pooled CPU load" />
            </Panel>
            <Panel className="metric-panel">
              <div className="metric-label"><Gauge size={16} /><span>Pooled GPU</span></div>
              <strong>{displayValue(pooled.gpuPct, '%', 1)}</strong>
              <p className="metric-context">{pooled.gpuCount ? `${pooled.gpuCount} ${pooled.gpuCount === 1 ? 'GPU' : 'GPUs'} across ${pooled.gpuNodes} ${pooled.gpuNodes === 1 ? 'node' : 'nodes'}${pooled.measuredGpus === pooled.gpuCount ? '' : ` · ${pooled.measuredGpus} measured`}` : 'GPU telemetry unavailable'}</p>
              <MetricBar value={pooled.gpuPct} label="Pooled GPU utilization" />
            </Panel>
            <Panel className="metric-panel">
              <div className="metric-label"><HardDrive size={16} /><span>Pooled RAM</span></div>
              <strong>{pooled.ramTotal ? `${(pooled.ramUsed / 1024 ** 3).toFixed(1)} GB` : '—'}</strong>
              <p className="metric-context">{pooled.ramTotal ? `of ${(pooled.ramTotal / 1024 ** 3).toFixed(1)} GB across ${pooled.ramNodes} ${pooled.ramNodes === 1 ? 'node' : 'nodes'}` : 'RAM telemetry unavailable'}</p>
              <MetricBar value={pooled.ramPct} label="Pooled RAM allocation" />
            </Panel>
            <Panel className="metric-panel">
              <div className="metric-label"><Activity size={16} /><span>Inference</span></div>
              <strong>{inferenceAvailable ? runningSessions : '—'}</strong>
              <p className="metric-context">{inferenceAvailable ? `active ${runningSessions === 1 ? 'session' : 'sessions'} · ${queueSummary}` : statsResource.error || admissionResource.error ? 'Inference telemetry unavailable' : 'Loading inference telemetry'}</p>
              <div className="metric-status"><Status status={inferenceStatus}>{inferenceStatusLabel}</Status></div>
            </Panel>
          </section>

          <section className="cluster-health" aria-labelledby="cluster-health-title">
            <div className="section-heading"><div><h2 id="cluster-health-title">Cluster nodes</h2><p>{nodesResource.loading && !nodesResource.data ? 'Loading cluster inventory' : `${clusterNodes.filter((node) => node.online).length} of ${clusterNodes.length} visible nodes online · pooled above, telemetry per machine${hiddenNodeCount ? ` · ${hiddenNodeCount} hidden` : ''}`}</p></div><Link className="text-link" to="/cluster">Manage cluster</Link></div>
            {nodesResource.error && nodesResource.data && <p className="dashboard-stale" role="status">Cluster inventory refresh paused: {nodesResource.error}</p>}
            <div className="cluster-health-grid">
              {nodesResource.loading && !nodesResource.data && <LoadingState label="Loading cluster nodes" />}
              {!nodesResource.loading && !clusterNodes.length && hiddenNodeCount > 0 && <EmptyState title="No nodes shown on the dashboard" description="Use Manage cluster to show a hidden machine." action={<Link className="button button-primary" to="/cluster">Manage cluster</Link>} />}
              {!nodesResource.loading && !clusterNodes.length && hiddenNodeCount === 0 && <EmptyState title="Cluster inventory unavailable" description="Refresh to retry loading per-machine telemetry." />}
              {clusterNodes.map((node) => {
                const nodeStats = node.stats
                const nodeGpu = nodeStats?.gpus?.find((item) => !item.error)
                const nodeMemory = memorySnapshot(nodeStats)
                const sessions = Object.values(nodeStats?.active_requests ?? {}).reduce((sum, request) => sum + (request.connections ?? 0), 0)
                return <Panel className="cluster-health-card" key={node.id}>
                  <div className="cluster-health-heading"><div><Server size={16} /><div><h3>{node.name}</h3><p>{node.local ? 'Current entry node' : node.id}</p></div></div><Status status={node.online ? 'running' : 'offline'}>{node.online ? 'Online' : 'Offline'}</Status></div>
                  {node.online ? <dl>
                    <div><dt>CPU temp</dt><dd className={temperatureTone(nodeStats?.cpu_temp_c)}>{displayValue(nodeStats?.cpu_temp_c, '°C', 1)}</dd></div>
                    <div><dt>GPU temp</dt><dd className={temperatureTone(nodeGpu?.temp)}>{displayValue(nodeGpu?.temp, '°C', 1)}</dd></div>
                    <div><dt>{nodeMemory?.label ?? 'Memory'}</dt><dd>{nodeMemory ? `${nodeMemory.used.toFixed(1)} / ${nodeMemory.total.toFixed(1)} GB` : '—'}</dd></div>
                    <div><dt>Sessions</dt><dd>{sessions}</dd></div>
                  </dl> : <p className="cluster-health-offline">Telemetry unavailable while this node is offline.</p>}
                </Panel>
              })}
            </div>
          </section>

          <div className="dashboard-grid">
            <Panel className="dashboard-panel">
              <div className="dashboard-panel-heading">
                <div><span className="panel-icon"><Server size={17} /></span><div><h2>Running models</h2><p>{deploymentsResource.loading && !deploymentsResource.data ? 'Loading deployments' : `${activeDeployments.length} of ${deployments.length} deployments active`}</p></div></div>
                <Link className="text-link" to="/models">Manage</Link>
              </div>
              {deploymentsResource.error && deploymentsResource.data && <p className="dashboard-stale" role="status">Deployment refresh paused: {deploymentsResource.error}</p>}
              {deploymentsResource.loading && !deploymentsResource.data ? (
                <LoadingState label="Loading deployments" />
              ) : deploymentsResource.error && !deploymentsResource.data ? (
                <EmptyState title="Deployment status unavailable" description="Refresh to retry loading model status." />
              ) : activeDeployments.length === 0 ? (
                <EmptyState title="No models running" description="Start a deployment to make it available for chat and comparison." action={<Link className="button button-primary" to="/models">Open models</Link>} />
              ) : (
                <div className="dashboard-list">
                  {activeDeployments.map((deployment) => (
                    <div className="dashboard-list-row" key={deployment.id}>
                      <span className={`status-dot status-${deployment.status}`} aria-hidden="true" />
                      <span className="sr-only">Status: {deployment.status}</span>
                      <div><strong>{deployment.alias}</strong><small>{deployment.model_id}</small></div>
                      <RuntimeMark runtime={deployment.runtime} />
                    </div>
                  ))}
                </div>
              )}
            </Panel>

            <Panel className="dashboard-panel">
              <div className="dashboard-panel-heading">
                <div><span className="panel-icon"><Users size={17} /></span><div><h2>Current inference</h2><p>{inferenceAvailable ? `${runningSessions} active` : 'Active sessions loading'} · {queueSummary}</p></div></div>
                <Link className="text-link" to="/chat">Open chat</Link>
              </div>
              {admissionResource.error && <p className="dashboard-stale" role="status">{admission ? 'Queue refresh paused' : 'Queue status unavailable'}: {admissionResource.error}</p>}
              {activeRequests.length > 0 ? (
                <div className="dashboard-list">
                  {activeRequests.map(([model, request]) => <SessionRow key={model} model={model} request={request} />)}
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

function SessionRow({ model, request }: { model: string; request: ActiveRequestStats }) {
  const rate = (request.thinking_tok_s ?? 0) + (request.output_tok_s ?? 0)
  const callers = Object.entries(request.caller_ips ?? {})
    .sort(([left], [right]) => left.localeCompare(right, undefined, { numeric: true }))
    .map(([ip, connections]) => `${connections} from ${ip}`)
  return (
    <div className="dashboard-list-row session-row">
      <span className="status-dot status-running" aria-hidden="true" />
      <div>
        <strong>{model}</strong>
        <small>{request.connections} active · {request.queued ?? 0} queued</small>
        {callers.length > 0 && <small>{callers.join(' · ')}</small>}
      </div>
      <span className="session-rate">{rate > 0 ? `${rate.toFixed(1)} tok/s` : 'Measuring…'}</span>
    </div>
  )
}
