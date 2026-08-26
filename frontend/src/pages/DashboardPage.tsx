import { useEffect } from 'react'
import {
  Activity,
  Cloud,
  Cpu,
  Gauge,
  HardDrive,
  RefreshCw,
  Server,
  Thermometer,
  Users,
} from 'lucide-react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { ActiveRequestStats, GpuStats } from '../api/types'
import { Button, EmptyState, ErrorState, LoadingState, PageHeader, Panel, RuntimeMark, Status } from '../components/ui'
import { useResource } from '../hooks/useResource'

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

function MetricBar({ value, label }: { value: number | null | undefined; label: string }) {
  return (
    <div className="metric-bar" role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(percent(value))}>
      <span style={{ width: `${percent(value)}%` }} />
    </div>
  )
}

function GpuSummary({ gpu }: { gpu?: GpuStats }) {
  if (!gpu || gpu.error) return <p className="metric-unavailable">GPU telemetry unavailable</p>
  return <p className="metric-context">{gpu.name ?? `GPU ${gpu.index}`} · {displayValue(gpu.util, '%')} utilized</p>
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

export function DashboardPage() {
  const resource = useResource((signal) => api.dashboard.load(signal))

  useEffect(() => {
    const timer = window.setInterval(resource.reload, 10_000)
    return () => window.clearInterval(timer)
  }, [resource.reload])

  const stats = resource.data?.stats
  const gpu = stats?.gpus?.find((item) => !item.error)
  const activeRequests = Object.entries(stats?.active_requests ?? {})
  const runningSessions = activeRequests.reduce((sum, [, item]) => sum + (item.connections ?? 0), 0)
  const queuedRequests = Object.values(resource.data?.admission ?? {}).reduce((sum, item) => sum + (item.queued ?? 0), 0)
  const runningDeployments = resource.data?.deployments.filter((item) => item.status === 'running') ?? []
  const memory = memorySnapshot(stats)
  const updatedAt = stats?.ts ? new Date(stats.ts * 1000) : undefined
  const clusterNodes = resource.data?.nodes ?? []

  return (
    <div className="page dashboard-page">
      <PageHeader
        eyebrow="Cluster command center"
        title="Dashboard"
        description="Live health for this entry node and every SparkDeck node in the cluster."
        actions={
          <div className="dashboard-refresh">
            <span>{updatedAt ? `Updated ${updatedAt.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: '2-digit' })}` : 'Waiting for telemetry'}</span>
            <Button onClick={resource.reload} disabled={resource.loading}><RefreshCw size={15} /> Refresh</Button>
          </div>
        }
      />

      {resource.loading && !resource.data && <LoadingState label="Loading system overview" />}
      {resource.error && !resource.data && <ErrorState message={resource.error} onRetry={resource.reload} />}
      {resource.error && resource.data && <p className="dashboard-stale" role="status">Live refresh paused: {resource.error}</p>}

      {resource.data && (
        <>
          <section className="metric-grid" aria-label="System overview">
            <Panel className="metric-panel">
              <div className="metric-label"><Cpu size={16} /><span>CPU</span></div>
              <strong className={temperatureTone(stats?.cpu_temp_c)}>{displayValue(stats?.cpu_temp_c, '°C', 1)}</strong>
              <p className="metric-context">{displayValue(stats?.cpu_pct, '%', 1)} load</p>
              <MetricBar value={stats?.cpu_pct} label="CPU load" />
            </Panel>
            <Panel className="metric-panel">
              <div className="metric-label"><Thermometer size={16} /><span>GPU</span></div>
              <strong className={temperatureTone(gpu?.temp)}>{displayValue(gpu?.temp, '°C', 1)}</strong>
              <GpuSummary gpu={gpu} />
              <MetricBar value={gpu?.util} label="GPU utilization" />
            </Panel>
            <Panel className="metric-panel">
              <div className="metric-label"><HardDrive size={16} /><span>{memory?.label ?? 'GPU memory'}</span></div>
              <strong>{memory ? `${memory.used.toFixed(1)} GB` : '—'}</strong>
              <p className="metric-context">{memory ? `of ${memory.total.toFixed(1)} GB · ${memory.context}` : 'Memory telemetry unavailable'}</p>
              <MetricBar value={memory?.percent} label={`${memory?.label ?? 'GPU memory'} allocation`} />
            </Panel>
            <Panel className="metric-panel">
              <div className="metric-label"><Activity size={16} /><span>Inference</span></div>
              <strong>{runningSessions}</strong>
              <p className="metric-context">active {runningSessions === 1 ? 'session' : 'sessions'} · {queuedRequests} queued</p>
              <div className="metric-status"><Status status={runningSessions > 0 ? 'running' : queuedRequests > 0 ? 'waiting' : 'stopped'}>{runningSessions > 0 ? 'Processing' : queuedRequests > 0 ? 'Waiting' : 'Idle'}</Status></div>
            </Panel>
          </section>

          <section className="cluster-health" aria-labelledby="cluster-health-title">
            <div className="section-heading"><div><h2 id="cluster-health-title">Cluster nodes</h2><p>{clusterNodes.filter((node) => node.online).length} of {clusterNodes.length} nodes online · telemetry is shown per machine</p></div><Link className="text-link" to="/cluster">Manage cluster</Link></div>
            <div className="cluster-health-grid">
              {clusterNodes.map((node) => {
                const nodeStats = node.local ? stats : node.stats
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
                <div><span className="panel-icon"><Server size={17} /></span><div><h2>Running models</h2><p>{runningDeployments.length} of {resource.data.deployments.length} deployments online</p></div></div>
                <Link className="text-link" to="/models">Manage</Link>
              </div>
              {runningDeployments.length === 0 ? (
                <EmptyState title="No models running" description="Start a deployment to make it available for chat and comparison." action={<Link className="button button-primary" to="/models">Open models</Link>} />
              ) : (
                <div className="dashboard-list">
                  {runningDeployments.map((deployment) => (
                    <div className="dashboard-list-row" key={deployment.id}>
                      <span className="status-dot status-running" aria-hidden="true" />
                      <div><strong>{deployment.alias}</strong><small>{deployment.model_id}</small></div>
                      <RuntimeMark runtime={deployment.runtime} />
                    </div>
                  ))}
                </div>
              )}
            </Panel>

            <Panel className="dashboard-panel">
              <div className="dashboard-panel-heading">
                <div><span className="panel-icon"><Users size={17} /></span><div><h2>Current inference</h2><p>{runningSessions} active · {queuedRequests} queued requests</p></div></div>
                <Link className="text-link" to="/chat">Open chat</Link>
              </div>
              {activeRequests.length === 0 ? (
                <EmptyState title="No active inference" description="Current sessions and queue pressure will appear here." />
              ) : (
                <div className="dashboard-list">
                  {activeRequests.map(([model, request]) => <SessionRow key={model} model={model} request={request} />)}
                </div>
              )}
              {queuedRequests > 0 && (
                <div className="queue-note"><Gauge size={15} /><span><strong>{queuedRequests} queued</strong> · oldest wait {displayValue(Math.max(...Object.values(resource.data.admission).map((item) => item.oldest_wait_seconds ?? 0)), 's', 1)}</span></div>
              )}
            </Panel>
          </div>

          <Panel className="community-strip">
            <span className="panel-icon"><Cloud size={17} /></span>
            <div><h2>Community benchmark sync</h2><p>Share aggregation-safe performance measurements without prompts or responses.</p></div>
            <Status status={resource.data.sync.sharing_enabled ? (resource.data.sync.account_paired ? 'running' : 'waiting') : 'stopped'}>
              {resource.data.sync.sharing_enabled ? (resource.data.sync.account_paired ? 'Connected' : 'Waiting for account') : 'Sharing off'}
            </Status>
            <span className="community-counts">{resource.data.sync.pending_count} pending · {resource.data.sync.synced_count} synced</span>
            <Link className="text-link" to="/benchmarks">View benchmarks</Link>
          </Panel>
        </>
      )}
    </div>
  )
}

function SessionRow({ model, request }: { model: string; request: ActiveRequestStats }) {
  const rate = (request.thinking_tok_s ?? 0) + (request.output_tok_s ?? 0)
  return (
    <div className="dashboard-list-row session-row">
      <span className="status-dot status-running" aria-hidden="true" />
      <div><strong>{model}</strong><small>{request.connections} active · {request.queued ?? 0} queued</small></div>
      <span className="session-rate">{rate > 0 ? `${rate.toFixed(1)} tok/s` : 'Measuring…'}</span>
    </div>
  )
}
