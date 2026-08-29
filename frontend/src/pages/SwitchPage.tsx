import { useEffect, useState, type FormEvent } from 'react'
import { Cable, CheckCircle2, CircleAlert, Fan, Gauge, Link2, RefreshCw, Save, Server, Thermometer, Trash2 } from 'lucide-react'
import { api } from '../api/client'
import type { RouterOSConnectionInput, RouterOSNodeOverview, RouterOSPresenceNode } from '../api/types'
import { Button, EmptyState, ErrorState, LoadingState, PageHeader, Panel, Status } from '../components/ui'
import { useConfirmDialog } from '../components/useConfirmDialog'
import { useResource } from '../hooks/useResource'

const presenceEvent = 'sparkdeck:routeros-presence-changed'
const pollDelayMs = 3000
const defaultRouterOSUrl = 'https://192.168.88.1'

function humanize(value: string) {
  return value.replaceAll('-', ' ').replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function displayValue(value: unknown): string {
  if (value === undefined || value === null || value === '') return '—'
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (typeof value === 'number') return Intl.NumberFormat('en', { maximumFractionDigits: 2 }).format(value)
  if (typeof value === 'string') return value
  return JSON.stringify(value) ?? '—'
}

function formatRate(value: unknown) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric) || numeric < 0) return '—'
  const units = ['bps', 'Kbps', 'Mbps', 'Gbps', 'Tbps']
  let amount = numeric
  let unit = 0
  while (amount >= 1000 && unit < units.length - 1) {
    amount /= 1000
    unit += 1
  }
  return `${Intl.NumberFormat('en', { maximumFractionDigits: 1 }).format(amount)} ${units[unit]}`
}

function discoveredUrl(node?: RouterOSPresenceNode) {
  const address = node?.discovery?.[0]?.address?.trim() ?? ''
  if (!address) return defaultRouterOSUrl
  return /^https?:\/\//i.test(address) ? address : `https://${address}`
}

function isRunning(value: unknown) {
  return ['1', 'true', 'yes', 'running', 'link-ok'].includes(String(value ?? '').toLowerCase())
}

function ConnectionPanel({
  nodes,
  gateway,
  selectedNodeId,
  onSelected,
  onChanged,
}: {
  nodes: RouterOSPresenceNode[]
  gateway?: RouterOSNodeOverview | null
  selectedNodeId: string
  onSelected: (nodeId: string) => void
  onChanged: () => void
}) {
  const { confirm, confirmationDialog } = useConfirmDialog()
  const selectedNode = nodes.find((node) => node.node_id === selectedNodeId)
  const selectedGateway = gateway?.node_id === selectedNodeId ? gateway : undefined
  const [form, setForm] = useState<RouterOSConnectionInput>({
    base_url: selectedGateway?.base_url ?? discoveredUrl(selectedNode),
    username: '',
    password: '',
    verify_tls: selectedGateway?.verify_tls ?? true,
  })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string>()
  const [notice, setNotice] = useState<string>()

  const save = async (event: FormEvent) => {
    event.preventDefault()
    if (!selectedNode) return
    setBusy(true)
    setError(undefined)
    setNotice(undefined)
    try {
      await api.routeros.connect(selectedNode.node_id, {
        ...form,
        base_url: form.base_url.trim(),
        username: form.username.trim(),
      })
      setForm((current) => ({ ...current, password: '' }))
      setNotice(`Connected RouterOS through ${selectedNode.node_name}.`)
      window.dispatchEvent(new Event(presenceEvent))
      onChanged()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not connect to RouterOS')
    } finally {
      setBusy(false)
    }
  }

  const remove = async () => {
    if (!selectedGateway || !await confirm({
      title: 'Remove RouterOS connection?',
      message: `SparkDeck will remove the saved RouterOS credentials from ${selectedGateway.node_name}.`,
      confirmLabel: 'Remove connection',
      danger: true,
    })) return
    setBusy(true)
    setError(undefined)
    setNotice(undefined)
    try {
      await api.routeros.disconnect(selectedGateway.node_id)
      setNotice('Removed the RouterOS connection.')
      window.dispatchEvent(new Event(presenceEvent))
      onChanged()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not remove the RouterOS connection')
    } finally {
      setBusy(false)
    }
  }

  return <><Panel className="switch-gateway-panel">
    <div className="switch-panel-heading"><span><Link2 size={17} aria-hidden="true" /></span><div><h2>RouterOS connection</h2><p>Choose the one SparkDeck node physically connected to the switch by Ethernet.</p></div></div>
    <form onSubmit={(event) => void save(event)}>
      <div className="switch-gateway-fields">
        <label className="field"><span>Ethernet-connected node</span><select required value={selectedNodeId} onChange={(event) => onSelected(event.target.value)}>
          <option value="">Select a node</option>
          {nodes.map((node) => <option key={node.node_id} value={node.node_id} disabled={!node.online}>{node.node_name}{!node.online ? ' (offline)' : node.detected ? ' (switch detected)' : ''}</option>)}
        </select><small>This node must have a direct Ethernet path to the RouterOS management interface.</small></label>
        <label className="field"><span>Username</span><input required autoComplete="username" value={form.username} onChange={(event) => setForm({ ...form, username: event.target.value })} /></label>
        <label className="field"><span>Password</span><input type="password" required autoComplete="current-password" value={form.password} onChange={(event) => setForm({ ...form, password: event.target.value })} /></label>
      </div>
      <details className="switch-advanced-connection">
        <summary>Advanced connection settings</summary>
        <div className="switch-advanced-fields">
          <label className="field"><span>RouterOS URL</span><input type="url" required value={form.base_url} onChange={(event) => setForm({ ...form, base_url: event.target.value })} placeholder={defaultRouterOSUrl} /><small>Discovery fills this automatically; the factory default is {defaultRouterOSUrl}.</small></label>
          <label className="check-field"><input type="checkbox" checked={form.verify_tls} onChange={(event) => setForm({ ...form, verify_tls: event.target.checked })} /><span><strong>Verify TLS certificate</strong><small>Recommended when RouterOS has a trusted certificate.</small></span></label>
        </div>
      </details>
      {selectedNode?.discovery?.[0] && <p className="switch-discovery-note"><Cable size={14} aria-hidden="true" /> Detected {selectedNode.discovery[0].identity || selectedNode.discovery[0].address} at {selectedNode.discovery[0].address} from this node.</p>}
      {error && <p className="inline-error" role="alert">{error}</p>}
      {notice && <p className="inline-success" role="status">{notice}</p>}
      <div className="switch-form-actions"><Button type="submit" variant="primary" disabled={busy || !selectedNode?.online}>{busy ? 'Connecting…' : selectedGateway?.configured ? 'Reconnect and validate' : 'Connect and validate'}</Button>{selectedGateway?.configured && <Button type="button" variant="danger" disabled={busy} onClick={() => void remove()}><Trash2 size={14} aria-hidden="true" /> Remove</Button>}</div>
    </form>
  </Panel>{confirmationDialog}</>
}

function ConfigurationStatus({ node }: { node: RouterOSNodeOverview }) {
  return <Panel className="switch-detail-panel switch-validation-panel">
    <div className="switch-panel-heading"><span><CheckCircle2 size={17} aria-hidden="true" /></span><div><h3>Configuration status</h3><p>Live validation from the selected Ethernet gateway.</p></div></div>
    <div className="switch-check-list">
      {node.configuration_checks.map((check) => <div className={`switch-check switch-check-${check.status}`} key={check.id}>
        {check.status === 'passed' ? <CheckCircle2 size={17} aria-hidden="true" /> : <CircleAlert size={17} aria-hidden="true" />}
        <div><strong>{check.label}</strong><small>{check.detail}</small></div>
        <span>{check.status === 'passed' ? 'Passed' : check.status === 'failed' ? 'Failed' : 'Check'}</span>
      </div>)}
    </div>
  </Panel>
}

const fanFields: Record<string, { label: string; kind: 'number' | 'boolean' | 'duration'; unit?: string; min?: number; max?: number }> = {
  'fan-target-temp': { label: 'Target temperature', kind: 'number', unit: '°C', min: -273, max: 65 },
  'fan-full-speed-temp': { label: 'Full-speed temperature', kind: 'number', unit: '°C', min: -273, max: 65 },
  'fan-min-speed-percent': { label: 'Minimum fan speed', kind: 'number', unit: '%', min: 0, max: 100 },
  'fan-control-interval': { label: 'Fan control interval', kind: 'number', unit: 'seconds', min: 5, max: 30 },
  'cpu-overtemp-check': { label: 'CPU over-temperature check', kind: 'boolean' },
  'cpu-overtemp-threshold': { label: 'CPU over-temperature threshold', kind: 'number', unit: '°C', min: 0, max: 105 },
  'cpu-overtemp-startup-delay': { label: 'CPU over-temperature startup delay', kind: 'duration' },
}

function fanForm(node: RouterOSNodeOverview) {
  return Object.fromEntries((node.fan_capabilities ?? [])
    .filter((key) => key in fanFields)
    .map((key) => {
      const value = node.fan_settings?.[key]
      if (fanFields[key].kind !== 'boolean') return [key, value ?? '']
      return [key, ['1', 'true', 'yes', 'on', 'enabled'].includes(String(value ?? '').toLowerCase())]
    }))
}

function FanCurvePreview({ settings }: { settings: Record<string, unknown> }) {
  const target = Number(settings['fan-target-temp'])
  const full = Number(settings['fan-full-speed-temp'])
  const minimum = Number(settings['fan-min-speed-percent'])
  if (![target, full, minimum].every(Number.isFinite)) return null
  const x = (temperature: number) => 28 + Math.max(0, Math.min(70, temperature)) / 70 * 268
  const y = (speed: number) => 116 - Math.max(0, Math.min(100, speed)) / 100 * 92
  const points = `${x(0)},${y(minimum)} ${x(target)},${y(minimum)} ${x(full)},${y(100)} ${x(70)},${y(100)}`
  return <div className="switch-curve-preview">
    <svg viewBox="0 0 320 140" role="img" aria-label={`Fan curve from ${minimum}% at ${target} degrees to full speed at ${full} degrees`}>
      <path d="M28 20V116H300" className="switch-curve-axis" />
      <polyline points={points} className="switch-curve-line" />
      <circle cx={x(target)} cy={y(minimum)} r="4" /><circle cx={x(full)} cy={y(100)} r="4" />
      <text x={x(target)} y={Math.min(133, y(minimum) + 17)} textAnchor="middle">{target}° / {minimum}%</text>
      <text x={x(full)} y={Math.min(133, y(100) + 17)} textAnchor="middle">{full}° / 100%</text>
      <text x="4" y="27">100%</text><text x="8" y="118">0%</text>
    </svg>
  </div>
}

function FanSettings({ node, onChanged }: { node: RouterOSNodeOverview; onChanged: () => void }) {
  const [settings, setSettings] = useState<Record<string, unknown>>(() => fanForm(node))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string>()
  const [saved, setSaved] = useState(false)

  useEffect(() => setSettings(fanForm(node)), [node])

  const save = async (event: FormEvent) => {
    event.preventDefault()
    setError(undefined)
    setSaved(false)
    const allowed = (node.fan_capabilities ?? []).filter((key) => key in fanFields)
    const parsed = Object.fromEntries(allowed.map((key) => {
      const field = fanFields[key]
      return [key, field.kind === 'boolean' ? Boolean(settings[key]) : String(settings[key] ?? '').trim()]
    }))
    setBusy(true)
    try {
      await api.routeros.updateFanSettings(node.node_id, parsed)
      setSaved(true)
      onChanged()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not update fan settings')
    } finally {
      setBusy(false)
    }
  }

  return <Panel className="switch-detail-panel switch-fan-panel">
    <div className="switch-panel-heading"><span><Fan size={17} aria-hidden="true" /></span><div><h3>Temperature fan curve</h3><p>Edit the curve and safety thresholds this RouterOS device actually supports.</p></div></div>
    <FanCurvePreview settings={settings} />
    <form onSubmit={(event) => void save(event)}>
      <div className="switch-fan-fields">
        {(node.fan_capabilities ?? []).filter((key) => key in fanFields).map((key) => {
          const field = fanFields[key]
          if (field.kind === 'boolean') return <label className="check-field" key={key}><input type="checkbox" checked={Boolean(settings[key])} onChange={(event) => setSettings({ ...settings, [key]: event.target.checked })} /><span><strong>{field.label}</strong><small>{humanize(key)}</small></span></label>
          return <label className="field" key={key}><span>{field.label}{field.unit ? ` (${field.unit})` : ''}</span><input type={field.kind === 'number' ? 'number' : 'text'} required min={field.min} max={field.max} step={field.kind === 'number' ? 1 : undefined} value={String(settings[key] ?? '')} onChange={(event) => setSettings({ ...settings, [key]: event.target.value })} /><small>{field.kind === 'duration' ? 'Use a RouterOS duration such as 30s or 5m.' : humanize(key)}</small></label>
        })}
      </div>
      {error && <p className="inline-error" role="alert">{error}</p>}
      {saved && <p className="inline-success" role="status">Fan settings saved and validation refreshed.</p>}
      <Button type="submit" variant="primary" disabled={busy}><Save size={14} aria-hidden="true" /> {busy ? 'Saving…' : 'Save fan curve'}</Button>
    </form>
  </Panel>
}

function HealthPanel({ node }: { node: RouterOSNodeOverview }) {
  const health = node.health ?? []
  const temperatures = health.filter((item) => /temp|thermal/i.test(`${item.name} ${item.type ?? ''}`))
  const otherHealth = health.filter((item) => !temperatures.includes(item))
  return <Panel className="switch-detail-panel">
    <div className="switch-panel-heading"><span><Thermometer size={17} aria-hidden="true" /></span><div><h3>Temperatures and health</h3><p>Live sensors reported by the RouterOS hardware.</p></div></div>
    {health.length === 0 ? <p className="switch-panel-empty">No health sensors were reported.</p> : <dl className="switch-health-grid">
      {[...temperatures, ...otherHealth].map((item, index) => <div className={temperatures.includes(item) ? 'temperature' : ''} key={`${item.name}-${index}`}><dt>{humanize(item.name)}</dt><dd>{displayValue(item.value)}</dd>{item.type && <small>{humanize(item.type)}</small>}</div>)}
    </dl>}
  </Panel>
}

function NetworkPanel({ node }: { node: RouterOSNodeOverview }) {
  return <Panel className="switch-interfaces-panel">
    <div className="switch-panel-heading"><span><Gauge size={17} aria-hidden="true" /></span><div><h3>Network speeds</h3><p>Live receive/transmit traffic and negotiated Ethernet link state.</p></div></div>
    <div className="switch-network-summary">
      <div><span>Receive</span><strong>{formatRate(node.network.rx_bits_per_second)}</strong></div>
      <div><span>Transmit</span><strong>{formatRate(node.network.tx_bits_per_second)}</strong></div>
      <div><span>Active links</span><strong>{node.network.active_interfaces} / {node.network.total_interfaces}</strong></div>
    </div>
    {node.interfaces.length === 0 ? <p className="switch-panel-empty">No interface statistics were reported.</p> : <div className="switch-port-grid" aria-label={`RouterOS interfaces through ${node.node_name}`}>
      {node.interfaces.map((item, index) => {
        const name = displayValue(item.name ?? item['default-name'] ?? `Interface ${index + 1}`)
        const running = isRunning(item.running) || isRunning(item.status)
        const counters = Object.entries(item).filter(([key]) => ![
          'name', 'default-name', 'type', 'running', 'status', 'rate', 'full-duplex',
          'rx-bits-per-second', 'tx-bits-per-second',
        ].includes(key))
        return <article className="switch-port-card" key={`${name}-${index}`}>
          <header><div><strong>{name}</strong><small>{displayValue(item.type)}</small></div><Status status={running ? 'running' : 'stopped'}>{running ? 'Link up' : 'Link down'}</Status></header>
          <dl><div><dt>Link speed</dt><dd>{displayValue(item.rate)}</dd></div><div><dt>Receive</dt><dd>{formatRate(item['rx-bits-per-second'])}</dd></div><div><dt>Transmit</dt><dd>{formatRate(item['tx-bits-per-second'])}</dd></div><div><dt>Duplex</dt><dd>{isRunning(item['full-duplex']) ? 'Full' : displayValue(item['full-duplex'])}</dd></div></dl>
          {counters.length > 0 && <details><summary>View port counters</summary><dl>{counters.map(([key, value]) => <div key={key}><dt>{humanize(key)}</dt><dd>{displayValue(value)}</dd></div>)}</dl></details>}
        </article>
      })}
    </div>}
  </Panel>
}

function GatewayOverview({ node, onChanged }: { node: RouterOSNodeOverview; onChanged: () => void }) {
  const identity = displayValue(node.device?.identity ?? node.device?.['board-name'] ?? 'RouterOS switch')
  const deviceFields = ['board-name', 'version', 'uptime', 'cpu-load', 'free-memory']
    .filter((key) => node.device?.[key] !== undefined)
  return <article className="switch-gateway-overview">
    <Panel className="switch-node-summary">
      <div className="switch-node-heading"><div><span className="panel-icon"><Server size={17} aria-hidden="true" /></span><div><h2>{identity}</h2><p>Managed through {node.node_name}</p></div></div><Status status={node.connected ? 'running' : node.configured ? 'waiting' : 'stopped'}>{node.connected ? 'Connected' : node.configured ? 'Unavailable' : 'Not configured'}</Status></div>
      {node.error && <p className="inline-error" role="alert">{node.error}</p>}
      {deviceFields.length > 0 && <dl className="switch-device-grid">{deviceFields.map((key) => <div key={key}><dt>{humanize(key)}</dt><dd>{displayValue(node.device?.[key])}</dd></div>)}</dl>}
    </Panel>
    <ConfigurationStatus node={node} />
    {node.connected && <>
      <div className="switch-details-grid"><HealthPanel node={node} />{node.fan_capabilities?.length ? <FanSettings node={node} onChanged={onChanged} /> : <Panel className="switch-detail-panel switch-fan-panel"><div className="switch-panel-heading"><span><Fan size={17} aria-hidden="true" /></span><div><h3>Temperature fan curve</h3><p>This RouterOS model reports no writable fan-curve settings.</p></div></div><p className="switch-panel-empty">Temperature and RPM monitoring remain available when the hardware exposes those sensors.</p></Panel>}</div>
      <NetworkPanel node={node} />
    </>}
  </article>
}

export function SwitchPage() {
  const resource = useResource((signal) => api.routeros.get(signal))
  const [selectedNodeId, setSelectedNodeId] = useState('')

  useEffect(() => {
    if (resource.loading) return
    const timeout = window.setTimeout(resource.reload, pollDelayMs)
    return () => window.clearTimeout(timeout)
  }, [resource.loading, resource.reload])

  const availableNodes = resource.data?.nodes ?? []
  const preferredNode = availableNodes.find((node) => node.node_id === resource.data?.gateway_node_id)
    ?? availableNodes.find((node) => node.online && node.detected)
    ?? availableNodes.find((node) => node.online)
    ?? availableNodes[0]
  const activeNodeId = availableNodes.some((node) => node.node_id === selectedNodeId)
    ? selectedNodeId
    : preferredNode?.node_id ?? ''
  const selectedNode = availableNodes.find((node) => node.node_id === activeNodeId)
  const gateway = resource.data?.gateway
  const selectedGateway = gateway?.node_id === activeNodeId ? gateway : undefined
  const refresh = () => {
    resource.reload()
    window.dispatchEvent(new Event(presenceEvent))
  }

  return <div className="page switch-page">
    <PageHeader eyebrow="RouterOS network control" title="Switch" description="Use one Ethernet-connected SparkDeck node to validate RouterOS configuration, tune device-supported temperature and fan behavior, and monitor network speeds." actions={<Button type="button" onClick={refresh} disabled={resource.loading}><RefreshCw size={15} aria-hidden="true" /> Refresh</Button>} />
    {resource.loading && !resource.data && <LoadingState label="Loading RouterOS gateway" />}
    {resource.error && !resource.data && <ErrorState message={resource.error} onRetry={refresh} />}
    {resource.error && resource.data && <p className="inline-error" role="status">Live refresh failed: {resource.error}</p>}
    {resource.data && <>
      <Panel className="switch-cluster-summary"><Cable size={19} aria-hidden="true" /><div><strong>{selectedNode ? `${selectedNode.node_name} selected` : 'No Ethernet gateway selected'}</strong><span>{selectedNode ? 'Only this node manages the RouterOS connection' : 'Choose the node connected to the switch'}</span></div><Status status={selectedGateway?.connected ? 'running' : selectedNode?.detected ? 'waiting' : 'stopped'}>{selectedGateway?.connected ? 'Validated' : selectedNode?.detected ? 'Switch detected' : 'Not detected'}</Status></Panel>
      {resource.data.nodes.length === 0 ? <Panel className="switch-no-nodes"><h2>No cluster nodes available</h2><p>Join or restore a SparkDeck node before connecting a RouterOS switch.</p></Panel> : <>
        {!selectedNode?.detected && <EmptyState title="Switch not detected" description="Connect the selected SparkDeck node to the RouterOS switch by Ethernet, then enter the RouterOS username and password below." />}
        <ConnectionPanel key={`${activeNodeId}:${selectedGateway?.base_url ?? ''}`} nodes={resource.data.nodes} gateway={gateway} selectedNodeId={activeNodeId} onSelected={setSelectedNodeId} onChanged={refresh} />
        {selectedGateway && <GatewayOverview node={selectedGateway} onChanged={refresh} />}
      </>}
    </>}
  </div>
}
