import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { Cable, Fan, Link2, RefreshCw, Save, Server, Thermometer, Trash2 } from 'lucide-react'
import { api } from '../api/client'
import type { RouterOSConnectionInput, RouterOSDiscoveryCandidate, RouterOSNodeOverview } from '../api/types'
import { Button, EmptyState, ErrorState, LoadingState, PageHeader, Panel, Status } from '../components/ui'
import { useResource } from '../hooks/useResource'

const presenceEvent = 'sparkdeck:routeros-presence-changed'

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

function displayInterfaceValue(key: string, value: unknown): string {
  if (key.endsWith('bits-per-second')) {
    const numeric = Number(value)
    if (Number.isFinite(numeric) && numeric >= 0) {
      const units = ['bps', 'Kbps', 'Mbps', 'Gbps', 'Tbps']
      let amount = numeric
      let unit = 0
      while (amount >= 1000 && unit < units.length - 1) {
        amount /= 1000
        unit += 1
      }
      return `${Intl.NumberFormat('en', { maximumFractionDigits: 1 }).format(amount)} ${units[unit]}`
    }
  }
  return displayValue(value)
}

function resourceEntries(resource?: Record<string, unknown>) {
  return Object.entries(resource ?? {}).filter(([, value]) => value !== undefined && value !== null && value !== '')
}

function discoveredUrl(candidate?: RouterOSDiscoveryCandidate) {
  const address = candidate?.address?.trim() ?? ''
  if (!address || /^https?:\/\//i.test(address)) return address
  return `https://${address}`
}

function ConnectionForm({ node, onChanged }: { node: RouterOSNodeOverview; onChanged: () => void }) {
  const [form, setForm] = useState<RouterOSConnectionInput>({
    base_url: node.base_url?.trim() || discoveredUrl(node.discovery?.[0]),
    username: '',
    password: '',
    verify_tls: node.verify_tls ?? true,
  })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string>()
  const [notice, setNotice] = useState<string>()

  useEffect(() => {
    setForm((current) => {
      if (node.configured && node.base_url) {
        return {
          ...current,
          base_url: node.base_url.trim(),
          verify_tls: node.verify_tls ?? true,
        }
      }
      return current.base_url ? current : {
        ...current,
        base_url: discoveredUrl(node.discovery?.[0]),
      }
    })
  }, [node.base_url, node.configured, node.discovery, node.verify_tls])

  const save = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError(undefined)
    setNotice(undefined)
    try {
      await api.routeros.connect(node.node_id, {
        ...form,
        base_url: form.base_url.trim(),
        username: form.username.trim(),
      })
      setForm((current) => ({ ...current, password: '' }))
      setNotice(`Connected RouterOS through ${node.node_name}.`)
      window.dispatchEvent(new Event(presenceEvent))
      onChanged()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not connect to RouterOS')
    } finally {
      setBusy(false)
    }
  }

  const remove = async () => {
    if (!window.confirm(`Remove the RouterOS connection from ${node.node_name}?`)) return
    setBusy(true)
    setError(undefined)
    setNotice(undefined)
    try {
      await api.routeros.disconnect(node.node_id)
      setNotice(`Removed the RouterOS connection from ${node.node_name}.`)
      window.dispatchEvent(new Event(presenceEvent))
      onChanged()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not remove the RouterOS connection')
    } finally {
      setBusy(false)
    }
  }

  return <details className="switch-connection" open={!node.configured}>
    <summary><span><Link2 size={15} aria-hidden="true" /> {node.configured ? 'Update connection' : 'Connect this switch'}</span><small>Credentials stay on {node.node_name}</small></summary>
    <form onSubmit={(event) => void save(event)}>
      {(node.discovery?.length ?? 0) > 0 && <div className="switch-discovery-list" aria-label={`Discovered RouterOS candidates on ${node.node_name}`}>
        {node.discovery?.map((candidate) => <button type="button" key={`${candidate.address}-${candidate.mac ?? ''}`} onClick={() => setForm({ ...form, base_url: discoveredUrl(candidate) })}>
          <Cable size={15} aria-hidden="true" /><span><strong>{candidate.identity || candidate.address}</strong><small>{[candidate.board, candidate.platform, candidate.version, candidate.address].filter(Boolean).join(' · ')}</small></span>
        </button>)}
      </div>}
      <div className="switch-connection-fields">
        <label className="field"><span>RouterOS URL</span><input type="url" required value={form.base_url} onChange={(event) => setForm({ ...form, base_url: event.target.value })} placeholder="https://192.168.88.1" /><small>Use the REST API address reachable from this SparkDeck node.</small></label>
        <label className="field"><span>Username</span><input required autoComplete="username" value={form.username} onChange={(event) => setForm({ ...form, username: event.target.value })} /></label>
        <label className="field"><span>Password</span><input type="password" required={!node.configured} autoComplete="current-password" value={form.password} onChange={(event) => setForm({ ...form, password: event.target.value })} />{node.configured && <small>Leave blank to keep the existing password.</small>}</label>
        <label className="check-field switch-tls"><input type="checkbox" checked={form.verify_tls} onChange={(event) => setForm({ ...form, verify_tls: event.target.checked })} /><span><strong>Verify TLS certificate</strong><small>Keep enabled for a trusted RouterOS certificate.</small></span></label>
      </div>
      {error && <p className="inline-error" role="alert">{error}</p>}
      {notice && <p className="inline-success" role="status">{notice}</p>}
      <div className="switch-form-actions"><Button type="submit" variant="primary" disabled={busy}>{busy ? 'Saving…' : 'Save connection'}</Button>{node.configured && <Button type="button" variant="danger" disabled={busy} onClick={() => void remove()}><Trash2 size={14} aria-hidden="true" /> Remove</Button>}</div>
    </form>
  </details>
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
      const value = settings[key]
      if (field.kind === 'boolean') return [key, Boolean(value)]
      if (field.kind === 'number') return [key, String(value ?? '').trim()]
      return [key, String(value ?? '').trim()]
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
    <div className="switch-panel-heading"><span><Fan size={17} aria-hidden="true" /></span><div><h3>Fan settings</h3><p>Only settings supported by this RouterOS device are accepted.</p></div></div>
    <form onSubmit={(event) => void save(event)}>
      <div className="switch-fan-fields">
        {(node.fan_capabilities ?? []).filter((key) => key in fanFields).map((key) => {
          const field = fanFields[key]
          if (field.kind === 'boolean') return <label className="check-field" key={key}><input type="checkbox" checked={Boolean(settings[key])} onChange={(event) => setSettings({ ...settings, [key]: event.target.checked })} /><span><strong>{field.label}</strong><small>{humanize(key)}</small></span></label>
          return <label className="field" key={key}><span>{field.label}{field.unit ? ` (${field.unit})` : ''}</span><input type={field.kind === 'number' ? 'number' : 'text'} required min={field.min} max={field.max} step={field.kind === 'number' ? 1 : undefined} value={String(settings[key] ?? '')} onChange={(event) => setSettings({ ...settings, [key]: event.target.value })} /><small>{field.kind === 'duration' ? 'Use a RouterOS duration such as 30s or 5m.' : humanize(key)}</small></label>
        })}
      </div>
      {error && <p className="inline-error" role="alert">{error}</p>}
      {saved && <p className="inline-success" role="status">Fan settings saved.</p>}
      <Button type="submit" variant="primary" disabled={busy}><Save size={14} aria-hidden="true" /> {busy ? 'Saving…' : 'Save fan settings'}</Button>
    </form>
  </Panel>
}

function NodeOverview({ node, onChanged }: { node: RouterOSNodeOverview; onChanged: () => void }) {
  const health = node.health ?? []
  const temperatures = health.filter((item) => /temp|thermal/i.test(`${item.name} ${item.type ?? ''}`))
  const otherHealth = health.filter((item) => !temperatures.includes(item))
  const deviceEntries = resourceEntries(node.device)

  return <article className="switch-node">
    <Panel className="switch-node-summary">
      <div className="switch-node-heading"><div><span className="panel-icon"><Server size={17} aria-hidden="true" /></span><div><h2>{node.node_name}</h2><p className="mono">{node.node_id}</p></div></div><Status status={node.connected ? 'running' : node.detected ? 'waiting' : 'stopped'}>{node.connected ? 'Connected' : node.configured ? 'Unavailable' : node.detected ? 'Detected' : 'Not detected'}</Status></div>
      {node.error && <p className="inline-error" role="alert">{node.error}</p>}
      {deviceEntries.length > 0 && <dl className="switch-device-grid">{deviceEntries.map(([key, value]) => <div key={key}><dt>{humanize(key)}</dt><dd>{displayValue(value)}</dd></div>)}</dl>}
      <ConnectionForm node={node} onChanged={onChanged} />
    </Panel>

    {node.connected && <div className="switch-details-grid">
      <Panel className="switch-detail-panel">
        <div className="switch-panel-heading"><span><Thermometer size={17} aria-hidden="true" /></span><div><h3>Health and temperatures</h3><p>Live RouterOS sensor readings.</p></div></div>
        {health.length === 0 ? <p className="switch-panel-empty">No health sensors were reported.</p> : <dl className="switch-health-grid">
          {[...temperatures, ...otherHealth].map((item, index) => <div className={temperatures.includes(item) ? 'temperature' : ''} key={`${item.name}-${index}`}><dt>{humanize(item.name)}</dt><dd>{displayValue(item.value)}</dd>{item.type && <small>{humanize(item.type)}</small>}</div>)}
        </dl>}
      </Panel>
      {Boolean(node.fan_capabilities?.length) && <FanSettings node={node} onChanged={onChanged} />}
    </div>}

    {node.connected && <Panel className="switch-interfaces-panel">
      <div className="switch-panel-heading"><span><Cable size={17} aria-hidden="true" /></span><div><h3>Interfaces</h3><p>Port state and traffic statistics reported by RouterOS.</p></div></div>
      {node.interfaces.length === 0 ? <p className="switch-panel-empty">No interface statistics were reported.</p> : <div className="switch-interface-list" role="table" aria-label={`RouterOS interfaces on ${node.node_name}`}>
        {node.interfaces.map((item, index) => {
          const name = displayValue(item.name ?? item['default-name'] ?? `Interface ${index + 1}`)
          const values = Object.entries(item).filter(([key]) => !['name', 'default-name'].includes(key))
          return <div className="switch-interface-row" role="row" key={`${name}-${index}`}><div role="cell"><strong>{name}</strong><small>{displayValue(item.type)}</small></div><dl>{values.map(([key, value]) => <div key={key}><dt>{humanize(key)}</dt><dd>{displayInterfaceValue(key, value)}</dd></div>)}</dl></div>
        })}
      </div>}
    </Panel>}
  </article>
}

export function SwitchPage() {
  const resource = useResource((signal) => api.routeros.get(signal))
  const connectedCount = useMemo(() => resource.data?.nodes.filter((node) => node.connected).length ?? 0, [resource.data])
  const refresh = () => {
    resource.reload()
    window.dispatchEvent(new Event(presenceEvent))
  }

  return <div className="page switch-page">
    <PageHeader eyebrow="RouterOS network control" title="Switch" description="Monitor RouterOS switch health, temperatures, fans, and interfaces from every SparkDeck node." actions={<Button type="button" onClick={refresh} disabled={resource.loading}><RefreshCw size={15} aria-hidden="true" /> Refresh</Button>} />
    {resource.loading && !resource.data && <LoadingState label="Looking for RouterOS switches" />}
    {resource.error && !resource.data && <ErrorState message={resource.error} onRetry={refresh} />}
    {resource.error && resource.data && <p className="inline-error" role="status">Live refresh failed: {resource.error}</p>}
    {resource.data && <>
      <Panel className="switch-cluster-summary"><Cable size={19} aria-hidden="true" /><div><strong>{connectedCount} connected</strong><span>{resource.data.nodes.length} cluster {resource.data.nodes.length === 1 ? 'node' : 'nodes'} checked</span></div><Status status={resource.data.detected ? 'running' : 'stopped'}>{resource.data.detected ? 'Switch detected' : 'Not detected'}</Status></Panel>
      {!resource.data.detected && <EmptyState title="Switch not detected" description="No SparkDeck node can currently reach a RouterOS switch. Choose a discovered candidate or enter a RouterOS REST API address below." />}
      <div className="switch-node-list">{resource.data.nodes.map((node) => <NodeOverview node={node} onChanged={refresh} key={node.node_id} />)}</div>
      {resource.data.nodes.length === 0 && <Panel className="switch-no-nodes"><h2>No cluster nodes available</h2><p>Join or restore a SparkDeck node before connecting a RouterOS switch.</p></Panel>}
    </>}
  </div>
}
