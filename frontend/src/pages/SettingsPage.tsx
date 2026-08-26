import { useEffect, useState, type FormEvent } from 'react'
import { Check, Cloud, DownloadCloud, KeyRound, MonitorCog, Network, RefreshCw, Save } from 'lucide-react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { AppSettings, RuntimeKind } from '../api/types'
import { Button, ErrorState, LoadingState, PageHeader, Panel, Status } from '../components/ui'
import { useResource } from '../hooks/useResource'
import { applyTheme, persistTheme, storedTheme } from '../theme'

function shortRevision(value?: string) {
  return value ? value.slice(0, 8) : 'Unknown'
}

function SoftwareUpdatePanel() {
  const resource = useResource((signal) => api.updates.overview(signal))
  const [starting, setStarting] = useState(false)
  const [actionError, setActionError] = useState<string>()
  const active = Boolean(resource.data?.job?.active)

  useEffect(() => {
    if (!active) return
    const timer = window.setInterval(resource.reload, 2500)
    return () => window.clearInterval(timer)
  }, [active, resource.reload])

  const start = async () => {
    const nodeCount = resource.data?.nodes?.length ?? 0
    if (!window.confirm(`Update all ${nodeCount} cluster node${nodeCount === 1 ? '' : 's'}? Workers restart one at a time and the controller restarts last.`)) return
    setStarting(true)
    setActionError(undefined)
    try {
      await api.updates.start()
      resource.reload()
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : 'Could not start the cluster update')
    } finally {
      setStarting(false)
    }
  }

  const data = resource.data
  const blockers = data?.blockers ?? []
  const nodes = data?.job?.nodes ?? data?.nodes ?? []
  return (
    <Panel className="settings-section software-update-section">
      <div className="settings-heading"><span><DownloadCloud size={18} /></span><div><h2>Software update</h2><p>Install the latest published GitHub release across the entire cluster.</p></div></div>
      <div className="settings-fields">
        {resource.loading && !data && <LoadingState label="Checking for updates" />}
        {resource.error && !data && <ErrorState message={resource.error} onRetry={resource.reload} />}
        {data && <>
          <div className="credential-state wide-field">
            <DownloadCloud size={17} />
            <div>
              <strong>{data.latest_release ? `${data.latest_release.name} is the latest release` : 'No published release available'}</strong>
              <span className="muted">Installed {shortRevision(data.current_revision)} · {data.nodes?.length ?? 0} cluster node{data.nodes?.length === 1 ? '' : 's'}</span>
            </div>
            <Button type="button" variant="primary" disabled={!data.can_update || starting || active} onClick={() => void start()}>{active ? <RefreshCw className="spin" size={16} /> : <DownloadCloud size={16} />} {starting ? 'Starting…' : active ? 'Updating…' : 'Update all nodes'}</Button>
          </div>
          {(data.job?.message || data.job?.error || actionError) && <p className={data.job?.error || actionError ? 'form-error wide-field' : 'muted wide-field'} role="status" aria-live="polite">{data.job?.error || actionError || data.job?.message}</p>}
          {blockers.length > 0 && <div className="update-blockers wide-field"><strong>Update unavailable</strong><ul>{blockers.map((blocker) => <li key={blocker}>{blocker}</li>)}</ul></div>}
          {nodes.length > 0 && <div className="update-node-list wide-field" aria-label="Cluster update status">{nodes.map((node) => <div className="update-node" key={node.id}><span><strong>{node.name}</strong><small>{shortRevision(node.current_revision)}</small></span><Status status={node.error ? 'error' : node.phase === 'succeeded' ? 'running' : node.online === false ? 'stopped' : 'starting'}>{node.error || node.phase || (node.online === false ? 'Offline' : 'Ready')}</Status></div>)}</div>}
        </>}
      </div>
    </Panel>
  )
}

function editableSettingsFingerprint(settings: AppSettings) {
  return JSON.stringify({
    theme: settings.theme ?? 'system',
    default_runtime: settings.default_runtime ?? 'vllm',
    default_context_length: settings.default_context_length ?? 8192,
    community_api_url: settings.community_api_url ?? '',
  })
}

export function SettingsPage() {
  const resource = useResource((signal) => api.settings.get(signal))
  const [form, setForm] = useState<AppSettings>({ theme: storedTheme(), default_runtime: 'vllm', default_context_length: 8192 })
  const [savedFingerprint, setSavedFingerprint] = useState<string>()
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState<string>()

  useEffect(() => {
    if (resource.data) {
      setForm((current) => {
        const loaded = { ...current, ...resource.data }
        setSavedFingerprint(editableSettingsFingerprint(loaded))
        return loaded
      })
      persistTheme(resource.data.theme ?? storedTheme())
    }
  }, [resource.data])

  useEffect(() => {
    applyTheme(form.theme ?? 'system')
  }, [form.theme])

  const save = async (event: FormEvent) => {
    event.preventDefault()
    if (savedFingerprint === undefined || editableSettingsFingerprint(form) === savedFingerprint) return
    setSaving(true)
    setSaved(false)
    setError(undefined)
    try {
      const savedSettings = await api.settings.update(form)
      const savedForm = { ...form, ...savedSettings }
      setForm(savedForm)
      setSavedFingerprint(editableSettingsFingerprint(savedForm))
      persistTheme(savedSettings.theme ?? form.theme ?? 'system')
      setSaved(true)
      window.setTimeout(() => setSaved(false), 2400)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not save settings')
    } finally {
      setSaving(false)
    }
  }

  const hasUnsavedChanges = savedFingerprint !== undefined
    && editableSettingsFingerprint(form) !== savedFingerprint

  return (
    <div className="page settings-page">
      <PageHeader eyebrow="Preferences" title="Settings" description="Configure local defaults, community connectivity, and the SparkDeck interface." />
      {resource.loading && <LoadingState label="Loading settings" />}
      {resource.error && <ErrorState message={resource.error} onRetry={resource.reload} />}
      {!resource.loading && <form onSubmit={(event) => void save(event)}>
        {error && <p className="form-error" role="alert">{error}</p>}
        <Panel className="settings-section">
          <div className="settings-heading"><span><MonitorCog size={18} /></span><div><h2>Interface and defaults</h2><p>Preferences used when creating local model servers.</p></div></div>
          <div className="settings-fields">
            <label className="field"><span>Appearance</span><select value={form.theme} onChange={(event) => setForm({ ...form, theme: event.target.value as AppSettings['theme'] })}><option value="system">Follow system</option><option value="light">Light</option><option value="dark">Dark</option></select></label>
            <label className="field"><span>Default runtime</span><select value={form.default_runtime} onChange={(event) => setForm({ ...form, default_runtime: event.target.value as RuntimeKind })}><option value="vllm">vLLM</option><option value="llama.cpp">llama.cpp</option><option value="sglang">SGLang</option></select></label>
            <label className="field"><span>Default context length</span><input type="number" min="256" value={form.default_context_length} onChange={(event) => setForm({ ...form, default_context_length: Number(event.target.value) })} /><small>Applied as a starting value; each deployment can override it.</small></label>
          </div>
        </Panel>
        <Panel className="settings-section">
          <div className="settings-heading"><span><Network size={18} /></span><div><h2>DGX Spark cluster</h2><p>Connect nodes privately over Tailscale for targeted pulls and deployments.</p></div></div>
          <div className="settings-fields"><div className="credential-state wide-field"><Network size={17} /><div><strong>Cluster onboarding</strong><span className="muted">Review this node’s role, private access URL, and join instructions.</span></div><Link className="button button-secondary" to="/cluster">Open cluster setup</Link></div></div>
        </Panel>
        <Panel className="settings-section">
          <div className="settings-heading"><span><Cloud size={18} /></span><div><h2>Community service</h2><p>Optional hosted service for account pairing and benchmark aggregation.</p></div></div>
          <div className="settings-fields">
            <label className="field wide-field"><span>Community API URL</span><input type="url" value={form.community_api_url ?? ''} onChange={(event) => setForm({ ...form, community_api_url: event.target.value })} placeholder="Not configured" /><small>Leave empty to keep SparkDeck entirely local.</small></label>
            <div className="credential-state"><KeyRound size={17} /><div><strong>Device credential</strong><Status status="stopped">Not paired</Status></div><Button type="button" disabled>Pair account</Button></div>
          </div>
        </Panel>
        <div className="settings-save"><span aria-live="polite">{saved && <><Check size={15} /> Saved</>}</span><Button type="submit" variant="primary" disabled={saving || !hasUnsavedChanges}><Save size={16} /> {saving ? 'Saving…' : 'Save settings'}</Button></div>
      </form>}
      <SoftwareUpdatePanel />
    </div>
  )
}
