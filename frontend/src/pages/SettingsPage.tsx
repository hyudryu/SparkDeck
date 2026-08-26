import { useEffect, useState, type FormEvent } from 'react'
import { Check, Cloud, KeyRound, MonitorCog, Network, Save } from 'lucide-react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { AppSettings, RuntimeKind } from '../api/types'
import { Button, ErrorState, LoadingState, PageHeader, Panel, Status } from '../components/ui'
import { useResource } from '../hooks/useResource'
import { applyTheme, persistTheme, storedTheme } from '../theme'

export function SettingsPage() {
  const resource = useResource((signal) => api.settings.get(signal))
  const [form, setForm] = useState<AppSettings>({ theme: storedTheme(), default_runtime: 'vllm', default_context_length: 8192 })
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState<string>()

  useEffect(() => {
    if (resource.data) {
      setForm((current) => ({ ...current, ...resource.data }))
      persistTheme(resource.data.theme ?? storedTheme())
    }
  }, [resource.data])

  useEffect(() => {
    applyTheme(form.theme ?? 'system')
  }, [form.theme])

  const save = async (event: FormEvent) => {
    event.preventDefault()
    setSaving(true)
    setSaved(false)
    setError(undefined)
    try {
      const savedSettings = await api.settings.update(form)
      setForm((current) => ({ ...current, ...savedSettings }))
      persistTheme(savedSettings.theme ?? form.theme ?? 'system')
      setSaved(true)
      window.setTimeout(() => setSaved(false), 2400)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not save settings')
    } finally {
      setSaving(false)
    }
  }

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
        <div className="settings-save"><span aria-live="polite">{saved && <><Check size={15} /> Saved</>}</span><Button type="submit" variant="primary" disabled={saving}><Save size={16} /> {saving ? 'Saving…' : 'Save settings'}</Button></div>
      </form>}
    </div>
  )
}
