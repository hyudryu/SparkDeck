import { useEffect, useState, type FormEvent } from 'react'
import { Check, Cloud, KeyRound, MonitorCog, Network, Save } from 'lucide-react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { AppSettings } from '../api/types'
import { Button, ErrorState, LoadingState, PageHeader, Panel, Status } from '../components/ui'
import { useResource } from '../hooks/useResource'
import { applyTheme, persistTheme, storedTheme } from '../theme'

function editableSettingsFingerprint(settings: AppSettings) {
  return JSON.stringify({
    theme: settings.theme ?? 'system',
    community_api_url: settings.community_api_url ?? '',
  })
}

export function SettingsPage() {
  const resource = useResource((signal) => api.settings.get(signal))
  const [form, setForm] = useState<AppSettings>({ theme: storedTheme(), community_api_url: '' })
  const [huggingFaceApiKey, setHuggingFaceApiKey] = useState('')
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
    if (savedFingerprint === undefined
      || (editableSettingsFingerprint(form) === savedFingerprint && !huggingFaceApiKey.trim())) return
    setSaving(true)
    setSaved(false)
    setError(undefined)
    try {
      const savedSettings = await api.settings.update({
        ...form,
        hf_token: huggingFaceApiKey.trim() || undefined,
      })
      const savedForm = { ...form, ...savedSettings, hf_token: '' }
      setForm(savedForm)
      setHuggingFaceApiKey('')
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
    && (editableSettingsFingerprint(form) !== savedFingerprint || Boolean(huggingFaceApiKey.trim()))

  return (
    <div className="page settings-page">
      <PageHeader eyebrow="Preferences" title="Settings" description="Configure cluster credentials, community connectivity, and the SparkDeck interface." />
      {resource.loading && <LoadingState label="Loading settings" />}
      {resource.error && <ErrorState message={resource.error} onRetry={resource.reload} />}
      {!resource.loading && <form onSubmit={(event) => void save(event)}>
        {error && <p className="form-error" role="alert">{error}</p>}
        <Panel className="settings-section">
          <div className="settings-heading"><span><MonitorCog size={18} /></span><div><h2>Interface</h2><p>Choose how SparkDeck looks on this browser.</p></div></div>
          <div className="settings-fields">
            <label className="field"><span>Appearance</span><select value={form.theme} onChange={(event) => setForm({ ...form, theme: event.target.value as AppSettings['theme'] })}><option value="system">Follow system</option><option value="light">Light</option><option value="dark">Dark</option></select></label>
          </div>
        </Panel>
        <Panel className="settings-section">
          <div className="settings-heading"><span><Network size={18} /></span><div><h2>DGX Spark cluster</h2><p>Connect nodes privately over Tailscale for targeted pulls and deployments.</p></div></div>
          <div className="settings-fields"><div className="credential-state wide-field"><Network size={17} /><div><strong>Cluster onboarding</strong><span className="muted">Review this node’s role, private access URL, and join instructions.</span></div><Link className="button button-secondary" to="/cluster">Open cluster setup</Link></div></div>
        </Panel>
        <Panel className="settings-section">
          <div className="settings-heading"><span><KeyRound size={18} /></span><div><h2>Hugging Face access</h2><p>Use one credential for gated and private models across the cluster.</p></div></div>
          <div className="settings-fields">
            <label className="field wide-field"><span>Hugging Face API key</span><input aria-label="Hugging Face API key" type="password" autoComplete="new-password" value={huggingFaceApiKey} onChange={(event) => setHuggingFaceApiKey(event.target.value)} placeholder={form.hf_token_configured ? 'Enter a new key to replace the saved key' : 'hf_…'} /><small>The controller stores this key privately and sends it only over authenticated cluster channels when selected nodes start Hugging Face models. Leave blank to keep the current key.</small></label>
            <div className="credential-state"><KeyRound size={17} /><div><strong>Cluster credential</strong><Status status={form.hf_token_configured ? 'running' : 'stopped'}>{form.hf_token_configured ? 'Configured' : 'Not configured'}</Status></div></div>
          </div>
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
    </div>
  )
}
