import { useEffect, useState, type FormEvent, type KeyboardEvent } from 'react'
import { Check, Cloud, DownloadCloud, KeyRound, MonitorCog, Network, RefreshCw, Save } from 'lucide-react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { AppSettings } from '../api/types'
import { useAuth } from '../auth/AuthContext'
import { UserNotConfirmedError } from '../auth/cognitoAuth'
import { Button, ErrorState, LoadingState, PageHeader, Panel, Status } from '../components/ui'
import { useResource } from '../hooks/useResource'
import { applyTheme, persistTheme, storedTheme } from '../theme'
import { SPARKDECK_VERSION } from '../buildInfo'

function shortRevision(value?: string) {
  return value ? value.slice(0, 8) : 'Unknown'
}

function SoftwareUpdatePanel() {
  const resource = useResource((signal) => api.updates.overview(signal))
  const [starting, setStarting] = useState(false)
  const [actionError, setActionError] = useState<string>()
  const [selectedTag, setSelectedTag] = useState('')
  const active = Boolean(resource.data?.job?.active)

  useEffect(() => {
    const releases = resource.data?.releases ?? []
    const jobTag = resource.data?.job?.active ? resource.data.job.target_tag : undefined
    if (jobTag && selectedTag !== jobTag) {
      setSelectedTag(jobTag)
      return
    }
    if (releases.length && !releases.some((release) => release.tag === selectedTag)) {
      setSelectedTag(releases[0].tag)
    }
  }, [resource.data?.job?.active, resource.data?.job?.target_tag, resource.data?.releases, selectedTag])

  useEffect(() => {
    if (!active) return
    const timer = window.setInterval(resource.reload, 2500)
    return () => window.clearInterval(timer)
  }, [active, resource.reload])

  const start = async () => {
    const nodeCount = resource.data?.nodes?.length ?? 0
    if (!selectedTag) return
    if (!window.confirm(`Install ${selectedTag} on all ${nodeCount} cluster node${nodeCount === 1 ? '' : 's'}? This may upgrade or downgrade SparkDeck. Workers restart one at a time and the controller restarts last.`)) return
    setStarting(true)
    setActionError(undefined)
    try {
      await api.updates.start(selectedTag)
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
  const releases = data?.releases ?? []
  const selectedRelease = releases.find((release) => release.tag === selectedTag)
  const selectedEverywhere = Boolean(
    data?.current_release_tag === selectedTag
    && data.nodes?.length
    && data.nodes.every((node) => node.current_revision === data.current_revision),
  )
  return (
    <Panel className="settings-section software-update-section">
      <div className="settings-heading"><span><DownloadCloud size={18} /></span><div><h2>Software update</h2><p>Install or roll back to a published GitHub release across the entire cluster.</p></div></div>
      <div className="settings-fields">
        {resource.loading && !data && <LoadingState label="Checking for updates" />}
        {resource.error && !data && <ErrorState message={resource.error} onRetry={resource.reload} />}
        {data && <>
          <div className="credential-state wide-field">
            <DownloadCloud size={17} />
            <div>
              <strong>{data.current_release_tag ? `Running ${data.current_release_tag}` : `Running ${shortRevision(data.current_revision)}`}</strong>
              <span className="muted">Latest {data.latest_release?.tag ?? 'unavailable'} · {data.nodes?.length ?? 0} cluster node{data.nodes?.length === 1 ? '' : 's'}</span>
            </div>
            <Button type="button" variant="primary" disabled={!data.can_update || !selectedTag || selectedEverywhere || starting || active} onClick={() => void start()}>{active ? <RefreshCw className="spin" size={16} /> : <DownloadCloud size={16} />} {starting ? 'Starting…' : active ? 'Installing…' : selectedEverywhere ? 'Installed on all nodes' : 'Install on all nodes'}</Button>
          </div>
          {releases.length > 0 && <label className="field wide-field"><span>Release</span><select aria-label="Release version" value={selectedTag} disabled={active} onChange={(event) => setSelectedTag(event.target.value)}>{releases.map((release, index) => <option value={release.tag} key={release.tag}>{release.name} ({release.tag}){index === 0 ? ' — latest' : ''}{release.tag === data.current_release_tag ? ' — installed' : ''}{release.prerelease ? ' — prerelease' : ''}</option>)}</select><small>{selectedRelease?.published_at ? `Published ${new Date(selectedRelease.published_at).toLocaleDateString()}. ` : ''}Choosing an older release performs a cluster-wide rollback.</small></label>}
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
    community_api_url: settings.community_api_url ?? '',
  })
}

type CommunityAuthMode = 'sign-in' | 'sign-up' | 'confirm' | 'reset-request' | 'reset-confirm'

function CommunitySignInForm() {
  const auth = useAuth()
  const [mode, setMode] = useState<CommunityAuthMode>('sign-in')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [code, setCode] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string>()
  const [notice, setNotice] = useState<string>()

  const switchMode = (next: CommunityAuthMode) => {
    setMode(next)
    setError(undefined)
    setNotice(undefined)
  }

  const run = async (action: () => Promise<void>) => {
    setBusy(true)
    setError(undefined)
    try {
      await action()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Something went wrong')
    } finally {
      setBusy(false)
    }
  }

  const submitSignIn = () => void run(async () => {
    try {
      await auth.signIn(email.trim(), password)
    } catch (reason) {
      if (reason instanceof UserNotConfirmedError) {
        switchMode('confirm')
        setNotice('This account is not confirmed yet — enter the code from your email.')
        return
      }
      throw reason
    }
  })

  const submitSignUp = () => {
    if (password !== confirmPassword) {
      setError('Passwords do not match')
      return
    }
    void run(async () => {
      await auth.signUp(email.trim(), password)
      switchMode('confirm')
      setNotice(`We emailed a confirmation code to ${email.trim()}.`)
    })
  }

  const submitConfirm = () => void run(async () => {
    await auth.confirmSignUp(email.trim(), code.trim())
    setPassword('')
    setConfirmPassword('')
    setCode('')
    switchMode('sign-in')
    setNotice('Account confirmed — sign in with your password.')
  })

  const resend = () => void run(async () => {
    await auth.resendCode(email.trim())
    setNotice('A new confirmation code is on its way.')
  })

  const submitResetRequest = () => void run(async () => {
    await auth.forgotPassword(email.trim())
    switchMode('reset-confirm')
    setNotice(`If an account exists for ${email.trim()}, we emailed a reset code.`)
  })

  const submitResetConfirm = () => {
    if (password !== confirmPassword) {
      setError('Passwords do not match')
      return
    }
    void run(async () => {
      await auth.confirmForgotPassword(email.trim(), code.trim(), password)
      setPassword('')
      setConfirmPassword('')
      setCode('')
      switchMode('sign-in')
      setNotice('Password updated — sign in with your new password.')
    })
  }

  const submit = mode === 'sign-in' ? submitSignIn
    : mode === 'sign-up' ? submitSignUp
      : mode === 'confirm' ? submitConfirm
        : mode === 'reset-request' ? submitResetRequest
          : submitResetConfirm
  const onEnter = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && !busy) submit()
  }

  return (
    <div className="community-auth wide-field">
      {(mode === 'sign-in' || mode === 'sign-up' || mode === 'reset-request') && <label className="field"><span>Email</span><input type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} onKeyDown={onEnter} /></label>}
      {(mode === 'sign-in' || mode === 'sign-up') && <label className="field"><span>Password</span><input type="password" autoComplete={mode === 'sign-up' ? 'new-password' : 'current-password'} value={password} onChange={(event) => setPassword(event.target.value)} onKeyDown={onEnter} /></label>}
      {mode === 'sign-up' && <>
        <label className="field"><span>Confirm password</span><input type="password" autoComplete="new-password" value={confirmPassword} onChange={(event) => setConfirmPassword(event.target.value)} onKeyDown={onEnter} /></label>
        <small className="muted">At least 8 characters with upper and lower case letters and a number.</small>
      </>}
      {(mode === 'confirm' || mode === 'reset-confirm') && <label className="field"><span>Confirmation code</span><input inputMode="numeric" autoComplete="one-time-code" value={code} onChange={(event) => setCode(event.target.value)} onKeyDown={onEnter} /></label>}
      {mode === 'reset-confirm' && <>
        <label className="field"><span>New password</span><input type="password" autoComplete="new-password" value={password} onChange={(event) => setPassword(event.target.value)} onKeyDown={onEnter} /></label>
        <label className="field"><span>Confirm new password</span><input type="password" autoComplete="new-password" value={confirmPassword} onChange={(event) => setConfirmPassword(event.target.value)} onKeyDown={onEnter} /></label>
        <small className="muted">At least 8 characters with upper and lower case letters and a number.</small>
      </>}
      {error && <p className="form-error" role="alert">{error}</p>}
      {notice && !error && <p className="muted" role="status">{notice}</p>}
      <div className="credential-state">
        {mode === 'sign-in' && <>
          <Button type="button" variant="primary" disabled={busy || auth.status === 'signing-in' || !email.trim() || !password} onClick={submitSignIn}>{busy ? 'Signing in…' : 'Sign in'}</Button>
          <Button type="button" variant="tertiary" disabled={busy} onClick={() => switchMode('sign-up')}>Create account</Button>
          <Button type="button" variant="tertiary" disabled={busy} onClick={() => switchMode('reset-request')}>Forgot password?</Button>
        </>}
        {mode === 'sign-up' && <>
          <Button type="button" variant="primary" disabled={busy || !email.trim() || !password || !confirmPassword} onClick={submitSignUp}>{busy ? 'Creating…' : 'Create account'}</Button>
          <Button type="button" variant="tertiary" disabled={busy} onClick={() => switchMode('sign-in')}>Back to sign in</Button>
        </>}
        {mode === 'confirm' && <>
          <Button type="button" variant="primary" disabled={busy || !code.trim()} onClick={submitConfirm}>{busy ? 'Confirming…' : 'Confirm'}</Button>
          <Button type="button" variant="tertiary" disabled={busy} onClick={resend}>Resend code</Button>
          <Button type="button" variant="tertiary" disabled={busy} onClick={() => switchMode('sign-in')}>Back to sign in</Button>
        </>}
        {mode === 'reset-request' && <>
          <Button type="button" variant="primary" disabled={busy || !email.trim()} onClick={submitResetRequest}>{busy ? 'Sending…' : 'Send reset code'}</Button>
          <Button type="button" variant="tertiary" disabled={busy} onClick={() => switchMode('sign-in')}>Back to sign in</Button>
        </>}
        {mode === 'reset-confirm' && <>
          <Button type="button" variant="primary" disabled={busy || !code.trim() || !password || !confirmPassword} onClick={submitResetConfirm}>{busy ? 'Resetting…' : 'Reset password'}</Button>
          <Button type="button" variant="tertiary" disabled={busy} onClick={() => switchMode('sign-in')}>Back to sign in</Button>
        </>}
      </div>
    </div>
  )
}

export function SettingsPage() {
  const resource = useResource((signal) => api.settings.get(signal))
  const auth = useAuth()
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

  const clearHuggingFaceKey = async () => {
    if (!window.confirm('Remove the saved Hugging Face API key for the entire cluster?')) return
    setSaving(true)
    setSaved(false)
    setError(undefined)
    try {
      const savedSettings = await api.settings.clearHfToken()
      setForm((current) => ({
        ...current,
        hf_token: '',
        hf_token_configured: savedSettings.hf_token_configured,
      }))
      setHuggingFaceApiKey('')
      setSaved(true)
      window.setTimeout(() => setSaved(false), 2400)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not remove the saved Hugging Face API key')
    } finally {
      setSaving(false)
    }
  }

  const hasUnsavedChanges = savedFingerprint !== undefined
    && (editableSettingsFingerprint(form) !== savedFingerprint || Boolean(huggingFaceApiKey.trim()))

  return (
    <div className="page settings-page">
      <PageHeader eyebrow="Preferences" title="Settings" description="Configure cluster credentials, community connectivity, and the SparkDeck interface." actions={<span className="build-version" aria-label="SparkDeck version">Version {SPARKDECK_VERSION}</span>} />
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
            <div className="credential-state"><KeyRound size={17} /><div><strong>Cluster credential</strong><Status status={form.hf_token_configured ? 'running' : 'stopped'}>{form.hf_token_configured ? 'Configured' : 'Not configured'}</Status></div>{form.hf_token_configured && <Button type="button" variant="danger" disabled={saving} onClick={() => void clearHuggingFaceKey()}>Remove saved key</Button>}</div>
          </div>
        </Panel>
        <Panel className="settings-section">
          <div className="settings-heading"><span><Cloud size={18} /></span><div><h2>Community Features</h2><p>Create an account or sign in to share anonymized benchmark telemetry and see community data.</p></div></div>
          <div className="settings-fields">
            <label className="field wide-field"><span>Community API URL</span><input type="url" value={form.community_api_url ?? ''} onChange={(event) => setForm({ ...form, community_api_url: event.target.value })} placeholder="Not configured" /><small>Leave empty to keep SparkDeck entirely local.</small></label>
            {auth.status === 'signed-in'
              ? <div className="credential-state"><KeyRound size={17} /><div><strong>{auth.email ?? 'Community account'}</strong><Status status="running">Signed in</Status></div><Button type="button" onClick={() => void auth.signOut()}>Sign out</Button></div>
              : <CommunitySignInForm />}
            {auth.status === 'signed-in' && auth.clusterSync && <>
              {auth.clusterSync.conflicts.map((conflict) => (
                <p className="muted wide-field" role="status" key={conflict.node}>
                  Sign-in was not applied to: {conflict.node}{conflict.email ? ` (already signed in as ${conflict.email})` : ''}.
                </p>
              ))}
              {auth.clusterSync.errors.length > 0 && <p className="muted wide-field" role="status">
                Could not reach: {auth.clusterSync.errors.map((entry) => entry.split(':')[0]).join(', ')} — they'll stay signed out until synced.
              </p>}
              {auth.clusterSync.applied.length > 0 && auth.clusterSync.conflicts.length === 0 && <p className="muted wide-field" role="status">
                Sign-in synced to {auth.clusterSync.applied.length} node{auth.clusterSync.applied.length === 1 ? '' : 's'}.
              </p>}
            </>}
          </div>
        </Panel>
        <div className="settings-save"><span aria-live="polite">{saved && <><Check size={15} /> Saved</>}</span><Button type="submit" variant="primary" disabled={saving || !hasUnsavedChanges}><Save size={16} /> {saving ? 'Saving…' : 'Save settings'}</Button></div>
      </form>}
      <SoftwareUpdatePanel />
    </div>
  )
}
