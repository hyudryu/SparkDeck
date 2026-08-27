import { useCallback, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import { Bug, Cable, Check, Cloud, DownloadCloud, ExternalLink, FileText, KeyRound, MonitorCog, Network, RefreshCw, Save, ShieldCheck } from 'lucide-react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { AppSettings } from '../api/types'
import { useAuth } from '../auth/AuthContext'
import { UserNotConfirmedError } from '../auth/cognitoAuth'
import { Button, ErrorState, LoadingState, PageHeader, Panel, Status } from '../components/ui'
import { LegalDialog } from '../components/LegalDialog'
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
  const active = Boolean(resource.data?.job?.active)

  useEffect(() => {
    if (!active) return
    const timer = window.setInterval(resource.reload, 2500)
    return () => window.clearInterval(timer)
  }, [active, resource.reload])

  const start = async () => {
    const nodeCount = resource.data?.nodes?.length ?? 0
    const targetRevision = resource.data?.target?.revision
    if (!targetRevision) return
    if (!window.confirm(`Update all ${nodeCount} cluster node${nodeCount === 1 ? '' : 's'} to origin/main at ${shortRevision(targetRevision)}? Workers restart one at a time and the controller restarts last.`)) return
    setStarting(true)
    setActionError(undefined)
    try {
      await api.updates.start(targetRevision)
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
  const targetRevision = data?.target?.revision
  const upToDate = Boolean(data?.up_to_date)
  return (
    <Panel className="settings-section software-update-section">
      <div className="settings-heading"><span><DownloadCloud size={18} /></span><div><h2>Software update</h2><p>Update every cluster node to the latest commit on the main branch.</p></div></div>
      <div className="settings-fields">
        {resource.loading && !data && <LoadingState label="Checking for updates" />}
        {resource.error && !data && <ErrorState message={resource.error} onRetry={resource.reload} />}
        {data && <>
          <div className="credential-state wide-field">
            <DownloadCloud size={17} />
            <div>
              <strong>Running {shortRevision(data.current_revision)}</strong>
              <span className="muted">origin/main {targetRevision ? shortRevision(targetRevision) : 'unavailable'} · {data.nodes?.length ?? 0} cluster node{data.nodes?.length === 1 ? '' : 's'}</span>
            </div>
            <Button type="button" variant="primary" disabled={!data.can_update || !targetRevision || upToDate || starting || active} onClick={() => void start()}>{active ? <RefreshCw className="spin" size={16} /> : <DownloadCloud size={16} />} {starting ? 'Starting…' : active ? 'Updating…' : upToDate ? 'Up to date' : 'Update to main'}</Button>
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
  const [ageConfirmed, setAgeConfirmed] = useState(false)

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
    if (!ageConfirmed) {
      setError('You must confirm that you are at least 18 years old')
      return
    }
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
    if (event.key !== 'Enter') return
    event.preventDefault()
    if (!busy) submit()
  }

  return (
    <div className="community-auth wide-field">
      {(mode === 'sign-in' || mode === 'sign-up' || mode === 'reset-request') && <label className="field"><span>Email</span><input type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} onKeyDown={onEnter} /></label>}
      {(mode === 'sign-in' || mode === 'sign-up') && <label className="field"><span>Password</span><input type="password" autoComplete={mode === 'sign-up' ? 'new-password' : 'current-password'} value={password} onChange={(event) => setPassword(event.target.value)} onKeyDown={onEnter} /></label>}
      {mode === 'sign-up' && <>
        <label className="field"><span>Confirm password</span><input type="password" autoComplete="new-password" value={confirmPassword} onChange={(event) => setConfirmPassword(event.target.value)} onKeyDown={onEnter} /></label>
        <small className="muted">At least 8 characters with upper and lower case letters and a number.</small>
        <label className="check-field community-age-confirm"><input type="checkbox" checked={ageConfirmed} onChange={(event) => setAgeConfirmed(event.target.checked)} /><span><strong>I confirm that I am at least 18 years old.</strong><small>By creating an account, you agree to the Terms & Conditions and acknowledge the Privacy Policy available below.</small></span></label>
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
          <Button type="button" variant="primary" disabled={busy || !email.trim() || !password || !confirmPassword || !ageConfirmed} onClick={submitSignUp}>{busy ? 'Creating…' : 'Create account'}</Button>
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
  const [signingOut, setSigningOut] = useState(false)
  const [legalDialog, setLegalDialog] = useState<'privacy' | 'terms'>()
  const privacyButtonRef = useRef<HTMLButtonElement>(null)
  const termsButtonRef = useRef<HTMLButtonElement>(null)
  const closeLegalDialog = useCallback(() => setLegalDialog(undefined), [])

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

  useEffect(() => {
    if (auth.status === 'signed-in') setError(undefined)
  }, [auth.status])

  const signOut = async () => {
    setSigningOut(true)
    setError(undefined)
    try {
      await auth.signOut()
    } catch (reason) {
      const detail = reason instanceof Error ? reason.message : 'The controller could not be reached'
      setError(`Could not sign out: ${detail} Your account may still be paired with this node. Sign in again if prompted, then retry Sign out.`)
    } finally {
      setSigningOut(false)
    }
  }

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
          <div className="settings-heading"><span><Cable size={18} /></span><div><h2>RouterOS switch</h2><p>Connect and manage a MikroTik switch from a SparkDeck node.</p></div></div>
          <div className="settings-fields"><div className="credential-state wide-field"><Cable size={17} /><div><strong>Switch connection</strong><span className="muted">Enter a RouterOS REST API URL manually when discovery cannot cross a routed network.</span></div><Link className="button button-secondary" to="/switch">Open switch setup</Link></div></div>
        </Panel>
        <Panel className="settings-section">
          <div className="settings-heading"><span><Cloud size={18} /></span><div><h2>Community Features</h2><p>Create an account or sign in to access community data. Benchmark sharing remains off until you explicitly enable it.</p></div></div>
          <div className="settings-fields">
            <label className="field wide-field"><span>Community API URL</span><input type="url" value={form.community_api_url ?? ''} onChange={(event) => setForm({ ...form, community_api_url: event.target.value })} placeholder="Not configured" /><small>Leave empty to disable hosted community aggregate lookups. Account sign-in and optional telemetry use their separately configured services.</small></label>
            {auth.status === 'restoring'
              ? <p className="muted wide-field" role="status">Restoring community session…</p>
              : auth.status === 'signed-in'
                ? <div className="credential-state"><KeyRound size={17} /><div><strong>{auth.email ?? 'Community account'}</strong><Status status="running">Signed in</Status></div><Button type="button" disabled={signingOut} onClick={() => void signOut()}>{signingOut ? 'Signing out…' : 'Sign out'}</Button></div>
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
            {auth.status !== 'signed-in' && auth.clusterSync && <>
              {auth.clusterSync.conflicts.map((conflict) => (
                <p className="muted wide-field" role="status" key={conflict.node}>
                  Sign-out was not applied to: {conflict.node}{conflict.email ? ` (signed in as ${conflict.email})` : ''}.
                </p>
              ))}
              {auth.clusterSync.errors.length > 0 && <p className="muted wide-field" role="status">
                Some nodes are still signed in: {auth.clusterSync.errors.map((entry) => entry.split(':')[0]).join(', ')} — they could not be reached.
              </p>}
            </>}
          </div>
        </Panel>
        <div className="settings-save"><span aria-live="polite">{saved && <><Check size={15} /> Saved</>}</span><Button type="submit" variant="primary" disabled={saving || !hasUnsavedChanges}><Save size={16} /> {saving ? 'Saving…' : 'Save settings'}</Button></div>
      </form>}
      <SoftwareUpdatePanel />
      <Panel className="settings-section support-legal-section">
        <div className="settings-heading"><span><ShieldCheck size={18} /></span><div><h2>Support & legal</h2><p>Review how Community Features handle data, read the service terms, or report a problem.</p></div></div>
        <div className="settings-link-list">
          <div><FileText size={17} /><span><strong>Privacy Policy</strong><small>Optional telemetry, Cognito, retention, and California disclosures.</small></span><button ref={privacyButtonRef} className="button button-secondary" type="button" onClick={() => setLegalDialog('privacy')}>View policy</button></div>
          <div><FileText size={17} /><span><strong>Terms & Conditions</strong><small>18+ eligibility, acceptable use, benchmark limitations, and disclaimers.</small></span><button ref={termsButtonRef} className="button button-secondary" type="button" onClick={() => setLegalDialog('terms')}>View terms</button></div>
          <div><Bug size={17} /><span><strong>Report a bug</strong><small>Open a GitHub issue. Never include passwords, tokens, prompts, model outputs, or other sensitive data.</small></span><a className="button button-secondary" href="https://github.com/hyudryu/SparkDeck/issues/new" target="_blank" rel="noopener noreferrer">Report a bug <ExternalLink size={14} /></a></div>
        </div>
      </Panel>
      {legalDialog === 'privacy' && <LegalDialog eyebrow="Your data" title="SparkDeck Privacy Policy" titleId="privacy-policy-title" onClose={closeLegalDialog} returnFocusRef={privacyButtonRef}>
        <p className="legal-effective">Effective August 27, 2026</p>
        <section><h3>Local-first by default</h3><p>SparkDeck's core app runs on systems you control. It keeps benchmark history, runtime details, settings, and operational records locally on your device or cluster. Local storage is not collection by SparkDeck's hosted Community Features service.</p><p>If you do not create or sign in to a Community Features account, SparkDeck does not send account data or benchmark telemetry to the Community Features service.</p></section>
        <section><h3>Community account and authentication</h3><p>SparkDeck and Community Features are intended only for people age 18 or older. Sign-up and sign-in are handled by Amazon Cognito. The account information used by SparkDeck is your email address as username and Cognito account identifier. SparkDeck's Community Features servers do not store your password. Cognito processes credentials and authentication data, and your browser stores session tokens locally to keep you signed in.</p></section>
        <section><h3>Optional benchmark telemetry</h3><p>Telemetry is off unless you sign in and explicitly opt in from Benchmarks. When enabled, eligible existing local samples may be queued along with future samples. If an update expands these upload fields, SparkDeck disables the prior consent and asks you to review and opt in again. The benchmark JSON is limited to:</p><ul><li>model identifier;</li><li>measured inference speed in tokens per second;</li><li>request concurrency, when recorded;</li><li>tensor-parallel (TP) size, when recorded; and</li><li>context-window size.</li></ul><p>Prompt text, system messages, retrieved context, uploaded content, and model output are never included in benchmark telemetry or stored by the Community Features service.</p></section>
        <section><h3>How information is used</h3><p>Account information authenticates Community Features. Benchmark telemetry is used to group comparable results, show expected performance for the same model and configuration, detect invalid submissions, and operate the service. Published results may be aggregated with other users' results.</p><p>Telemetry uploads use a node-scoped credential and idempotency identifier. Hosting and network providers may also process ordinary connection metadata such as IP address, request time, and user agent for security and service operation. This metadata is not part of the benchmark JSON.</p></section>
        <section><h3>Your controls and retention</h3><p>You can turn telemetry off at any time. This stops future uploads and removes unsent queued uploads; it does not delete local benchmark history or recall data already received. Account information and received telemetry are retained only while reasonably needed to provide Community Features, protect the service, meet legal obligations, or maintain aggregated benchmark results. You may request access, correction, or deletion by emailing <a href="mailto:hydryu@gmail.com">hydryu@gmail.com</a>. Backup and aggregated records may persist where legally permitted or where they can no longer reasonably be linked to an account.</p></section>
        <section><h3>Service providers and external links</h3><p>Amazon Cognito provides authentication. GitHub receives information under its own privacy notice if you open or submit a bug report. Services you choose separately, including Hugging Face and Tailscale, operate under their own terms and privacy notices. SparkDeck does not sell personal information or share it for cross-context behavioral advertising.</p></section>
        <section><h3>California privacy disclosures</h3><p>California residents may have rights under applicable law to know, access, correct, or delete personal information; opt out of sale or sharing; limit certain uses of sensitive personal information; and receive equal service when exercising privacy rights. SparkDeck does not sell personal information, use it for targeted advertising, or offer financial incentives for it. Submit a request through the email above; identity may need to be verified before a request is completed.</p><p>SparkDeck does not track people over time across third-party websites for targeted advertising, so it does not respond differently to browser Do Not Track signals. Where legally applicable, Global Privacy Control signals are honored for sale or sharing opt-outs; SparkDeck does not currently conduct either activity.</p></section>
        <section><h3>Security and changes</h3><p>Reasonable safeguards are used, but no system is completely secure. Protect your local SparkDeck installation and credentials. Material policy changes will be identified by a new effective date and, where required, an in-app notice.</p></section>
      </LegalDialog>}
      {legalDialog === 'terms' && <LegalDialog eyebrow="Community Features" title="Terms & Conditions" titleId="terms-title" onClose={closeLegalDialog} returnFocusRef={termsButtonRef}>
        <p className="legal-effective">Effective August 27, 2026</p>
        <section><h3>Agreement and eligibility</h3><p>These Terms govern hosted Community Features. Local SparkDeck software remains governed by the license in its repository. By creating an account or using Community Features, you agree to these Terms, confirm you are at least 18 years old, and confirm you can legally enter this agreement. If you do not agree, do not create an account or use Community Features.</p></section>
        <section><h3>Work in progress</h3><p>SparkDeck and Community Features are works in progress. Features, schemas, model compatibility, benchmark methods, and availability may change, be interrupted, or be discontinued. Community results may be delayed, incomplete, or incorrect.</p></section>
        <section><h3>Benchmark and operational disclaimer</h3><p>Tokens-per-second results and other comparisons are observations, not guarantees. Results vary with model revision, quantization, runtime, hardware, drivers, networking, thermals, power limits, context length, concurrency, TP size, and workload. Verify results on your own systems before making purchasing, capacity, safety, or operational decisions. SparkDeck is not a substitute for hardware, electrical, thermal, network, or security expertise.</p></section>
        <section><h3>Your responsibilities</h3><p>Use Community Features only on systems and with models and data you are authorized to use. Keep credentials and local management interfaces secure. Do not disrupt the service, bypass access controls, submit fabricated or manipulated results, probe other users' systems, infringe rights, or use the service unlawfully.</p></section>
        <section><h3>Submitted telemetry</h3><p>You retain any rights you have in submitted telemetry. By opting in, you grant the SparkDeck project a non-exclusive, worldwide, royalty-free license to host, process, validate, aggregate, analyze, and publish the permitted telemetry solely to operate and improve Community Features and benchmark comparisons. Turning sharing off ends future submissions but does not require already published aggregate statistics to be withdrawn.</p></section>
        <section><h3>Third-party services</h3><p>Models, runtimes, AWS, GitHub, Hugging Face, Tailscale, and other third-party services are governed by their own terms. SparkDeck is not responsible for third-party services, models, licenses, availability, or content.</p></section>
        <section><h3>No warranty</h3><p>To the maximum extent permitted by law, SparkDeck, Community Features, and benchmark information are provided "as is" and "as available," without warranties of accuracy, availability, merchantability, fitness for a particular purpose, or non-infringement. Nothing in these Terms limits rights or warranties that cannot lawfully be excluded.</p></section>
        <section><h3>Limitation of liability</h3><p>To the maximum extent permitted by law, the SparkDeck project maintainer and contributors are not liable for indirect, incidental, special, consequential, exemplary, or punitive damages, or loss of data, profits, use, goodwill, or business opportunity arising from SparkDeck or Community Features. Nothing excludes liability that cannot lawfully be excluded, including liability for fraud or willful misconduct where applicable.</p></section>
        <section><h3>Suspension, termination, and changes</h3><p>You may sign out and turn off telemetry at any time. Access may be suspended when reasonably necessary to protect users, the service, or benchmark integrity, or to address misuse. Material changes will be identified by a new effective date and reasonable notice where required.</p></section>
        <section><h3>Contact</h3><p>Questions about these Terms may be sent to <a href="mailto:hydryu@gmail.com">hydryu@gmail.com</a>. Applicable California and United States law applies without limiting mandatory consumer protections in your place of residence.</p></section>
      </LegalDialog>}
    </div>
  )
}
