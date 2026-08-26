import { useEffect, useState, type FormEvent } from 'react'
import { Check, Clipboard, Link2, Network, Server, ShieldCheck, Unlink } from 'lucide-react'
import { api } from '../api/client'
import type { JoinClusterInput } from '../api/types'
import { Button, ErrorState, LoadingState, PageHeader, Panel, Status } from '../components/ui'
import { useResource } from '../hooks/useResource'

const emptyJoin: JoinClusterInput = { controller_url: '', join_code: '', advertise_url: '', name: '' }

export function ClusterPage() {
  const resource = useResource((signal) => api.onboarding.get(signal))
  const [joining, setJoining] = useState(false)
  const [form, setForm] = useState(emptyJoin)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string>()
  const [notice, setNotice] = useState<string>()
  const [copied, setCopied] = useState<string>()

  useEffect(() => {
    if (!resource.data) return
    setForm((current) => ({
      ...current,
      name: current.name || resource.data?.node.name || '',
      advertise_url: current.advertise_url || resource.data?.node.access_urls[0] || '',
    }))
  }, [resource.data])

  const copy = async (value: string, label: string) => {
    try {
      await navigator.clipboard.writeText(value)
      setCopied(label)
      window.setTimeout(() => setCopied(undefined), 1800)
    } catch {
      setError('Copy was blocked by the browser. Select the value and copy it manually.')
    }
  }

  const join = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError(undefined)
    setNotice(undefined)
    try {
      const status = await api.onboarding.join({
        controller_url: form.controller_url.trim(),
        join_code: form.join_code.trim(),
        advertise_url: form.advertise_url.trim(),
        name: form.name.trim(),
      })
      resource.setData(status)
      setForm((current) => ({ ...current, join_code: '' }))
      setJoining(false)
      setNotice(`Joined ${status.controller_url ?? 'the controller'} as ${status.node.name}.`)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not join the controller')
    } finally {
      setBusy(false)
    }
  }

  const leave = async () => {
    if (!window.confirm('Leave this cluster and make this node its own controller?')) return
    setBusy(true)
    setError(undefined)
    setNotice(undefined)
    try {
      const status = await api.onboarding.leave()
      resource.setData(status)
      setNotice('This node left the cluster and is now a controller.')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not leave the cluster')
    } finally {
      setBusy(false)
    }
  }

  const status = resource.data
  const controller = status?.role === 'controller'

  return (
    <div className="page cluster-page">
      <PageHeader eyebrow="DGX Spark cluster" title="Cluster onboarding" description="Connect SparkDeck nodes over Tailscale, then manage model pulls and deployments from one controller." />
      {resource.loading && <LoadingState label="Loading cluster status" />}
      {resource.error && <ErrorState message={resource.error} onRetry={resource.reload} />}
      {error && <p className="form-error" role="alert">{error}</p>}
      {notice && <p className="inline-success" role="status">{notice}</p>}

      {status && <>
        <section className="cluster-summary" aria-label="Cluster status">
          <Panel><span className="panel-icon"><Server size={17} /></span><div><small>This node</small><strong>{status.node.name}</strong><p>{status.node.id}</p></div></Panel>
          <Panel><span className="panel-icon"><Network size={17} /></span><div><small>Role</small><strong>{controller ? 'Controller' : 'Worker node'}</strong><Status status={controller || status.controller_reachable ? 'running' : 'error'}>{controller ? 'Ready for nodes' : status.controller_reachable ? 'Controller online' : 'Controller unreachable'}</Status></div></Panel>
          <Panel><span className="panel-icon"><Link2 size={17} /></span><div><small>Local port</small><strong>{status.node.port}</strong><p>{status.node.access_urls.length} access {status.node.access_urls.length === 1 ? 'URL' : 'URLs'}</p></div></Panel>
        </section>

        <Panel className="tailnet-note">
          <ShieldCheck size={20} aria-hidden="true" />
          <div><strong>Keep SparkDeck inside your tailnet</strong><p>Use tailnet grants or ACLs and either a Tailscale-IP-only bind or <code>tailscale serve --bg --https=443 localhost:7878</code>. This makes SparkDeck available only inside the tailnet. Do not expose a raw <code>0.0.0.0</code> bind to the public internet.</p><p>Joined nodes are alternate private entry points, not automatic failover. If the controller is offline, running models continue but cluster management is unavailable; local onboarding remains available for recovery.</p></div>
        </Panel>

        <div className="role-grid">
          <Panel><p className="eyebrow">Controller</p><h2>Coordinates the cluster</h2><p>Keep this node online. It holds the node inventory and sends pull and deployment work to joined DGX Spark systems.</p></Panel>
          <Panel><p className="eyebrow">Joined node</p><h2>Runs assigned work</h2><p>A worker advertises its private Tailscale URL and provides another private entry point to the same controller. It does not become a standby controller.</p></Panel>
        </div>

        {controller && !joining && <div className="onboarding-grid">
          <Panel className="onboarding-panel">
            <div className="section-heading"><div><h2>Create from this controller</h2><p>This node is already a controller by default—no creation step is required. Use these details on the DGX Spark you want to join.</p></div></div>
            <ol className="setup-steps">
              {(status.instructions?.length ? status.instructions : [
                'Connect both systems to the same Tailscale tailnet.',
                'Open Cluster onboarding on the node you want to join.',
                'Enter this controller URL and the one-time pairing code.',
              ]).map((instruction, index) => <li key={instruction}><span>{index + 1}</span><p>{instruction}</p></li>)}
            </ol>
            <div className="access-list">
              <h3>Controller access URLs</h3>
              {status.node.access_urls.length === 0 && <p className="muted">No private access URL detected. Configure Tailscale Serve, then refresh this page.</p>}
              {status.node.access_urls.map((url) => <div className="copy-row" key={url}><code>{url}</code><Button type="button" variant="tertiary" aria-label={`Copy ${url}`} onClick={() => void copy(url, url)}>{copied === url ? <Check size={15} /> : <Clipboard size={15} />}{copied === url ? 'Copied' : 'Copy'}</Button></div>)}
            </div>
            <div className="join-code"><span>One-time pairing code</span><div><code>{status.join_code ?? 'Unavailable'}</code>{status.join_code && <Button type="button" variant="tertiary" onClick={() => void copy(status.join_code!, 'code')}>{copied === 'code' ? <Check size={15} /> : <Clipboard size={15} />}{copied === 'code' ? 'Copied' : 'Copy'}</Button>}</div><small>Share this only with the node you are joining. It may expire after use.</small></div>
            <Button type="button" onClick={() => setJoining(true)}>Join this node to another controller</Button>
          </Panel>
        </div>}

        {controller && joining && <Panel className="join-panel">
          <div><p className="eyebrow">Join an existing controller</p><h2>Connect this DGX Spark</h2><p>Enter details from the other controller. The browser sends them only to this node’s local SparkDeck service.</p></div>
          <form onSubmit={(event) => void join(event)}>
            <label className="field"><span>Controller Tailscale URL</span><input type="url" required value={form.controller_url} onChange={(event) => setForm({ ...form, controller_url: event.target.value })} placeholder="https://controller.your-tailnet.ts.net:7878" /></label>
            <label className="field"><span>Pairing code</span><input required autoComplete="one-time-code" value={form.join_code} onChange={(event) => setForm({ ...form, join_code: event.target.value })} /></label>
            <label className="field"><span>This node’s advertised Tailscale URL</span><input type="url" required value={form.advertise_url} onChange={(event) => setForm({ ...form, advertise_url: event.target.value })} placeholder="https://spark-2.your-tailnet.ts.net:7878" /></label>
            <label className="field"><span>This node’s name</span><input required value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} placeholder="Studio Spark" /></label>
            <div className="modal-actions"><Button type="button" onClick={() => setJoining(false)}>Cancel</Button><Button type="submit" variant="primary" disabled={busy}>{busy ? 'Joining…' : 'Join controller'}</Button></div>
          </form>
        </Panel>}

        {!controller && <Panel className="joined-panel">
          <div><p className="eyebrow">Joined node</p><h2>{status.controller_reachable ? 'Connected to controller' : 'Controller unavailable'}</h2><p className="mono">{status.controller_url}</p><Status status={status.controller_reachable ? 'running' : 'error'}>{status.controller_reachable ? 'Healthy connection' : 'Check Tailscale and controller service'}</Status></div>
          <Button type="button" variant="danger" disabled={busy} onClick={() => void leave()}><Unlink size={16} /> {busy ? 'Leaving…' : 'Leave cluster'}</Button>
        </Panel>}
      </>}
    </div>
  )
}
