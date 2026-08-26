import { useEffect, useState, type FormEvent } from 'react'
import { Check, Clipboard, Edit3, Link2, Network, RefreshCw, Server, ShieldCheck, Unlink } from 'lucide-react'
import { api } from '../api/client'
import type { JoinClusterInput, NodeInventoryItem, OnboardingStatus } from '../api/types'
import { Button, ErrorState, LoadingState, PageHeader, Panel, Status } from '../components/ui'
import { useResource } from '../hooks/useResource'

const emptyJoin: JoinClusterInput = { controller_url: '', join_code: '', advertise_url: '', name: '' }
const maximumNodeNameLength = 80

function isCurrentEntryNode(node: NodeInventoryItem, onboarding: OnboardingStatus) {
  return onboarding.role === 'controller' ? node.local === true : node.id === onboarding.node.id
}

function nodeRole(node: NodeInventoryItem, onboarding: OnboardingStatus) {
  if (isCurrentEntryNode(node, onboarding)) return `${onboarding.role === 'controller' ? 'Controller' : 'Worker node'} · Current entry node`
  if (node.id === 'local' || node.local) return 'Controller'
  return 'Worker node'
}

export function ClusterPage() {
  const resource = useResource((signal) => api.onboarding.get(signal))
  const nodes = useResource((signal) => api.nodes.list(signal))
  const [joining, setJoining] = useState(false)
  const [form, setForm] = useState(emptyJoin)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string>()
  const [notice, setNotice] = useState<string>()
  const [copied, setCopied] = useState<string>()
  const [editingNodeId, setEditingNodeId] = useState<string>()
  const [nodeName, setNodeName] = useState('')
  const [renamingNodeId, setRenamingNodeId] = useState<string>()
  const [nodeError, setNodeError] = useState<{ id: string; message: string }>()

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

  const editNode = (node: NodeInventoryItem) => {
    setEditingNodeId(node.id)
    setNodeName(node.name)
    setNodeError(undefined)
    setNotice(undefined)
  }

  const cancelNodeEdit = () => {
    setEditingNodeId(undefined)
    setNodeName('')
    setNodeError(undefined)
  }

  const renameNode = async (event: FormEvent, node: NodeInventoryItem) => {
    event.preventDefault()
    const name = nodeName.trim()
    if (!name) {
      setNodeError({ id: node.id, message: 'Enter a node name.' })
      return
    }
    if (name.length > maximumNodeNameLength) {
      setNodeError({ id: node.id, message: `Node names must be ${maximumNodeNameLength} characters or fewer.` })
      return
    }
    if (name === node.name) {
      cancelNodeEdit()
      return
    }

    setRenamingNodeId(node.id)
    setNodeError(undefined)
    setNotice(undefined)
    try {
      const updated = await api.nodes.rename(node.id, { name })
      nodes.setData((nodes.data ?? []).map((item) => item.id === node.id ? { ...item, ...updated } : item))
      if (resource.data && isCurrentEntryNode(node, resource.data)) {
        resource.setData({ ...resource.data, node: { ...resource.data.node, name: updated.name } })
      }
      setEditingNodeId(undefined)
      setNodeName('')
      setNotice(`Renamed ${node.name} to ${updated.name}.`)
    } catch (reason) {
      setNodeError({ id: node.id, message: reason instanceof Error ? reason.message : 'Could not rename this node' })
    } finally {
      setRenamingNodeId(undefined)
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

        <section className="node-management" aria-labelledby="node-management-title">
          <div className="section-heading">
            <div><h2 id="node-management-title">Cluster nodes</h2><p>Use a recognizable name for each system. Runtime status and node identity are unchanged.</p></div>
            <Button type="button" variant="tertiary" onClick={nodes.reload} disabled={nodes.loading || Boolean(renamingNodeId)}><RefreshCw size={15} aria-hidden="true" /> Refresh</Button>
          </div>
          {nodes.loading && <LoadingState label="Loading cluster nodes" />}
          {nodes.error && <ErrorState message={nodes.error} onRetry={nodes.reload} />}
          {!nodes.loading && !nodes.error && nodes.data?.length === 0 && <p className="node-management-empty">No cluster nodes are registered yet.</p>}
          {!nodes.loading && !nodes.error && Boolean(nodes.data?.length) && <ul className="node-management-list">
            {nodes.data?.map((node) => {
              const editing = editingNodeId === node.id
              const saving = renamingNodeId === node.id
              return <li key={node.id} className="node-management-row" aria-busy={saving}>
                <div className="node-management-icon" aria-hidden="true"><Server size={17} /></div>
                <div className="node-management-identity">
                  <strong>{node.name}</strong>
                  <span className="mono">{node.id}</span>
                </div>
                <div className="node-management-state">
                  <span>{nodeRole(node, status)}</span>
                  <Status status={node.online === false ? 'error' : 'running'}>{node.online === false ? 'Offline' : 'Online'}</Status>
                </div>
                {editing ? <form className="node-rename-form" noValidate onSubmit={(event) => void renameNode(event, node)}>
                  <label className="field"><span className="sr-only">New name for {node.name}</span><input autoFocus required maxLength={maximumNodeNameLength} value={nodeName} disabled={saving} onChange={(event) => setNodeName(event.target.value)} aria-invalid={nodeError?.id === node.id} aria-describedby={nodeError?.id === node.id ? `node-error-${node.id}` : undefined} /></label>
                  <div className="node-rename-actions"><Button type="submit" variant="primary" disabled={saving}>{saving ? 'Saving…' : 'Save'}</Button><Button type="button" disabled={saving} onClick={cancelNodeEdit}>Cancel</Button></div>
                  {nodeError?.id === node.id && <p id={`node-error-${node.id}`} className="inline-error" role="alert">{nodeError.message}</p>}
                </form> : <Button type="button" className="node-edit-button" onClick={() => editNode(node)} disabled={Boolean(renamingNodeId)} aria-label={`Edit name for ${node.name}`}><Edit3 size={15} aria-hidden="true" /> Edit</Button>}
              </li>
            })}
          </ul>}
        </section>

        <Panel className="tailnet-note">
          <ShieldCheck size={20} aria-hidden="true" />
          <div><strong>Prefer a Tailscale HTTPS address</strong><p>Keep SparkDeck on <code>127.0.0.1:7878</code>, run <code>tailscale serve --bg --https=443 http://127.0.0.1:7878</code>, and join with the resulting <code>https://machine.tailnet.ts.net</code> URL. Tailnet grants or ACLs still control who can connect. A direct <code>http://100.x.x.x:7878</code> URL is WireGuard-encrypted but is a browser-insecure origin, so use it only when HTTPS is unavailable. Never use Tailscale Funnel or expose a raw <code>0.0.0.0</code> bind.</p><p>Joined nodes are alternate private entry points, not automatic failover. If the controller is offline, running models continue but cluster management is unavailable; local onboarding remains available for recovery.</p></div>
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
          <div><p className="eyebrow">Join an existing controller</p><h2>Connect this DGX Spark</h2><p>Choose the SparkDeck machine that will remain the controller. If one machine already owns workloads, choose that machine. Enter that controller’s URL and pairing code here—not this node’s URL.</p></div>
          <form onSubmit={(event) => void join(event)}>
            <div className="field"><label htmlFor="controller-tailnet-url">Chosen controller Tailscale URL</label><input id="controller-tailnet-url" type="url" required aria-describedby="controller-tailnet-url-help" value={form.controller_url} onChange={(event) => setForm({ ...form, controller_url: event.target.value })} placeholder="https://controller.your-tailnet.ts.net" /><small id="controller-tailnet-url-help">Use the HTTPS URL printed by Tailscale Serve on the chosen controller.</small></div>
            <label className="field"><span>Pairing code</span><input required autoComplete="one-time-code" value={form.join_code} onChange={(event) => setForm({ ...form, join_code: event.target.value })} /></label>
            <div className="field"><label htmlFor="advertised-tailnet-url">This node’s advertised Tailscale URL</label><input id="advertised-tailnet-url" type="url" required aria-describedby="advertised-tailnet-url-help" value={form.advertise_url} onChange={(event) => setForm({ ...form, advertise_url: event.target.value })} placeholder="https://spark-2.your-tailnet.ts.net" /><small id="advertised-tailnet-url-help">Use this node’s own Tailscale Serve HTTPS URL.</small></div>
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
