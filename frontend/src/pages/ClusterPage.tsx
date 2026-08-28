import { useEffect, useState, type FormEvent } from 'react'
import { Check, Clipboard, Edit3, Eye, EyeOff, Link2, Network, RefreshCw, Server, ShieldCheck, Trash2, Unlink } from 'lucide-react'
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
  const [removingNodeId, setRemovingNodeId] = useState<string>()
  const [visibilityNodeId, setVisibilityNodeId] = useState<string>()
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
        // The shell caches the entry node's name for the sidebar chip. The
        // rename only changed this node's own settings once the write reached
        // it; the controller reports "pending" when an offline worker still
        // holds its old name.
        if (updated.name_sync !== 'pending') {
          window.dispatchEvent(new Event('sparkdeck:node-name-changed'))
        }
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

  const removeNode = async (node: NodeInventoryItem) => {
    const force = node.online === false
    const warning = force
      ? `Forget offline node ${node.name}? SparkDeck cannot notify it, so it may still show the old cluster until you use Leave cluster or join it again. Cached weights stay on that machine.`
      : `Remove ${node.name} from this cluster? SparkDeck will disconnect only this machine and make it a standalone controller. Cached weights stay on that machine.`
    if (!window.confirm(warning)) return

    setRemovingNodeId(node.id)
    setNodeError(undefined)
    setNotice(undefined)
    try {
      await api.nodes.remove(node.id, force)
      nodes.setData((nodes.data ?? []).filter((item) => item.id !== node.id))
      cancelNodeEdit()
      setNotice(force
        ? `Forgot offline node ${node.name}. Its local cluster assignment could not be cleared.`
        : `Removed ${node.name} from the cluster.`)
    } catch (reason) {
      setNodeError({ id: node.id, message: reason instanceof Error ? reason.message : 'Could not remove this node' })
    } finally {
      setRemovingNodeId(undefined)
    }
  }

  const toggleDashboardVisibility = async (node: NodeInventoryItem) => {
    const hidden = node.hidden_from_dashboard !== true
    setVisibilityNodeId(node.id)
    setNodeError(undefined)
    setNotice(undefined)
    try {
      const updated = await api.nodes.setDashboardHidden(node.id, hidden)
      nodes.setData((nodes.data ?? []).map((item) => item.id === node.id ? { ...item, ...updated } : item))
      setNotice(`${node.name} ${hidden ? 'is hidden from' : 'will appear on'} the dashboard.`)
    } catch (reason) {
      setNodeError({ id: node.id, message: reason instanceof Error ? reason.message : 'Could not change dashboard visibility' })
    } finally {
      setVisibilityNodeId(undefined)
    }
  }

  const status = resource.data
  const controller = status?.role === 'controller'
  const canRegisterWorkers = Boolean(status?.join_code) && Boolean(controller || status?.controller_reachable)

  return (
    <div className="page cluster-page">
      <PageHeader eyebrow="DGX Spark cluster" title="Cluster Management" description="Connect SparkDeck nodes over Tailscale, then manage model pulls and deployments from one controller." />
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
            <div><h2 id="node-management-title">Cluster nodes</h2><p>Membership is per machine. Joining one node does not bring along machines from its former cluster; join each machine separately.</p></div>
            <Button type="button" variant="tertiary" onClick={nodes.reload} disabled={nodes.loading || Boolean(renamingNodeId) || Boolean(removingNodeId)}><RefreshCw size={15} aria-hidden="true" /> Refresh</Button>
          </div>
          {nodes.loading && <LoadingState label="Loading cluster nodes" />}
          {nodes.error && <ErrorState message={nodes.error} onRetry={nodes.reload} />}
          {!nodes.loading && !nodes.error && nodes.data?.length === 0 && <p className="node-management-empty">No cluster nodes are registered yet.</p>}
          {!nodes.loading && !nodes.error && Boolean(nodes.data?.length) && <ul className="node-management-list">
            {nodes.data?.map((node) => {
              const editing = editingNodeId === node.id
              const saving = renamingNodeId === node.id
              const removing = removingNodeId === node.id
              const removable = !isCurrentEntryNode(node, status) && node.id !== 'local' && !node.local
              const changingVisibility = visibilityNodeId === node.id
              const dashboardHidden = node.hidden_from_dashboard === true
              return <li key={node.id} className="node-management-row" aria-busy={saving || removing || changingVisibility}>
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
                  <label className="field"><span className="sr-only">New name for {node.name}</span><input autoFocus required maxLength={maximumNodeNameLength} value={nodeName} disabled={saving || removing} onChange={(event) => setNodeName(event.target.value)} aria-invalid={nodeError?.id === node.id} aria-describedby={nodeError?.id === node.id ? `node-error-${node.id}` : undefined} /></label>
                  <div className="node-rename-actions"><Button type="submit" variant="primary" disabled={saving || removing}>{saving ? 'Saving…' : 'Save'}</Button><Button type="button" disabled={saving || removing} onClick={cancelNodeEdit}>Cancel</Button>{removable && <Button type="button" variant="danger" className="node-remove-button" disabled={saving || removing} onClick={() => void removeNode(node)}><Trash2 size={15} aria-hidden="true" /> {removing ? 'Removing…' : node.online === false ? 'Forget node' : 'Remove node'}</Button>}</div>
                  {nodeError?.id === node.id && <p id={`node-error-${node.id}`} className="inline-error" role="alert">{nodeError.message}</p>}
                </form> : <div className="node-management-actions"><Button type="button" variant="tertiary" onClick={() => void toggleDashboardVisibility(node)} disabled={Boolean(renamingNodeId) || Boolean(removingNodeId) || Boolean(visibilityNodeId)} aria-label={`${dashboardHidden ? 'Show' : 'Hide'} ${node.name} ${dashboardHidden ? 'on' : 'from'} dashboard`}>{dashboardHidden ? <Eye size={15} aria-hidden="true" /> : <EyeOff size={15} aria-hidden="true" />}{changingVisibility ? 'Saving…' : dashboardHidden ? 'Show' : 'Hide'}</Button><Button type="button" className="node-edit-button" onClick={() => editNode(node)} disabled={Boolean(renamingNodeId) || Boolean(removingNodeId) || Boolean(visibilityNodeId)} aria-label={`Edit name for ${node.name}`}><Edit3 size={15} aria-hidden="true" /> Edit</Button></div>}
                {!editing && nodeError?.id === node.id && <p id={`node-error-${node.id}`} className="inline-error node-management-error" role="alert">{nodeError.message}</p>}
              </li>
            })}
          </ul>}
        </section>

        <Panel className="tailnet-note">
          <ShieldCheck size={20} aria-hidden="true" />
          <div><strong>Use a private Tailscale address</strong><p>A direct <code>http://100.x.x.x:7878</code> URL works when both machines are on the same tailnet. Tailscale encrypts the connection and tailnet grants or ACLs control who can reach it. Include <code>http://</code> because SparkDeck expects a complete URL. A Tailscale Serve <code>https://machine.tailnet.ts.net</code> URL also works when Serve is configured. Never use Tailscale Funnel or expose SparkDeck to the public internet.</p><p>Joined nodes are alternate private entry points, not automatic failover. If the controller is offline, running models continue but cluster management is unavailable; local onboarding remains available for recovery.</p></div>
        </Panel>

        <div className="role-grid">
          <Panel><p className="eyebrow">Controller</p><h2>Coordinates the cluster</h2><p>Keep this node online. It holds the node inventory and sends pull and deployment work to joined DGX Spark systems.</p></Panel>
          <Panel><p className="eyebrow">Joined node</p><h2>Runs assigned work</h2><p>A worker advertises its private Tailscale URL and provides another private entry point to the same controller. It does not become a standby controller.</p></Panel>
        </div>

        {canRegisterWorkers && !joining && <div className="onboarding-grid">
          <Panel className="onboarding-panel">
            <div className="section-heading"><div><h2>{controller ? 'Add a node from this controller' : 'Add a node through this worker'}</h2><p>{controller ? 'Use these details on the DGX Spark you want to join.' : 'This worker refers the joining node directly to the existing controller. Use this worker’s details on the DGX Spark you want to join.'}</p></div></div>
            <ol className="setup-steps">
              {(status.instructions?.length ? status.instructions : [
                'Connect both systems to the same Tailscale tailnet.',
                'Open Cluster Management on the node you want to join.',
                'Enter this controller URL and the one-time pairing code.',
              ]).map((instruction, index) => <li key={instruction}><span>{index + 1}</span><p>{instruction}</p></li>)}
            </ol>
            <div className="access-list">
              <h3>{controller ? 'Controller access URLs' : 'Worker entry URLs'}</h3>
              {status.node.access_urls.length === 0 && <p className="muted">No private Tailscale URL detected. Confirm Tailscale is connected, then refresh this page.</p>}
              {status.node.access_urls.map((url) => <div className="copy-row" key={url}><code>{url}</code><Button type="button" variant="tertiary" aria-label={`Copy ${url}`} onClick={() => void copy(url, url)}>{copied === url ? <Check size={15} /> : <Clipboard size={15} />}{copied === url ? 'Copied' : 'Copy'}</Button></div>)}
            </div>
            <div className="join-code"><span>One-time pairing code</span><div><code>{status.join_code ?? 'Unavailable'}</code>{status.join_code && <Button type="button" variant="tertiary" onClick={() => void copy(status.join_code!, 'code')}>{copied === 'code' ? <Check size={15} /> : <Clipboard size={15} />}{copied === 'code' ? 'Copied' : 'Copy'}</Button>}</div><small>Share this only with the node you are joining. It may expire after use.</small></div>
            {controller && <Button type="button" onClick={() => setJoining(true)}>Join this node to another controller</Button>}
          </Panel>
        </div>}

        {controller && joining && <Panel className="join-panel">
          <div><p className="eyebrow">Join an existing cluster</p><h2>Connect this DGX Spark</h2><p>Enter the URL and pairing code shown by either the controller or an online worker in the cluster. Only this machine moves; nodes from its previous cluster must join the destination separately.</p></div>
          <form onSubmit={(event) => void join(event)}>
            <div className="field"><label htmlFor="controller-tailnet-url">Existing cluster entry URL</label><input id="controller-tailnet-url" type="url" required aria-describedby="controller-tailnet-url-help" value={form.controller_url} onChange={(event) => setForm({ ...form, controller_url: event.target.value })} placeholder="http://100.x.x.x:7878" /><small id="controller-tailnet-url-help">Use a private Tailscale URL shown by the controller or an online worker, such as <code>http://100.x.x.x:7878</code>.</small></div>
            <label className="field"><span>Pairing code</span><input required autoComplete="one-time-code" value={form.join_code} onChange={(event) => setForm({ ...form, join_code: event.target.value })} /></label>
            <div className="field"><label htmlFor="advertised-tailnet-url">This node’s advertised Tailscale URL</label><input id="advertised-tailnet-url" type="url" required aria-describedby="advertised-tailnet-url-help" value={form.advertise_url} onChange={(event) => setForm({ ...form, advertise_url: event.target.value })} placeholder="http://100.x.x.x:7878" /><small id="advertised-tailnet-url-help">Use this node’s own private Tailscale URL, such as <code>http://100.x.x.x:7878</code>.</small></div>
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
