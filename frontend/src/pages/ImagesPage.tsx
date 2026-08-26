import { useEffect, useState, type FormEvent } from 'react'
import { Box, Download, HardDrive, Trash2 } from 'lucide-react'
import { api } from '../api/client'
import { Button, EmptyState, ErrorState, LoadingState, PageHeader, Panel, RuntimeMark } from '../components/ui'
import { isNodeSelectable, NodeSelector, selectedNodeLabel } from '../components/NodeSelector'
import { useResource } from '../hooks/useResource'

function formatBytes(bytes?: number) {
  if (bytes === undefined) return 'Size unavailable'
  const gb = bytes / 1024 / 1024 / 1024
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(bytes / 1024 / 1024).toFixed(0)} MB`
}

export function ImagesPage() {
  const resource = useResource((signal) => api.images.list(signal))
  const nodes = useResource((signal) => api.nodes.list(signal))
  const onboarding = useResource((signal) => api.onboarding.get(signal))
  const [image, setImage] = useState('')
  const [selectedNodeIds, setSelectedNodeIds] = useState(['local'])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string>()
  const [notice, setNotice] = useState<string>()

  useEffect(() => {
    const inventory = nodes.data
    if (!inventory?.length) return
    setSelectedNodeIds((current) => {
      const available = current.filter((id) => inventory.some((node) => node.id === id && isNodeSelectable(node)))
      const fallback = inventory.find((node) => node.local && isNodeSelectable(node)) ?? inventory.find(isNodeSelectable)
      return available.length ? available : fallback ? [fallback.id] : []
    })
  }, [nodes.data])

  const selectionReady = !nodes.loading && !nodes.error && selectedNodeIds.length > 0
    && selectedNodeIds.every((id) => nodes.data?.some((node) => node.id === id && isNodeSelectable(node)))
  const localLabel = onboarding.data?.role === 'worker' ? 'Controller' : 'This device'

  const pull = async (event: FormEvent) => {
    event.preventDefault()
    if (!image.trim()) return
    setBusy(true)
    setError(undefined)
    setNotice(undefined)
    try {
      const requestedImage = image.trim()
      const result = await api.images.pull(requestedImage, selectedNodeIds)
      setImage('')
      const failures = result.results?.filter((item) => !item.ok) ?? []
      if (!result.ok || failures.length > 0) {
        const detail = failures.map((item) => `${item.node_name}: ${item.error ?? 'pull failed'}`).join('; ')
        setError(detail || 'The image could not be pulled on every selected node.')
        resource.reload()
        return
      }
      const selected = result.selected_nodes?.map((node) => node.id === 'local' ? localLabel : node.name).join(', ')
        || selectedNodeLabel(nodes.data ?? [], result.node_ids, localLabel)
      setNotice(`Pulled ${requestedImage} on ${selected}.`)
      resource.reload()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not pull image')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="page">
      <PageHeader eyebrow="Runtime storage" title="Images" description="Manage the container images used by local model servers." />
      <Panel className="pull-panel">
        <div><h2>Pull an image</h2><p>Use a trusted registry reference with an explicit version or digest.</p></div>
        <form onSubmit={(event) => void pull(event)}>
          <label className="field"><span>Container image</span><input value={image} onChange={(event) => setImage(event.target.value)} placeholder="vllm/vllm-openai:v0.10.0" /></label>
          <Button type="submit" variant="primary" disabled={busy || !image.trim() || !selectionReady}><Download size={16} /> {busy ? 'Pulling…' : `Pull on ${selectedNodeIds.length} ${selectedNodeIds.length === 1 ? 'node' : 'nodes'}`}</Button>
          <NodeSelector nodes={nodes.data ?? []} selectedIds={selectedNodeIds} onChange={setSelectedNodeIds} loading={nodes.loading} error={nodes.error} onRetry={nodes.reload} disabled={busy} localLabel={localLabel} />
        </form>
        {error && <p className="inline-error" role="alert">{error}</p>}
        {notice && <p className="inline-success" role="status">{notice}</p>}
      </Panel>
      <div className="section-heading"><div><h2>Local images</h2><p>Images currently available to managed deployments.</p></div></div>
      {resource.loading && <LoadingState label="Loading images" />}
      {resource.error && <ErrorState message={resource.error} onRetry={resource.reload} />}
      {!resource.loading && !resource.error && resource.data?.length === 0 && <EmptyState title="No runtime images" description="Pull an image above or launch a managed model to get started." />}
      {resource.data && resource.data.length > 0 && <div className="image-list">{resource.data.map((item) => <Panel className="image-row" key={item.id}>
        <span className="image-icon"><Box size={18} /></span><div className="image-main"><h2>{item.repository ?? item.id}{item.tag ? `:${item.tag}` : ''}</h2><p className="mono">{item.id}</p><div className="runtime-row">{item.runtimes?.map((runtime) => <RuntimeMark runtime={runtime} key={runtime} />)}</div></div><div className="image-size"><HardDrive size={14} /> {formatBytes(item.size)}</div><Button variant="tertiary" disabled={item.in_use} title={item.in_use ? 'Stop deployments using this image first' : 'Remove image'} aria-label={`Remove ${item.repository ?? item.id}`} onClick={() => void api.images.remove(item.id).then(resource.reload)}><Trash2 size={16} /></Button>
      </Panel>)}</div>}
    </div>
  )
}
