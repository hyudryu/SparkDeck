import { useState, type FormEvent } from 'react'
import { Box, Download, HardDrive, Trash2 } from 'lucide-react'
import { api } from '../api/client'
import { Button, EmptyState, ErrorState, LoadingState, PageHeader, Panel, RuntimeMark } from '../components/ui'
import { useResource } from '../hooks/useResource'

function formatBytes(bytes?: number) {
  if (bytes === undefined) return 'Size unavailable'
  const gb = bytes / 1024 / 1024 / 1024
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(bytes / 1024 / 1024).toFixed(0)} MB`
}

export function ImagesPage() {
  const resource = useResource((signal) => api.images.list(signal))
  const [image, setImage] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string>()

  const pull = async (event: FormEvent) => {
    event.preventDefault()
    if (!image.trim()) return
    setBusy(true)
    setError(undefined)
    try {
      await api.images.pull(image.trim())
      setImage('')
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
        <form onSubmit={(event) => void pull(event)}><label className="field"><span>Container image</span><input value={image} onChange={(event) => setImage(event.target.value)} placeholder="vllm/vllm-openai:v0.10.0" /></label><Button type="submit" variant="primary" disabled={busy || !image.trim()}><Download size={16} /> {busy ? 'Pulling…' : 'Pull image'}</Button></form>
        {error && <p className="inline-error" role="alert">{error}</p>}
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
