import { useEffect, useMemo, useRef, useState, type DragEvent, type FormEvent } from 'react'
import { AlertTriangle, ArrowRight, Database, GripVertical, HardDrive, RefreshCw, Trash2, UploadCloud } from 'lucide-react'
import { api } from '../api/client'
import type { StorageModel, StorageNode, StorageTransferJob } from '../api/types'
import { Button, EmptyState, ErrorState, LoadingState, PageHeader, Panel, Status } from '../components/ui'
import { useResource } from '../hooks/useResource'
import { formatBytes } from '../utils/format'

type DraggedModel = { modelId: string; sourceNodeId: string; sourceNodeName: string }

function formatTimestamp(value?: string | number) {
  if (!value) return 'Not reported'
  const date = new Date(typeof value === 'number' && value < 1_000_000_000_000 ? value * 1000 : value)
  return Number.isNaN(date.valueOf())
    ? value
    : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date)
}

function jobProgress(job: StorageTransferJob) {
  const reported = job.progress === undefined ? Number.NaN : job.progress <= 1 ? job.progress * 100 : job.progress
  const measured = job.bytes_total > 0 ? (job.bytes_transferred / job.bytes_total) * 100 : 0
  return Math.max(0, Math.min(100, Number.isFinite(reported) ? reported : measured))
}

function canCancel(status: string) {
  return !['completed', 'failed', 'cancelled', 'canceled'].includes(status.toLowerCase())
}

export function StoragePage() {
  const resource = useResource((signal) => api.storage.get(signal))
  const [sourceNodeId, setSourceNodeId] = useState('')
  const [modelId, setModelId] = useState('')
  const [targetNodeIds, setTargetNodeIds] = useState<string[]>([])
  const [busy, setBusy] = useState<string>()
  const [error, setError] = useState<string>()
  const [notice, setNotice] = useState<string>()
  const [draggedModel, setDraggedModel] = useState<DraggedModel>()
  const [dropTargetId, setDropTargetId] = useState<string>()
  const draggedModelRef = useRef<DraggedModel | undefined>(undefined)

  const nodes = useMemo(() => resource.data?.nodes ?? [], [resource.data?.nodes])
  const sourceNode = nodes.find((node) => node.id === sourceNodeId)
  const sourceModels = sourceNode?.models.filter((model) => !model.partial) ?? []
  const inventory = useMemo(() => {
    const models = new Map<string, { model: StorageModel; nodes: Map<string, StorageModel> }>()
    nodes.forEach((node) => node.models.forEach((model) => {
      const current = models.get(model.model_id)
      if (current) {
        current.nodes.set(node.id, model)
        if (current.model.partial && !model.partial) current.model = model
      } else models.set(model.model_id, { model, nodes: new Map([[node.id, model]]) })
    }))
    return [...models.values()]
  }, [nodes])

  useEffect(() => {
    if (!nodes.length) return
    const selected = nodes.find((node) => node.id === sourceNodeId && node.online && node.models.some((model) => !model.partial))
      ?? nodes.find((node) => node.online && node.models.some((model) => !model.partial))
    const completeModels = selected?.models.filter((model) => !model.partial) ?? []
    const nextSourceId = selected?.id ?? ''
    const nextModelId = completeModels.some((model) => model.model_id === modelId)
      ? modelId
      : completeModels[0]?.model_id ?? ''
    setSourceNodeId(nextSourceId)
    setModelId(nextModelId)
    setTargetNodeIds((current) => current.filter((id) => id !== nextSourceId && nodes.some((node) => node.id === id && node.online && !node.models.some((model) => model.model_id === nextModelId))))
  }, [modelId, nodes, sourceNodeId])

  const hasActiveJobs = resource.data?.jobs.some((job) => canCancel(job.status)) ?? false
  useEffect(() => {
    if (!hasActiveJobs) return
    const timer = window.setInterval(resource.reload, 3000)
    return () => window.clearInterval(timer)
  }, [hasActiveJobs, resource.reload])

  const enable = async (enabled: boolean) => {
    setBusy('settings')
    setError(undefined)
    setNotice(undefined)
    try {
      await api.storage.setEnabled(enabled)
      setNotice(enabled ? 'Virtual NAS enabled.' : 'Virtual NAS disabled.')
      resource.reload()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not update Virtual NAS')
    } finally {
      setBusy(undefined)
    }
  }

  const queueTransfer = async (transferModelId: string, source: StorageNode | DraggedModel, targets: string[]) => {
    if (!targets.length) return
    setBusy('transfer')
    setError(undefined)
    setNotice(undefined)
    try {
      await api.storage.transfer({
        model_id: transferModelId,
        source_node_id: 'id' in source ? source.id : source.sourceNodeId,
        target_node_ids: targets,
      })
      const targetNames = targets.map((id) => nodes.find((node) => node.id === id)?.name ?? id).join(', ')
      setNotice(`Queued ${transferModelId} for transfer to ${targetNames}.`)
      setTargetNodeIds([])
      resource.reload()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not queue model transfer')
    } finally {
      setBusy(undefined)
      setDraggedModel(undefined)
      setDropTargetId(undefined)
      draggedModelRef.current = undefined
    }
  }

  const submitTransfer = (event: FormEvent) => {
    event.preventDefault()
    if (sourceNode && modelId && targetNodeIds.length) void queueTransfer(modelId, sourceNode, targetNodeIds)
  }

  const startDrag = (event: DragEvent, model: StorageModel, node: StorageNode) => {
    if (model.partial) return
    const payload = { modelId: model.model_id, sourceNodeId: node.id, sourceNodeName: node.name }
    draggedModelRef.current = payload
    setDraggedModel(payload)
    event.dataTransfer.effectAllowed = 'copy'
    event.dataTransfer.setData('text/plain', model.model_id)
  }

  const dropModel = (event: DragEvent, target: StorageNode) => {
    const payload = draggedModelRef.current
    if (!payload || !target.online || payload.sourceNodeId === target.id || target.models.some((model) => model.model_id === payload.modelId)) return
    event.preventDefault()
    void queueTransfer(payload.modelId, payload, [target.id])
  }

  const removeModel = async (node: StorageNode, model: StorageModel) => {
    if (!window.confirm(`Delete ${model.model_id} from ${node.name}? Other node copies are not affected.`)) return
    const busyKey = `delete:${node.id}:${model.model_id}`
    setBusy(busyKey)
    setError(undefined)
    setNotice(undefined)
    try {
      await api.storage.removeModel(node.id, model.model_id)
      setNotice(`Deleted ${model.model_id} from ${node.name}.`)
      resource.reload()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not delete model weights')
    } finally {
      setBusy(undefined)
    }
  }

  const cancel = async (job: StorageTransferJob) => {
    setBusy(job.id)
    setError(undefined)
    setNotice(undefined)
    try {
      await api.storage.cancel(job.id)
      setNotice(`Cancelled transfer of ${job.model_id} to ${job.target_node_name}.`)
      resource.reload()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not cancel transfer')
    } finally {
      setBusy(undefined)
    }
  }

  return (
    <div className="page storage-page">
      <PageHeader
        eyebrow="Cluster model weights"
        title="Storage"
        description="Treat model weights across your SparkDeck nodes as one managed inventory, without exposing host file paths."
        actions={resource.data?.enabled ? <>
          <Button onClick={resource.reload}><RefreshCw size={15} /> Refresh</Button>
          <Button onClick={() => void enable(false)} disabled={busy === 'settings'}>Disable Virtual NAS</Button>
        </> : undefined}
      />

      {resource.loading && !resource.data && <LoadingState label="Loading Virtual NAS" />}
      {resource.error && !resource.data && <ErrorState message={resource.error} onRetry={resource.reload} />}
      {error && <p className="inline-error" role="alert">{error}</p>}
      {notice && <p className="inline-success" role="status">{notice}</p>}

      {resource.data && !resource.data.enabled && <Panel className="storage-disabled">
        <span className="storage-feature-icon"><Database size={20} /></span>
        <div><h2>Virtual NAS is off</h2><p>Enable it to inventory model weights by node and queue copies between online systems. SparkDeck reports model identity and size only, never filesystem paths.</p></div>
        <Button variant="primary" onClick={() => void enable(true)} disabled={busy === 'settings'}>{busy === 'settings' ? 'Enabling…' : 'Enable Virtual NAS'}</Button>
      </Panel>}

      {resource.data?.enabled && <>
        <div className="section-heading"><div><h2>Node storage</h2><p>Drag an individual model to another online node, or use the transfer form below.</p></div></div>
        {nodes.length === 0 ? <EmptyState title="No storage nodes" description="Join a node to the cluster before transferring model weights." /> : <div className="storage-node-grid">
          {nodes.map((node) => {
            const used = node.models.reduce((total, model) => total + model.size_bytes, 0)
            // "Used" counts SparkDeck-managed model weights only, so the
            // capacity denominator must be that usage plus the free space on
            // the model-cache mount (reported with the inventory). The free
            // reading is only meaningful while the node's inventory is valid —
            // the manager marks a node offline when that request fails — and
            // zero is a genuinely full disk, not missing data.
            const cacheFree = node.online && typeof node.free_size === 'number' ? node.free_size : undefined
            const hasFree = cacheFree !== undefined
            const capacity = hasFree ? used + cacheFree : (node.total_size ?? 0)
            const alreadyStored = Boolean(draggedModel && node.models.some((model) => model.model_id === draggedModel.modelId))
            const validDrop = Boolean(draggedModel && node.online && draggedModel.sourceNodeId !== node.id && !alreadyStored)
            return <Panel
              className={`storage-node-panel ${validDrop ? 'drop-target' : ''} ${dropTargetId === node.id ? 'drop-active' : ''}`}
              key={node.id}
              aria-label={`Storage on ${node.name}`}
              onDragEnter={(event) => {
                if (!validDrop) return
                event.preventDefault()
                setDropTargetId(node.id)
              }}
              onDragOver={(event) => {
                if (!validDrop) return
                event.preventDefault()
                event.dataTransfer.dropEffect = 'copy'
              }}
              onDragLeave={(event) => {
                if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDropTargetId(undefined)
              }}
              onDrop={(event) => dropModel(event, node)}
            >
              <div className="storage-node-heading"><HardDrive size={18} /><div><h3>{node.name}</h3><Status status={node.online ? 'running' : 'offline'}>{node.online ? 'Online' : 'Offline'}</Status></div></div>
              <div className="storage-capacity">
                <span>{formatBytes(used)} used</span>
                {hasFree && <span>{formatBytes(cacheFree)} free</span>}
                <span>{formatBytes(capacity)} total</span>
              </div>
              <div className="storage-capacity-track" aria-label={`${node.name} used model storage`}><span style={{ width: `${capacity > 0 ? Math.min(100, (used / capacity) * 100) : 0}%` }} /></div>
              <p className="storage-drop-hint">{dropTargetId === node.id ? `Drop to copy ${draggedModel?.modelId}` : alreadyStored ? 'This model is already available here' : node.online ? 'Drop model weights here to queue a copy' : 'Node must be online to receive transfers'}</p>
              {node.models.length === 0 ? <p className="storage-node-empty">No model weights reported</p> : <ul className="storage-weight-list">
                {node.models.map((model) => <li
                  key={`${node.id}:${model.model_id}`}
                  draggable={node.online && !model.partial}
                  aria-label={model.partial ? `Partial cache ${model.model_id} on ${node.name}` : `Transfer ${model.model_id} from ${node.name}`}
                  onDragStart={(event) => startDrag(event, model, node)}
                  onDragEnd={() => {
                    draggedModelRef.current = undefined
                    setDraggedModel(undefined)
                    setDropTargetId(undefined)
                  }}
                >
                  {model.partial ? <AlertTriangle className="storage-partial-icon" size={15} aria-label="Partial cache" /> : <GripVertical size={15} aria-hidden="true" />}
                  <div><strong>{model.model_id}</strong><small>{formatBytes(model.size_bytes)}{model.revision ? ` · ${model.revision}` : ''}{model.partial ? ' · Partial' : ''}</small></div>
                  <Button
                    variant="tertiary"
                    aria-label={`Delete ${model.model_id} from ${node.name}`}
                    title={`Delete from ${node.name}`}
                    disabled={!node.online || busy === `delete:${node.id}:${model.model_id}`}
                    onClick={() => void removeModel(node, model)}
                  ><Trash2 size={15} /></Button>
                </li>)}
              </ul>}
            </Panel>
          })}
        </div>}

        <Panel className="storage-transfer-panel">
          <div><h2>Queue a transfer</h2><p>This form is the keyboard and touch-friendly alternative to drag and drop. Choose one source and one or more online targets.</p></div>
          <form onSubmit={submitTransfer} aria-label="Queue model transfer">
            <div className="field-grid">
              <label className="field"><span>Source node</span><select value={sourceNodeId} onChange={(event) => {
                const next = nodes.find((node) => node.id === event.target.value)
                setSourceNodeId(event.target.value)
                setModelId(next?.models.find((model) => !model.partial)?.model_id ?? '')
                setTargetNodeIds((current) => current.filter((id) => id !== event.target.value))
              }}><option value="">Select a source</option>{nodes.map((node) => {
                const completeCount = node.models.filter((model) => !model.partial).length
                return <option key={node.id} value={node.id} disabled={!node.online || completeCount === 0}>{node.name}{!node.online ? ' (offline)' : completeCount === 0 ? ' (no complete models)' : ''}</option>
              })}</select></label>
              <label className="field"><span>Model weights</span><select value={modelId} onChange={(event) => {
                const nextModelId = event.target.value
                setModelId(nextModelId)
                setTargetNodeIds((current) => current.filter((id) => !nodes.find((node) => node.id === id)?.models.some((model) => model.model_id === nextModelId)))
              }} disabled={!sourceNode}><option value="">Select a model</option>{sourceModels.map((model) => <option key={model.model_id} value={model.model_id}>{model.model_id}</option>)}</select></label>
            </div>
            <fieldset className="storage-targets"><legend>Target nodes</legend><p>Select every online node that should receive a copy.</p><div>
              {nodes.filter((node) => node.id !== sourceNodeId).map((node) => {
                const alreadyAvailable = Boolean(modelId && node.models.some((model) => model.model_id === modelId))
                const unavailable = !node.online || alreadyAvailable
                return <label className={`check-field ${unavailable ? 'unavailable' : ''}`} key={node.id}><input type="checkbox" checked={targetNodeIds.includes(node.id)} disabled={unavailable || busy === 'transfer'} onChange={(event) => setTargetNodeIds((current) => event.target.checked ? [...current, node.id] : current.filter((id) => id !== node.id))} /><span><strong>{node.name}</strong><small>{!node.online ? 'Offline' : alreadyAvailable ? 'Model already available' : `${node.models.length} models available`}</small></span></label>
              })}
              {nodes.filter((node) => node.id !== sourceNodeId).length === 0 && <p className="storage-target-empty">No other nodes are available.</p>}
            </div></fieldset>
            <div className="storage-transfer-actions"><span>{targetNodeIds.length ? `${targetNodeIds.length} target${targetNodeIds.length === 1 ? '' : 's'} selected` : 'Select at least one target'}</span><Button type="submit" variant="primary" disabled={busy === 'transfer' || !modelId || !sourceNodeId || targetNodeIds.length === 0}><UploadCloud size={16} /> {busy === 'transfer' ? 'Queueing…' : 'Queue transfer'}</Button></div>
          </form>
        </Panel>

        <div className="section-heading"><div><h2>Model inventory</h2><p>Availability matrix across every storage node.</p></div></div>
        {inventory.length === 0 ? <EmptyState title="No model weights found" description="Models downloaded on participating nodes will appear here." /> : <Panel className="table-panel">
          <div className="responsive-table storage-model-table" role="table" aria-label="Model storage inventory">
            <div className="table-row table-header" role="row"><span role="columnheader">Model</span><span role="columnheader">Size</span><span role="columnheader">Files</span><span role="columnheader">Node availability</span></div>
            {inventory.map(({ model, nodes: locations }) => <div className="table-row" role="row" key={model.model_id}>
              <div role="cell" data-label="Model"><strong>{model.model_id}</strong><small>{model.partial ? 'Partial cache' : model.revision ?? 'Default revision'} · {formatTimestamp(model.last_modified)}</small></div>
              <div role="cell" data-label="Size">{formatBytes(model.size_bytes)}</div>
              <div role="cell" data-label="Files">{model.file_count ?? 'Not reported'}</div>
              <div role="cell" data-label="Node availability" className="storage-availability" aria-label={`Model availability for ${model.model_id}`}>{nodes.map((node) => {
                const location = locations.get(node.id)
                return <span key={node.id} className={location?.partial ? 'partial' : location ? 'available' : ''}><span aria-hidden="true">{location?.partial ? '!' : location ? '✓' : '—'}</span> {node.name}</span>
              })}</div>
            </div>)}
          </div>
        </Panel>}

        <div className="section-heading"><div><h2>Transfer queue</h2><p>Queued, active, completed, and failed model copies.</p></div></div>
        {resource.data.jobs.length === 0 ? <EmptyState title="No transfers yet" description="Drag a model between node cards or use the transfer form to queue a copy." /> : <Panel className="table-panel">
          <div className="responsive-table storage-job-table" role="table" aria-label="Model transfer queue">
            <div className="table-row table-header" role="row"><span role="columnheader">Model</span><span role="columnheader">Route</span><span role="columnheader">Status</span><span role="columnheader">Progress</span><span role="columnheader">Actions</span></div>
            {resource.data.jobs.map((job) => {
              const progress = jobProgress(job)
              return <div className="table-row" role="row" key={job.id}>
                <div role="cell" data-label="Model"><strong>{job.model_id}</strong><small>Created {formatTimestamp(job.created_at)}</small></div>
                <div role="cell" data-label="Route" className="storage-route"><span>{job.source_node_name}</span><ArrowRight size={13} aria-label="to" /><span>{job.target_node_name}</span></div>
                <div role="cell" data-label="Status"><Status status={job.status} />{job.error && <small className="storage-job-error" role="alert">{job.error}</small>}</div>
                <div role="cell" data-label="Progress" className="storage-job-progress"><progress max="100" value={progress} aria-label={`Transfer ${job.model_id} progress`} /><span>{Math.round(progress)}% · {formatBytes(job.bytes_transferred)} of {formatBytes(job.bytes_total)}</span></div>
                <div role="cell" data-label="Actions" className="row-actions">{canCancel(job.status) && <Button variant="tertiary" disabled={busy === job.id} onClick={() => void cancel(job)}>Cancel</Button>}</div>
              </div>
            })}
          </div>
        </Panel>}
      </>}

      {resource.data && <Panel className="storage-guidance"><h2>How Virtual NAS works</h2><ol>{resource.data.instructions.length ? resource.data.instructions.map((instruction) => <li key={instruction}>{instruction}</li>) : <><li>Keep source and target nodes online while a transfer is active.</li><li>SparkDeck copies model weights between nodes and tracks progress in this queue.</li><li>Deleting a node copy does not remove the same model from other nodes.</li></>}</ol></Panel>}
    </div>
  )
}
