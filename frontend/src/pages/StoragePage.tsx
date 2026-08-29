import { useEffect, useMemo, useRef, useState, type CSSProperties, type DragEvent, type FormEvent } from 'react'
import { AlertTriangle, ArrowRight, Database, DownloadCloud, GripVertical, HardDrive, RefreshCw, Trash2, UploadCloud } from 'lucide-react'
import { api } from '../api/client'
import type { StorageModel, StorageNode, StorageTransferJob } from '../api/types'
import { Button, EmptyState, ErrorState, LoadingState, PageHeader, Panel, Status } from '../components/ui'
import { useConfirmDialog } from '../components/useConfirmDialog'
import { useResource } from '../hooks/useResource'
import { formatBytes } from '../utils/format'

type DraggedModel = { modelId: string; sourceNodeId: string; sourceNodeName: string }

function compareModelIdsDescending(leftModelId: string, rightModelId: string) {
  return rightModelId.localeCompare(leftModelId, undefined, {
    numeric: true,
    sensitivity: 'base',
  })
}

function compareModels(left: StorageModel, right: StorageModel) {
  return right.size_bytes - left.size_bytes || compareModelIdsDescending(left.model_id, right.model_id)
}

function isPartialModel(model: StorageModel) {
  // A repository can contain a complete usable revision alongside leftovers
  // from a different interrupted revision. Only a repository with no complete
  // snapshot should be presented as partial in the generic storage inventory.
  return model.partial === true
}

function isTransferableModel(model: StorageModel) {
  return !isPartialModel(model) && model.transferable !== false
}

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

function formatProgress(value: number) {
  const rounded = Math.round(value * 10) / 10
  return Number.isInteger(rounded) ? rounded.toFixed(0) : rounded.toFixed(1)
}

function timestampValue(value?: string | number) {
  if (!value) return 0
  const timestamp = new Date(typeof value === 'number' && value < 1_000_000_000_000 ? value * 1000 : value).valueOf()
  return Number.isFinite(timestamp) ? timestamp : 0
}

function compareJobsNewestFirst(left: StorageTransferJob, right: StorageTransferJob) {
  return timestampValue(right.created_at) - timestampValue(left.created_at) || right.id.localeCompare(left.id)
}

function formatTransferRate(job: StorageTransferJob) {
  if (!job.bytes_per_second || job.bytes_per_second <= 0) return undefined
  const gigabytesPerSecond = job.bytes_per_second / 1_000_000_000
  const rate = gigabytesPerSecond < 0.001
    ? '<0.001'
    : gigabytesPerSecond.toFixed(gigabytesPerSecond < 0.01 ? 3 : 2)
  return `${rate} GB/s avg`
}

function SmoothProgress({ value, label }: { value: number; label: string }) {
  const progress = Math.max(0, Math.min(100, value))
  return <div
    className="storage-progress"
    role="progressbar"
    aria-label={label}
    aria-valuemin={0}
    aria-valuemax={100}
    aria-valuenow={Math.round(progress * 10) / 10}
  ><span style={{ transform: `scaleX(${progress / 100})` }} /></div>
}

function isActive(job: StorageTransferJob) {
  return !['completed', 'failed', 'cancelled', 'canceled'].includes(job.status.toLowerCase())
}

function canCancel(job: StorageTransferJob) {
  return isActive(job) && !(job.kind === 'download' && job.status.toLowerCase() === 'running')
}

export function StoragePage() {
  const { confirm, confirmationDialog } = useConfirmDialog()
  const resource = useResource((signal) => api.storage.get(signal))
  const [sourceNodeId, setSourceNodeId] = useState('')
  const [modelId, setModelId] = useState('')
  const [targetNodeIds, setTargetNodeIds] = useState<string[]>([])
  const [busy, setBusy] = useState<string>()
  const [error, setError] = useState<string>()
  const [notice, setNotice] = useState<string>()
  const [queuedJobs, setQueuedJobs] = useState<StorageTransferJob[]>([])
  const [draggedModel, setDraggedModel] = useState<DraggedModel>()
  const [dropTargetId, setDropTargetId] = useState<string>()
  const draggedModelRef = useRef<DraggedModel | undefined>(undefined)

  const nodes = useMemo(() => (resource.data?.nodes ?? []).map((node) => {
    return {
      ...node,
      models: [...node.models].sort(compareModels),
    }
  }), [resource.data?.nodes])
  // Nodes hidden from the dashboard are hidden from every storage view too.
  const visibleNodes = useMemo(
    () => nodes.filter((node) => node.hidden_from_dashboard !== true),
    [nodes],
  )
  const sourceNode = visibleNodes.find((node) => node.id === sourceNodeId)
  const sourceModels = sourceNode?.models.filter(isTransferableModel) ?? []
  const inventory = useMemo(() => {
    const models = new Map<string, { model: StorageModel; nodes: Map<string, StorageModel> }>()
    visibleNodes.forEach((node) => node.models.forEach((model) => {
      const current = models.get(model.model_id)
      if (current) {
        current.nodes.set(node.id, model)
        if (current.model.partial && !model.partial) current.model = model
      } else models.set(model.model_id, { model, nodes: new Map([[node.id, model]]) })
    }))
    return [...models.values()].sort((left, right) => compareModels(left.model, right.model))
  }, [visibleNodes])
  const transferJobs = useMemo(() => {
    const serverJobs = resource.data?.jobs ?? []
    const knownJobIds = new Set(serverJobs.map((job) => job.id))
    return [...serverJobs, ...queuedJobs.filter((job) => !knownJobIds.has(job.id))]
  }, [queuedJobs, resource.data?.jobs])
  const visibleNodeIds = useMemo(() => new Set(visibleNodes.map((node) => node.id)), [visibleNodes])
  const visibleTransferJobs = useMemo(
    () => transferJobs.filter((job) =>
      (job.kind === 'download' || !job.source_node_id || visibleNodeIds.has(job.source_node_id))
      && visibleNodeIds.has(job.target_node_id),
    ),
    [transferJobs, visibleNodeIds],
  )
  const visibleRecentJobs = useMemo(() => [...visibleTransferJobs].sort(compareJobsNewestFirst).slice(0, 5), [visibleTransferJobs])
  const activeJobsByNode = useMemo(() => {
    const jobsByNode = new Map<string, Map<string, StorageTransferJob>>()
    const activeJobs = [...visibleTransferJobs].filter(isActive).sort(compareJobsNewestFirst)
    activeJobs.forEach((job) => {
      const nodeJobs = jobsByNode.get(job.target_node_id) ?? new Map<string, StorageTransferJob>()
      if (!nodeJobs.has(job.model_id)) nodeJobs.set(job.model_id, job)
      jobsByNode.set(job.target_node_id, nodeJobs)
    })
    return jobsByNode
  }, [visibleTransferJobs])

  useEffect(() => {
    if (!visibleNodes.length) return
    const selected = visibleNodes.find((node) => node.id === sourceNodeId && node.online && node.models.some(isTransferableModel))
      ?? visibleNodes.find((node) => node.online && node.models.some(isTransferableModel))
    const completeModels = selected?.models.filter(isTransferableModel) ?? []
    const nextSourceId = selected?.id ?? ''
    const nextModelId = completeModels.some((model) => model.model_id === modelId)
      ? modelId
      : completeModels[0]?.model_id ?? ''
    setSourceNodeId(nextSourceId)
    setModelId(nextModelId)
    setTargetNodeIds((current) => current.filter((id) => id !== nextSourceId && visibleNodes.some((node) => node.id === id && node.online && !node.models.some((model) => model.model_id === nextModelId))))
  }, [modelId, visibleNodes, sourceNodeId])

  const hasActiveJobs = Boolean(resource.data?.enabled && visibleTransferJobs.some(isActive))
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
      const result = await api.storage.transfer({
        model_id: transferModelId,
        source_node_id: 'id' in source ? source.id : source.sourceNodeId,
        target_node_ids: targets,
      })
      setQueuedJobs((current) => {
        const currentIds = new Set(current.map((job) => job.id))
        return [...current, ...(result.jobs ?? []).filter((job) => !currentIds.has(job.id))]
      })
      const targetNames = targets.map((id) => visibleNodes.find((node) => node.id === id)?.name ?? id).join(', ')
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
    if (sourceNode?.models.some((model) => model.model_id === modelId && isTransferableModel(model)) && targetNodeIds.length) {
      void queueTransfer(modelId, sourceNode, targetNodeIds)
    }
  }

  const startDrag = (event: DragEvent, model: StorageModel, node: StorageNode) => {
    if (!isTransferableModel(model)) return
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
    if (model.deletable === false) return
    if (!await confirm({
      title: `Delete ${model.model_id}?`,
      message: `Delete these weights from ${node.name}? Copies on other nodes are not affected.`,
      confirmLabel: 'Delete weights',
      danger: true,
    })) return
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

  const finishDownload = async (node: StorageNode, model: StorageModel) => {
    if (!isPartialModel(model) || !node.online) return
    if (!await confirm({
      title: `Finish downloading ${model.model_id}?`,
      message: `Resume the Hugging Face download on ${node.name}? SparkDeck will reuse the partial cache and remove the warning when the download completes.`,
      confirmLabel: 'Finish download',
    })) return
    const busyKey = `download:${node.id}:${model.model_id}`
    setBusy(busyKey)
    setError(undefined)
    setNotice(undefined)
    try {
      const result = await api.storage.finishDownload(node.id, model.model_id, model.revision)
      setQueuedJobs((current) => {
        const currentIds = new Set(current.map((job) => job.id))
        return [...current, ...(result.jobs ?? []).filter((job) => !currentIds.has(job.id))]
      })
      setNotice(`Queued ${model.model_id} to finish downloading on ${node.name}.`)
      resource.reload()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not finish model download')
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
        {visibleNodes.length === 0 ? <EmptyState title="No storage nodes" description="Join a node to the cluster before transferring model weights." /> : <div className="storage-node-grid">
          {visibleNodes.map((node) => {
            const activeJobs = activeJobsByNode.get(node.id) ?? new Map<string, StorageTransferJob>()
            const modelsById = new Map(node.models.map((model) => [model.model_id, model]))
            const weightRows = [
              ...[...activeJobs.values()].sort(compareJobsNewestFirst).map((job) => job.model_id),
              ...node.models.filter((model) => !activeJobs.has(model.model_id)).sort(compareModels).map((model) => model.model_id),
            ]
            const used = node.models.reduce(
              (total, model) => total + (model.externally_managed ? 0 : model.size_bytes),
              0,
            )
            const comfyUiUsed = node.models.reduce(
              (total, model) => total + (model.externally_managed ? model.size_bytes : 0),
              0,
            )
            // The capacity bar describes the Hugging Face cache mount, so its
            // used and free values exclude weights found in ComfyUI storage.
            // The cache free reading is reported with the inventory and is
            // only meaningful while the node's inventory is valid —
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
              <div className="storage-node-heading"><HardDrive size={18} /><div><h3 title={node.name}>{node.name}</h3><Status status={node.online ? 'running' : 'offline'}>{node.online ? 'Online' : 'Offline'}</Status></div></div>
              <div className="storage-capacity">
                <span>{formatBytes(used)} used</span>
                {hasFree && <span>{formatBytes(cacheFree)} free</span>}
                <span>{formatBytes(capacity)} total</span>
                {comfyUiUsed > 0 && <span>{formatBytes(comfyUiUsed)} in ComfyUI</span>}
              </div>
              <div className="storage-capacity-track" aria-label={`${node.name} used model storage`}><span style={{ width: `${capacity > 0 ? Math.min(100, (used / capacity) * 100) : 0}%` }} /></div>
              <p className="storage-drop-hint">{dropTargetId === node.id ? `Drop to copy ${draggedModel?.modelId}` : alreadyStored ? 'This model is already available here' : node.online ? 'Drop model weights here to queue a copy' : 'Node must be online to receive transfers'}</p>
              {weightRows.length === 0 ? <p className="storage-node-empty">No model weights reported</p> : <ul className="storage-weight-list">
                {weightRows.map((modelId) => {
                  const job = activeJobs.get(modelId)
                  const model = modelsById.get(modelId)
                  if (!job && model) return <li
                    key={`${node.id}:${model.model_id}`}
                    draggable={node.online && isTransferableModel(model)}
                    aria-label={model.partial
                      ? `Partial cache ${model.model_id} on ${node.name}`
                      : model.externally_managed
                        ? `Installed weights ${model.model_id} on ${node.name}`
                        : `Transfer ${model.model_id} from ${node.name}`}
                    onDragStart={(event) => startDrag(event, model, node)}
                    onDragEnd={() => {
                      draggedModelRef.current = undefined
                      setDraggedModel(undefined)
                      setDropTargetId(undefined)
                    }}
                  >
                    {isPartialModel(model) ? <button
                      type="button"
                      className="storage-partial-action"
                      aria-label={`Finish download of ${model.model_id} on ${node.name}`}
                      title={`Finish download on ${node.name}`}
                      disabled={!node.online || busy === `download:${node.id}:${model.model_id}` || busy === `delete:${node.id}:${model.model_id}`}
                      onClick={() => void finishDownload(node, model)}
                    ><AlertTriangle className="storage-partial-icon" size={15} aria-hidden="true" /></button> : <GripVertical size={15} aria-hidden="true" />}
                    <div><strong title={model.model_id}>{model.model_id}</strong><small>{model.partial && model.expected_size_bytes ? `${formatBytes(model.size_bytes)} of ${formatBytes(model.expected_size_bytes)}` : formatBytes(model.size_bytes)}{model.revision && !model.externally_managed ? ` · ${model.revision}` : ''}{model.partial ? ' · Partial' : ''}</small></div>
                    {model.deletable !== false && <Button
                      variant="tertiary"
                      aria-label={`Delete ${model.model_id} from ${node.name}`}
                      title={`Delete from ${node.name}`}
                      disabled={!node.online || busy === `delete:${node.id}:${model.model_id}` || busy === `download:${node.id}:${model.model_id}`}
                      onClick={() => void removeModel(node, model)}
                    ><Trash2 size={15} /></Button>}
                  </li>
                  if (!job) return null
                  const progress = jobProgress(job)
                  const transferRate = formatTransferRate(job)
                  const downloading = job.kind === 'download'
                  const running = job.status.toLowerCase() === 'running'
                  const activity = downloading
                    ? running ? 'Downloading from Hugging Face' : 'Download queued'
                    : running ? `Transferring from ${job.source_node_name}` : 'Transfer queued'
                  return <li
                    className="storage-active-weight"
                    key={`job:${job.id}`}
                    aria-label={`${activity} ${job.model_id} on ${node.name}`}
                    style={{ '--storage-active-progress': `${progress}%` } as CSSProperties}
                  >
                    {downloading ? <DownloadCloud size={15} aria-hidden="true" /> : <UploadCloud size={15} aria-hidden="true" />}
                    <div><strong>{job.model_id}</strong><small>{activity}{job.bytes_total > 0 ? ` · ${formatBytes(job.bytes_total)}` : ''}</small><SmoothProgress value={progress} label={`${activity} ${job.model_id} progress`} /><small>{formatProgress(progress)}% · {formatBytes(job.bytes_transferred)} of {formatBytes(job.bytes_total)}{transferRate ? ` · ${transferRate}` : ''}</small></div>
                    {canCancel(job) && <Button variant="tertiary" aria-label={`Cancel ${job.model_id} ${job.kind ?? 'transfer'}`} disabled={busy === job.id} onClick={() => void cancel(job)}>Cancel</Button>}
                  </li>
                })}
              </ul>}
            </Panel>
          })}
        </div>}

        <Panel className="storage-transfer-panel">
          <div><h2>Queue a transfer</h2><p>This form is the keyboard and touch-friendly alternative to drag and drop. Choose one source and one or more online targets.</p></div>
          <form onSubmit={submitTransfer} aria-label="Queue model transfer">
            <div className="field-grid">
              <label className="field"><span>Source node</span><select value={sourceNodeId} onChange={(event) => {
                const next = visibleNodes.find((node) => node.id === event.target.value)
                setSourceNodeId(event.target.value)
                setModelId(next?.models.find(isTransferableModel)?.model_id ?? '')
                setTargetNodeIds((current) => current.filter((id) => id !== event.target.value))
              }}><option value="">Select a source</option>{visibleNodes.map((node) => {
                const completeCount = node.models.filter(isTransferableModel).length
                return <option key={node.id} value={node.id} disabled={!node.online || completeCount === 0}>{node.name}{!node.online ? ' (offline)' : completeCount === 0 ? ' (no transferable models)' : ''}</option>
              })}</select></label>
              <label className="field"><span>Model weights</span><select value={modelId} onChange={(event) => {
                const nextModelId = event.target.value
                setModelId(nextModelId)
                setTargetNodeIds((current) => current.filter((id) => !visibleNodes.find((node) => node.id === id)?.models.some((model) => model.model_id === nextModelId)))
              }} disabled={!sourceNode}><option value="">Select a model</option>{sourceModels.map((model) => <option key={model.model_id} value={model.model_id}>{model.model_id}</option>)}</select></label>
            </div>
            <fieldset className="storage-targets"><legend>Target nodes</legend><p>Select every online node that should receive a copy.</p><div>
              {visibleNodes.filter((node) => node.id !== sourceNodeId).map((node) => {
                const alreadyAvailable = Boolean(modelId && node.models.some((model) => model.model_id === modelId))
                const unavailable = !node.online || alreadyAvailable
                return <label className={`check-field ${unavailable ? 'unavailable' : ''}`} key={node.id}><input type="checkbox" checked={targetNodeIds.includes(node.id)} disabled={unavailable || busy === 'transfer'} onChange={(event) => setTargetNodeIds((current) => event.target.checked ? [...current, node.id] : current.filter((id) => id !== node.id))} /><span><strong>{node.name}</strong><small>{!node.online ? 'Offline' : alreadyAvailable ? 'Model already available' : `${node.models.length} models available`}</small></span></label>
              })}
              {visibleNodes.filter((node) => node.id !== sourceNodeId).length === 0 && <p className="storage-target-empty">No other nodes are available.</p>}
            </div></fieldset>
            <div className="storage-transfer-actions"><span>{targetNodeIds.length ? `${targetNodeIds.length} target${targetNodeIds.length === 1 ? '' : 's'} selected` : 'Select at least one target'}</span><Button type="submit" variant="primary" disabled={busy === 'transfer' || !modelId || !sourceNodeId || targetNodeIds.length === 0}><UploadCloud size={16} /> {busy === 'transfer' ? 'Queueing…' : 'Queue transfer'}</Button></div>
          </form>
        </Panel>

        <div className="section-heading"><div><h2>Transfer queue</h2><p>Queued, active, completed, and failed model copies.</p></div></div>
        {visibleTransferJobs.length === 0 ? <EmptyState title="No transfers yet" description="Drag a model between node cards or use the transfer form to queue a copy." /> : <Panel className="table-panel">
          <div className="responsive-table storage-job-table" role="table" aria-label="Model transfer queue">
            <div className="table-row table-header" role="row"><span role="columnheader">Model</span><span role="columnheader">Route</span><span role="columnheader">Status</span><span role="columnheader">Progress</span><span role="columnheader">Actions</span></div>
            {visibleRecentJobs.map((job) => {
              const progress = jobProgress(job)
              const transferRate = formatTransferRate(job)
              return <div className="table-row" role="row" key={job.id}>
                <div role="cell" data-label="Model"><strong>{job.model_id}</strong><small>Created {formatTimestamp(job.created_at)}</small></div>
                <div role="cell" data-label="Route" className="storage-route"><span>{job.source_node_name}</span><ArrowRight size={13} aria-label="to" /><span>{job.target_node_name}</span></div>
                <div role="cell" data-label="Status"><Status status={job.status} />{job.error && <small className="storage-job-error" role="alert">{job.error}</small>}</div>
                <div role="cell" data-label="Progress" className="storage-job-progress"><SmoothProgress value={progress} label={`Transfer ${job.model_id} progress`} /><span>{formatProgress(progress)}% · {formatBytes(job.bytes_transferred)} of {formatBytes(job.bytes_total)}{transferRate ? ` · ${transferRate}` : ''}</span></div>
                <div role="cell" data-label="Actions" className="row-actions">{canCancel(job) && <Button variant="tertiary" aria-label={`Cancel ${job.model_id} ${job.kind ?? 'transfer'}`} disabled={busy === job.id} onClick={() => void cancel(job)}>Cancel</Button>}</div>
              </div>
            })}
          </div>
        </Panel>}

        <div className="section-heading"><div><h2>Model inventory</h2><p>Availability matrix across every storage node.</p></div></div>
        {inventory.length === 0 ? <EmptyState title="No model weights found" description="Models downloaded on participating nodes will appear here." /> : <Panel className="table-panel">
          <div className="responsive-table storage-model-table" role="table" aria-label="Model storage inventory">
            <div className="table-row table-header" role="row"><span role="columnheader">Model</span><span role="columnheader">Size</span><span role="columnheader">Files</span><span role="columnheader">Node availability</span></div>
            {inventory.map(({ model, nodes: locations }) => <div className="table-row" role="row" key={model.model_id}>
              <div role="cell" data-label="Model"><strong title={model.model_id}>{model.model_id}</strong><small>{model.partial ? 'Partial cache' : model.externally_managed ? 'Installed' : model.revision ?? 'Default revision'} · {formatTimestamp(model.last_modified)}</small></div>
              <div role="cell" data-label="Size">{model.partial && model.expected_size_bytes ? `${formatBytes(model.size_bytes)} of ${formatBytes(model.expected_size_bytes)}` : formatBytes(model.size_bytes)}</div>
              <div role="cell" data-label="Files">{model.file_count ?? 'Not reported'}</div>
              <div role="cell" data-label="Node availability" className="storage-availability" aria-label={`Model availability for ${model.model_id}`}>{visibleNodes.map((node) => {
                const location = locations.get(node.id)
                const resumable = Boolean(location && isPartialModel(location))
                return <span key={node.id} className={resumable ? 'partial' : location ? 'available' : ''}>{resumable && location ? <button
                  type="button"
                  className="storage-availability-action"
                  aria-label={`Finish download of ${location.model_id} on ${node.name} from inventory`}
                  title={`Finish download on ${node.name}`}
                  disabled={!node.online || busy === `download:${node.id}:${location.model_id}` || busy === `delete:${node.id}:${location.model_id}`}
                  onClick={() => void finishDownload(node, location)}
                >!</button> : <span aria-hidden="true">{location ? '✓' : '—'}</span>} {node.name}</span>
              })}</div>
            </div>)}
          </div>
        </Panel>}
      </>}

      {resource.data && <Panel className="storage-guidance"><h2>How Virtual NAS works</h2><ol>{resource.data.instructions.length ? resource.data.instructions.map((instruction) => <li key={instruction}>{instruction}</li>) : <><li>Keep source and target nodes online while a transfer is active.</li><li>SparkDeck copies model weights between nodes and tracks progress in this queue.</li><li>Deleting a node copy does not remove the same model from other nodes.</li></>}</ol></Panel>}
      {confirmationDialog}
    </div>
  )
}
