import { useEffect, useRef, useState, type FormEvent } from 'react'
import { Plus, Server, Trash2 } from 'lucide-react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import type { AppSettings, CreateDeploymentInput, Deployment, RuntimeKind } from '../api/types'
import { Button, EmptyState, ErrorState, LoadingState, PageHeader, Panel, RuntimeMark, Status } from '../components/ui'
import { isNodeSelectable, NodeSelector, selectedNodeLabel } from '../components/NodeSelector'
import { useResource } from '../hooks/useResource'

const initialForm: CreateDeploymentInput = {
  alias: '',
  model_id: '',
  runtime: 'vllm',
  managed: true,
  endpoint_url: '',
  settings: { context_length: 8192, tensor_parallel_size: 1 },
  node_ids: ['local'],
  deployment_mode: 'single',
}

type DefaultFieldEdits = { runtime: boolean; contextLength: boolean }

function mergeSavedDefaults(
  current: CreateDeploymentInput,
  defaults: AppSettings,
  edits: DefaultFieldEdits,
): CreateDeploymentInput {
  const runtime = edits.runtime ? current.runtime : defaults.default_runtime ?? 'vllm'
  const contextLength = edits.contextLength
    ? current.settings.context_length ?? 8192
    : defaults.default_context_length ?? 8192
  const runtimeChanged = runtime !== current.runtime
  return {
    ...current,
    runtime,
    settings: runtimeChanged
      ? runtime === 'llama.cpp'
        ? { context_length: contextLength, parallel_slots: 1, gpu_layers: 99 }
        : { context_length: contextLength, tensor_parallel_size: 1 }
      : { ...current.settings, context_length: contextLength },
  }
}

export function ModelsPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const resource = useResource((signal) => api.deployments.list(signal))
  const defaults = useResource((signal) => api.settings.get(signal))
  const nodes = useResource((signal) => api.nodes.list(signal))
  const onboarding = useResource((signal) => api.onboarding.get(signal))
  const [creating, setCreating] = useState(false)
  const [form, setForm] = useState<CreateDeploymentInput>(initialForm)
  const [busy, setBusy] = useState<string>()
  const [formError, setFormError] = useState<string>()
  const [actionError, setActionError] = useState<string>()
  const [actionNotice, setActionNotice] = useState<string>()
  const defaultFieldEdits = useRef<DefaultFieldEdits>({ runtime: false, contextLength: false })

  useEffect(() => {
    const modelId = searchParams.get('model')?.trim()
    if (!modelId) return
    defaultFieldEdits.current = { runtime: false, contextLength: false }
    setForm((current) => {
      const prefilled = {
        ...current,
        model_id: modelId,
        alias: current.alias || modelId.split('/').at(-1) || modelId,
      }
      return defaults.data
        ? mergeSavedDefaults(prefilled, defaults.data, defaultFieldEdits.current)
        : prefilled
    })
    setCreating(true)
    setSearchParams({}, { replace: true })
  }, [defaults.data, searchParams, setSearchParams])

  useEffect(() => {
    const inventory = nodes.data
    if (!inventory?.length) return
    setForm((current) => {
      const local = inventory.find((node) => node.local)
      const available = (current.node_ids ?? []).filter((id) => inventory.some((node) => node.id === id && isNodeSelectable(node)))
      const fallback = local && isNodeSelectable(local) ? local : inventory.find(isNodeSelectable)
      const nodeIds = available.length ? available : fallback ? [fallback.id] : []
      return { ...current, node_ids: nodeIds, deployment_mode: nodeIds.length > 1 ? 'replicated' : 'single' }
    })
  }, [nodes.data])

  const localNodeId = nodes.data?.find((node) => node.local)?.id
  const selectionReady = !nodes.loading && !nodes.error && (form.node_ids?.length ?? 0) > 0
    && (form.node_ids ?? []).every((id) => nodes.data?.some((node) => node.id === id && isNodeSelectable(node)))
    && (form.runtime !== 'llama.cpp' || (form.node_ids?.length === 1 && form.node_ids[0] === localNodeId))
  const localLabel = onboarding.data?.role === 'worker' ? 'Controller' : 'This device'

  useEffect(() => {
    if (!defaults.data) return
    const savedDefaults = defaults.data
    setForm((current) => mergeSavedDefaults(current, savedDefaults, defaultFieldEdits.current))
  }, [defaults.data])

  const openCreator = () => {
    defaultFieldEdits.current = { runtime: false, contextLength: false }
    setForm((current) => defaults.data
      ? mergeSavedDefaults(current, defaults.data, defaultFieldEdits.current)
      : current)
    setCreating(true)
  }

  const create = async (event: FormEvent) => {
    event.preventDefault()
    setBusy('create')
    setFormError(undefined)
    setActionNotice(undefined)
    try {
      const deployment = await api.deployments.create(form)
      const selected = deployment.selected_nodes?.map((node) => node.id === 'local' ? localLabel : node.name).join(', ')
        || selectedNodeLabel(nodes.data ?? [], deployment.node_ids ?? form.node_ids ?? ['local'], localLabel)
      setActionNotice(`Added ${deployment.alias} on ${selected}.`)
      setCreating(false)
      setForm(initialForm)
      resource.reload()
    } catch (reason) {
      setFormError(reason instanceof Error ? reason.message : 'Could not create deployment')
    } finally {
      setBusy(undefined)
    }
  }

  const act = async (deployment: Deployment, action: 'start' | 'stop' | 'remove') => {
    setBusy(deployment.id)
    setActionError(undefined)
    try {
      await api.deployments.action(deployment.id, action)
      resource.reload()
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : 'Could not update deployment')
    } finally {
      setBusy(undefined)
    }
  }

  const updateRuntime = (runtime: RuntimeKind) => {
    defaultFieldEdits.current.runtime = true
    setForm((current) => {
      const contextLength = current.settings.context_length ?? defaults.data?.default_context_length ?? 8192
      const localId = localNodeId ?? 'local'
      const nodeIds = runtime === 'llama.cpp' ? [localId] : current.node_ids
      return {
        ...current,
        runtime,
        node_ids: nodeIds,
        deployment_mode: (nodeIds?.length ?? 0) > 1 ? 'replicated' : 'single',
        settings: runtime === 'llama.cpp'
          ? { context_length: contextLength, parallel_slots: 1, gpu_layers: 99 }
          : { context_length: contextLength, tensor_parallel_size: 1 },
      }
    })
  }

  return (
    <div className="page">
      <PageHeader
        eyebrow="Local runtimes"
        title="Models"
        description="Manage model servers across vLLM, llama.cpp, and SGLang from one place."
        actions={<Button variant="primary" onClick={openCreator}><Plus size={16} /> Add model</Button>}
      />
      {resource.loading && <LoadingState label="Loading deployments" />}
      {resource.error && <ErrorState message={resource.error} onRetry={resource.reload} />}
      {actionError && <p className="form-error" role="alert">{actionError}</p>}
      {actionNotice && <p className="inline-success" role="status">{actionNotice}</p>}
      {!resource.loading && !resource.error && resource.data?.length === 0 && (
        <EmptyState title="No model servers yet" description="Launch a managed runtime or connect an existing OpenAI-compatible endpoint." action={<Button variant="primary" onClick={openCreator}>Add your first model</Button>} />
      )}
      {resource.data && resource.data.length > 0 && (
        <Panel className="table-panel">
          <div className="responsive-table deployments-table" role="table" aria-label="Model deployments">
            <div className="table-row table-header" role="row">
              <span role="columnheader">Model</span><span role="columnheader">Runtime</span><span role="columnheader">Configuration</span><span role="columnheader">Target</span><span role="columnheader">Status</span><span role="columnheader">Actions</span>
            </div>
            {resource.data.map((deployment) => (
              <div className="table-row" role="row" key={deployment.id} tabIndex={0}>
                <div role="cell" data-label="Model"><strong>{deployment.alias}</strong><small>{deployment.model_id}</small></div>
                <div role="cell" data-label="Runtime"><RuntimeMark runtime={deployment.runtime} /><small>{deployment.runtime_version ?? (deployment.managed ? 'Managed' : 'External')}</small></div>
                <div role="cell" data-label="Configuration"><span>{deployment.settings.context_length?.toLocaleString() ?? '—'} ctx</span><small>{deployment.settings.quantization ?? 'Default precision'}</small></div>
                <div role="cell" data-label="Target"><span>{deployment.selected_nodes?.map((node, index) => `${node.id === 'local' ? localLabel : node.name}${deployment.selected_nodes!.length > 1 && index === 0 ? ' (primary)' : ''}`).join(', ') || deployment.node_ids?.map((id, index) => `${id === 'local' ? localLabel : id}${deployment.node_ids!.length > 1 && index === 0 ? ' (primary)' : ''}`).join(', ') || localLabel}</span></div>
                <div role="cell" data-label="Status"><Status status={deployment.status} /></div>
                <div role="cell" data-label="Actions" className="row-actions">
                  {deployment.managed && (deployment.status === 'running'
                    ? <Button variant="tertiary" disabled={busy === deployment.id} onClick={() => void act(deployment, 'stop')}>Stop</Button>
                    : <Button variant="tertiary" disabled={busy === deployment.id} onClick={() => void act(deployment, 'start')}>Start</Button>)}
                  <Button variant="tertiary" disabled={busy === deployment.id} aria-label={`Remove ${deployment.alias}`} onClick={() => {
                    if (window.confirm(`Remove ${deployment.alias} from SparkDeck?`)) void act(deployment, 'remove')
                  }}><Trash2 size={17} /></Button>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      )}

      {creating && (
        <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && setCreating(false)}>
          <section className="modal" role="dialog" aria-modal="true" aria-labelledby="create-deployment-title">
            <div className="modal-heading"><div><p className="eyebrow">New deployment</p><h2 id="create-deployment-title">Add a model server</h2></div><button className="icon-button" onClick={() => setCreating(false)} aria-label="Close dialog">×</button></div>
            <form onSubmit={(event) => void create(event)}>
              {formError && <p className="form-error" role="alert">{formError}</p>}
              <div className="field-grid">
                <label className="field"><span>Display name</span><input autoFocus required value={form.alias} onChange={(event) => setForm({ ...form, alias: event.target.value })} /></label>
                <label className="field"><span>Runtime</span><select value={form.runtime} onChange={(event) => updateRuntime(event.target.value as RuntimeKind)}><option value="vllm">vLLM</option><option value="llama.cpp">llama.cpp</option><option value="sglang">SGLang</option></select></label>
              </div>
              <label className="field"><span>Model repository or GGUF artifact</span><input required value={form.model_id} onChange={(event) => setForm({ ...form, model_id: event.target.value })} placeholder="org/model-name" /></label>
              <label className="check-field"><input type="checkbox" checked={!form.managed} onChange={(event) => setForm({ ...form, managed: !event.target.checked })} /><span><strong>Connect an existing endpoint</strong><small>SparkDeck will not manage its process or container.</small></span></label>
              {!form.managed && <label className="field"><span>Endpoint URL</span><input type="url" required value={form.endpoint_url} onChange={(event) => setForm({ ...form, endpoint_url: event.target.value })} placeholder="http://127.0.0.1:8001" /></label>}
              {!form.managed && <label className="field"><span>API key (optional)</span><input type="password" autoComplete="off" value={form.api_key ?? ''} onChange={(event) => setForm({ ...form, api_key: event.target.value })} /><small>Stored in your operating system credential store, never in SparkDeck's database.</small></label>}
              {form.managed && <NodeSelector
                nodes={nodes.data ?? []}
                selectedIds={form.node_ids ?? []}
                onChange={(nodeIds) => setForm({ ...form, node_ids: nodeIds, deployment_mode: nodeIds.length > 1 ? 'replicated' : 'single' })}
                loading={nodes.loading}
                error={nodes.error}
                onRetry={nodes.reload}
                multiple={form.runtime !== 'llama.cpp'}
                disabled={busy === 'create'}
                requiredIds={form.runtime === 'llama.cpp' && localNodeId ? [localNodeId] : []}
                allowedIds={form.runtime === 'llama.cpp' && localNodeId ? [localNodeId] : undefined}
                localLabel={localLabel}
                primaryId={(form.node_ids?.length ?? 0) > 1 ? form.node_ids?.[0] : undefined}
              />}
              {form.managed && form.runtime === 'llama.cpp' && <p className="field-note">llama.cpp deployments use the local node because GGUF artifacts are local to this device.</p>}
              <div className="field-grid">
                <label className="field"><span>Context length</span><input type="number" min="256" value={form.settings.context_length} onChange={(event) => {
                  defaultFieldEdits.current.contextLength = true
                  setForm({ ...form, settings: { ...form.settings, context_length: Number(event.target.value) } })
                }} /></label>
                {form.runtime === 'llama.cpp' ? (
                  <label className="field"><span>Parallel slots</span><input type="number" min="1" value={form.settings.parallel_slots} onChange={(event) => setForm({ ...form, settings: { ...form.settings, parallel_slots: Number(event.target.value) } })} /></label>
                ) : (
                  <label className="field"><span>Tensor parallel size</span><input type="number" min="1" value={form.settings.tensor_parallel_size} onChange={(event) => setForm({ ...form, settings: { ...form.settings, tensor_parallel_size: Number(event.target.value) } })} /></label>
                )}
              </div>
              <div className="modal-actions"><Button type="button" onClick={() => setCreating(false)}>Cancel</Button><Button type="submit" variant="primary" disabled={busy === 'create' || (form.managed && !selectionReady)}>{busy === 'create' ? 'Adding…' : <><Server size={16} /> Add to {form.node_ids?.length ?? 1} {(form.node_ids?.length ?? 1) === 1 ? 'node' : 'nodes'}</>}</Button></div>
            </form>
          </section>
        </div>
      )}
    </div>
  )
}
