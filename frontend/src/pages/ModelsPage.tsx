import { useEffect, useState, type FormEvent } from 'react'
import { Plus, Server, Trash2 } from 'lucide-react'
import { api } from '../api/client'
import type { CreateDeploymentInput, Deployment, RuntimeKind } from '../api/types'
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

export function ModelsPage() {
  const resource = useResource((signal) => api.deployments.list(signal))
  const defaults = useResource((signal) => api.settings.get(signal))
  const nodes = useResource((signal) => api.nodes.list(signal))
  const [creating, setCreating] = useState(false)
  const [form, setForm] = useState<CreateDeploymentInput>(initialForm)
  const [busy, setBusy] = useState<string>()
  const [formError, setFormError] = useState<string>()
  const [actionError, setActionError] = useState<string>()
  const [actionNotice, setActionNotice] = useState<string>()

  useEffect(() => {
    const inventory = nodes.data
    if (!inventory?.length) return
    setForm((current) => {
      const local = inventory.find((node) => node.local)
      if (!local) return { ...current, node_ids: [], deployment_mode: 'single' }
      const remotes = (current.node_ids ?? []).filter((id) => id !== local.id && inventory.some((node) => node.id === id && isNodeSelectable(node)))
      const nodeIds = [local.id, ...remotes]
      return { ...current, node_ids: nodeIds, deployment_mode: nodeIds.length > 1 ? 'replicated' : 'single' }
    })
  }, [nodes.data])

  const localNodeId = nodes.data?.find((node) => node.local)?.id
  const selectionReady = !nodes.loading && !nodes.error && Boolean(localNodeId)
    && (form.node_ids?.length ?? 0) > 0
    && (form.node_ids ?? []).every((id) => nodes.data?.some((node) => node.id === id && isNodeSelectable(node)))

  useEffect(() => {
    if (!defaults.data || creating) return
    const runtime = defaults.data.default_runtime ?? 'vllm'
    const contextLength = defaults.data.default_context_length ?? 8192
    setForm((current) => ({
      ...current,
      runtime,
      settings: runtime === 'llama.cpp'
        ? { context_length: contextLength, parallel_slots: 1, gpu_layers: 99 }
        : { context_length: contextLength, tensor_parallel_size: 1 },
    }))
  }, [creating, defaults.data])

  const create = async (event: FormEvent) => {
    event.preventDefault()
    setBusy('create')
    setFormError(undefined)
    setActionNotice(undefined)
    try {
      const deployment = await api.deployments.create(form)
      const selected = deployment.selected_nodes?.map((node) => node.name).join(', ')
        || selectedNodeLabel(nodes.data ?? [], deployment.node_ids ?? form.node_ids ?? ['local'])
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
        actions={<Button variant="primary" onClick={() => setCreating(true)}><Plus size={16} /> Add model</Button>}
      />
      {resource.loading && <LoadingState label="Loading deployments" />}
      {resource.error && <ErrorState message={resource.error} onRetry={resource.reload} />}
      {actionError && <p className="form-error" role="alert">{actionError}</p>}
      {actionNotice && <p className="inline-success" role="status">{actionNotice}</p>}
      {!resource.loading && !resource.error && resource.data?.length === 0 && (
        <EmptyState title="No model servers yet" description="Launch a managed runtime or connect an existing OpenAI-compatible endpoint." action={<Button variant="primary" onClick={() => setCreating(true)}>Add your first model</Button>} />
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
                <div role="cell" data-label="Target"><span>{deployment.selected_nodes?.map((node) => node.name).join(', ') || deployment.node_ids?.join(', ') || 'This device'}</span></div>
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
                requiredIds={localNodeId ? [localNodeId] : []}
              />}
              {form.managed && form.runtime === 'llama.cpp' && <p className="field-note">llama.cpp deployments use the local node because GGUF artifacts are local to this device.</p>}
              <div className="field-grid">
                <label className="field"><span>Context length</span><input type="number" min="256" value={form.settings.context_length} onChange={(event) => setForm({ ...form, settings: { ...form.settings, context_length: Number(event.target.value) } })} /></label>
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
