import { useState, type FormEvent } from 'react'
import { MoreHorizontal, Plus, Server } from 'lucide-react'
import { api } from '../api/client'
import type { CreateDeploymentInput, Deployment, RuntimeKind } from '../api/types'
import { Button, EmptyState, ErrorState, LoadingState, PageHeader, Panel, RuntimeMark, Status } from '../components/ui'
import { useResource } from '../hooks/useResource'

const initialForm: CreateDeploymentInput = {
  alias: '',
  model_id: '',
  runtime: 'vllm',
  managed: true,
  endpoint_url: '',
  settings: { context_length: 8192, tensor_parallel_size: 1 },
}

export function ModelsPage() {
  const resource = useResource((signal) => api.deployments.list(signal))
  const [creating, setCreating] = useState(false)
  const [form, setForm] = useState<CreateDeploymentInput>(initialForm)
  const [busy, setBusy] = useState<string>()
  const [formError, setFormError] = useState<string>()

  const create = async (event: FormEvent) => {
    event.preventDefault()
    setBusy('create')
    setFormError(undefined)
    try {
      await api.deployments.create(form)
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
    try {
      await api.deployments.action(deployment.id, action)
      resource.reload()
    } finally {
      setBusy(undefined)
    }
  }

  const updateRuntime = (runtime: RuntimeKind) => {
    setForm((current) => ({
      ...current,
      runtime,
      settings: runtime === 'llama.cpp'
        ? { context_length: 8192, parallel_slots: 1, gpu_layers: 99 }
        : { context_length: 8192, tensor_parallel_size: 1 },
    }))
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
      {!resource.loading && !resource.error && resource.data?.length === 0 && (
        <EmptyState title="No model servers yet" description="Launch a managed runtime or connect an existing OpenAI-compatible endpoint." action={<Button variant="primary" onClick={() => setCreating(true)}>Add your first model</Button>} />
      )}
      {resource.data && resource.data.length > 0 && (
        <Panel className="table-panel">
          <div className="responsive-table deployments-table" role="table" aria-label="Model deployments">
            <div className="table-row table-header" role="row">
              <span role="columnheader">Model</span><span role="columnheader">Runtime</span><span role="columnheader">Configuration</span><span role="columnheader">Status</span><span role="columnheader">Actions</span>
            </div>
            {resource.data.map((deployment) => (
              <div className="table-row" role="row" key={deployment.id} tabIndex={0}>
                <div role="cell" data-label="Model"><strong>{deployment.alias}</strong><small>{deployment.model_id}</small></div>
                <div role="cell" data-label="Runtime"><RuntimeMark runtime={deployment.runtime} /><small>{deployment.runtime_version ?? (deployment.managed ? 'Managed' : 'External')}</small></div>
                <div role="cell" data-label="Configuration"><span>{deployment.settings.context_length?.toLocaleString() ?? '—'} ctx</span><small>{deployment.settings.quantization ?? 'Default precision'}</small></div>
                <div role="cell" data-label="Status"><Status status={deployment.status} /></div>
                <div role="cell" data-label="Actions" className="row-actions">
                  {deployment.status === 'running'
                    ? <Button variant="tertiary" disabled={busy === deployment.id} onClick={() => void act(deployment, 'stop')}>Stop</Button>
                    : <Button variant="tertiary" disabled={busy === deployment.id} onClick={() => void act(deployment, 'start')}>Start</Button>}
                  <Button variant="tertiary" aria-label={`More actions for ${deployment.alias}`}><MoreHorizontal size={17} /></Button>
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
              <div className="field-grid">
                <label className="field"><span>Context length</span><input type="number" min="256" value={form.settings.context_length} onChange={(event) => setForm({ ...form, settings: { ...form.settings, context_length: Number(event.target.value) } })} /></label>
                {form.runtime === 'llama.cpp' ? (
                  <label className="field"><span>Parallel slots</span><input type="number" min="1" value={form.settings.parallel_slots} onChange={(event) => setForm({ ...form, settings: { ...form.settings, parallel_slots: Number(event.target.value) } })} /></label>
                ) : (
                  <label className="field"><span>Tensor parallel size</span><input type="number" min="1" value={form.settings.tensor_parallel_size} onChange={(event) => setForm({ ...form, settings: { ...form.settings, tensor_parallel_size: Number(event.target.value) } })} /></label>
                )}
              </div>
              <div className="modal-actions"><Button type="button" onClick={() => setCreating(false)}>Cancel</Button><Button type="submit" variant="primary" disabled={busy === 'create'}>{busy === 'create' ? 'Adding…' : <><Server size={16} /> Add model</>}</Button></div>
            </form>
          </section>
        </div>
      )}
    </div>
  )
}
