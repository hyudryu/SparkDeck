import { useEffect, useState, type FormEvent } from 'react'
import { Bookmark, HardDrive, Play, Plus, Server, Trash2 } from 'lucide-react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import type { CreateDeploymentInput, Deployment, RuntimeKind, SavedConfiguration } from '../api/types'
import { Button, EmptyState, ErrorState, LoadingState, PageHeader, Panel, RuntimeMark, Status } from '../components/ui'
import { isNodeSelectable, NodeSelector, selectedNodeLabel } from '../components/NodeSelector'
import { useResource } from '../hooks/useResource'
import { formatBytes } from '../utils/format'

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

const isLocalModelPath = (model: string) => model.startsWith('/') || model.startsWith('~')

export function ModelsPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const resource = useResource((signal) => api.deployments.list(signal))
  const nodes = useResource((signal) => api.nodes.list(signal))
  const onboarding = useResource((signal) => api.onboarding.get(signal))
  const modelCache = useResource((signal) => api.modelCache.get(signal))
  const recipes = useResource((signal) => api.recipes.list(signal))
  const [creating, setCreating] = useState(false)
  const [form, setForm] = useState<CreateDeploymentInput>(initialForm)
  const [busy, setBusy] = useState<string>()
  const [formError, setFormError] = useState<string>()
  const [actionError, setActionError] = useState<string>()
  const [actionNotice, setActionNotice] = useState<string>()
  const [recipeDeployment, setRecipeDeployment] = useState<{ recipe: SavedConfiguration; nodeIds: string[] }>()
  const [recipeError, setRecipeError] = useState<string>()

  useEffect(() => {
    const modelId = searchParams.get('model')?.trim()
    if (!modelId) return
    setForm((current) => ({
      ...current,
      model_id: modelId,
      alias: current.alias || modelId.split('/').at(-1) || modelId,
    }))
    setCreating(true)
    setSearchParams({}, { replace: true })
  }, [searchParams, setSearchParams])

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

  const openCreator = () => {
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

  const nodesWithWeights = (recipe: SavedConfiguration) => {
    if (isLocalModelPath(recipe.model)) {
      return new Set(localNodeId ? [localNodeId] : [])
    }
    return new Set((modelCache.data?.nodes ?? [])
      .filter((node) => node.models.some((model) => model.model_id === recipe.model
        && model.revisions?.includes(recipe.model_revision ?? 'main')))
      .map((node) => node.id))
  }

  const openRecipeDeployment = (recipe: SavedConfiguration) => {
    const weighted = nodesWithWeights(recipe)
    const eligible = (nodes.data ?? []).filter((node) => weighted.has(node.id) && isNodeSelectable(node))
    let preferred = [...new Set(recipe.node_ids)]
      .filter((id) => eligible.some((node) => node.id === id))
    if (recipe.deployment_mode === 'sharded' && localNodeId && eligible.some((node) => node.id === localNodeId)) {
      preferred = [localNodeId, ...preferred.filter((id) => id !== localNodeId)]
    }
    const nodeIds = [...preferred, ...eligible.map((node) => node.id).filter((id) => !preferred.includes(id))]
      .slice(0, recipe.required_node_count)
    setRecipeError(undefined)
    setRecipeDeployment({ recipe, nodeIds })
  }

  const deployRecipe = async () => {
    if (!recipeDeployment) return
    const { recipe, nodeIds } = recipeDeployment
    setBusy(`recipe:${recipe.id}`)
    setActionError(undefined)
    setActionNotice(undefined)
    setRecipeError(undefined)
    try {
      const deployment = await api.recipes.deploy(recipe.id, nodeIds)
      const selected = selectedNodeLabel(nodes.data ?? [], nodeIds, localLabel)
      setActionNotice(`Deployed saved configuration ${deployment.alias} on ${selected}.`)
      setRecipeDeployment(undefined)
      resource.reload()
    } catch (reason) {
      setRecipeError(reason instanceof Error ? reason.message : 'Could not deploy saved configuration')
    } finally {
      setBusy(undefined)
    }
  }

  const modelStorage = (deployment: Deployment) => {
    const locations = (modelCache.data?.nodes ?? []).flatMap((node) =>
      node.models
        .filter((model) => model.model_id === deployment.model_id)
        .map((model) => ({ ...model, nodeName: node.name })),
    )
    if (!locations.length) return 'Disk size unavailable'
    const total = locations.reduce((sum, location) => sum + location.size_bytes, 0)
    if (locations.length === 1) return `${formatBytes(total)} on ${locations[0].nodeName}`
    const perCopy = locations.every((location) => location.size_bytes === locations[0].size_bytes)
      ? `${formatBytes(locations[0].size_bytes)} each · `
      : ''
    return `${perCopy}${formatBytes(total)} total on ${locations.length} nodes`
  }

  const updateRuntime = (runtime: RuntimeKind) => {
    setForm((current) => {
      const contextLength = current.settings.context_length ?? 8192
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
      {recipes.error && <ErrorState message={`Saved configurations: ${recipes.error}`} onRetry={recipes.reload} />}
      {actionError && <p className="form-error" role="alert">{actionError}</p>}
      {actionNotice && <p className="inline-success" role="status">{actionNotice}</p>}
      {recipes.data && recipes.data.length > 0 && <section className="saved-configurations" aria-labelledby="saved-configurations-title">
        <div className="section-heading"><div><h2 id="saved-configurations-title">Saved cluster configurations</h2><p>Existing saved recipes are preserved and deploy through SparkDeck.</p></div></div>
        <div className="saved-configuration-grid">
          {recipes.data.map((recipe) => {
            const targets = (recipe.node_ids?.length ? recipe.node_ids : ['local']).map((id) => nodes.data?.find((node) => node.id === id))
            const targetNames = targets.map((node, index) => node?.name ?? recipe.node_ids?.[index] ?? 'This device')
            const unavailable = targets.some((node) => !node || !isNodeSelectable(node))
            const disabled = recipe.supported === false
            return <Panel className="saved-configuration-card" key={recipe.id}>
              <div className="saved-configuration-heading"><span className="panel-icon"><Bookmark size={17} /></span><div><h3>{recipe.name || recipe.model}</h3><p>{recipe.model}</p></div></div>
              <dl>
                <div><dt>Runtime</dt><dd><RuntimeMark runtime={recipe.engine || 'vllm'} /></dd></div>
                <div><dt>Layout</dt><dd>{recipe.tensor_parallel_size > 1 ? `TP${recipe.tensor_parallel_size} · ` : ''}{recipe.deployment_mode || 'single'} · {recipe.required_node_count} {recipe.required_node_count === 1 ? 'node' : 'nodes'}</dd></div>
                <div><dt>Targets</dt><dd>{targetNames.join(', ')}</dd></div>
                <div><dt>Arguments</dt><dd>{recipe.extra_args_count ?? 0} saved</dd></div>
              </dl>
              {(recipe.error || unavailable) && <p className="saved-configuration-warning">{recipe.error || 'One or more saved nodes are missing, offline, or not ready.'}</p>}
              <Button variant="primary" disabled={disabled || busy === `recipe:${recipe.id}`} onClick={() => openRecipeDeployment(recipe)}><Play size={15} /> Choose nodes &amp; deploy</Button>
            </Panel>
          })}
        </div>
      </section>}
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
                <div role="cell" data-label="Model"><strong>{deployment.alias}</strong><small>{deployment.model_id}</small><small className="model-disk-usage"><HardDrive size={12} /> {modelStorage(deployment)}</small></div>
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

      {recipeDeployment && (() => {
        const { recipe, nodeIds } = recipeDeployment
        const localPath = isLocalModelPath(recipe.model)
        const weighted = nodesWithWeights(recipe)
        const allowedIds = (nodes.data ?? []).filter((node) => weighted.has(node.id)).map((node) => node.id)
        const unavailableReasons = Object.fromEntries((nodes.data ?? []).filter((node) => !weighted.has(node.id)).map((node) => [node.id, localPath ? 'Local paths are available only on the controller' : 'Model weights not cached']))
        const localRequired = recipe.deployment_mode === 'sharded' && localNodeId && allowedIds.includes(localNodeId) ? [localNodeId] : []
        const exactCount = nodeIds.length === recipe.required_node_count
        const allEligible = nodeIds.every((id) => allowedIds.includes(id) && nodes.data?.some((node) => node.id === id && isNodeSelectable(node)))
        const coordinatorReady = recipe.deployment_mode !== 'sharded' || Boolean(localNodeId && nodeIds.includes(localNodeId))
        const ready = !nodes.loading && !nodes.error && (localPath || (!modelCache.loading && !modelCache.error)) && exactCount && allEligible && coordinatorReady
        const recipeBusy = busy === `recipe:${recipe.id}`
        return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && !recipeBusy && setRecipeDeployment(undefined)}>
          <section className="modal" role="dialog" aria-modal="true" aria-labelledby="deploy-saved-configuration-title">
            <div className="modal-heading"><div><p className="eyebrow">Saved cluster configuration</p><h2 id="deploy-saved-configuration-title">Deploy {recipe.name || recipe.model}</h2></div><button className="icon-button" disabled={recipeBusy} onClick={() => setRecipeDeployment(undefined)} aria-label="Close dialog">×</button></div>
            <p className="modal-description">{recipe.tensor_parallel_size > 1 ? `TP${recipe.tensor_parallel_size} requires exactly ${recipe.required_node_count} nodes.` : `Select exactly ${recipe.required_node_count} ${recipe.required_node_count === 1 ? 'node' : 'nodes'}.`} Nodes without the complete model weights are disabled.</p>
            {recipeError && <p className="form-error" role="alert">{recipeError}</p>}
            {!localPath && modelCache.error && <ErrorState message={`Model weights: ${modelCache.error}`} onRetry={modelCache.reload} />}
            <NodeSelector
              nodes={nodes.data ?? []}
              selectedIds={nodeIds}
              onChange={(next) => setRecipeDeployment({ recipe, nodeIds: next.length <= recipe.required_node_count ? next : nodeIds })}
              loading={nodes.loading || (!localPath && modelCache.loading)}
              error={nodes.error}
              onRetry={() => { nodes.reload(); modelCache.reload() }}
              multiple={recipe.required_node_count > 1}
              disabled={recipeBusy}
              requiredIds={localRequired}
              allowedIds={allowedIds}
              unavailableReasons={unavailableReasons}
              localLabel={localLabel}
              primaryId={recipe.deployment_mode === 'sharded'
                ? (localNodeId && nodeIds.includes(localNodeId) ? localNodeId : undefined)
                : nodeIds[0]}
              legend="Deployment nodes"
              help={localPath ? 'Local model paths can run only on the controller.' : `Only nodes with ${recipe.model} already cached can be selected.`}
            />
            {recipe.deployment_mode === 'sharded' && !coordinatorReady && <p className="field-note">Sharded deployments must include the controller. Transfer the model weights to the controller in Storage if it is disabled.</p>}
            {!exactCount && <p className="field-note" role="status">Select exactly {recipe.required_node_count} {recipe.required_node_count === 1 ? 'node' : 'nodes'} to continue.</p>}
            <div className="modal-actions"><Button type="button" disabled={recipeBusy} onClick={() => setRecipeDeployment(undefined)}>Cancel</Button><Button variant="primary" disabled={!ready || recipeBusy} onClick={() => void deployRecipe()}><Play size={15} /> {recipeBusy ? 'Deploying…' : `Deploy on ${recipe.required_node_count} ${recipe.required_node_count === 1 ? 'node' : 'nodes'}`}</Button></div>
          </section>
        </div>
      })()}

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
