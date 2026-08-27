import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { Bookmark, Check, HardDrive, Pencil, Play, Plus, Server, Settings2, Trash2, X } from 'lucide-react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import type { CreateDeploymentInput, Deployment, RecipeUpdateInput, RuntimeKind, SavedConfiguration, SavedConfigurationDetail } from '../api/types'
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

const PIN_STORAGE_KEY = 'sparkdeck:pinned-recipes'
const SORT_STORAGE_KEY = 'sparkdeck:models-sort'

type SortMode = 'recent' | 'name-asc' | 'name-desc'

// Flags the structured argument editor manages; everything else in a saved
// configuration's extra args is shown verbatim in the "Other flags" field.
const CONTROLLED_FLAGS = new Set([
  '--max-model-len', '--max-model-length', '--max-num-seqs', '--kv-cache-dtype',
  '--context-length', '--max-running-requests', '--max-cudagraph-capture-size',
  '--max-num-batched-tokens', '--speculative-config', '--default-chat-template-kwargs',
])

const shellQuote = (arg: string) => (/[\s"']/.test(arg) ? `'${arg.replace(/'/g, `'\\''`)}'` : arg)

// Shell-aware whitespace splitter: respects single/double quotes, strips them.
function shellSplit(input: string): string[] {
  const out: string[] = []
  let cur = ''
  let quote: string | null = null
  let inWord = false
  for (let i = 0; i < input.length; i++) {
    const ch = input[i]
    if (quote) {
      if (ch === quote) quote = null
      else cur += ch
      inWord = true
    } else if (ch === '"' || ch === "'") {
      quote = ch
      inWord = true
    } else if (/\s/.test(ch)) {
      if (inWord) { out.push(cur); cur = ''; inWord = false }
    } else if (ch === '\\' && i + 1 < input.length) {
      cur += input[i + 1]
      i++
      inWord = true
    } else {
      cur += ch
      inWord = true
    }
  }
  if (inWord) out.push(cur)
  return out
}

// Extra args minus the flags the structured editor controls, shell-quoted for display.
function remainingArgs(args: string[]): string {
  const out: string[] = []
  for (let i = 0; i < args.length; i++) {
    const token = args[i]
    const flag = token.split('=')[0]
    if (CONTROLLED_FLAGS.has(flag)) {
      if (!token.includes('=') && i + 1 < args.length && !args[i + 1].startsWith('-')) i++
      continue
    }
    out.push(token)
  }
  return out.map(shellQuote).join(' ')
}

const companyOf = (recipe: SavedConfiguration) => {
  const first = (recipe.model || recipe.name || '').split('/')[0]?.trim()
  return first || 'Other'
}

type ArgsForm = Record<string, string>

type ArgsEditorState = {
  open: boolean
  loading: boolean
  saving: boolean
  saved: boolean
  error?: string
  form: ArgsForm
}

const seedArgsForm = (detail: SavedConfigurationDetail): ArgsForm => {
  const controls = detail.launch_controls ?? {}
  return {
    context_window: controls.context_window?.toString() ?? '',
    max_concurrency: controls.max_concurrency?.toString() ?? '',
    kv_cache_dtype: controls.kv_cache_dtype ?? '',
    thinking_mode: controls.thinking_mode ?? 'default',
    dspark_num_speculative_tokens: controls.dspark_num_speculative_tokens?.toString() ?? '',
    max_cudagraph_capture_size: controls.max_cudagraph_capture_size?.toString() ?? '',
    max_num_batched_tokens: controls.max_num_batched_tokens?.toString() ?? '',
    gpu_memory_utilization: detail.gpu_memory_utilization?.toString() ?? '',
    gpu_memory_gb: detail.gpu_memory_gb?.toString() ?? '',
    remaining_flags: remainingArgs(detail.extra_args ?? []),
  }
}

const readPinned = (): string[] => {
  try {
    const parsed = JSON.parse(localStorage.getItem(PIN_STORAGE_KEY) ?? '[]')
    return Array.isArray(parsed) ? parsed.filter((id) => typeof id === 'string') : []
  } catch {
    return []
  }
}

const readSortMode = (): SortMode => {
  const value = localStorage.getItem(SORT_STORAGE_KEY)
  return value === 'name-asc' || value === 'name-desc' ? value : 'recent'
}

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
  const [sortMode, setSortMode] = useState<SortMode>(readSortMode)
  const [pinned, setPinned] = useState<string[]>(readPinned)
  const [renaming, setRenaming] = useState<{ id: string; value: string }>()
  const [argsEditors, setArgsEditors] = useState<Record<string, ArgsEditorState>>({})

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

  const saveRename = async () => {
    if (!renaming) return
    const alias = renaming.value.trim()
    if (!alias) return
    setBusy(renaming.id)
    setActionError(undefined)
    try {
      const updated = await api.deployments.rename(renaming.id, alias)
      setRenaming(undefined)
      setActionNotice(`Renamed deployment to ${updated.alias}.`)
      resource.reload()
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : 'Could not rename deployment')
    } finally {
      setBusy(undefined)
    }
  }

  const changeSortMode = (mode: SortMode) => {
    setSortMode(mode)
    localStorage.setItem(SORT_STORAGE_KEY, mode)
  }

  const togglePin = (recipeId: string) => {
    setPinned((current) => {
      const next = current.includes(recipeId)
        ? current.filter((id) => id !== recipeId)
        : [...current, recipeId]
      localStorage.setItem(PIN_STORAGE_KEY, JSON.stringify(next))
      return next
    })
  }

  const sortedDeployments = useMemo(() => {
    const items = [...(resource.data ?? [])]
    if (sortMode === 'name-asc' || sortMode === 'name-desc') {
      items.sort((a, b) => a.alias.localeCompare(b.alias, undefined, { sensitivity: 'base' }))
      if (sortMode === 'name-desc') items.reverse()
    } else {
      // Stable sort: deployments without a timestamp keep their existing order.
      items.sort((a, b) => (Date.parse(b.created_at ?? '') || 0) - (Date.parse(a.created_at ?? '') || 0))
    }
    return items
  }, [resource.data, sortMode])

  const recipeGroups = useMemo(() => {
    const groups = new Map<string, SavedConfiguration[]>()
    for (const recipe of recipes.data ?? []) {
      const company = companyOf(recipe)
      groups.set(company, [...(groups.get(company) ?? []), recipe])
    }
    const pinnedSet = new Set(pinned)
    return [...groups.entries()]
      .sort(([a], [b]) => a.localeCompare(b, undefined, { sensitivity: 'base' }))
      .map(([company, items]): [string, SavedConfiguration[]] => [company, [...items].sort((a, b) => {
        const pinDelta = Number(pinnedSet.has(b.id)) - Number(pinnedSet.has(a.id))
        if (pinDelta) return pinDelta
        return (a.name || a.model).localeCompare(b.name || b.model, undefined, { sensitivity: 'base' })
      })])
  }, [recipes.data, pinned])

  const setArgsEditor = (recipeId: string, next: Partial<ArgsEditorState>) => {
    setArgsEditors((current) => {
      const previous: ArgsEditorState = current[recipeId] ?? {
        open: false, loading: false, saving: false, saved: false, form: {},
      }
      return { ...current, [recipeId]: { ...previous, ...next } }
    })
  }

  const toggleArgsEditor = async (recipe: SavedConfiguration) => {
    const current = argsEditors[recipe.id]
    if (current?.open) {
      setArgsEditor(recipe.id, { open: false })
      return
    }
    if (current && Object.keys(current.form).length) {
      setArgsEditor(recipe.id, { open: true })
      return
    }
    setArgsEditor(recipe.id, { open: true, loading: true, error: undefined })
    try {
      const detail = await api.recipes.get(recipe.id)
      setArgsEditor(recipe.id, { loading: false, form: seedArgsForm(detail) })
    } catch (reason) {
      setArgsEditor(recipe.id, {
        loading: false,
        error: reason instanceof Error ? reason.message : 'Could not load arguments',
      })
    }
  }

  const saveArgs = async (recipe: SavedConfiguration) => {
    const editor = argsEditors[recipe.id]
    if (!editor) return
    const numeric = (value: string) => {
      const trimmed = (value ?? '').trim()
      if (!trimmed) return null
      const parsed = Number(trimmed)
      return Number.isFinite(parsed) ? parsed : null
    }
    const editorForm = editor.form
    const payload: RecipeUpdateInput = {
      extra_args: shellSplit(editorForm.remaining_flags ?? ''),
      launch_controls: {
        context_window: numeric(editorForm.context_window),
        max_concurrency: numeric(editorForm.max_concurrency),
        kv_cache_dtype: editorForm.kv_cache_dtype?.trim() || null,
        thinking_mode: editorForm.thinking_mode || 'default',
        dspark_num_speculative_tokens: numeric(editorForm.dspark_num_speculative_tokens),
        max_cudagraph_capture_size: numeric(editorForm.max_cudagraph_capture_size),
        max_num_batched_tokens: numeric(editorForm.max_num_batched_tokens),
      },
      gpu_memory_utilization: numeric(editorForm.gpu_memory_utilization),
      gpu_memory_gb: numeric(editorForm.gpu_memory_gb),
    }
    setArgsEditor(recipe.id, { saving: true, error: undefined, saved: false })
    try {
      await api.recipes.update(recipe.id, payload)
      setArgsEditor(recipe.id, { saving: false, saved: true })
      setTimeout(() => setArgsEditor(recipe.id, { saved: false }), 2500)
      recipes.reload()
    } catch (reason) {
      setArgsEditor(recipe.id, {
        saving: false,
        error: reason instanceof Error ? reason.message : 'Could not save arguments',
      })
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
        .map((model) => ({ ...model, nodeId: node.id, nodeName: node.name })),
    )
    if (!locations.length) return 'Disk size unavailable'
    const total = locations.reduce((sum, location) => sum + location.size_bytes, 0)
    let base: string
    if (locations.length === 1) {
      base = `${formatBytes(total)} on ${locations[0].nodeName}`
    } else {
      const perCopy = locations.every((location) => location.size_bytes === locations[0].size_bytes)
        ? `${formatBytes(locations[0].size_bytes)} each · `
        : ''
      base = `${perCopy}${formatBytes(total)} total on ${locations.length} nodes`
    }
    // Multi-node deployments: call out selected nodes whose weights were not
    // found in the cache inventory instead of silently omitting them.
    const cachedIds = new Set(locations.map((location) => location.nodeId))
    const expectedIds = deployment.selected_nodes?.map((node) => node.id) ?? deployment.node_ids ?? []
    const missing = expectedIds
      .filter((id) => !cachedIds.has(id))
      .map((id) => id === 'local' ? localLabel : (nodes.data?.find((node) => node.id === id)?.name ?? id))
    return missing.length ? `${base} · not cached on ${missing.join(', ')}` : base
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
      {!resource.loading && !resource.error && resource.data?.length === 0 && (
        <EmptyState title="No model servers yet" description="Launch a managed runtime or connect an existing OpenAI-compatible endpoint." action={<Button variant="primary" onClick={openCreator}>Add your first model</Button>} />
      )}
      {resource.data && resource.data.length > 0 && (
        <section className="deployments" aria-labelledby="deployments-title">
          <div className="section-heading">
            <div><h2 id="deployments-title">Deployments</h2></div>
            <label className="sort-field"><span>Sort</span>
              <select value={sortMode} onChange={(event) => changeSortMode(event.target.value as SortMode)} aria-label="Sort deployments">
                <option value="recent">Most recent</option>
                <option value="name-asc">Name A–Z</option>
                <option value="name-desc">Name Z–A</option>
              </select>
            </label>
          </div>
          <Panel className="table-panel">
            <div className="responsive-table deployments-table" role="table" aria-label="Model deployments">
              <div className="table-row table-header" role="row">
                <span role="columnheader">Model</span><span role="columnheader">Runtime</span><span role="columnheader">Configuration</span><span role="columnheader">Target</span><span role="columnheader">Status</span><span role="columnheader">Actions</span>
              </div>
              {sortedDeployments.map((deployment) => (
                <div className="table-row" role="row" key={deployment.id} tabIndex={0}>
                  <div role="cell" data-label="Model">
                    {renaming?.id === deployment.id ? (
                      <span className="rename-row">
                        <input
                          className="rename-input"
                          autoFocus
                          value={renaming.value}
                          aria-label={`New name for ${deployment.alias}`}
                          onChange={(event) => setRenaming({ id: deployment.id, value: event.target.value })}
                          onKeyDown={(event) => {
                            if (event.key === 'Enter') void saveRename()
                            if (event.key === 'Escape') setRenaming(undefined)
                          }}
                        />
                        <Button variant="tertiary" disabled={busy === deployment.id} aria-label="Save name" onClick={() => void saveRename()}><Check size={15} /></Button>
                        <Button variant="tertiary" aria-label="Cancel rename" onClick={() => setRenaming(undefined)}><X size={15} /></Button>
                      </span>
                    ) : (
                      <strong>{deployment.alias}</strong>
                    )}
                    <small>{deployment.model_id}</small>
                    <small className="model-disk-usage"><HardDrive size={12} /> {modelStorage(deployment)}</small>
                  </div>
                  <div role="cell" data-label="Runtime"><RuntimeMark runtime={deployment.runtime} /><small>{deployment.runtime_version ?? (deployment.managed ? 'Managed' : 'External')}</small></div>
                  <div role="cell" data-label="Configuration"><span>{deployment.settings.context_length?.toLocaleString() ?? '—'} ctx</span><small>{deployment.settings.quantization ?? 'Default precision'}</small></div>
                  <div role="cell" data-label="Target"><span>{deployment.selected_nodes?.map((node, index) => `${node.id === 'local' ? localLabel : node.name}${deployment.selected_nodes!.length > 1 && index === 0 ? ' (primary)' : ''}`).join(', ') || deployment.node_ids?.map((id, index) => `${id === 'local' ? localLabel : id}${deployment.node_ids!.length > 1 && index === 0 ? ' (primary)' : ''}`).join(', ') || localLabel}</span></div>
                  <div role="cell" data-label="Status"><Status status={deployment.status} /></div>
                  <div role="cell" data-label="Actions" className="row-actions">
                    {deployment.managed && (deployment.status === 'running'
                      ? <Button variant="tertiary" disabled={busy === deployment.id} onClick={() => void act(deployment, 'stop')}>Stop</Button>
                      : <Button variant="tertiary" disabled={busy === deployment.id} onClick={() => void act(deployment, 'start')}>Start</Button>)}
                    <Button variant="tertiary" disabled={busy === deployment.id} aria-label={`Rename ${deployment.alias}`} onClick={() => setRenaming({ id: deployment.id, value: deployment.alias })}><Pencil size={16} /></Button>
                    <Button variant="tertiary" disabled={busy === deployment.id} aria-label={`Remove ${deployment.alias}`} onClick={() => {
                      if (window.confirm(`Remove ${deployment.alias} from SparkDeck?`)) void act(deployment, 'remove')
                    }}><Trash2 size={17} /></Button>
                  </div>
                </div>
              ))}
            </div>
          </Panel>
        </section>
      )}
      {recipes.data && recipes.data.length > 0 && <section className="saved-configurations" aria-labelledby="saved-configurations-title">
        <div className="section-heading"><div><h2 id="saved-configurations-title">Saved cluster configurations</h2><p>Existing saved recipes are preserved and deploy through SparkDeck.</p></div></div>
        {recipeGroups.map(([company, items]) => (
          <div className="saved-configuration-group" key={company}>
            <h3 className="saved-configuration-group-title">{company} <span className="saved-configuration-group-count">{items.length}</span></h3>
            <div className="saved-configuration-grid">
              {items.map((recipe) => {
                const targets = (recipe.node_ids?.length ? recipe.node_ids : ['local']).map((id) => nodes.data?.find((node) => node.id === id))
                const targetNames = targets.map((node, index) => node?.name ?? recipe.node_ids?.[index] ?? 'This device')
                const unavailable = targets.some((node) => !node || !isNodeSelectable(node))
                const disabled = recipe.supported === false
                const isPinned = pinned.includes(recipe.id)
                const editor = argsEditors[recipe.id]
                const isVllm = (recipe.engine || 'vllm') !== 'sglang'
                return <Panel className="saved-configuration-card" key={recipe.id}>
                  <div className="saved-configuration-heading">
                    <button
                      type="button"
                      className={`icon-button pin-button${isPinned ? ' pinned' : ''}`}
                      aria-label={`${isPinned ? 'Unpin' : 'Pin'} ${recipe.name || recipe.model}`}
                      aria-pressed={isPinned}
                      onClick={() => togglePin(recipe.id)}
                    ><Bookmark size={17} fill={isPinned ? 'currentColor' : 'none'} /></button>
                    <div><h3>{recipe.name || recipe.model}</h3><p>{recipe.model}</p></div>
                  </div>
                  <dl>
                    <div><dt>Runtime</dt><dd><RuntimeMark runtime={recipe.engine || 'vllm'} /></dd></div>
                    <div><dt>Layout</dt><dd>{recipe.tensor_parallel_size > 1 ? `TP${recipe.tensor_parallel_size} · ` : ''}{recipe.deployment_mode || 'single'} · {recipe.required_node_count} {recipe.required_node_count === 1 ? 'node' : 'nodes'}</dd></div>
                    <div><dt>Targets</dt><dd>{targetNames.join(', ')}</dd></div>
                    <div><dt>Arguments</dt><dd>{recipe.extra_args_count ?? 0} saved</dd></div>
                  </dl>
                  {(recipe.error || unavailable) && <p className="saved-configuration-warning">{recipe.error || 'One or more saved nodes are missing, offline, or not ready.'}</p>}
                  <div className="saved-configuration-actions">
                    <Button variant="primary" disabled={disabled || busy === `recipe:${recipe.id}`} onClick={() => openRecipeDeployment(recipe)}><Play size={15} /> Choose nodes &amp; deploy</Button>
                    <Button variant="tertiary" aria-expanded={editor?.open ?? false} onClick={() => void toggleArgsEditor(recipe)}><Settings2 size={15} /> Arguments</Button>
                  </div>
                  {editor?.open && <div className="args-editor">
                    {editor.loading && <p className="field-note">Loading arguments…</p>}
                    {editor.error && <p className="form-error" role="alert">{editor.error}</p>}
                    {!editor.loading && Object.keys(editor.form).length > 0 && <>
                      <div className="field-grid">
                        <label className="field"><span>Context window</span><input type="number" min="1" value={editor.form.context_window} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, context_window: event.target.value } })} /></label>
                        <label className="field"><span>Max concurrency</span><input type="number" min="1" value={editor.form.max_concurrency} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, max_concurrency: event.target.value } })} /></label>
                        <label className="field"><span>KV cache dtype</span><input value={editor.form.kv_cache_dtype} placeholder="auto" onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, kv_cache_dtype: event.target.value } })} /></label>
                        <label className="field"><span>Thinking</span><select value={editor.form.thinking_mode} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, thinking_mode: event.target.value } })}><option value="default">Default</option><option value="enabled">Enabled</option><option value="disabled">Disabled</option></select></label>
                        {isVllm && <>
                          <label className="field"><span>Speculative tokens</span><input type="number" min="1" value={editor.form.dspark_num_speculative_tokens} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, dspark_num_speculative_tokens: event.target.value } })} /></label>
                          <label className="field"><span>Cudagraph capture size</span><input type="number" min="1" value={editor.form.max_cudagraph_capture_size} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, max_cudagraph_capture_size: event.target.value } })} /></label>
                          <label className="field"><span>Batched tokens</span><input type="number" min="1" value={editor.form.max_num_batched_tokens} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, max_num_batched_tokens: event.target.value } })} /></label>
                        </>}
                        <label className="field"><span>GPU memory util</span><input type="number" step="0.05" min="0.1" max="0.98" value={editor.form.gpu_memory_utilization} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, gpu_memory_utilization: event.target.value } })} /></label>
                        <label className="field"><span>Reserve GB</span><input type="number" min="1" value={editor.form.gpu_memory_gb} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, gpu_memory_gb: event.target.value } })} /></label>
                      </div>
                      <label className="field"><span>Other flags</span><textarea rows={3} value={editor.form.remaining_flags} spellCheck={false} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, remaining_flags: event.target.value } })} /></label>
                      <p className="field-note">Blank fields remove the flag. Structured fields above override matching flags in &quot;Other flags&quot;.</p>
                      <div className="args-editor-actions">
                        <Button variant="primary" disabled={editor.saving} onClick={() => void saveArgs(recipe)}>{editor.saving ? 'Saving…' : 'Save settings'}</Button>
                        {editor.saved && <span className="inline-success" role="status">Saved.</span>}
                      </div>
                    </>}
                  </div>}
                </Panel>
              })}
            </div>
          </div>
        ))}
      </section>}

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
