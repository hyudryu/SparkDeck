import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react'
import { Check, ChevronDown, ChevronRight, Download, ExternalLink, Heart, Search } from 'lucide-react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { BenchmarkAggregate, CatalogModel, NodeInventoryItem, RuntimeKind } from '../api/types'
import { Button, EmptyState, ErrorState, formatNumber, formatRate, LoadingState, PageHeader, RuntimeMark } from '../components/ui'
import { isNodeSelectable, NodeSelector, selectedNodeLabel } from '../components/NodeSelector'
import { useResource } from '../hooks/useResource'
import { communityAccessHint, useCommunityAccess } from '../hooks/useCommunityAccess'
import { formatBytes } from '../utils/format'
import { ggufArtifactOptions, type GgufArtifactOption } from '../utils/gguf'

type CatalogTab = 'hugging-face' | 'community'
type FitTone = 'easy' | 'tight' | 'no-fit' | 'unknown'
type DisplayCatalogModel = CatalogModel & {
  communityEvidenceSource?: 'community' | 'local'
  communityVariantKey?: string
}

const MIB = 1024 ** 2
const COMMUNITY_PAGE_SIZE = 50
const EMPTY_COMPATIBILITY: NonNullable<CatalogModel['runtime_compatibility']> = []
const EMPTY_QUANTIZATIONS: NonNullable<CatalogModel['quantizations']> = []

function formatParameters(value?: number | null) {
  if (!Number.isFinite(value) || Number(value) <= 0) return '—'
  return Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(Number(value))
}

function nodeMemoryBytes(node: NodeInventoryItem) {
  if (node.online === false || node.selectable === false || node.docker_ready === false) return undefined
  const gpuTotal = (node.stats?.gpus ?? []).reduce((sum, gpu) => (
    !gpu.error && Number.isFinite(gpu.mem_total_mib) && Number(gpu.mem_total_mib) > 0
      ? sum + Number(gpu.mem_total_mib) * MIB
      : sum
  ), 0)
  if (gpuTotal > 0) return gpuTotal
  const unifiedTotal = Number(node.stats?.mem?.total)
  return Number.isFinite(unifiedTotal) && unifiedTotal > 0 ? unifiedTotal : undefined
}

function deployableMemory(nodes: NodeInventoryItem[]) {
  const measured = nodes
    .map((node) => ({ node, capacity: nodeMemoryBytes(node) }))
    .filter((item): item is { node: NodeInventoryItem; capacity: number } => item.capacity !== undefined)
  if (measured.length === 0) {
    return { capacity: 0, localCapacity: 0, measuredNodes: 0, aggregate: false, workerCapacities: [] as number[] }
  }
  const isLocal = (node: NodeInventoryItem) => node.local === true || node.id === 'local'
  const localCapacity = measured.find(({ node }) => node.local === true)?.capacity
    ?? measured.find(({ node }) => node.id === 'local')?.capacity
    ?? 0
  const aggregate = measured.length > 1 && measured.some(({ node }) => isLocal(node))
  return {
    capacity: aggregate
      ? measured.reduce((sum, item) => sum + item.capacity, 0)
      : Math.max(...measured.map((item) => item.capacity)),
    measuredNodes: measured.length,
    aggregate,
    localCapacity,
    workerCapacities: measured.filter(({ node }) => !isLocal(node)).map((item) => item.capacity),
  }
}

function minimumFitNodes(
  weightSize: number | null | undefined,
  capacities: number[],
  pinnedCapacity?: number,
): number | undefined {
  // Sharded deployments always include the controller (the creation flow
  // prepends it), so its memory is pinned and only worker capacities are
  // pooled greedily, largest first — count-optimal either way. Returns
  // undefined when the weights do not fit at all.
  if (!Number.isFinite(weightSize) || Number(weightSize) <= 0) return undefined
  const weight = Number(weightSize)
  if (pinnedCapacity !== undefined) {
    if (pinnedCapacity >= weight) return 1
    if (capacities.length === 0) return undefined
  } else if (capacities.length === 0) return undefined
  let pooled = pinnedCapacity ?? 0
  const descending = [...capacities].sort((left, right) => right - left)
  for (let index = 0; index < descending.length; index += 1) {
    pooled += descending[index]
    if (pooled >= weight) return (pinnedCapacity === undefined ? 0 : 1) + index + 1
  }
  return undefined
}

function fitNodeLabel(count: number | undefined) {
  if (count === undefined) return ''
  return count === 1 ? 'Fits on 1 node' : `Fits on ${count}+ nodes`
}

function fitTone(weightSize: number | null | undefined, capacity: number): FitTone {
  if (!Number.isFinite(weightSize) || Number(weightSize) <= 0 || capacity <= 0) return 'unknown'
  const ratio = Number(weightSize) / capacity
  if (ratio <= 0.7) return 'easy'
  if (ratio <= 1) return 'tight'
  return 'no-fit'
}

function fitLabel(tone: FitTone) {
  if (tone === 'easy') return 'Fits easily'
  if (tone === 'tight') return 'Tight fit'
  if (tone === 'no-fit') return 'Does not fit'
  return 'Fit unknown'
}

function bestCommunityEstimate(items: BenchmarkAggregate[]) {
  return [...items].sort((left, right) => right.sample_count - left.sample_count)[0]
}

function aggregateQuantization(item: BenchmarkAggregate) {
  return item.quantization?.trim() || 'unknown'
}

function communityVariantKey(item: BenchmarkAggregate) {
  return `${item.model_id}::${aggregateQuantization(item)}::${item.prompt_tokens_bucket}`
}

function mergeRuntimeCompatibility(
  searchCompatibility: NonNullable<CatalogModel['runtime_compatibility']>,
  detailCompatibility: NonNullable<CatalogModel['runtime_compatibility']>,
) {
  const merged = new Map(detailCompatibility.map((item) => [item.runtime, item]))
  for (const item of searchCompatibility) {
    const existing = merged.get(item.runtime)
    merged.set(item.runtime, existing
      ? { ...existing, ...item, supported: existing.supported || item.supported }
      : item)
  }
  return [...merged.values()]
}

function requiresControllerCapacity(model: DisplayCatalogModel) {
  const compatibility = model.runtime_compatibility ?? EMPTY_COMPATIBILITY
  const hasNonLlamaRuntime = compatibility.some((item) => item.runtime !== 'llama.cpp' && item.supported)
  const hasLlamaRuntime = compatibility.some((item) => item.runtime === 'llama.cpp' && item.supported)
  const hasGgufArtifact = ggufArtifactOptions(model.quantizations ?? EMPTY_QUANTIZATIONS).length > 0
  return !hasNonLlamaRuntime && (hasLlamaRuntime || hasGgufArtifact)
}

function preferredGgufArtifact(
  artifactOptions: GgufArtifactOption[],
  communityQuantization?: string,
) {
  return artifactOptions.find((item) => (
    communityQuantization
    && item.quantization.toLocaleLowerCase() === communityQuantization.toLocaleLowerCase()
  )) ?? artifactOptions[0]
}

function defaultGgufWeightSize(model: DisplayCatalogModel, communityMode: boolean) {
  const artifactOptions = ggufArtifactOptions(model.quantizations ?? EMPTY_QUANTIZATIONS)
  const communityQuantization = communityMode && model.community
    ? aggregateQuantization(model.community)
    : undefined
  const defaultArtifact = preferredGgufArtifact(artifactOptions, communityQuantization)
  return defaultArtifact?.weightSize ?? model.weight_size_bytes ?? model.community?.weight_size_bytes
}

function deployHref(
  model: DisplayCatalogModel,
  runtime: RuntimeKind,
  artifact: GgufArtifactOption | undefined,
  sharded: boolean,
  communityMode: boolean,
) {
  const params = new URLSearchParams({ model: model.id, runtime })
  const quantization = runtime === 'llama.cpp'
    ? artifact?.quantization
    : communityMode && model.community ? aggregateQuantization(model.community) : undefined
  if (quantization && quantization !== 'unknown') params.set('quantization', quantization)
  if (runtime === 'llama.cpp' && artifact) params.set('artifact', artifact.filename)
  else if (runtime !== 'llama.cpp' && sharded) params.set('layout', 'sharded')
  return `/models?${params.toString()}`
}

function ModelRow({
  model,
  capacity,
  localCapacity,
  measuredNodes,
  aggregate,
  workerCapacities,
  expanded,
  communityEnabled,
  communityMode,
  requestedRuntime,
  onToggle,
  onPull,
}: {
  model: DisplayCatalogModel
  capacity: number
  localCapacity: number
  measuredNodes: number
  aggregate: boolean
  workerCapacities: number[]
  expanded: boolean
  communityEnabled: boolean
  communityMode: boolean
  requestedRuntime: RuntimeKind | ''
  onToggle: () => void
  onPull: (model: DisplayCatalogModel) => void
}) {
  const rowKey = model.communityVariantKey ?? model.id
  const panelId = `model-details-${rowKey.replace(/[^a-zA-Z0-9_-]/g, '-')}`
  const modelName = model.name ?? model.id.split('/').at(-1) ?? model.id
  const details = useResource(
    (signal) => api.catalog.details(model.id, signal),
    [model.id],
    expanded,
  )
  const detailedModel = details.data?.model
  const compatibility = useMemo(() => mergeRuntimeCompatibility(
    model.runtime_compatibility ?? EMPTY_COMPATIBILITY,
    detailedModel?.runtime_compatibility ?? EMPTY_COMPATIBILITY,
  ), [detailedModel?.runtime_compatibility, model.runtime_compatibility])
  const supportedRuntimes = useMemo(
    () => compatibility.filter((item) => item.supported),
    [compatibility],
  )
  const compatibilityByRuntime = useMemo(
    () => new Map(compatibility.map((item) => [item.runtime, item.supported])),
    [compatibility],
  )
  const quantizations = detailedModel?.quantizations ?? model.quantizations ?? EMPTY_QUANTIZATIONS
  const artifactOptions = useMemo(() => ggufArtifactOptions(quantizations), [quantizations])
  const initiallySupported = (candidate: RuntimeKind) => compatibilityByRuntime.get(candidate) !== false
  const initialRuntime = requestedRuntime && initiallySupported(requestedRuntime)
    ? requestedRuntime
    : supportedRuntimes.length === 1
      ? supportedRuntimes[0].runtime
      : initiallySupported('vllm') ? 'vllm' : initiallySupported('sglang') ? 'sglang' : 'llama.cpp'
  const [deploymentRuntime, setDeploymentRuntime] = useState<RuntimeKind>(initialRuntime)
  const communityQuantization = communityMode && model.community ? aggregateQuantization(model.community) : undefined
  const preferredArtifact = preferredGgufArtifact(artifactOptions, communityQuantization)
  const [artifactKey, setArtifactKey] = useState(preferredArtifact?.key ?? '')
  const selectedArtifact = artifactOptions.find((item) => item.key === artifactKey) ?? preferredArtifact
  const llamaSupported = compatibilityByRuntime.get('llama.cpp') !== false && artifactOptions.length > 0
  const deploymentReady = deploymentRuntime !== 'llama.cpp' || Boolean(selectedArtifact)
  const previousRequestedRuntime = useRef(requestedRuntime)

  useEffect(() => {
    const requestedRuntimeChanged = previousRequestedRuntime.current !== requestedRuntime
    previousRequestedRuntime.current = requestedRuntime
    setDeploymentRuntime((current) => {
      if (requestedRuntimeChanged && requestedRuntime && compatibilityByRuntime.get(requestedRuntime) !== false) {
        return requestedRuntime
      }
      return compatibilityByRuntime.get(current) === false
        ? supportedRuntimes[0]?.runtime ?? 'vllm'
        : current
    })
  }, [compatibilityByRuntime, requestedRuntime, supportedRuntimes])

  useEffect(() => {
    if (!artifactOptions.some((item) => item.key === artifactKey)) {
      setArtifactKey(preferredArtifact?.key ?? '')
    }
  }, [artifactKey, artifactOptions, preferredArtifact?.key])
  const rowLabel = communityMode && model.community
    ? `${model.id} (${aggregateQuantization(model.community)}, ${formatNumber(model.community.prompt_tokens_bucket)}-token prompt bucket)`
    : model.id
  const parameterCount = model.parameter_count ?? model.community?.parameter_count
  const weightSize = model.weight_size_bytes ?? model.community?.weight_size_bytes
  const fitWeightSize = deploymentRuntime === 'llama.cpp'
    ? selectedArtifact?.weightSize ?? weightSize
    : weightSize
  const fitCapacity = deploymentRuntime === 'llama.cpp' ? localCapacity : capacity
  const fitAggregate = deploymentRuntime !== 'llama.cpp' && aggregate
  const fitMeasuredNodes = deploymentRuntime === 'llama.cpp'
    ? localCapacity > 0 ? 1 : 0
    : measuredNodes
  // Sharded deployments always include the controller, then pool the largest
  // workers first; single-node and replicated deployments (and Llama server
  // on the controller) must fit within one node's memory.
  const minFitNodes = fitAggregate
    ? minimumFitNodes(fitWeightSize, workerCapacities, localCapacity)
    : minimumFitNodes(fitWeightSize, [fitCapacity])
  const minFitLabel = fitNodeLabel(minFitNodes)

  return <article className={`catalog-model-row${expanded ? ' expanded' : ''}`}>
    <button
      className="catalog-model-summary"
      type="button"
      aria-expanded={expanded}
      aria-controls={panelId}
      aria-label={`${expanded ? 'Collapse' : 'Expand'} ${rowLabel}`}
      onClick={onToggle}
    >
      <span className="catalog-model-identity"><strong>{modelName}</strong><small>{model.id}{communityMode && model.community ? ` · ${aggregateQuantization(model.community)} · ${formatNumber(model.community.prompt_tokens_bucket)}-token prompt bucket` : ''}</small></span>
      <span className="catalog-model-stat"><small>Parameters</small><strong>{formatParameters(parameterCount)}</strong></span>
      <span className={`catalog-model-stat catalog-model-size fit-${fitTone(fitWeightSize, fitCapacity)}`}><small>Weights</small><strong>{fitWeightSize ? formatBytes(fitWeightSize) : '—'}</strong><em>{fitLabel(fitTone(fitWeightSize, fitCapacity))}{minFitNodes ? ` · ${minFitNodes === 1 ? '1 node' : `${minFitNodes}+ nodes`}` : ''}</em></span>
      {communityMode
        ? <>
          <span className="catalog-model-stat"><small>Output speed</small><strong>{formatRate(model.community?.inference_tokens_per_second)}</strong></span>
          <span className="catalog-model-stat"><small>Unique clusters</small><strong>{formatNumber(model.community?.unique_cluster_count)}</strong></span>
        </>
        : <>
          <span className="catalog-model-stat"><small>Downloads</small><strong><Download size={13} aria-hidden="true" /> {formatNumber(model.downloads)}</strong></span>
          <span className="catalog-model-stat"><small>Likes</small><strong><Heart size={13} aria-hidden="true" /> {formatNumber(model.likes)}</strong></span>
        </>}
      <span className="catalog-model-chevron" aria-hidden="true">{expanded ? <ChevronDown size={17} /> : <ChevronRight size={17} />}</span>
    </button>
    {expanded && <div className="catalog-model-details" id={panelId}>
      <div className="catalog-model-detail-grid">
        <div>
          <span className="detail-label">{deploymentRuntime === 'llama.cpp' ? 'Controller fit' : 'Cluster fit'}</span>
          <strong className={`fit-${fitTone(fitWeightSize, fitCapacity)}`}>{fitLabel(fitTone(fitWeightSize, fitCapacity))} · {fitWeightSize ? formatBytes(fitWeightSize) : 'Weight size unavailable'}{minFitLabel ? ` · ${minFitLabel}` : ''}</strong>
          <p>{fitCapacity > 0
            ? deploymentRuntime === 'llama.cpp'
              ? `${formatBytes(fitCapacity)} on the controller node. Llama server deployments run on the controller and do not pool cluster memory. `
              : fitAggregate
              ? `${formatBytes(capacity)} aggregate memory across ${measuredNodes} measured nodes. Fit assumes a sharded deployment that can divide model weights across those nodes; replicated deployments still require the full model weights on every replica. `
              : `${formatBytes(fitCapacity)} on the largest of ${fitMeasuredNodes} measured ${fitMeasuredNodes === 1 ? 'node' : 'nodes'}. Fit assumes a single-node or replicated deployment, where every replica must hold the full model weights. `
            : deploymentRuntime === 'llama.cpp' ? 'Controller memory telemetry is unavailable. ' : 'Cluster memory telemetry is unavailable. '}Context and KV cache can increase runtime memory.</p>
        </div>
        <div>
          <span className="detail-label">Compatibility</span>
          <div className="runtime-row" aria-label="Compatible runtimes">
            {supportedRuntimes.map((item) => <RuntimeMark key={item.runtime} runtime={item.runtime} />)}
            {supportedRuntimes.length === 0 && <span className="muted">Compatibility unknown</span>}
            {model.local_deployment_ids && model.local_deployment_ids.length > 0 && <span className="local-chip"><Check size={12} /> Local</span>}
          </div>
        </div>
      </div>
      {details.loading && <p className="catalog-artifact-loading" role="status">Loading available quantizations…</p>}
      {quantizations.length > 0 && <div className="catalog-quantizations">
        <span className="detail-label">Available quantizations and artifacts</span>
        <div>{quantizations.map((variant) => <section key={variant.name}>
          <div><strong>{variant.name}</strong>{variant.weight_size_bytes ? <small>{formatBytes(variant.weight_size_bytes)}</small> : null}</div>
          {variant.files.length > 0 && <ul>{variant.files.map((file) => <li key={file.filename}><code>{file.filename}</code>{file.size_bytes ? <span>{formatBytes(file.size_bytes)}</span> : null}</li>)}</ul>}
        </section>)}</div>
      </div>}
      {model.community && communityEnabled && <div className="community-estimate" aria-label={`Community inference-speed estimate for ${rowLabel}`}>
        <div><span>{model.communityEvidenceSource === 'local' ? 'Aggregated from benchmarks on this controller' : 'Sampled from other SparkDeck users'}</span><strong>{formatRate(model.community.inference_tokens_per_second)}</strong></div>
        <p>{aggregateQuantization(model.community)} · inference-speed estimate for the {formatNumber(model.community.prompt_tokens_bucket)}-token prompt-length bucket · {formatNumber(model.community.sample_count)} {model.communityEvidenceSource === 'local' ? 'local' : 'shared'} samples</p>
        <small>{model.communityEvidenceSource === 'local' ? 'Local benchmark evidence only' : 'Aggregated community benchmark evidence only'} — an estimate, not a guarantee for your system.</small>
      </div>}
      <div className="catalog-model-actions">
        <a className="button" href={`https://huggingface.co/${model.id}`} target="_blank" rel="noreferrer"><ExternalLink size={14} /> Hugging Face</a>
        <button className="button" type="button" aria-label={`Pull weights for ${model.id}`} onClick={() => onPull({ ...model, revision: detailedModel?.revision ?? model.revision })}><Download size={14} /> Pull weights</button>
        <label className="catalog-deployment-type"><span>Deployment type</span><select aria-label={`Deployment type for ${model.id}`} value={deploymentRuntime} onChange={(event) => setDeploymentRuntime(event.target.value as RuntimeKind)}>
          <option value="vllm" disabled={compatibilityByRuntime.get('vllm') === false}>vLLM</option>
          <option value="sglang" disabled={compatibilityByRuntime.get('sglang') === false}>SGLang</option>
          <option value="llama.cpp" disabled={!llamaSupported}>Llama server</option>
        </select></label>
        {deploymentRuntime === 'llama.cpp' && artifactOptions.length > 0 && <label className="catalog-deployment-type catalog-artifact-select"><span>GGUF artifact</span><select aria-label={`GGUF artifact for ${model.id}`} value={selectedArtifact?.key ?? ''} onChange={(event) => setArtifactKey(event.target.value)}>
          {artifactOptions.map((item) => <option key={item.key} value={item.key}>{item.quantization} · {item.filename}{item.weightSize ? ` · ${formatBytes(item.weightSize)}` : ''}</option>)}
        </select></label>}
        {deploymentReady
          ? <Link className="button button-primary" aria-label={`Deploy ${model.id}`} title={`Deploy with ${deploymentRuntime === 'llama.cpp' ? 'Llama server' : deploymentRuntime === 'vllm' ? 'vLLM' : 'SGLang'}`} to={deployHref(model, deploymentRuntime, selectedArtifact, fitAggregate, communityMode)}>Deploy</Link>
          : <button className="button button-primary" type="button" disabled title={details.loading ? 'Loading GGUF artifacts' : 'No deployable GGUF artifact was found'}>Deploy</button>}
      </div>
    </div>}
  </article>
}

export function ExplorePage() {
  const [draft, setDraft] = useState('')
  const [query, setQuery] = useState('')
  const [runtime, setRuntime] = useState<RuntimeKind | ''>('')
  const [tab, setTab] = useState<CatalogTab>('hugging-face')
  const [fitsOnly, setFitsOnly] = useState(false)
  const [communityOnly, setCommunityOnly] = useState(false)
  const [communityLimit, setCommunityLimit] = useState(COMMUNITY_PAGE_SIZE)
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set())
  const [pullSelection, setPullSelection] = useState<{ model: DisplayCatalogModel; nodeIds: string[] }>()
  const [pullBusy, setPullBusy] = useState(false)
  const [pullError, setPullError] = useState<string>()
  const [pullNotice, setPullNotice] = useState<string>()
  const activeRuntime = tab === 'hugging-face' ? runtime : ''
  const catalog = useResource(
    (signal) => api.catalog.search(query, activeRuntime || undefined, undefined, signal),
    [activeRuntime, query],
  )
  const nodes = useResource((signal) => api.nodes.list(signal))
  const communityAccess = useCommunityAccess()
  const accessHint = communityAccessHint(communityAccess.signedIn)
  const aggregates = useResource(
    (signal) => api.benchmarks.aggregates(signal),
    [communityAccess.enabled],
    communityAccess.enabled,
  )

  useEffect(() => {
    const timeout = window.setTimeout(() => setQuery(draft.trim()), 350)
    return () => window.clearTimeout(timeout)
  }, [draft])

  useEffect(() => {
    setCommunityLimit(COMMUNITY_PAGE_SIZE)
  }, [aggregates.data?.items, communityOnly, fitsOnly, query, tab])

  const memory = useMemo(() => deployableMemory(nodes.data ?? []), [nodes.data])
  const catalogFitCapacity = activeRuntime === 'llama.cpp' ? memory.localCapacity : memory.capacity
  const models = useMemo(() => {
    const catalogItems = catalog.data?.items ?? []
    const evidence = new Map<string, BenchmarkAggregate[]>()
    for (const aggregate of aggregates.data?.items ?? []) {
      const key = communityVariantKey(aggregate)
      evidence.set(key, [...(evidence.get(key) ?? []), aggregate])
    }
    for (const model of catalogItems) {
      if (model.community) {
        const key = communityVariantKey(model.community)
        evidence.set(key, [...(evidence.get(key) ?? []), model.community])
      }
    }
    const catalogById = new Map(catalogItems.map((model) => [model.id, model]))
    const aggregateSource = aggregates.data?.availability === 'local' ? 'local' : 'community'
    const withEvidence: DisplayCatalogModel[] = catalogItems.map((model) => {
      const aggregate = bestCommunityEstimate(
        [...evidence.values()].flat().filter((item) => item.model_id === model.id),
      )
      return {
        ...model,
        community: model.community ?? aggregate,
        communityEvidenceSource: model.community ? 'community' : aggregate ? aggregateSource : undefined,
      }
    })
    const communityModels: DisplayCatalogModel[] = [...evidence.entries()].map(([variantKey, samples]) => {
      const modelId = samples[0]?.model_id ?? variantKey.split('::')[0]
      const catalogModel = catalogById.get(modelId)
      const community = bestCommunityEstimate(samples)
      return {
        ...(catalogModel ?? { id: modelId, name: modelId.split('/').at(-1) }),
        parameter_count: community.parameter_count ?? catalogModel?.parameter_count,
        weight_size_bytes: community.weight_size_bytes ?? catalogModel?.weight_size_bytes,
        community,
        communityVariantKey: variantKey,
        communityEvidenceSource: catalogModel?.community === community ? 'community' : aggregateSource,
      }
    })
    let visible = tab === 'community' ? communityModels : withEvidence
    if (tab === 'community' && query) {
      const folded = query.toLowerCase()
      visible = visible.filter((model) => (
        model.id.toLowerCase().includes(folded)
        || aggregateQuantization(model.community!).toLowerCase().includes(folded)
      ))
    }
    if (communityOnly || tab === 'community') visible = visible.filter((model) => Boolean(model.community))
    if (fitsOnly) visible = visible.filter((model) => {
      const usesControllerCapacity = activeRuntime === 'llama.cpp'
        || (activeRuntime === '' && requiresControllerCapacity(model))
      const applicableCapacity = usesControllerCapacity
        ? memory.localCapacity
        : memory.capacity
      const applicableWeightSize = usesControllerCapacity
        ? defaultGgufWeightSize(model, tab === 'community')
        : model.weight_size_bytes
      return ['easy', 'tight'].includes(fitTone(applicableWeightSize, applicableCapacity))
    })
    if (fitsOnly) {
      visible = [...visible].sort((left, right) => Number(right.weight_size_bytes ?? 0) - Number(left.weight_size_bytes ?? 0))
    } else if (tab === 'community') {
      visible = [...visible].sort((left, right) => Number(right.community?.sample_count ?? 0) - Number(left.community?.sample_count ?? 0))
    }
    return visible
  }, [activeRuntime, aggregates.data?.availability, aggregates.data?.items, catalog.data?.items, communityOnly, fitsOnly, memory, query, tab])
  const displayedModels = tab === 'community' ? models.slice(0, communityLimit) : models
  const remainingCommunityModels = tab === 'community' ? models.length - displayedModels.length : 0

  const submit = (event: FormEvent) => {
    event.preventDefault()
    setQuery(draft.trim())
  }

  const toggleExpanded = (modelId: string) => setExpandedIds((current) => {
    const next = new Set(current)
    if (next.has(modelId)) next.delete(modelId)
    else next.add(modelId)
    return next
  })

  const openPull = (model: DisplayCatalogModel) => {
    const firstNode = (nodes.data ?? []).find(isNodeSelectable)
    setPullError(undefined)
    setPullSelection({ model, nodeIds: firstNode ? [firstNode.id] : [] })
  }

  const pullWeights = async () => {
    if (!pullSelection?.nodeIds.length) return
    const { model, nodeIds } = pullSelection
    setPullBusy(true)
    setPullError(undefined)
    setPullNotice(undefined)
    try {
      const result = await api.storage.prepareModel(
        model.id,
        model.revision,
        nodeIds,
        nodeIds[0],
      )
      const transferCount = result.jobs.filter((job) => job.kind === 'transfer').length
      const downloadCount = result.jobs.filter((job) => job.kind === 'download').length
      setPullSelection(undefined)
      setPullNotice(result.jobs.length === 0
        ? `${model.id} is already available on every selected node.`
        : `Queued ${downloadCount ? `${downloadCount} Hugging Face ${downloadCount === 1 ? 'download' : 'downloads'}` : 'the cached source'}${transferCount ? ` and ${transferCount} Virtual NAS ${transferCount === 1 ? 'transfer' : 'transfers'}` : ''} for ${model.id}.`)
    } catch (reason) {
      setPullError(reason instanceof Error ? reason.message : 'Could not queue the model weights')
    } finally {
      setPullBusy(false)
    }
  }

  const communityEnabled = communityAccess.enabled
  const communityUnavailable = tab === 'community'
    && (Boolean(aggregates.error) || aggregates.data?.availability === 'unavailable')
  const activeError = tab === 'hugging-face' ? catalog.error : aggregates.error
  const loading = tab === 'hugging-face' ? catalog.loading : aggregates.loading

  return (
    <div className="page">
      <PageHeader
        eyebrow="Model catalog"
        title="Find the right model for your hardware"
        description="Search Hugging Face models, see what fits across your SparkDeck cluster, and compare community-sampled inference speed."
      />
      {pullNotice && <p className="inline-success" role="status">{pullNotice} <Link to="/storage">View Storage</Link></p>}

      <div className="usage-tabs catalog-tabs" role="tablist" aria-label="Model catalog source">
        <button role="tab" aria-selected={tab === 'hugging-face'} onClick={() => setTab('hugging-face')}>Hugging Face</button>
        <button role="tab" aria-selected={tab === 'community'} disabled={!communityEnabled} title={communityEnabled ? undefined : accessHint} onClick={() => setTab('community')}>Community Run Models</button>
      </div>

      <div className="catalog-toolbar">
        <form className={`catalog-search${tab === 'community' ? ' community-search' : ''}`} onSubmit={submit} role="search">
          <label className="search-field">
            <span className="sr-only">Search models</span>
            <Search size={18} aria-hidden="true" />
            <input value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Search models, tasks, or authors" />
          </label>
          {tab === 'hugging-face' && <label className="select-field compact-select">
            <span className="sr-only">Runtime</span>
            <select value={runtime} onChange={(event) => setRuntime(event.target.value as RuntimeKind | '')}>
              <option value="">All runtimes</option>
              <option value="vllm">vLLM</option>
              <option value="llama.cpp">Llama server</option>
              <option value="sglang">SGLang</option>
            </select>
          </label>}
          <button className="button button-primary" type="submit">Search</button>
        </form>
        <div className="catalog-filters" aria-label="Model filters">
          <label><input type="checkbox" checked={fitsOnly} disabled={!fitsOnly && catalogFitCapacity <= 0} onChange={(event) => setFitsOnly(event.target.checked)} /><span><strong>Only what fits</strong><small>{catalogFitCapacity > 0 ? activeRuntime === 'llama.cpp' ? `${formatBytes(catalogFitCapacity)} controller memory for Llama server` : memory.aggregate ? `${formatBytes(memory.capacity)} aggregate sharded memory across ${memory.measuredNodes} measured nodes` : `${formatBytes(memory.capacity)} largest per-node memory across ${memory.measuredNodes} measured ${memory.measuredNodes === 1 ? 'node' : 'nodes'}` : activeRuntime === 'llama.cpp' ? 'Controller memory unavailable' : 'Cluster memory unavailable'}</small></span></label>
          <label title={communityEnabled ? undefined : accessHint}><input type="checkbox" checked={communityOnly || tab === 'community'} disabled={!communityEnabled || tab === 'community'} onChange={(event) => setCommunityOnly(event.target.checked)} /><span><strong>Only with community data</strong><small>Benchmark samples shared by SparkDeck users</small></span></label>
          {(nodes.error || aggregates.error) && <Button variant="tertiary" onClick={() => { nodes.reload(); aggregates.reload() }}>Retry metadata</Button>}
        </div>
      </div>

      <div className="result-heading" aria-live="polite">
        <div>
          <h2>{tab === 'community' ? 'Community Run Models' : query ? `Results for “${query}”` : 'Popular Hugging Face models'}</h2>
          {!loading && <span>{formatNumber(models.length)} models</span>}
        </div>
        <p>{tab === 'community' ? 'Based on aggregated benchmark samples—not live session tracking.' : 'Sorted by Hugging Face downloads unless “Only what fits” is enabled.'}</p>
      </div>

      {loading && <LoadingState label={tab === 'community' ? 'Loading community models' : 'Searching Hugging Face'} />}
      {catalog.error && tab === 'hugging-face' && <ErrorState message={catalog.error} onRetry={catalog.reload} />}
      {communityUnavailable && <ErrorState message={aggregates.error ?? 'The community service is unavailable right now — try again later.'} onRetry={aggregates.reload} />}
      {!loading && !activeError && !communityUnavailable && models.length === 0 && (
        <EmptyState
          title={tab === 'community' ? 'No community-run models yet' : 'No models found'}
          description={tab === 'community' ? 'Models appear here only after aggregated benchmark samples are available.' : 'Try a broader name or turn off one of the filters.'}
        />
      )}
      {!loading && !activeError && !communityUnavailable && models.length > 0 && <section className="catalog-model-list" aria-label="Model results">
        <div className="catalog-model-header" aria-hidden="true"><span>Model</span><span>Parameters</span><span>Weights</span>{tab === 'community' ? <><span>Output speed</span><span>Unique clusters</span></> : <><span>Downloads</span><span>Likes</span></>}<span /></div>
        {displayedModels.map((model) => {
          const rowKey = model.communityVariantKey ?? model.id
          return <ModelRow key={rowKey} model={model} capacity={memory.capacity} localCapacity={memory.localCapacity} measuredNodes={memory.measuredNodes} aggregate={memory.aggregate} workerCapacities={memory.workerCapacities} expanded={expandedIds.has(rowKey)} communityEnabled={communityEnabled} communityMode={tab === 'community'} requestedRuntime={activeRuntime} onToggle={() => toggleExpanded(rowKey)} onPull={openPull} />
        })}
        {remainingCommunityModels > 0 && <div className="catalog-load-more"><Button type="button" onClick={() => setCommunityLimit((current) => current + COMMUNITY_PAGE_SIZE)}>Load more community models ({formatNumber(remainingCommunityModels)} remaining)</Button></div>}
      </section>}
      {pullSelection && <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && !pullBusy && setPullSelection(undefined)}>
        <section className="modal" role="dialog" aria-modal="true" aria-labelledby="pull-model-title">
          <div className="modal-heading"><div><p className="eyebrow">Model Explorer</p><h2 id="pull-model-title">Pull {pullSelection.model.id}</h2></div><button className="icon-button" disabled={pullBusy} onClick={() => setPullSelection(undefined)} aria-label="Close dialog">×</button></div>
          <p className="modal-description">Choose every node that should receive the weights. SparkDeck downloads once to the first selected node, then automatically transfers that copy to the rest through Virtual NAS.</p>
          {pullError && <p className="form-error" role="alert">{pullError}</p>}
          <NodeSelector
            nodes={nodes.data ?? []}
            selectedIds={pullSelection.nodeIds}
            onChange={(nodeIds) => setPullSelection({ ...pullSelection, nodeIds })}
            loading={nodes.loading}
            error={nodes.error}
            onRetry={nodes.reload}
            multiple
            disabled={pullBusy}
            legend="Weight targets"
            help="The first selected node is the download seed. Additional selected nodes receive queued Virtual NAS transfers after that download finishes."
          />
          {pullSelection.nodeIds.length > 0 && <p className="field-note"><strong>Download seed:</strong> {selectedNodeLabel(nodes.data ?? [], [pullSelection.nodeIds[0]])}</p>}
          <div className="modal-actions"><Button type="button" disabled={pullBusy} onClick={() => setPullSelection(undefined)}>Cancel</Button><Button type="button" variant="primary" disabled={pullBusy || pullSelection.nodeIds.length === 0} onClick={() => void pullWeights()}><Download size={15} /> {pullBusy ? 'Queueing…' : `Pull to ${pullSelection.nodeIds.length} ${pullSelection.nodeIds.length === 1 ? 'node' : 'nodes'}`}</Button></div>
        </section>
      </div>}
    </div>
  )
}
