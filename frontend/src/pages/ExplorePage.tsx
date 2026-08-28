import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { Check, ChevronDown, ChevronRight, Download, ExternalLink, Heart, Search } from 'lucide-react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { BenchmarkAggregate, CatalogModel, NodeInventoryItem, RuntimeKind } from '../api/types'
import { Button, EmptyState, ErrorState, formatNumber, formatRate, LoadingState, PageHeader, RuntimeMark } from '../components/ui'
import { useResource } from '../hooks/useResource'
import { communityAccessHint, useCommunityAccess } from '../hooks/useCommunityAccess'
import { formatBytes } from '../utils/format'

type CatalogTab = 'hugging-face' | 'community'
type FitTone = 'easy' | 'tight' | 'no-fit' | 'unknown'
type DisplayCatalogModel = CatalogModel & {
  communityEvidenceSource?: 'community' | 'local'
  communityVariantKey?: string
}

const MIB = 1024 ** 2
const COMMUNITY_PAGE_SIZE = 50

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
  if (measured.length === 0) return { capacity: 0, measuredNodes: 0, aggregate: false }
  const aggregate = measured.length > 1 && measured.some(({ node }) => node.local === true || node.id === 'local')
  return {
    capacity: aggregate
      ? measured.reduce((sum, item) => sum + item.capacity, 0)
      : Math.max(...measured.map((item) => item.capacity)),
    measuredNodes: measured.length,
    aggregate,
  }
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

function deployHref(model: DisplayCatalogModel, quantizations: NonNullable<CatalogModel['quantizations']>, sharded: boolean, communityMode: boolean) {
  const params = new URLSearchParams({ model: model.id })
  const quantization = communityMode && model.community ? aggregateQuantization(model.community) : undefined
  if (quantization && quantization !== 'unknown') params.set('quantization', quantization)
  const variant = quantization
    ? quantizations.find((item) => item.name.toLocaleLowerCase() === quantization.toLocaleLowerCase())
    : undefined
  const artifact = variant?.files.find((file) => file.filename.toLocaleLowerCase().endsWith('.gguf'))?.filename
  if (artifact) params.set('artifact', artifact)
  else if (sharded) params.set('layout', 'sharded')
  return `/models?${params.toString()}`
}

function ModelRow({
  model,
  capacity,
  measuredNodes,
  aggregate,
  expanded,
  communityEnabled,
  communityMode,
  onToggle,
}: {
  model: DisplayCatalogModel
  capacity: number
  measuredNodes: number
  aggregate: boolean
  expanded: boolean
  communityEnabled: boolean
  communityMode: boolean
  onToggle: () => void
}) {
  const rowKey = model.communityVariantKey ?? model.id
  const panelId = `model-details-${rowKey.replace(/[^a-zA-Z0-9_-]/g, '-')}`
  const modelName = model.name ?? model.id.split('/').at(-1) ?? model.id
  const supportedRuntimes = (model.runtime_compatibility ?? []).filter((item) => item.supported)
  const details = useResource(
    (signal) => api.catalog.details(model.id, signal),
    [model.id],
    expanded,
  )
  const quantizations = details.data?.model?.quantizations ?? model.quantizations ?? []
  const rowLabel = communityMode && model.community
    ? `${model.id} (${aggregateQuantization(model.community)}, ${formatNumber(model.community.prompt_tokens_bucket)}-token prompt bucket)`
    : model.id
  const parameterCount = model.parameter_count ?? model.community?.parameter_count
  const weightSize = model.weight_size_bytes ?? model.community?.weight_size_bytes

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
      <span className={`catalog-model-stat catalog-model-size fit-${fitTone(weightSize, capacity)}`}><small>Weights</small><strong>{weightSize ? formatBytes(weightSize) : '—'}</strong><em>{fitLabel(fitTone(weightSize, capacity))}</em></span>
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
          <span className="detail-label">Cluster fit</span>
          <strong className={`fit-${fitTone(weightSize, capacity)}`}>{fitLabel(fitTone(weightSize, capacity))} · {weightSize ? formatBytes(weightSize) : 'Weight size unavailable'}</strong>
          <p>{capacity > 0
            ? aggregate
              ? `${formatBytes(capacity)} aggregate memory across ${measuredNodes} measured nodes. Fit assumes a sharded deployment that can divide model weights across those nodes; replicated deployments still require the full model weights on every replica. `
              : `${formatBytes(capacity)} on the largest of ${measuredNodes} measured ${measuredNodes === 1 ? 'node' : 'nodes'}. Fit assumes a single-node or replicated deployment, where every replica must hold the full model weights. `
            : 'Cluster memory telemetry is unavailable. '}Context and KV cache can increase runtime memory.</p>
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
        <Link className="button button-primary" aria-label={`Deploy ${model.id}`} to={deployHref(model, quantizations, aggregate, communityMode)}>Deploy</Link>
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
  const catalog = useResource(
    (signal) => api.catalog.search(query, runtime || undefined, undefined, signal),
    [query, runtime],
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
    if (fitsOnly) visible = visible.filter((model) => ['easy', 'tight'].includes(fitTone(model.weight_size_bytes, memory.capacity)))
    if (fitsOnly) {
      visible = [...visible].sort((left, right) => Number(right.weight_size_bytes ?? 0) - Number(left.weight_size_bytes ?? 0))
    } else if (tab === 'community') {
      visible = [...visible].sort((left, right) => Number(right.community?.sample_count ?? 0) - Number(left.community?.sample_count ?? 0))
    }
    return visible
  }, [aggregates.data?.availability, aggregates.data?.items, catalog.data?.items, communityOnly, fitsOnly, memory.capacity, query, tab])
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
              <option value="llama.cpp">llama.cpp</option>
              <option value="sglang">SGLang</option>
            </select>
          </label>}
          <button className="button button-primary" type="submit">Search</button>
        </form>
        <div className="catalog-filters" aria-label="Model filters">
          <label><input type="checkbox" checked={fitsOnly} disabled={memory.capacity <= 0} onChange={(event) => setFitsOnly(event.target.checked)} /><span><strong>Only what fits</strong><small>{memory.capacity > 0 ? memory.aggregate ? `${formatBytes(memory.capacity)} aggregate sharded memory across ${memory.measuredNodes} measured nodes` : `${formatBytes(memory.capacity)} largest per-node memory across ${memory.measuredNodes} measured ${memory.measuredNodes === 1 ? 'node' : 'nodes'}` : 'Cluster memory unavailable'}</small></span></label>
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
          return <ModelRow key={rowKey} model={model} capacity={memory.capacity} measuredNodes={memory.measuredNodes} aggregate={memory.aggregate} expanded={expandedIds.has(rowKey)} communityEnabled={communityEnabled} communityMode={tab === 'community'} onToggle={() => toggleExpanded(rowKey)} />
        })}
        {remainingCommunityModels > 0 && <div className="catalog-load-more"><Button type="button" onClick={() => setCommunityLimit((current) => current + COMMUNITY_PAGE_SIZE)}>Load more community models ({formatNumber(remainingCommunityModels)} remaining)</Button></div>}
      </section>}
    </div>
  )
}
