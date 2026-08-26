import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { Check, ChevronDown, ChevronRight, Download, ExternalLink, Heart, Search } from 'lucide-react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { BenchmarkAggregate, CatalogModel, NodeInventoryItem, RuntimeKind } from '../api/types'
import { Button, EmptyState, ErrorState, formatNumber, formatRate, LoadingState, PageHeader, RuntimeMark } from '../components/ui'
import { useResource } from '../hooks/useResource'
import { formatBytes } from '../utils/format'

type CatalogTab = 'hugging-face' | 'community'
type FitTone = 'easy' | 'tight' | 'no-fit' | 'unknown'
type DisplayCatalogModel = CatalogModel & { communityEvidenceSource?: 'community' | 'local' }

const MIB = 1024 ** 2

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
  const capacities = nodes.map(nodeMemoryBytes).filter((value): value is number => value !== undefined)
  if (capacities.length === 0) return { capacity: 0, measuredNodes: 0 }
  return {
    capacity: Math.max(...capacities),
    measuredNodes: capacities.length,
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

function ModelRow({
  model,
  capacity,
  expanded,
  onToggle,
}: {
  model: DisplayCatalogModel
  capacity: number
  expanded: boolean
  onToggle: () => void
}) {
  const tone = fitTone(model.weight_size_bytes, capacity)
  const panelId = `model-details-${model.id.replace(/[^a-zA-Z0-9_-]/g, '-')}`
  const modelName = model.name ?? model.id.split('/').at(-1) ?? model.id
  const supportedRuntimes = (model.runtime_compatibility ?? []).filter((item) => item.supported)

  return <article className={`catalog-model-row${expanded ? ' expanded' : ''}`}>
    <button
      className="catalog-model-summary"
      type="button"
      aria-expanded={expanded}
      aria-controls={panelId}
      aria-label={`${expanded ? 'Collapse' : 'Expand'} ${model.id}`}
      onClick={onToggle}
    >
      <span className="catalog-model-identity"><strong>{modelName}</strong><small>{model.id}</small></span>
      <span className="catalog-model-stat"><small>Parameters</small><strong>{formatParameters(model.parameter_count)}</strong></span>
      <span className={`catalog-model-stat catalog-model-size fit-${tone}`}><small>Weights</small><strong>{model.weight_size_bytes ? formatBytes(model.weight_size_bytes) : '—'}</strong><em>{fitLabel(tone)}</em></span>
      <span className="catalog-model-stat"><small>Downloads</small><strong><Download size={13} aria-hidden="true" /> {formatNumber(model.downloads)}</strong></span>
      <span className="catalog-model-stat"><small>Likes</small><strong><Heart size={13} aria-hidden="true" /> {formatNumber(model.likes)}</strong></span>
      <span className="catalog-model-chevron" aria-hidden="true">{expanded ? <ChevronDown size={17} /> : <ChevronRight size={17} />}</span>
    </button>
    {expanded && <div className="catalog-model-details" id={panelId}>
      <div className="catalog-model-detail-grid">
        <div>
          <span className="detail-label">Cluster fit</span>
          <strong className={`fit-${tone}`}>{fitLabel(tone)} · {model.weight_size_bytes ? formatBytes(model.weight_size_bytes) : 'Weight size unavailable'}</strong>
          <p>{capacity > 0 ? `${formatBytes(capacity)} on the largest measured node. ` : 'Cluster memory telemetry is unavailable. '}Fit assumes a single-node or replicated deployment, where every replica must hold the full model weights; context and KV cache can increase runtime memory.</p>
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
      {model.community && <div className="community-estimate" aria-label={`Community inference-speed estimate for ${model.id}`}>
        <div><span>{model.communityEvidenceSource === 'local' ? 'Aggregated from benchmarks on this controller' : 'Sampled from other SparkDeck users'}</span><strong>{formatRate(model.community.inference_tokens_per_second)}</strong></div>
        <p>Inference-speed estimate at a {formatNumber(model.community.context_window_size)}-token context window · {formatNumber(model.community.sample_count)} {model.communityEvidenceSource === 'local' ? 'local' : 'shared'} samples</p>
        <small>{model.communityEvidenceSource === 'local' ? 'Local benchmark evidence only' : 'Aggregated community benchmark evidence only'} — an estimate, not a guarantee for your system.</small>
      </div>}
      <div className="catalog-model-actions">
        <a className="button" href={`https://huggingface.co/${model.id}`} target="_blank" rel="noreferrer"><ExternalLink size={14} /> Hugging Face</a>
        <Link className="button button-primary" aria-label={`Deploy ${model.id}`} to={`/models?model=${encodeURIComponent(model.id)}`}>Deploy</Link>
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
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set())
  const catalog = useResource(
    (signal) => api.catalog.search(query, runtime || undefined, undefined, signal),
    [query, runtime],
  )
  const nodes = useResource((signal) => api.nodes.list(signal))
  const aggregates = useResource((signal) => api.benchmarks.aggregates(signal))

  useEffect(() => {
    const timeout = window.setTimeout(() => setQuery(draft.trim()), 350)
    return () => window.clearTimeout(timeout)
  }, [draft])

  const memory = useMemo(() => deployableMemory(nodes.data ?? []), [nodes.data])
  const models = useMemo(() => {
    const catalogItems = catalog.data?.items ?? []
    const evidence = new Map<string, BenchmarkAggregate[]>()
    for (const aggregate of aggregates.data?.items ?? []) {
      evidence.set(aggregate.model_id, [...(evidence.get(aggregate.model_id) ?? []), aggregate])
    }
    for (const model of catalogItems) {
      if (model.community) evidence.set(model.id, [...(evidence.get(model.id) ?? []), model.community])
    }
    const catalogById = new Map(catalogItems.map((model) => [model.id, model]))
    const aggregateSource = aggregates.data?.availability === 'local' ? 'local' : 'community'
    const withEvidence: DisplayCatalogModel[] = catalogItems.map((model) => {
      const aggregate = bestCommunityEstimate(evidence.get(model.id) ?? [])
      return {
        ...model,
        community: model.community ?? aggregate,
        communityEvidenceSource: model.community ? 'community' : aggregate ? aggregateSource : undefined,
      }
    })
    const communityModels: DisplayCatalogModel[] = [...evidence.entries()].map(([modelId, samples]) => {
      const catalogModel = catalogById.get(modelId)
      const community = bestCommunityEstimate(samples)
      return {
        ...(catalogModel ?? { id: modelId, name: modelId.split('/').at(-1) }),
        community,
        communityEvidenceSource: catalogModel?.community === community ? 'community' : aggregateSource,
      }
    })
    let visible = tab === 'community' ? communityModels : withEvidence
    if (tab === 'community' && query) {
      const folded = query.toLowerCase()
      visible = visible.filter((model) => model.id.toLowerCase().includes(folded))
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

  const communityUnavailable = Boolean(aggregates.error) && tab === 'community'
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
        <button role="tab" aria-selected={tab === 'community'} onClick={() => setTab('community')}>Community Run Models</button>
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
          <label><input type="checkbox" checked={fitsOnly} disabled={memory.capacity <= 0} onChange={(event) => setFitsOnly(event.target.checked)} /><span><strong>Only what fits</strong><small>{memory.capacity > 0 ? `${formatBytes(memory.capacity)} largest per-node memory across ${memory.measuredNodes} measured ${memory.measuredNodes === 1 ? 'node' : 'nodes'}` : 'Cluster memory unavailable'}</small></span></label>
          <label><input type="checkbox" checked={communityOnly || tab === 'community'} disabled={tab === 'community'} onChange={(event) => setCommunityOnly(event.target.checked)} /><span><strong>Only with community data</strong><small>Benchmark samples shared by SparkDeck users</small></span></label>
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
      {communityUnavailable && <ErrorState message={aggregates.error ?? 'Community data unavailable'} onRetry={aggregates.reload} />}
      {!loading && !activeError && !communityUnavailable && models.length === 0 && (
        <EmptyState
          title={tab === 'community' ? 'No community-run models yet' : 'No models found'}
          description={tab === 'community' ? 'Models appear here only after aggregated benchmark samples are available.' : 'Try a broader name or turn off one of the filters.'}
        />
      )}
      {!loading && !activeError && !communityUnavailable && models.length > 0 && <section className="catalog-model-list" aria-label="Model results">
        <div className="catalog-model-header" aria-hidden="true"><span>Model</span><span>Parameters</span><span>Weights</span><span>Downloads</span><span>Likes</span><span /></div>
        {models.map((model) => <ModelRow key={model.id} model={model} capacity={memory.capacity} expanded={expandedIds.has(model.id)} onToggle={() => toggleExpanded(model.id)} />)}
      </section>}
    </div>
  )
}
