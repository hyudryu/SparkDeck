import { useCallback, useEffect, useRef, useState } from 'react'
import { BarChart3, Check, ChevronRight, Trash2 } from 'lucide-react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { Button, EmptyState, ErrorState, formatDuration, formatRate, LoadingState, PageHeader, Panel, RuntimeMark, Status } from '../components/ui'
import { useResource } from '../hooks/useResource'
import { communityAccessHint, useCommunityAccess } from '../hooks/useCommunityAccess'
import { BenchmarkLineChart } from '../components/BenchmarkLineChart'
import { BenchmarkRunner } from '../components/BenchmarkRunner'
import { TemperatureRuns } from '../components/TemperatureRuns'
import { LegalDialog } from '../components/LegalDialog'

export function BenchmarksPage() {
  const [tab, setTab] = useState<'speed' | 'temp'>('speed')
  const samples = useResource((signal) => api.benchmarks.list(signal))
  const benchmarkModels = useResource((signal) => api.benchmarks.models(signal))
  const communityAccess = useCommunityAccess()
  const accessHint = communityAccessHint(communityAccess.signedIn)
  const aggregates = useResource(
    (signal) => api.benchmarks.aggregates(signal),
    [communityAccess.enabled],
    communityAccess.enabled,
  )
  const [selectedModel, setSelectedModel] = useState<string>()
  const [selectedTp, setSelectedTp] = useState<number>()
  const modelTriggerRef = useRef<HTMLButtonElement>(null)
  const modelDetail = useResource(
    (signal) => api.benchmarks.model(selectedModel ?? '', signal),
    [selectedModel],
    Boolean(selectedModel),
  )
  const activeModelDetail = modelDetail.data?.model_id === selectedModel
    ? modelDetail.data
    : undefined
  const closeModelDetail = useCallback(() => setSelectedModel(undefined), [])
  const aggregateResponse = aggregates.data
  const localAggregates = aggregateResponse?.availability === 'local'

  useEffect(() => {
    const sizes = [...new Set(activeModelDetail?.points.map((point) => point.tensor_parallel_size) ?? [])]
    if (sizes.length && !sizes.includes(selectedTp ?? -1)) setSelectedTp(sizes[0])
  }, [activeModelDetail, selectedTp])

  const remove = async (modelId: string) => {
    await api.benchmarks.deleteLocalModel(modelId)
    samples.reload()
    benchmarkModels.reload()
    aggregates.reload()
  }

  return (
    <div className="page">
      <PageHeader eyebrow="Performance evidence" title="Benchmarks" description="Run llama-benchy against served models, review measurements captured from SparkDeck requests, and compare them with privacy-preserving community results." />
      <div className="benchmark-tp-tabs benchmark-page-tabs" role="tablist" aria-label="Benchmark view">
        <button type="button" role="tab" aria-selected={tab === 'speed'} onClick={() => setTab('speed')}>Speed</button>
        <button type="button" role="tab" aria-selected={tab === 'temp'} onClick={() => setTab('temp')}>Temp</button>
      </div>
      {tab === 'speed' && <>
      <BenchmarkRunner />
      <Panel className="privacy-panel">
        <p className="eyebrow">Always private</p>
        <h2>Your content stays local</h2>
        <p>Prompts and outputs never enter benchmark JSON. Ordinary authenticated request and network metadata may still be processed to operate the service.</p>
        <span><Check size={15} /> Sharing is off until you opt in</span>
      </Panel>

      <div className="section-heading"><div><h2>Benchmark runs by model</h2><p>Run a deployed model at C1, C2, C5, or C10. Select a model to compare prompt and generation throughput across context windows.</p></div></div>
      {benchmarkModels.loading && <LoadingState label="Loading benchmark runs" />}
      {benchmarkModels.error && <ErrorState message={benchmarkModels.error} onRetry={benchmarkModels.reload} />}
      {!benchmarkModels.loading && !benchmarkModels.error && benchmarkModels.data?.length === 0 && <EmptyState title="No coordinated benchmark runs yet" description="Use benchmark_cluster_deployment for a ready model at concurrency 1, 2, 5, or 10. Every completed run is recorded here without its prompts or outputs." />}
      {benchmarkModels.data && benchmarkModels.data.length > 0 && <Panel className="benchmark-model-panel"><div className="benchmark-model-list" aria-label="Benchmarked models">
        {benchmarkModels.data.map((model) => <button
          className="benchmark-model-row"
          type="button"
          aria-haspopup="dialog"
          key={model.model_id}
          onClick={(event) => { modelTriggerRef.current = event.currentTarget; setSelectedTp(undefined); setSelectedModel(model.model_id) }}
        >
          <span className="benchmark-model-icon"><BarChart3 size={17} /></span>
          <span className="benchmark-model-main"><strong>{model.model_id}</strong><small>{model.run_count} run{model.run_count === 1 ? '' : 's'} · Updated {new Date(model.latest_at).toLocaleString()}</small></span>
          <span><small>Best prompt</small><strong>{formatRate(model.best_prompt_tokens_per_second)}</strong></span>
          <span><small>Best generation</small><strong>{formatRate(model.best_generation_tokens_per_second)}</strong></span>
          <span><small>Context windows</small><strong>{model.context_windows.map((window) => `${Math.round(window / 1024)}K`).join(', ')}</strong></span>
          <span><small>TP sizes</small><strong>{model.tensor_parallel_sizes.map((size) => `TP${size}`).join(', ')}</strong></span>
          <ChevronRight size={17} aria-hidden="true" />
        </button>)}
      </div></Panel>}

      <div className="section-heading" title={communityAccess.enabled ? undefined : accessHint}><div><h2>{localAggregates ? 'Local aggregate estimates' : 'Community estimates'}</h2><p>Evidence is matched only by exact model name, quantization, TP size, and prompt-length bucket. Results are estimates, not guarantees.</p></div></div>
      {!communityAccess.enabled && !communityAccess.loading && <EmptyState
        title="Community estimates are locked"
        description={accessHint}
        action={<Link className="button button-secondary" to="/settings">Open community settings</Link>}
      />}
      {communityAccess.enabled && aggregates.loading && <LoadingState label="Loading community aggregates" />}
      {communityAccess.enabled && aggregates.error && <ErrorState message={aggregates.error} onRetry={aggregates.reload} />}
      {communityAccess.enabled && !aggregates.loading && !aggregates.error && aggregateResponse?.availability === 'unavailable' && <EmptyState
        title="Community service unavailable"
        description="The hosted community service could not be reached. Your local benchmarks are unaffected — try again later."
        action={<Button variant="secondary" onClick={aggregates.reload}>Retry</Button>}
      />}
      {communityAccess.enabled && !aggregates.loading && !aggregates.error && aggregateResponse?.availability !== 'unavailable' && aggregateResponse?.items.length === 0 && <EmptyState title="No community estimates yet" description="Estimates will appear when enough samples share the same model name, quantization, TP size, and prompt-length bucket." />}
      {communityAccess.enabled && aggregateResponse && aggregateResponse.items.length > 0 && <div className="aggregate-grid">{aggregateResponse.items.map((item) => (
        <Panel className="aggregate-item" key={`${item.model_id}-${item.quantization}-${item.tensor_parallel_size}-${item.prompt_tokens_bucket}`}>
          <div><div><p className="aggregate-model">{item.model_id}</p><small className="aggregate-quantization">{item.quantization}</small></div><span className="estimate-label">{localAggregates ? 'Local estimate' : 'Community estimate'}</span></div>
          <dl><div><dt>Inference speed</dt><dd>{formatRate(item.inference_tokens_per_second)}</dd></div><div><dt>Tensor parallel</dt><dd>TP{item.tensor_parallel_size}</dd></div><div><dt>Prompt-length bucket</dt><dd>{item.prompt_tokens_bucket.toLocaleString()} tokens</dd></div><div><dt>Evidence</dt><dd>{item.sample_count} contributors</dd></div></dl>
          {item.sample_count >= aggregateResponse.evidence_policy.minimum_samples ? <span className="proven"><Check size={14} /> Evidence threshold met</span> : <span className="muted">Collecting more evidence</span>}
        </Panel>
      ))}</div>}
      {communityAccess.enabled && aggregateResponse && <p className="aggregate-policy">Evidence threshold: {aggregateResponse.evidence_policy.minimum_samples} contributors, matched only on model name, quantization, TP size, and prompt-length bucket. Each contributor has at most one average per TP setting after output-speed outliers are removed, and only single-stream inference is included. Inference speed is {localAggregates ? 'aggregated from this controller' : 'a community estimate'} and may differ on your system.</p>}

      <div className="section-heading"><div><h2>Local history</h2><p>One latest result per identified model. Eligible results stay local while signed out and upload after Community sign-in when sharing is enabled.</p></div></div>
      {samples.loading && <LoadingState label="Loading benchmark history" />}
      {samples.error && <ErrorState message={samples.error} onRetry={samples.reload} />}
      {!samples.loading && !samples.error && samples.data?.length === 0 && <EmptyState title="No identified benchmark models yet" description="Local history is captured from consented startup benchmarks and coordinated (parallel) benchmarks. Start a consented model or run a coordinated benchmark to produce a measurement." />}
      {samples.data && samples.data.length > 0 && <Panel className="table-panel"><div className="responsive-table benchmark-table" role="table" aria-label="Local benchmark history">
        <div className="table-row table-header" role="row"><span role="columnheader">Model</span><span role="columnheader">Runtime</span><span role="columnheader">Speed</span><span role="columnheader">TTFT</span><span role="columnheader">Sync</span><span role="columnheader">Actions</span></div>
        {samples.data.map((sample) => <div className="table-row" role="row" tabIndex={0} key={sample.model_id}>
          <div role="cell" data-label="Model"><strong>{sample.model_id}</strong><small>{sample.sample_count ?? 1} saved result{(sample.sample_count ?? 1) === 1 ? '' : 's'} · Latest {new Date(sample.created_at).toLocaleString()}</small></div>
          <div role="cell" data-label="Runtime"><RuntimeMark runtime={sample.runtime} /><small>{sample.quantization ?? 'Default precision'}</small></div>
          <div role="cell" data-label="Speed"><strong>{formatRate(sample.tokens_per_second)}</strong><small>{sample.output_tokens ?? '—'} output tokens</small></div>
          <div role="cell" data-label="TTFT">{formatDuration(sample.ttft_ms)}</div>
          <div role="cell" data-label="Sync"><Status status={sample.sync_state ?? 'local'} /></div>
          <div role="cell" data-label="Actions"><Button variant="tertiary" aria-label={`Delete all benchmarks for ${sample.model_id}`} onClick={() => void remove(sample.model_id)}><Trash2 size={15} /></Button></div>
        </div>)}
      </div></Panel>}
      {selectedModel && <LegalDialog eyebrow="Benchmark detail" title={selectedModel} titleId="benchmark-model-title" onClose={closeModelDetail} returnFocusRef={modelTriggerRef}>
        <p className="modal-description">Measured results only. Missing concurrency or context combinations remain blank.</p>
        {modelDetail.loading && <LoadingState label="Loading model benchmark" />}
        {modelDetail.error && <ErrorState message={modelDetail.error} onRetry={modelDetail.reload} />}
        {activeModelDetail && <>
          {[...new Set(activeModelDetail.points.map((point) => point.tensor_parallel_size))].length > 1 && <div className="benchmark-tp-tabs" role="tablist" aria-label="Tensor parallel size">{[...new Set(activeModelDetail.points.map((point) => point.tensor_parallel_size))].map((size) => <button type="button" role="tab" aria-selected={selectedTp === size} key={size} onClick={() => setSelectedTp(size)}>TP {size}</button>)}</div>}
          <div className="benchmark-chart-stack">
            <BenchmarkLineChart title="Prompt throughput" metric="prompt_tokens_per_second" points={activeModelDetail.points.filter((point) => point.tensor_parallel_size === selectedTp)} />
            <BenchmarkLineChart title="Text generation throughput" metric="generation_tokens_per_second" points={activeModelDetail.points.filter((point) => point.tensor_parallel_size === selectedTp)} />
          </div>
          <p className="benchmark-method-note">Each point is the average of completed coordinated runs for the exact model, context window, concurrency, and TP size. Prompt throughput uses measured time to first token; generation throughput uses concurrent batch wall time. Results vary with runtime, thermals, networking, and workload.</p>
        </>}
      </LegalDialog>}
      </>}
      {tab === 'temp' && <TemperatureRuns />}
    </div>
  )
}
