import { useCallback, useEffect, useRef, useState } from 'react'
import { BarChart3, Check, ChevronRight, CloudOff, RotateCw, ShieldCheck, Trash2, UploadCloud } from 'lucide-react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { Button, EmptyState, ErrorState, formatDuration, formatRate, LoadingState, PageHeader, Panel, RuntimeMark, Status } from '../components/ui'
import { useResource } from '../hooks/useResource'
import { communityAccessHint, useCommunityAccess } from '../hooks/useCommunityAccess'
import { BenchmarkLineChart } from '../components/BenchmarkLineChart'
import { LegalDialog } from '../components/LegalDialog'

export function BenchmarksPage() {
  const samples = useResource((signal) => api.benchmarks.list(signal))
  const benchmarkModels = useResource((signal) => api.benchmarks.models(signal))
  const sync = useResource((signal) => api.benchmarks.syncStatus(signal))
  const communityAccess = useCommunityAccess()
  const accessHint = communityAccessHint(communityAccess.signedIn)
  const aggregates = useResource(
    (signal) => api.benchmarks.aggregates(signal),
    [communityAccess.enabled],
    communityAccess.enabled,
  )
  const [syncBusy, setSyncBusy] = useState(false)
  const [syncActionError, setSyncActionError] = useState<string>()
  const [reviewingConsent, setReviewingConsent] = useState(false)
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

  const applyConsent = async (enabled: boolean) => {
    if (!sync.data) return
    setSyncBusy(true)
    setSyncActionError(undefined)
    try {
      const updated = await api.benchmarks.setConsent(enabled)
      sync.setData(updated)
      communityAccess.reload()
      setReviewingConsent(false)
      if (updated.cluster_errors?.length) {
        setSyncActionError(`Sharing was updated on this controller, but not on: ${updated.cluster_errors.join('; ')}. Retry when those nodes are reachable.`)
      }
    } catch (reason) {
      setSyncActionError(reason instanceof Error ? reason.message : 'Could not update community sharing')
    } finally {
      setSyncBusy(false)
    }
  }

  const toggleSharing = () => applyConsent(!sync.data?.sharing_enabled)

  const retry = async () => {
    setSyncBusy(true)
    try {
      sync.setData(await api.benchmarks.retry())
    } finally {
      setSyncBusy(false)
    }
  }

  const remove = async (id: string) => {
    await api.benchmarks.deleteLocal(id)
    samples.reload()
    benchmarkModels.reload()
    aggregates.reload()
    sync.reload()
  }

  return (
    <div className="page">
      <PageHeader eyebrow="Performance evidence" title="Benchmarks" description="Review measurements captured from SparkDeck requests and compare them with privacy-preserving community results." />
      <div className="benchmark-summary-grid">
        <Panel className="sync-panel">
          <div className="sync-heading"><span className="sync-icon"><UploadCloud size={19} /></span><div><h2>Community sharing</h2><p>Optional, account-linked benchmark evidence with a strict data allowlist.</p></div></div>
          {sync.loading && <p className="muted">Checking sync status…</p>}
          {sync.error && <p className="inline-error">{sync.error}</p>}
          {syncActionError && <p className="inline-error" role="alert">{syncActionError}</p>}
          {sync.data && <>
            <div className="sync-state"><Status status={sync.data.sharing_enabled ? (sync.data.account_paired && sync.data.upload_configured ? 'running' : 'waiting') : 'stopped'}>{sync.data.sharing_enabled ? (sync.data.account_paired ? (sync.data.upload_configured ? 'Sharing enabled' : 'Queued locally') : 'Waiting for account') : 'Sharing off'}</Status><span>{sync.data.pending_count} pending · {sync.data.synced_count} synced</span></div>
            <div className="sync-actions"><Button variant={sync.data.sharing_enabled ? 'secondary' : 'primary'} disabled={syncBusy} onClick={() => sync.data?.sharing_enabled ? void toggleSharing() : setReviewingConsent(true)}>{sync.data.sharing_enabled ? <><CloudOff size={15} /> Turn off</> : <><ShieldCheck size={15} /> Review & enable</>}</Button>{syncActionError && !sync.data.sharing_enabled && <Button variant="secondary" disabled={syncBusy} onClick={() => void applyConsent(false)}>Retry turn off everywhere</Button>}{sync.data.failed_count > 0 && <Button disabled={syncBusy} onClick={() => void retry()}><RotateCw size={15} /> Retry {sync.data.failed_count}</Button>}</div>
          </>}
        </Panel>
        <Panel className="privacy-panel">
          <p className="eyebrow">Always private</p>
          <h2>Your content stays local</h2>
          <p>Prompts and outputs never enter benchmark JSON. Ordinary authenticated request and network metadata may still be processed to operate the service.</p>
          <span><Check size={15} /> Sharing is off until you opt in</span>
        </Panel>
      </div>

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

      <div className="section-heading" title={communityAccess.enabled ? undefined : accessHint}><div><h2>{localAggregates ? 'Local aggregate estimates' : 'Community estimates'}</h2><p>Evidence is matched only by exact model name, quantization, and prompt-length bucket. Results are estimates, not guarantees.</p></div></div>
      {!communityAccess.enabled && !communityAccess.loading && <EmptyState
        title="Community estimates are locked"
        description={accessHint}
        action={communityAccess.signedIn
          ? <Button variant="secondary" onClick={() => setReviewingConsent(true)}>Review sharing</Button>
          : <Link className="button button-secondary" to="/settings">Open community settings</Link>}
      />}
      {communityAccess.enabled && aggregates.loading && <LoadingState label="Loading community aggregates" />}
      {communityAccess.enabled && aggregates.error && <ErrorState message={aggregates.error} onRetry={aggregates.reload} />}
      {communityAccess.enabled && !aggregates.loading && !aggregates.error && aggregateResponse?.availability === 'unavailable' && <EmptyState
        title="Community service unavailable"
        description="The hosted community service could not be reached. Your local benchmarks are unaffected — try again later."
        action={<Button variant="secondary" onClick={aggregates.reload}>Retry</Button>}
      />}
      {communityAccess.enabled && !aggregates.loading && !aggregates.error && aggregateResponse?.availability !== 'unavailable' && aggregateResponse?.items.length === 0 && <EmptyState title="No community estimates yet" description="Estimates will appear when enough samples share the same model name, quantization, and prompt-length bucket." />}
      {communityAccess.enabled && aggregateResponse && aggregateResponse.items.length > 0 && <div className="aggregate-grid">{aggregateResponse.items.map((item) => (
        <Panel className="aggregate-item" key={`${item.model_id}-${item.quantization}-${item.prompt_tokens_bucket}`}>
          <div><div><p className="aggregate-model">{item.model_id}</p><small className="aggregate-quantization">{item.quantization}</small></div><span className="estimate-label">{localAggregates ? 'Local estimate' : 'Community estimate'}</span></div>
          <dl><div><dt>Inference speed</dt><dd>{formatRate(item.inference_tokens_per_second)}</dd></div><div><dt>Prompt-length bucket</dt><dd>{item.prompt_tokens_bucket.toLocaleString()} tokens</dd></div><div><dt>Evidence</dt><dd>{item.sample_count} samples</dd></div></dl>
          {item.sample_count >= aggregateResponse.evidence_policy.minimum_samples ? <span className="proven"><Check size={14} /> Evidence threshold met</span> : <span className="muted">Collecting more evidence</span>}
        </Panel>
      ))}</div>}
      {communityAccess.enabled && aggregateResponse && <p className="aggregate-policy">Evidence threshold: {aggregateResponse.evidence_policy.minimum_samples} samples, matched only on model name, quantization, and prompt-length bucket. Inference speed is {localAggregates ? 'aggregated from this controller' : 'a community estimate'} and may differ on your system.</p>}

      <div className="section-heading"><div><h2>Local history</h2><p>Successful proxied runs are captured automatically.</p></div></div>
      {samples.loading && <LoadingState label="Loading benchmark history" />}
      {samples.error && <ErrorState message={samples.error} onRetry={samples.reload} />}
      {!samples.loading && !samples.error && samples.data?.length === 0 && <EmptyState title="No benchmark samples yet" description="Chat with or compare a running model to capture your first measurement." />}
      {samples.data && samples.data.length > 0 && <Panel className="table-panel"><div className="responsive-table benchmark-table" role="table" aria-label="Local benchmark history">
        <div className="table-row table-header" role="row"><span role="columnheader">Model</span><span role="columnheader">Runtime</span><span role="columnheader">Speed</span><span role="columnheader">TTFT</span><span role="columnheader">Sync</span><span role="columnheader">Actions</span></div>
        {samples.data.map((sample) => <div className="table-row" role="row" tabIndex={0} key={sample.id}>
          <div role="cell" data-label="Model"><strong>{sample.model_id}</strong><small>{new Date(sample.created_at).toLocaleString()}</small></div>
          <div role="cell" data-label="Runtime"><RuntimeMark runtime={sample.runtime} /><small>{sample.quantization ?? 'Default precision'}</small></div>
          <div role="cell" data-label="Speed"><strong>{formatRate(sample.tokens_per_second)}</strong><small>{sample.output_tokens ?? '—'} output tokens</small></div>
          <div role="cell" data-label="TTFT">{formatDuration(sample.ttft_ms)}</div>
          <div role="cell" data-label="Sync"><Status status={sample.sync_state ?? 'local'} /></div>
          <div role="cell" data-label="Actions"><Button variant="tertiary" aria-label={`Delete benchmark for ${sample.model_id}`} onClick={() => void remove(sample.id)}><Trash2 size={15} /></Button></div>
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
      {reviewingConsent && <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && setReviewingConsent(false)}><section className="modal" role="dialog" aria-modal="true" aria-labelledby="sharing-review-title"><div className="modal-heading"><div><p className="eyebrow">Privacy review</p><h2 id="sharing-review-title">Enable community sharing?</h2></div><button className="icon-button" onClick={() => setReviewingConsent(false)} aria-label="Close dialog">×</button></div><p><strong>Benchmark JSON:</strong> canonical model identifier, quantization, prompt-length/context-occupancy bucket, measured inference tok/s, concurrency when recorded, and a stable opaque telemetry cluster ID used to count unique contributing clusters.</p><p><strong>Eligibility:</strong> only samples captured after you enable sharing can be uploaded; existing benchmark history stays local.</p><p><strong>The opaque ID:</strong> is randomly generated and contains no account ID, hostname, node name, or endpoint alias.</p><p><strong>Never in benchmark JSON:</strong> prompts or outputs, runtime, revision, hardware, settings, account email, paths, or endpoint aliases. The service still receives ordinary authenticated request and network metadata.</p><div className="modal-actions"><Button onClick={() => setReviewingConsent(false)}>Keep sharing off</Button><Button variant="primary" disabled={syncBusy} onClick={() => void toggleSharing()}><ShieldCheck size={15} /> I understand, enable sharing</Button></div></section></div>}
    </div>
  )
}
