import { useState } from 'react'
import { Check, CloudOff, RotateCw, ShieldCheck, Trash2, UploadCloud } from 'lucide-react'
import { api } from '../api/client'
import { Button, EmptyState, ErrorState, formatDuration, formatRate, LoadingState, PageHeader, Panel, RuntimeMark, Status } from '../components/ui'
import { useResource } from '../hooks/useResource'

export function BenchmarksPage() {
  const samples = useResource((signal) => api.benchmarks.list(signal))
  const aggregates = useResource((signal) => api.benchmarks.aggregates(signal))
  const sync = useResource((signal) => api.benchmarks.syncStatus(signal))
  const [syncBusy, setSyncBusy] = useState(false)
  const [reviewingConsent, setReviewingConsent] = useState(false)
  const aggregateResponse = aggregates.data
  const localAggregates = aggregateResponse?.availability === 'local'

  const toggleSharing = async () => {
    if (!sync.data) return
    setSyncBusy(true)
    try {
      sync.setData(await api.benchmarks.setConsent(!sync.data.sharing_enabled))
      setReviewingConsent(false)
    } finally {
      setSyncBusy(false)
    }
  }

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
    aggregates.reload()
    sync.reload()
  }

  return (
    <div className="page">
      <PageHeader eyebrow="Performance evidence" title="Benchmarks" description="Review measurements captured from SparkDeck requests and compare them with privacy-preserving community results." />
      <div className="benchmark-summary-grid">
        <Panel className="sync-panel">
          <div className="sync-heading"><span className="sync-icon"><UploadCloud size={19} /></span><div><h2>Community sharing</h2><p>Share only model name, context window size, and measured inference tok/s.</p></div></div>
          {sync.loading && <p className="muted">Checking sync status…</p>}
          {sync.error && <p className="inline-error">{sync.error}</p>}
          {sync.data && <>
            <div className="sync-state"><Status status={sync.data.sharing_enabled ? (sync.data.account_paired ? 'running' : 'waiting') : 'stopped'}>{sync.data.sharing_enabled ? (sync.data.account_paired ? 'Sharing enabled' : 'Waiting for account') : 'Sharing off'}</Status><span>{sync.data.pending_count} pending · {sync.data.synced_count} synced</span></div>
            <div className="sync-actions"><Button variant={sync.data.sharing_enabled ? 'secondary' : 'primary'} disabled={syncBusy} onClick={() => sync.data?.sharing_enabled ? void toggleSharing() : setReviewingConsent(true)}>{sync.data.sharing_enabled ? <><CloudOff size={15} /> Turn off</> : <><ShieldCheck size={15} /> Review & enable</>}</Button>{sync.data.failed_count > 0 && <Button disabled={syncBusy} onClick={() => void retry()}><RotateCw size={15} /> Retry {sync.data.failed_count}</Button>}</div>
          </>}
        </Panel>
        <Panel className="privacy-panel">
          <p className="eyebrow">Always private</p>
          <h2>Your content stays local</h2>
          <p>Prompts and outputs, runtime, revision, quantization, hardware, settings, host or network identity, and paths are not shared.</p>
          <span><Check size={15} /> Sharing is off until you opt in</span>
        </Panel>
      </div>

      <div className="section-heading"><div><h2>{localAggregates ? 'Local aggregate estimates' : 'Community estimates'}</h2><p>Evidence is matched only by exact model name and context window. Results are estimates, not guarantees.</p></div></div>
      {aggregates.loading && <LoadingState label="Loading community aggregates" />}
      {aggregates.error && <ErrorState message={aggregates.error} onRetry={aggregates.reload} />}
      {!aggregates.loading && !aggregates.error && aggregateResponse?.items.length === 0 && <EmptyState title="No community estimates yet" description="Estimates will appear when enough samples share the same model name and context window." />}
      {aggregateResponse && aggregateResponse.items.length > 0 && <div className="aggregate-grid">{aggregateResponse.items.map((item) => (
        <Panel className="aggregate-item" key={`${item.model_id}-${item.context_window_size}`}>
          <div><p className="aggregate-model">{item.model_id}</p><span className="estimate-label">{localAggregates ? 'Local estimate' : 'Community estimate'}</span></div>
          <dl><div><dt>Inference speed</dt><dd>{formatRate(item.inference_tokens_per_second)}</dd></div><div><dt>Context window</dt><dd>{item.context_window_size.toLocaleString()} tokens</dd></div><div><dt>Evidence</dt><dd>{item.sample_count} samples</dd></div></dl>
          {item.sample_count >= aggregateResponse.evidence_policy.minimum_samples ? <span className="proven"><Check size={14} /> Evidence threshold met</span> : <span className="muted">Collecting more evidence</span>}
        </Panel>
      ))}</div>}
      {aggregateResponse && <p className="aggregate-policy">Evidence threshold: {aggregateResponse.evidence_policy.minimum_samples} samples, matched only on model name and context window. Inference speed is {localAggregates ? 'aggregated from this controller' : 'a community estimate'} and may differ on your system.</p>}

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
      {reviewingConsent && <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && setReviewingConsent(false)}><section className="modal" role="dialog" aria-modal="true" aria-labelledby="sharing-review-title"><div className="modal-heading"><div><p className="eyebrow">Privacy review</p><h2 id="sharing-review-title">Enable community sharing?</h2></div><button className="icon-button" onClick={() => setReviewingConsent(false)} aria-label="Close dialog">×</button></div><p><strong>Only shared:</strong> model name, context window size, and measured inference tok/s.</p><p><strong>Not shared:</strong> prompts or outputs, runtime, revision, quantization, hardware, settings, host or network identity, or paths.</p><div className="modal-actions"><Button onClick={() => setReviewingConsent(false)}>Keep sharing off</Button><Button variant="primary" disabled={syncBusy} onClick={() => void toggleSharing()}><ShieldCheck size={15} /> I understand, enable sharing</Button></div></section></div>}
    </div>
  )
}
