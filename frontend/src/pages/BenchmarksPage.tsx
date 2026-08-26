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

  const toggleSharing = async () => {
    if (!sync.data) return
    setSyncBusy(true)
    try {
      sync.setData(await api.benchmarks.setConsent(!sync.data.sharing_enabled))
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
    sync.reload()
  }

  return (
    <div className="page">
      <PageHeader eyebrow="Performance evidence" title="Benchmarks" description="Review measurements captured from SparkDeck requests and compare them with privacy-preserving community results." />
      <div className="benchmark-summary-grid">
        <Panel className="sync-panel">
          <div className="sync-heading"><span className="sync-icon"><UploadCloud size={19} /></span><div><h2>Community sharing</h2><p>Upload timing, model, runtime, and hardware metadata automatically.</p></div></div>
          {sync.loading && <p className="muted">Checking sync status…</p>}
          {sync.error && <p className="inline-error">{sync.error}</p>}
          {sync.data && <>
            <div className="sync-state"><Status status={sync.data.sharing_enabled ? (sync.data.account_paired ? 'running' : 'waiting') : 'stopped'}>{sync.data.sharing_enabled ? (sync.data.account_paired ? 'Sharing enabled' : 'Waiting for account') : 'Sharing off'}</Status><span>{sync.data.pending_count} pending · {sync.data.synced_count} synced</span></div>
            <div className="sync-actions"><Button variant={sync.data.sharing_enabled ? 'secondary' : 'primary'} disabled={syncBusy} onClick={() => void toggleSharing()}>{sync.data.sharing_enabled ? <><CloudOff size={15} /> Turn off</> : <><ShieldCheck size={15} /> Review & enable</>}</Button>{sync.data.failed_count > 0 && <Button disabled={syncBusy} onClick={() => void retry()}><RotateCw size={15} /> Retry {sync.data.failed_count}</Button>}</div>
          </>}
        </Panel>
        <Panel className="privacy-panel">
          <p className="eyebrow">Always private</p>
          <h2>Your content stays local</h2>
          <p>Prompts, responses, endpoints, hostnames, paths, and credentials are never recorded or uploaded.</p>
          <span><Check size={15} /> Sharing is off until you opt in</span>
        </Panel>
      </div>

      <div className="section-heading"><div><h2>Community comparison</h2><p>Matching model, runtime, hardware, quantization, and context.</p></div></div>
      {aggregates.loading && <LoadingState label="Loading community aggregates" />}
      {aggregates.error && <ErrorState message={aggregates.error} onRetry={aggregates.reload} />}
      {!aggregates.loading && !aggregates.error && aggregates.data?.length === 0 && <EmptyState title="No matching community results" description="Community comparisons will appear as eligible samples are contributed." />}
      {aggregates.data && aggregates.data.length > 0 && <div className="aggregate-grid">{aggregates.data.map((item, index) => (
        <Panel className="aggregate-item" key={`${item.model_id}-${item.runtime}-${index}`}>
          <div><p className="aggregate-model">{item.model_id}</p><RuntimeMark runtime={item.runtime} /></div>
          <dl><div><dt>Median speed</dt><dd>{formatRate(item.median_tokens_per_second)}</dd></div><div><dt>Median TTFT</dt><dd>{formatDuration(item.median_ttft_ms)}</dd></div><div><dt>Evidence</dt><dd>{item.sample_count} runs · {item.distinct_device_count} devices</dd></div></dl>
          {item.community_proven ? <span className="proven"><Check size={14} /> Community proven</span> : <span className="muted">Collecting more evidence</span>}
        </Panel>
      ))}</div>}

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
    </div>
  )
}
