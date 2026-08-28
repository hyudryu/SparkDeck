import { useEffect, useRef, useState } from 'react'
import { Download, Gauge, Play, Square, Trash2 } from 'lucide-react'
import { api } from '../api/client'
import type { BenchyRunSummary } from '../api/types'
import {
  Button,
  EmptyState,
  ErrorState,
  LoadingState,
  PageHeader,
  Panel,
  Status,
  formatRate,
} from '../components/ui'
import { useResource } from '../hooks/useResource'
import { BenchyChart } from '../components/BenchyChart'

function parseNumberList(text: string): { values: number[]; invalid: string[] } {
  const values: number[] = []
  const invalid: string[] = []
  for (const part of text.split(/[\s,]+/)) {
    if (!part) continue
    // Whole positive integers only: parseInt would silently truncate
    // "1.5" to 1 and "2foo" to 2, running a different workload than entered.
    if (!/^[1-9]\d*$/.test(part)) {
      invalid.push(part)
      continue
    }
    const value = Number.parseInt(part, 10)
    if (!values.includes(value)) values.push(value)
  }
  return { values, invalid }
}

function formatSeconds(value?: number | null) {
  if (value === undefined || value === null) return '—'
  if (value < 90) return `${Math.round(value)} s`
  return `${Math.floor(value / 60)} min ${Math.round(value % 60)} s`
}

function configSummary(run: BenchyRunSummary) {
  const config = run.config
  const concurrency = config.concurrency_levels.map((level) => `C${level}`).join(' ')
  const prompt = config.prompt_sizes.map((size) => size.toLocaleString()).join(', ')
  const response = config.response_sizes.map((size) => size.toLocaleString()).join(', ')
  return `${prompt} → ${response} tok · ${concurrency}`
}

const RUN_STATUS_LABEL: Record<string, string> = {
  pending: 'Queued',
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

export function BenchyPage() {
  const status = useResource((signal) => api.benchy.status(signal))
  const installed = status.data?.installed ?? false
  const models = useResource((signal) => api.benchy.models(signal), [installed], installed)
  const runs = useResource((signal) => api.benchy.list(signal))

  const [modelId, setModelId] = useState('')
  const [concurrencyText, setConcurrencyText] = useState('1, 2')
  const [promptText, setPromptText] = useState('2048')
  const [responseText, setResponseText] = useState('128')
  const [runsPerTest, setRunsPerTest] = useState(3)
  const [exactTg, setExactTg] = useState(false)
  const [formError, setFormError] = useState<string>()
  const [starting, setStarting] = useState(false)
  const [installing, setInstalling] = useState(false)
  const [actionError, setActionError] = useState<string>()
  const [selectedRunId, setSelectedRunId] = useState<string>()

  const activeRunId = status.data?.active_run_id
  const activeRun = useResource(
    (signal) => api.benchy.get(activeRunId ?? '', signal),
    [activeRunId],
    Boolean(activeRunId),
  )
  const detail = useResource(
    (signal) => api.benchy.get(selectedRunId ?? '', signal),
    [selectedRunId],
    Boolean(selectedRunId),
  )

  useEffect(() => {
    if (!modelId && models.data?.length) setModelId(models.data[0].id)
  }, [models.data, modelId])

  // Poll the active run while it is in flight; refresh history when the active
  // run changes, including completing (which clears active_run_id).
  const previousActiveRef = useRef<string | undefined>(undefined)
  useEffect(() => {
    if (previousActiveRef.current && previousActiveRef.current !== activeRunId) {
      runs.reload()
      // The selected run may just have become terminal: fetch its results.
      detail.reload()
    }
    previousActiveRef.current = activeRunId ?? undefined
    if (!activeRunId) return
    const timer = window.setInterval(() => {
      activeRun.reload()
      status.reload()
      runs.reload()
      // Keep an open detail of the active run live while it progresses.
      detail.reload()
    }, 2_000)
    return () => window.clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeRunId])

  const install = async () => {
    setInstalling(true)
    setActionError(undefined)
    try {
      status.setData(await api.benchy.install())
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : 'llama-benchy installation failed')
    } finally {
      setInstalling(false)
    }
  }

  const start = async () => {
    const concurrency = parseNumberList(concurrencyText)
    const promptSizes = parseNumberList(promptText)
    const responseSizes = parseNumberList(responseText)
    const invalid = [...promptSizes.invalid, ...responseSizes.invalid, ...concurrency.invalid]
    if (invalid.length) {
      setFormError(`Invalid values: ${invalid.join(', ')}. Use whole numbers, e.g. 2048.`)
      return
    }
    if (!modelId) { setFormError('Select a served model to benchmark.'); return }
    if (!promptSizes.values.length) { setFormError('Enter at least one prompt size.'); return }
    if (!responseSizes.values.length) { setFormError('Enter at least one output token count.'); return }
    if (!concurrency.values.length) { setFormError('Enter at least one concurrency level.'); return }
    setFormError(undefined)
    setActionError(undefined)
    setStarting(true)
    try {
      await api.benchy.start({
        model_id: modelId,
        prompt_sizes: promptSizes.values,
        response_sizes: responseSizes.values,
        concurrency_levels: concurrency.values,
        context_depths: [0],
        runs: Math.min(10, Math.max(1, runsPerTest || 3)),
        warmup_runs: 1,
        exact_tg: exactTg,
      })
      status.reload()
      runs.reload()
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : 'Could not start the benchmark run')
    } finally {
      setStarting(false)
    }
  }

  const cancel = async () => {
    if (!activeRunId) return
    try {
      await api.benchy.cancel(activeRunId)
      activeRun.reload()
      status.reload()
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : 'Could not cancel the run')
    }
  }

  const remove = async (id: string) => {
    await api.benchy.remove(id)
    if (selectedRunId === id) setSelectedRunId(undefined)
    runs.reload()
    status.reload()
  }

  const detailIsStale = detail.data !== undefined && detail.data.id !== selectedRunId
  const activeDetail = detailIsStale ? undefined : detail.data
  const activeData = activeRun.data?.id === activeRunId ? activeRun.data : undefined

  return (
    <div className="page">
      <PageHeader
        eyebrow="Performance measurement"
        title="Benchy"
        description="Run llama-benchy against the models SparkDeck is currently serving, keep the CSV results, and compare prompt processing and generation speed across concurrency levels."
      />

      <Panel className="benchy-tool-panel">
        <div className="benchy-tool-row">
          <span className="benchy-tool-icon"><Gauge size={19} aria-hidden="true" /></span>
          <div className="benchy-tool-info">
            <h2>llama-benchy</h2>
            {status.loading && <p className="muted">Checking for llama-benchy…</p>}
            {status.data && (status.data.installed
              ? <p>Installed{status.data.version ? ` · v${status.data.version}` : ''}{status.data.launch_mode === 'python_module' ? ' · Python module' : ''}</p>
              : <p>Not installed. llama-benchy drives benchmark requests against a running model endpoint and reports llama-bench style statistics. Install it to enable this page.</p>)}
            {status.error && <p className="inline-error">{status.error}</p>}
          </div>
          {status.data && !status.data.installed && (
            <Button variant="primary" disabled={installing} onClick={() => void install()}>
              <Download size={15} aria-hidden="true" /> {installing ? 'Installing…' : 'Install llama-benchy'}
            </Button>
          )}
        </div>
        {actionError && <p className="inline-error" role="alert">{actionError}</p>}
      </Panel>

      {installed && <>
        <div className="section-heading"><div><h2>New benchmark run</h2><p>Pick a served model, then sweep prompt sizes and concurrency levels. Each combination runs {runsPerTest || 3} measured passes after a warm-up.</p></div></div>
        <Panel>
          <div className="field-grid">
            <label className="field" htmlFor="benchy-model">
              <span>Served model</span>
              <select
                id="benchy-model"
                value={modelId}
                onChange={(event) => setModelId(event.target.value)}
                disabled={models.loading || !models.data?.length || Boolean(activeRunId)}
              >
                {!models.data?.length && <option value="">{models.loading ? 'Loading served models…' : 'No models currently served'}</option>}
                {models.data?.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.label}{model.quantization ? ` · ${model.quantization}` : ''}{model.runtime ? ` · ${model.runtime}` : ''}
                  </option>
                ))}
              </select>
              <small>{models.data?.length
                ? 'Benchmarks run against the model endpoint over HTTP.'
                : 'Load a model on the Models page first — the list fills with currently served models.'}</small>
            </label>
            <label className="field" htmlFor="benchy-concurrency">
              <span>Concurrency levels</span>
              <input
                id="benchy-concurrency"
                inputMode="numeric"
                value={concurrencyText}
                disabled={Boolean(activeRunId)}
                onChange={(event) => setConcurrencyText(event.target.value)}
              />
              <small>Concurrent requests per test, e.g. “1, 2, 4”.</small>
            </label>
            <label className="field" htmlFor="benchy-prompt">
              <span>Prompt sizes (context)</span>
              <input
                id="benchy-prompt"
                inputMode="numeric"
                value={promptText}
                disabled={Boolean(activeRunId)}
                onChange={(event) => setPromptText(event.target.value)}
              />
              <small>Prompt token counts to sweep, e.g. “512, 2048”.</small>
            </label>
            <label className="field" htmlFor="benchy-response">
              <span>Output tokens</span>
              <input
                id="benchy-response"
                inputMode="numeric"
                value={responseText}
                disabled={Boolean(activeRunId)}
                onChange={(event) => setResponseText(event.target.value)}
              />
              <small>Generated tokens per test, e.g. “128”.</small>
            </label>
            <label className="field" htmlFor="benchy-runs">
              <span>Runs per test</span>
              <input
                id="benchy-runs"
                type="number"
                min={1}
                max={10}
                value={runsPerTest}
                disabled={Boolean(activeRunId)}
                onChange={(event) => setRunsPerTest(Number.parseInt(event.target.value, 10) || 1)}
              />
              <small>Measured passes per combination (1–10).</small>
            </label>
            <label className="field benchy-checkbox-field" htmlFor="benchy-exact-tg">
              <span>Force exact output length</span>
              <span className="benchy-checkbox">
                <input
                  id="benchy-exact-tg"
                  type="checkbox"
                  checked={exactTg}
                  disabled={Boolean(activeRunId)}
                  onChange={(event) => setExactTg(event.target.checked)}
                />
                <small>Request exactly the configured output tokens (ignore_eos).</small>
              </span>
            </label>
          </div>
          {formError && <p className="inline-error" role="alert">{formError}</p>}
          <div className="benchy-run-actions">
            <Button variant="primary" disabled={starting || Boolean(activeRunId)} onClick={() => void start()}>
              <Play size={15} aria-hidden="true" /> {activeRunId ? 'Benchmark running…' : 'Start benchmark'}
            </Button>
          </div>
        </Panel>

        {activeRunId && (activeRun.loading || activeData) && (
          <Panel className="benchy-active-panel" aria-live="polite">
            <div className="benchy-active-head">
              <Status status="running">{activeData ? 'Benchmark running' : 'Benchmark queued'}</Status>
              <Button variant="secondary" onClick={() => void cancel()}><Square size={14} aria-hidden="true" /> Cancel</Button>
            </div>
            {activeData && <>
              <p className="benchy-active-shape">
                {activeData.progress?.current
                  ? <>Testing {activeData.progress.current.prompt_size?.toLocaleString()} → {activeData.progress.current.response_size} tok at C{activeData.progress.current.concurrency}</>
                  : 'Starting benchmark…'}
              </p>
              <p className="muted">
                {activeData.progress?.requests_done ?? 0} requests completed
                {(activeData.progress?.requests_failed ?? 0) > 0 && <> · {activeData.progress?.requests_failed} failed</>}
              </p>
            </>}
          </Panel>
        )}
      </>}

      <div className="section-heading"><div><h2>Output history</h2><p>Select a completed run to chart prompt processing and generation speed per concurrency level.</p></div></div>
      {runs.loading && <LoadingState label="Loading benchmark runs" />}
      {runs.error && <ErrorState message={runs.error} onRetry={runs.reload} />}
      {!runs.loading && !runs.error && runs.data?.length === 0 && (
        <EmptyState
          title="No benchmark runs yet"
          description="Start a run above — results are stored as CSV with the model name, quantization, and full configuration."
        />
      )}
      {runs.data && runs.data.length > 0 && (
        <Panel className="table-panel"><div className="responsive-table benchy-history-table" role="table" aria-label="Benchmark run history">
          <div className="table-row table-header" role="row">
            <div role="columnheader">Run</div><div role="columnheader">Model</div>
            <div role="columnheader">Quantization</div><div role="columnheader">Configuration</div>
            <div role="columnheader">Results</div><div role="columnheader">Duration</div>
            <div role="columnheader">Status</div><div role="columnheader">Actions</div>
          </div>
          {runs.data.map((run) => (
            <div
              key={run.id}
              className={`table-row benchy-run-row ${selectedRunId === run.id ? 'benchy-run-row-selected' : ''}`}
              role="row"
              tabIndex={0}
              aria-pressed={selectedRunId === run.id}
              onClick={() => setSelectedRunId(selectedRunId === run.id ? undefined : run.id)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault()
                  setSelectedRunId(selectedRunId === run.id ? undefined : run.id)
                }
              }}
            >
              <div role="cell" data-label="Run"><strong>{new Date(run.created_at).toLocaleString()}</strong></div>
              <div role="cell" data-label="Model">{run.model}</div>
              <div role="cell" data-label="Quantization">{run.quantization ?? 'Default precision'}</div>
              <div role="cell" data-label="Configuration"><small>{configSummary(run)}</small></div>
              <div role="cell" data-label="Results">{run.result_count ?? 0} rows</div>
              <div role="cell" data-label="Duration">{formatSeconds(run.duration_seconds)}</div>
              <div role="cell" data-label="Status">
                <Status status={run.status === 'completed' ? 'completed' : run.status}>
                  {RUN_STATUS_LABEL[run.status] ?? run.status}
                </Status>
              </div>
              <div role="cell" data-label="Actions" className="benchy-run-actions-cell">
                {run.status === 'completed' && (
                  <a
                    className="icon-button"
                    href={api.benchy.csvUrl(run.id)}
                    download={`benchy-${run.id}.csv`}
                    aria-label={`Download CSV for run ${run.id}`}
                    onClick={(event) => event.stopPropagation()}
                  ><Download size={15} /></a>
                )}
                {run.status !== 'running' && run.status !== 'pending' && (
                  <Button
                    variant="tertiary"
                    aria-label={`Delete run ${run.id}`}
                    onClick={(event) => { event.stopPropagation(); void remove(run.id) }}
                  ><Trash2 size={15} /></Button>
                )}
              </div>
            </div>
          ))}
        </div></Panel>
      )}

      {selectedRunId && detail.loading && <LoadingState label="Loading run results" />}
      {selectedRunId && detail.error && <ErrorState message={detail.error} onRetry={detail.reload} />}
      {activeDetail && (
        <section className="benchy-detail" aria-label={`Results for run ${activeDetail.id}`}>
          <div className="benchy-detail-head">
            <div>
              <h3>{activeDetail.model}</h3>
              <p className="muted">
                {activeDetail.quantization ?? 'Default precision'}
                {activeDetail.runtime ? ` · ${activeDetail.runtime}` : ''}
                {activeDetail.benchy_version ? ` · llama-benchy v${activeDetail.benchy_version}` : ''}
                {' · '}{configSummary(activeDetail)}
                {activeDetail.report?.latency_mode ? ` · latency mode ${activeDetail.report.latency_mode}` : ''}
              </p>
            </div>
            {activeDetail.csv_filename && (
              <a className="button button-secondary" href={api.benchy.csvUrl(activeDetail.id)} download={`benchy-${activeDetail.id}.csv`}>
                <Download size={15} aria-hidden="true" /> Download CSV
              </a>
            )}
          </div>
          {activeDetail.error && <p className="inline-error" role="alert">{activeDetail.error}</p>}
          {activeDetail.results.length === 0 && <p className="muted">This run produced no measurements.</p>}
          {activeDetail.results.length > 0 && <>
            <div className="benchmark-chart-stack">
              <BenchyChart
                title="Processing speed (generation)"
                subtitle="Output tokens per second, all concurrent requests combined."
                rows={activeDetail.results}
                metric="tg_tokens_per_second"
              />
              <BenchyChart
                title="Prompt processing speed"
                subtitle="Prompt tokens per second per concurrency level."
                rows={activeDetail.results}
                metric="pp_tokens_per_second"
              />
              <BenchyChart
                title="Tokens per output"
                subtitle="Generation speed per request at each concurrency level."
                rows={activeDetail.results}
                metric="tg_tokens_per_second_request"
              />
            </div>
            <Panel className="table-panel"><div className="responsive-table benchy-measure-table" role="table" aria-label="Run measurements">
              <div className="table-row table-header" role="row">
                <div role="columnheader">Prompt</div><div role="columnheader">Output</div>
                <div role="columnheader">Concurrency</div><div role="columnheader">Prompt tok/s</div>
                <div role="columnheader">Generation tok/s</div><div role="columnheader">Per-request tok/s</div>
                <div role="columnheader">TTFR</div>
              </div>
              {activeDetail.results.map((row, index) => (
                <div className="table-row" role="row" key={index}>
                  <div role="cell" data-label="Prompt">{row.prompt_size?.toLocaleString()}</div>
                  <div role="cell" data-label="Output">{row.response_size}</div>
                  <div role="cell" data-label="Concurrency">C{row.concurrency}</div>
                  <div role="cell" data-label="Prompt tok/s">{formatRate(row.pp_tokens_per_second ?? undefined)}</div>
                  <div role="cell" data-label="Generation tok/s">{formatRate(row.tg_tokens_per_second ?? undefined)}</div>
                  <div role="cell" data-label="Per-request tok/s">{formatRate(row.tg_tokens_per_second_request ?? undefined)}</div>
                  <div role="cell" data-label="TTFR">{row.ttfr_ms ? `${Math.round(row.ttfr_ms)} ms` : '—'}</div>
                </div>
              ))}
            </div></Panel>
          </>}
        </section>
      )}
    </div>
  )
}
