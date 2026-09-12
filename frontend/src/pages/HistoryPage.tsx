import { useEffect, useState } from 'react'
import { RefreshCw } from 'lucide-react'
import { api } from '../api/client'
import type { LiveHistorySeries } from '../api/types'
import { HistoryChart, type HistoryMetric } from '../components/HistoryChart'
import { Button, EmptyState, ErrorState, LoadingState, PageHeader, Panel, Status } from '../components/ui'
import { useResource } from '../hooks/useResource'
import {
  MAX_HISTORY_SAMPLE_SECONDS,
  MIN_HISTORY_SAMPLE_SECONDS,
  clampHistorySampleSeconds,
} from '../utils/historySettings'

// The panel polls on a short fixed beat so a changed interval takes effect
// promptly; the chart's own resolution is the sampling interval, not this.
const POLL_SECONDS = 3

const RANGES = [
  { label: '5 min', seconds: 300 },
  { label: '15 min', seconds: 900 },
  { label: '30 min', seconds: 1_800 },
  { label: '1 hour', seconds: 3_600 },
] as const

const INTERVALS = [1, 2, 5, 10, 15, 30] as const

const METRICS: Array<{ id: HistoryMetric; label: string }> = [
  { id: 'output', label: 'Token generation' },
  { id: 'thinking', label: 'Thinking' },
  { id: 'prefill', label: 'Prompt processing' },
]

function seriesTitle(series: LiveHistorySeries): string {
  return series.model || series.group_id || series.key
}

function seriesSubtitle(series: LiveHistorySeries): string {
  const nodes = series.node_names.length > 0 ? series.node_names.join(' + ') : 'This node'
  return `${nodes}${series.instance_id === null || series.instance_id === undefined ? '' : ` · instance ${series.instance_id}`}`
}

export function HistoryPage() {
  const [rangeSeconds, setRangeSeconds] = useState<number>(RANGES[2].seconds)
  const [metrics, setMetrics] = useState<ReadonlySet<HistoryMetric>>(() => new Set(METRICS.map((metric) => metric.id)))
  const [colorByConcurrency, setColorByConcurrency] = useState(true)
  const [savingSettings, setSavingSettings] = useState(false)
  const [settingsError, setSettingsError] = useState<string>()
  const resource = useResource((signal) => api.liveHistory.get(signal))
  const snapshot = resource.data
  const interval = snapshot?.sample_seconds ?? MIN_HISTORY_SAMPLE_SECONDS
  const enabled = snapshot?.enabled !== false

  useEffect(() => {
    if (resource.loading || resource.error) return
    const timer = window.setTimeout(resource.reload, POLL_SECONDS * 1_000)
    return () => window.clearTimeout(timer)
  }, [resource.error, resource.loading, resource.reload, snapshot])

  const applySettings = async (change: { enabled?: boolean; sample_seconds?: number }) => {
    setSavingSettings(true)
    setSettingsError(undefined)
    try {
      // The response is the new snapshot, so the panel reflects the change
      // without waiting for the next poll.
      resource.apply(await api.liveHistory.updateSettings(change))
    } catch (reason) {
      setSettingsError(reason instanceof Error ? reason.message : 'Could not save the history setting')
    } finally {
      setSavingSettings(false)
    }
  }

  const toggleMetric = (metric: HistoryMetric) => {
    setMetrics((current) => {
      const next = new Set(current)
      if (next.has(metric)) {
        // Keep at least one line: an empty graph is never a useful state.
        if (next.size > 1) next.delete(metric)
      } else {
        next.add(metric)
      }
      return next
    })
  }

  const series: LiveHistorySeries[] = enabled ? snapshot?.series ?? [] : []
  const active = series.filter((item) => item.live_sessions > 0).length
  const totalSessions = series.reduce((sum, item) => sum + item.live_sessions, 0)

  return (
    <div className="page">
      <PageHeader
        eyebrow="Live throughput"
        title="History"
        description="One graph per deployment, pair, or serving group. Each line trails the last hour of token throughput, coloured by how many sessions were running."
        actions={<Button type="button" onClick={resource.reload} disabled={resource.loading}>
          <RefreshCw size={16} /> Refresh
        </Button>}
      />

      <Panel className="history-controls-panel">
        <div className="history-controls">
          <div className="history-control-group">
            <span className="history-control-label">Recording</span>
            <div className="history-toggle" role="group" aria-label="History recording">
              <button
                type="button"
                aria-pressed={enabled}
                disabled={savingSettings || !snapshot}
                onClick={() => void applySettings({ enabled: true })}
              >On</button>
              <button
                type="button"
                aria-pressed={!enabled}
                disabled={savingSettings || !snapshot}
                onClick={() => void applySettings({ enabled: false })}
              >Off</button>
            </div>
          </div>
          <div className="history-control-group">
            <span className="history-control-label">Sample every</span>
            <label className="history-interval">
              <input
                aria-label="History sampling interval in seconds"
                type="number"
                min={MIN_HISTORY_SAMPLE_SECONDS}
                max={MAX_HISTORY_SAMPLE_SECONDS}
                step="1"
                inputMode="numeric"
                list="history-interval-presets"
                disabled={savingSettings || !snapshot || !enabled}
                value={interval}
                onChange={(event) => {
                  // Commit only a usable value: every keystroke must not rewrite
                  // the saved cadence, and 1-30 is the supported range.
                  if (Number.isNaN(event.target.valueAsNumber)) return
                  const next = clampHistorySampleSeconds(event.target.valueAsNumber)
                  if (next !== interval) void applySettings({ sample_seconds: next })
                }}
              />
              <datalist id="history-interval-presets">
                {INTERVALS.map((value) => <option key={value} value={value} />)}
              </datalist>
              <span className="history-interval-hint" aria-hidden="true">seconds, 1–{MAX_HISTORY_SAMPLE_SECONDS}</span>
            </label>
          </div>
          <div className="history-control-group">
            <span className="history-control-label">Window</span>
            <div className="history-toggle" role="group" aria-label="History window">
              {RANGES.map((range) => (
                <button
                  key={range.seconds}
                  type="button"
                  aria-pressed={rangeSeconds === range.seconds}
                  onClick={() => setRangeSeconds(range.seconds)}
                >{range.label}</button>
              ))}
            </div>
          </div>
          <div className="history-control-group">
            <span className="history-control-label">Lines</span>
            <div className="history-toggle" role="group" aria-label="History lines">
              {METRICS.map((metric) => (
                <button
                  key={metric.id}
                  type="button"
                  aria-pressed={metrics.has(metric.id)}
                  onClick={() => toggleMetric(metric.id)}
                >{metric.label}</button>
              ))}
            </div>
          </div>
          <div className="history-control-group">
            <span className="history-control-label">Colour</span>
            <div className="history-toggle" role="group" aria-label="History line colour">
              <button type="button" aria-pressed={colorByConcurrency} onClick={() => setColorByConcurrency(true)}>Concurrent sessions</button>
              <button type="button" aria-pressed={!colorByConcurrency} onClick={() => setColorByConcurrency(false)}>Metric</button>
            </div>
          </div>
          <div className="history-summary">
            <Status status={active > 0 ? 'running' : 'stopped'}>{active > 0 ? 'Serving' : 'Idle'}</Status>
            <span>{series.length} graph{series.length === 1 ? '' : 's'} · {totalSessions} live session{totalSessions === 1 ? '' : 's'}</span>
          </div>
        </div>
        {settingsError && <p className="form-error" role="alert">{settingsError}</p>}
        <p className="history-note">
          Output and thinking use the left axis and describe tokens a client is receiving. Prompt processing uses the
          right axis: a completed prefill reports the rate the engine measured, and a prefill still running reports the
          best live estimate from the prompt tokens it holds, marked <em>estimated</em> in the card. Hovering any point
          also shows the prompt size and the seconds its prompt-processing rate was computed from. A blank segment means
          no prefill has been measured in that point yet. Recording off stops all sampling; the sampling interval is also
          the span of one graph point.
        </p>
      </Panel>

      {resource.error && resource.data && <p className="history-stale" role="status">Refresh paused: {resource.error}</p>}

      {!resource.data && resource.loading && <LoadingState label="Loading throughput history" />}
      {!resource.data && resource.error && <ErrorState message={resource.error} onRetry={resource.reload} />}
      {snapshot && !enabled && (
        <EmptyState
          title="History recording is off"
          description="Nothing is sampled and nothing is kept in memory while recording is off. Switch it back on to start a fresh trailing hour."
        />
      )}
      {snapshot && enabled && series.length === 0 && (
        <EmptyState
          title="No throughput history yet"
          description={`History is recorded while this controller runs, one point every ${interval} second${interval === 1 ? '' : 's'}. Send a request to a deployment and its graph appears shortly after.`}
        />
      )}

      <div className="history-grid">
        {series.map((item) => (
          <Panel className="history-panel" key={item.key}>
            <div className="history-panel-heading">
              <div>
                <h2>{seriesTitle(item)}</h2>
                <p>{seriesSubtitle(item)}</p>
              </div>
              <div className="history-panel-state">
                <Status status={item.live_sessions > 0 ? 'running' : 'stopped'}>
                  {item.live_sessions > 0 ? `${item.live_sessions} live` : 'No sessions'}
                </Status>
                <span className="history-panel-counts">
                  {item.state.output_sessions} outputting · {item.state.thinking_sessions} thinking · {item.state.prefill_sessions} prompt processing
                </span>
              </div>
            </div>
            <HistoryChart
              series={item}
              rangeSeconds={rangeSeconds}
              sampleSeconds={interval}
              metrics={metrics}
              colorByConcurrency={colorByConcurrency}
            />
          </Panel>
        ))}
      </div>
    </div>
  )
}
