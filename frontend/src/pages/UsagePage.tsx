import { useMemo, useState, type FormEvent } from 'react'
import {
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  BarChart3,
  ChevronDown,
  ChevronRight,
  Coins,
  Pencil,
  RefreshCw,
  RotateCcw,
  Trash2,
  X,
} from 'lucide-react'
import { api } from '../api/client'
import type { DailyUsagePoint, HourlyUsagePoint, UsageGroup, UsageMember } from '../api/types'
import { Button, EmptyState, ErrorState, LoadingState, PageHeader, Panel } from '../components/ui'
import { useResource } from '../hooks/useResource'

type View = 'models' | 'analysis'
type SortKey = 'model' | 'input' | 'cached' | 'output' | 'requests' | 'speed' | 'cost'

function formatTokens(value = 0) {
  return Intl.NumberFormat('en', {
    notation: Math.abs(value) >= 10_000 ? 'compact' : 'standard',
    maximumFractionDigits: 1,
  }).format(value)
}

function formatCost(value = 0) {
  return value > 0
    ? new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(value)
    : '--'
}

function formatDuration(seconds = 0) {
  if (!seconds) return '--'
  if (seconds < 60) return `${seconds.toFixed(1)} s`
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)} min`
  return `${(seconds / 3600).toFixed(1)} hr`
}

function inputMisses(row: UsageGroup) {
  return Math.max(0, Number(row.stats.input_miss ?? row.stats.input - row.stats.cached))
}

function sortValue(row: UsageGroup, key: SortKey): string | number {
  if (key === 'model') return row.label.toLocaleLowerCase()
  if (key === 'input') return inputMisses(row)
  if (key === 'speed') return row.speed?.tok_s ?? -1
  if (key === 'cost') return row.total_cost
  return Number(row.stats[key] ?? 0)
}

function UsageBars({ points }: { points: HourlyUsagePoint[] }) {
  const visible = points.slice(-48)
  const maximum = Math.max(1, ...visible.map((point) => point.input + point.output))
  if (!visible.length) return <EmptyState title="No hourly activity" description="Hourly token history appears after SparkDeck serves requests." />
  return <div className="usage-chart" role="img" aria-label="Hourly input and output token activity">
    <div className="usage-chart-bars">
      {visible.map((point) => {
        const total = point.input + point.output
        const height = Math.max(3, (total / maximum) * 100)
        const outputShare = total ? (point.output / total) * 100 : 0
        return <span
          className="usage-chart-column"
          key={point.hour}
          style={{ height: `${height}%` }}
          title={`${point.hour}: ${formatTokens(point.input)} input, ${formatTokens(point.output)} output`}
          aria-label={`${point.hour}, ${point.input} input tokens, ${point.output} output tokens`}
        ><span className="usage-chart-output" style={{ height: `${outputShare}%` }} /></span>
      })}
    </div>
    <div className="usage-chart-axis"><span>{visible[0]?.hour.replace('T', ' ')}</span><span>{visible.at(-1)?.hour.replace('T', ' ')}</span></div>
    <div className="usage-chart-legend"><span><i className="usage-legend-input" /> Input</span><span><i className="usage-legend-output" /> Output</span></div>
  </div>
}

export function UsagePage() {
  const summary = useResource((signal) => api.usage.get(signal))
  const [view, setView] = useState<View>('models')
  const [sortKey, setSortKey] = useState<SortKey>('output')
  const [sortAscending, setSortAscending] = useState(false)
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const [editing, setEditing] = useState<UsageMember>()
  const [alias, setAlias] = useState('')
  const [mergeGroup, setMergeGroup] = useState('')
  const [busy, setBusy] = useState<string>()
  const [actionError, setActionError] = useState<string>()
  const [notice, setNotice] = useState<string>()
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const analysis = useResource(
    (signal) => api.usage.analysis(start, end, signal),
    [start, end],
  )

  const rows = useMemo(() => [...(summary.data?.groups ?? [])].sort((left, right) => {
    const a = sortValue(left, sortKey)
    const b = sortValue(right, sortKey)
    const result = typeof a === 'string'
      ? a.localeCompare(String(b), undefined, { numeric: true, sensitivity: 'base' })
      : Number(a) - Number(b)
    return (result || left.key.localeCompare(right.key)) * (sortAscending ? 1 : -1)
  }), [sortAscending, sortKey, summary.data?.groups])

  const totals = useMemo(() => rows.reduce((value, row) => ({
    input: value.input + inputMisses(row),
    cached: value.cached + Number(row.stats.cached || 0),
    output: value.output + Number(row.stats.output || 0),
    requests: value.requests + Number(row.stats.requests || 0),
    cost: value.cost + Number(row.total_cost || 0),
  }), { input: 0, cached: 0, output: 0, requests: 0, cost: 0 }), [rows])

  const setSort = (next: SortKey) => {
    if (next === sortKey) setSortAscending((value) => !value)
    else {
      setSortKey(next)
      setSortAscending(next === 'model')
    }
  }

  const reload = () => {
    summary.reload()
    analysis.reload()
  }

  const reset = async () => {
    if (!window.confirm('Reset lifetime token counters for every model? This cannot be undone.')) return
    setBusy('reset')
    setActionError(undefined)
    try {
      await api.usage.reset()
      setNotice('Lifetime usage counters reset.')
      reload()
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : 'Could not reset lifetime usage')
    } finally {
      setBusy(undefined)
    }
  }

  const beginEdit = (member: UsageMember) => {
    setEditing(member)
    setAlias(member.alias ?? '')
    setMergeGroup(member.merge_group ?? '')
  }

  const saveAlias = async (event: FormEvent) => {
    event.preventDefault()
    if (!editing) return
    setBusy(`alias:${editing.model}`)
    setActionError(undefined)
    try {
      await api.usage.updateAlias(editing.model, alias.trim() || null, mergeGroup.trim() || null)
      setEditing(undefined)
      setNotice(`Updated usage display for ${editing.model}.`)
      summary.reload()
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : 'Could not update the usage alias')
    } finally {
      setBusy(undefined)
    }
  }

  const erase = async (model: string) => {
    if (!window.confirm(`Erase all lifetime and hourly usage for ${model}? This cannot be undone.`)) return
    setBusy(`erase:${model}`)
    setActionError(undefined)
    try {
      await api.usage.erase(model)
      setNotice(`Erased usage for ${model}.`)
      reload()
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : 'Could not erase model usage')
    } finally {
      setBusy(undefined)
    }
  }

  return <div className="page usage-page">
    <PageHeader
      eyebrow="Local inference accounting"
      title="Usage"
      description="Review lifetime model totals, cache reuse, rolling decode speed, estimated cost, and hourly activity recorded by SparkDeck."
      actions={<><Button onClick={reload}><RefreshCw size={15} /> Refresh</Button><Button variant="danger" disabled={busy === 'reset'} onClick={() => void reset()}><RotateCcw size={15} /> Reset lifetime</Button></>}
    />

    <div className="usage-tabs" role="tablist" aria-label="Usage views">
      <button role="tab" aria-selected={view === 'models'} onClick={() => setView('models')}><Coins size={15} /> Lifetime & cost</button>
      <button role="tab" aria-selected={view === 'analysis'} onClick={() => setView('analysis')}><BarChart3 size={15} /> Analysis</button>
    </div>
    {actionError && <p className="inline-error" role="alert">{actionError}</p>}
    {notice && <p className="inline-success" role="status">{notice}</p>}

    {summary.loading && !summary.data && <LoadingState label="Loading lifetime usage" />}
    {summary.error && !summary.data && <ErrorState message={summary.error} onRetry={summary.reload} />}

    {summary.data && view === 'models' && <>
      <section className="usage-metrics" aria-label="Lifetime totals">
        <Panel><span>Input misses</span><strong>{formatTokens(totals.input)}</strong><small>New prompt tokens processed</small></Panel>
        <Panel><span>Cache hits</span><strong>{formatTokens(totals.cached)}</strong><small>Prompt tokens reused</small></Panel>
        <Panel><span>Output</span><strong>{formatTokens(totals.output)}</strong><small>Generated tokens</small></Panel>
        <Panel><span>Requests</span><strong>{formatTokens(totals.requests)}</strong><small>Lifetime completions</small></Panel>
        <Panel><span>Estimated cost</span><strong>{formatCost(totals.cost)}</strong><small>From configured model pricing</small></Panel>
      </section>

      <div className="section-heading"><div><h2>Model usage</h2><p>Aliases, merge groups, and routing affect display only; raw counters remain intact.</p></div></div>
      {!rows.length ? <EmptyState title="No token usage recorded" description="Run a model through SparkDeck to begin collecting local lifetime counters." /> : <Panel className="table-panel usage-table-panel">
        <div className="responsive-table usage-table" role="table" aria-label="Lifetime model usage">
          <div className="table-row table-header" role="row">
            {([['model', 'Model / alias'], ['input', 'Input miss'], ['cached', 'Cache hit'], ['output', 'Output'], ['requests', 'Requests'], ['speed', 'Avg speed'], ['cost', 'Cost']] as Array<[SortKey, string]>).map(([key, label]) => <button role="columnheader" key={key} onClick={() => setSort(key)} aria-sort={sortKey === key ? (sortAscending ? 'ascending' : 'descending') : 'none'}>{label}<span aria-hidden="true">{sortKey === key ? (sortAscending ? <ArrowUp size={12} /> : <ArrowDown size={12} />) : <ArrowUpDown size={12} />}</span></button>)}
            <span role="columnheader">Details</span>
          </div>
          {rows.map((row) => {
            const open = expanded.has(row.key)
            return <div className="usage-row-group" key={row.key}>
              <div className="table-row" role="row">
                <div role="cell" data-label="Model / alias"><strong>{row.label}</strong><small>{row.merge_group ? `${row.models.length} merged models` : row.route_target ?? row.models[0]}</small></div>
                <div role="cell" data-label="Input miss"><strong>{row.stats.estimated_cached ? '~' : ''}{formatTokens(inputMisses(row))}</strong></div>
                <div role="cell" data-label="Cache hit"><strong>{row.stats.estimated_cached ? '~' : ''}{formatTokens(row.stats.cached)}</strong><small>{row.stats.estimated_cached ? `${formatTokens(row.stats.measured_cached)} measured` : 'Measured reuse'}</small></div>
                <div role="cell" data-label="Output"><strong>{formatTokens(row.stats.output)}</strong><small>{formatDuration(row.stats.gen_time_s)} decode</small></div>
                <div role="cell" data-label="Requests">{formatTokens(row.stats.requests)}</div>
                <div role="cell" data-label="Avg speed"><strong>{row.speed?.tok_s == null ? '--' : `${row.speed.tok_s.toFixed(1)} tok/s`}</strong><small>{row.speed?.legacy ? 'Legacy lifetime average' : row.speed?.tokens ? `Last ${formatTokens(row.speed.tokens)} output` : 'No speed samples'}</small></div>
                <div role="cell" data-label="Cost"><strong>{formatCost(row.total_cost)}</strong><small>{row.cost_estimated ? 'Includes cache estimate' : 'Recorded pricing'}</small></div>
                <div role="cell" data-label="Details" className="row-actions"><Button variant="tertiary" aria-expanded={open} aria-label={`${open ? 'Hide' : 'Show'} details for ${row.label}`} onClick={() => setExpanded((current) => {
                  const next = new Set(current)
                  if (next.has(row.key)) next.delete(row.key)
                  else next.add(row.key)
                  return next
                })}>{open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}</Button></div>
              </div>
              {open && <div className="usage-members" aria-label={`Models in ${row.label}`}>
                {row.members.map((member) => <div className="usage-member" key={member.model}>
                  <div><strong>{member.alias || member.model}</strong><code>{member.model}</code>{member.routed_to && <small>Displayed under {member.routed_to}</small>}</div>
                  <div className="row-actions"><Button variant="tertiary" aria-label={`Edit alias for ${member.model}`} onClick={() => beginEdit(member)}><Pencil size={14} /> Edit</Button><Button variant="danger" disabled={busy === `erase:${member.model}`} onClick={() => void erase(member.model)}><Trash2 size={14} /> Erase</Button></div>
                </div>)}
              </div>}
            </div>
          })}
        </div>
      </Panel>}
    </>}

    {view === 'analysis' && <>
      <Panel className="usage-range-panel"><div><h2>Activity range</h2><p>Filter the persisted hourly and daily token history.</p></div><div className="usage-range-fields"><label className="field"><span>Start date</span><input type="date" value={start} max={end || undefined} onChange={(event) => setStart(event.target.value)} /></label><label className="field"><span>End date</span><input type="date" value={end} min={start || undefined} onChange={(event) => setEnd(event.target.value)} /></label></div></Panel>
      {analysis.loading && !analysis.data && <LoadingState label="Loading usage analysis" />}
      {analysis.error && !analysis.data && <ErrorState message={analysis.error} onRetry={analysis.reload} />}
      {analysis.data && <div className="usage-analysis-grid">
        <Panel className="usage-chart-panel"><div className="section-heading"><div><h2>Hourly tokens</h2><p>Latest 48 recorded hours in the selected range.</p></div></div><UsageBars points={analysis.data.hourly} /></Panel>
        <Panel className="usage-daily-panel"><div className="section-heading"><div><h2>Daily totals</h2><p>Input, cache reuse, output, and requests by day.</p></div></div>{analysis.data.daily.length ? <div className="usage-daily-list">{analysis.data.daily.slice(-14).reverse().map((day: DailyUsagePoint) => <div key={day.date}><time dateTime={day.date}>{new Date(`${day.date}T00:00:00`).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}</time><span><strong>{formatTokens(day.input)}</strong> input</span><span><strong>{formatTokens(day.cached)}</strong> cached</span><span><strong>{formatTokens(day.output)}</strong> output</span><span><strong>{formatTokens(day.requests)}</strong> requests</span></div>)}</div> : <EmptyState title="No daily activity" description="No persisted token activity matches this range." />}</Panel>
      </div>}
    </>}

    {editing && <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && setEditing(undefined)}><section className="modal usage-alias-modal" role="dialog" aria-modal="true" aria-labelledby="usage-alias-title"><div className="modal-heading"><div><p className="eyebrow">Display settings</p><h2 id="usage-alias-title">Edit usage model</h2></div><button className="icon-button" onClick={() => setEditing(undefined)} aria-label="Close dialog"><X size={17} /></button></div><code>{editing.model}</code><form onSubmit={(event) => void saveAlias(event)}><label className="field"><span>Display alias</span><input maxLength={120} value={alias} onChange={(event) => setAlias(event.target.value)} placeholder="Optional friendly name" /></label><label className="field"><span>Merge group</span><input maxLength={120} value={mergeGroup} onChange={(event) => setMergeGroup(event.target.value)} placeholder="Optional group name" /></label><p>Merge groups combine display rows without changing the underlying counters.</p><div className="modal-actions"><Button type="button" onClick={() => setEditing(undefined)}>Cancel</Button><Button type="submit" variant="primary" disabled={busy === `alias:${editing.model}`}>Save usage display</Button></div></form></section></div>}
  </div>
}
