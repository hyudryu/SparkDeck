import { useMemo, useState, type CSSProperties, type FormEvent } from 'react'
import { ArrowDown, ArrowRight, ArrowUp, ArrowUpDown, ChevronDown, ChevronRight, Pencil, RefreshCw, RotateCcw, Trash2, X } from 'lucide-react'
import { api } from '../api/client'
import type { DailyUsagePoint, UsageCounters, UsageGroup, UsageMember, UsageSummary } from '../api/types'
import { Button, EmptyState, ErrorState, LoadingState, PageHeader, Panel } from '../components/ui'
import { useConfirmDialog } from '../components/useConfirmDialog'
import { useResource } from '../hooks/useResource'

type SortKey = 'model' | 'input' | 'cached' | 'output' | 'requests' | 'speed' | 'cost'
type RangeDays = 7 | 30
type MeterRange = 'lifetime' | 7 | 30 | 365

const chartColors = ['#4a9eff', '#48c774', '#8c6bff', '#ff626a', '#ff9238', '#4cc9c0', '#d66efd', '#e4c34e']
const modelShareLimit = 7

function isoDate(value: Date) { return value.toISOString().slice(0, 10) }
function daysAgo(days: number) {
  const value = new Date()
  value.setUTCHours(0, 0, 0, 0)
  value.setUTCDate(value.getUTCDate() - days)
  return isoDate(value)
}
function formatTokens(value = 0) { return Intl.NumberFormat('en', { notation: Math.abs(value) >= 10_000 ? 'compact' : 'standard', maximumFractionDigits: 1 }).format(value) }
function formatCost(value = 0) { return value > 0 ? new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(value) : '--' }
function formatDuration(seconds = 0) {
  if (!seconds) return '--'
  if (seconds < 60) return `${seconds.toFixed(1)} s`
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)} min`
  return `${(seconds / 3600).toFixed(1)} hr`
}
function inputMisses(row: UsageGroup) { return Math.max(0, Number(row.stats.input_miss ?? row.stats.input - row.stats.cached)) }
function totalTokens(counters?: UsageCounters) { return counters ? Number(counters.input ?? 0) + Number(counters.output ?? 0) : 0 }
function emptyCounters(): UsageCounters { return { input: 0, cached: 0, output: 0, requests: 0 } }
function addCounters(target: UsageCounters, source?: UsageCounters) {
  if (!source) return target
  target.input += Number(source.input ?? 0)
  target.cached += Number(source.cached ?? 0)
  target.output += Number(source.output ?? 0)
  target.requests += Number(source.requests ?? 0)
  return target
}

export function usageMeterSnapshot(
  summary: UsageSummary | undefined,
  daily: DailyUsagePoint[],
  range: MeterRange,
  reference = new Date(),
) {
  if (range === 'lifetime') {
    return {
      totals: addCounters(emptyCounters(), summary?.total),
      models: Object.entries(summary?.models ?? {}).map(([model, counters]) => ({ model, counters: addCounters(emptyCounters(), counters) })),
    }
  }
  const end = new Date(reference); end.setUTCHours(0, 0, 0, 0)
  const start = new Date(end); start.setUTCDate(end.getUTCDate() - range + 1)
  const startDate = isoDate(start); const endDate = isoDate(end)
  const totals = emptyCounters(); const models = new Map<string, UsageCounters>()
  daily.filter((point) => point.date >= startDate && point.date <= endDate).forEach((point) => {
    addCounters(totals, point)
    Object.entries(point.models ?? {}).forEach(([model, counters]) => {
      models.set(model, addCounters(models.get(model) ?? emptyCounters(), counters))
    })
  })
  return { totals, models: [...models].map(([model, counters]) => ({ model, counters })) }
}
function sortValue(row: UsageGroup, key: SortKey): string | number {
  if (key === 'model') return row.label.toLocaleLowerCase()
  if (key === 'input') return inputMisses(row)
  if (key === 'speed') return row.speed?.tok_s ?? -1
  if (key === 'cost') return row.total_cost
  return Number(row.stats[key] ?? 0)
}

function usageStreaks(points: DailyUsagePoint[]) {
  const active = [...new Set(points.filter((point) => totalTokens(point) > 0).map((point) => point.date))].sort()
  let longest = 0
  let run = 0
  let previous = Number.NaN
  active.forEach((date) => {
    const current = Date.parse(`${date}T00:00:00Z`) / 86_400_000
    run = current === previous + 1 ? run + 1 : 1
    longest = Math.max(longest, run)
    previous = current
  })
  const latest = active.at(-1)
  let current = latest === daysAgo(0) || latest === daysAgo(1) ? 1 : 0
  if (current) for (let index = active.length - 2; index >= 0; index -= 1) {
    if (Date.parse(`${active[index + 1]}T00:00:00Z`) - Date.parse(`${active[index]}T00:00:00Z`) !== 86_400_000) break
    current += 1
  }
  return { activeDays: active.length, current, longest }
}

export function activityHeatmapCalendar(points: DailyUsagePoint[], reference = new Date()) {
  const byDate = new Map(points.map((point) => [point.date, totalTokens(point)]))
  const end = new Date(reference)
  end.setUTCHours(0, 0, 0, 0)
  const start = new Date(end)
  start.setUTCDate(end.getUTCDate() - 364 - end.getUTCDay())
  const cells = Array.from({ length: 371 }, (_, index) => {
    const date = new Date(start); date.setUTCDate(start.getUTCDate() + index)
    const key = isoDate(date); return { date: key, value: byDate.get(key) ?? 0 }
  })
  const months: Array<{ key: string; label: string; column: number }> = []
  let previousMonth = ''
  cells.forEach((cell, index) => {
    const date = new Date(`${cell.date}T00:00:00Z`)
    const key = cell.date.slice(0, 7)
    if (key === previousMonth) return
    previousMonth = key
    months.push({
      key,
      label: date.toLocaleDateString(undefined, { month: 'short', timeZone: 'UTC' }),
      column: Math.floor(index / 7) + 1,
    })
  })
  return { cells, months }
}

function ActivityHeatmap({ points }: { points: DailyUsagePoint[] }) {
  const { cells, months } = activityHeatmapCalendar(points)
  const maximum = Math.max(1, ...cells.map((cell) => cell.value))
  const [hover, setHover] = useState<{ date: string; value: number; x: number; y: number }>()
  return <div className="usage-heatmap-scroll" role="img" aria-label="Daily token activity for the last year" onMouseLeave={() => setHover(undefined)}><div className="usage-heatmap">
    {cells.map((cell) => { const ratio = cell.value / maximum; const level = cell.value === 0 ? 0 : ratio < .08 ? 1 : ratio < .25 ? 2 : ratio < .55 ? 3 : 4; return <span key={cell.date} className={`usage-heatmap-cell level-${level}`} onMouseEnter={(event) => setHover({ date: cell.date, value: cell.value, x: event.clientX, y: event.clientY })} /> })}
  </div><div className="usage-heatmap-months" aria-hidden="true">{months.map((month) => <span key={month.key} style={{ gridColumn: `${month.column} / span 4` }}>{month.label}</span>)}</div><div className="usage-heatmap-scale" aria-hidden="true"><span>Less</span>{[0, 1, 2, 3, 4].map((level) => <i key={level} className={`usage-heatmap-cell level-${level}`} />)}<span>More</span></div>
    {hover && <div className="usage-heatmap-tooltip" style={{ left: Math.min(Math.max(hover.x, 90), window.innerWidth - 90), top: hover.y }}><strong>{formatTokens(hover.value)} tokens</strong><span>{new Date(`${hover.date}T00:00:00Z`).toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' })}</span></div>}
  </div>
}

function TrendChart({ points, range }: { points: DailyUsagePoint[]; range: RangeDays }) {
  const dates = Array.from({ length: range }, (_, index) => daysAgo(range - index - 1))
  const byDate = new Map(points.map((point) => [point.date, point]))
  const modelNames = [...new Set(points.flatMap((point) => Object.keys(point.models ?? {})))]
  const names = modelNames.length ? modelNames : ['All models']
  const series = names.map((name) => ({ name, values: dates.map((date) => name === 'All models' ? totalTokens(byDate.get(date)) : totalTokens(byDate.get(date)?.models?.[name])) })).filter((item) => item.values.some(Boolean))
  const maximum = Math.max(1, ...series.flatMap((item) => item.values)); const width = 720; const height = 220
  const x = (index: number) => 16 + index / Math.max(1, dates.length - 1) * (width - 32)
  const y = (value: number) => height - 20 - value / maximum * (height - 42)
  if (!series.length) return <EmptyState title="No activity in this range" description="Daily model trends appear after SparkDeck serves requests." />
  return <div className="usage-trend" role="img" aria-label={`Daily model token trend for the last ${range} days`}><div className="usage-chart-legend">{series.map((item, index) => <span key={item.name}><i style={{ background: chartColors[index % chartColors.length] }} />{item.name}</span>)}</div><svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
    {[.25, .5, .75].map((ratio) => <line key={ratio} x1="0" x2={width} y1={height * ratio} y2={height * ratio} className="usage-grid-line" />)}
    {series.map((item, index) => <polyline key={item.name} points={item.values.map((value, pointIndex) => `${x(pointIndex)},${y(value)}`).join(' ')} style={{ stroke: chartColors[index % chartColors.length] }} />)}
  </svg><div className="usage-chart-axis"><span>{new Date(`${dates[0]}T00:00:00`).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}</span><span>{new Date(`${dates.at(-1)}T00:00:00`).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}</span></div></div>
}

export function modelShareItems(rows: UsageGroup[]) {
  const ranked = rows
    .map((row) => ({ key: row.key, label: row.label, value: inputMisses(row) + Number(row.stats.cached || 0) + Number(row.stats.output || 0) }))
    .filter((item) => item.value > 0)
    .sort((left, right) => right.value - left.value || left.label.localeCompare(right.label, undefined, { numeric: true, sensitivity: 'base' }) || left.key.localeCompare(right.key))
  const values = ranked.slice(0, modelShareLimit)
  const others = ranked.slice(modelShareLimit).reduce((sum, item) => sum + item.value, 0)
  if (others > 0) values.push({ key: 'others', label: 'Others', value: others })
  return values
}

function ModelShare({ rows }: { rows: UsageGroup[] }) {
  const values = modelShareItems(rows)
  const total = values.reduce((sum, item) => sum + item.value, 0)
  const stops = values.map((item, index) => {
    const start = values.slice(0, index).reduce((sum, previous) => sum + previous.value, 0) / total * 100
    const end = start + item.value / total * 100
    return `${chartColors[index % chartColors.length]} ${start}% ${end}%`
  })
  if (!values.length) return <EmptyState title="No model usage" description="Model share appears after token usage is recorded." />
  return <div className="usage-share"><div className="usage-donut" style={{ '--usage-donut': `conic-gradient(${stops.join(',')})` } as CSSProperties}><strong>{formatTokens(total)}</strong><span>tokens</span></div><div className="usage-share-list">{values.map((item, index) => <div key={item.key}><i style={{ background: chartColors[index % chartColors.length] }} /><span><strong>{item.label}</strong><small>{formatTokens(item.value)} tokens</small></span><b>{Math.round(item.value / total * 100)}%</b></div>)}</div></div>
}

export function UsagePage() {
  const { confirm, confirmationDialog } = useConfirmDialog()
  const summary = useResource((signal) => api.usage.get(signal))
  const analysis = useResource((signal) => api.usage.analysis(daysAgo(364), daysAgo(0), signal))
  const [range, setRange] = useState<RangeDays>(30)
  const [meterRange, setMeterRange] = useState<MeterRange>('lifetime')
  const [sortKey, setSortKey] = useState<SortKey>('output'); const [sortAscending, setSortAscending] = useState(false)
  const [expanded, setExpanded] = useState<Set<string>>(new Set()); const [editing, setEditing] = useState<UsageMember>()
  const [alias, setAlias] = useState(''); const [mergeGroup, setMergeGroup] = useState(''); const [routeTo, setRouteTo] = useState(''); const [busy, setBusy] = useState<string>()
  const [routeSource, setRouteSource] = useState(''); const [routeDestination, setRouteDestination] = useState('')
  const [actionError, setActionError] = useState<string>(); const [routeError, setRouteError] = useState<string>(); const [notice, setNotice] = useState<string>()
  const rows = useMemo(() => [...(summary.data?.groups ?? [])].sort((left, right) => { const a = sortValue(left, sortKey); const b = sortValue(right, sortKey); const result = typeof a === 'string' ? a.localeCompare(String(b), undefined, { numeric: true, sensitivity: 'base' }) : Number(a) - Number(b); return (result || left.key.localeCompare(right.key)) * (sortAscending ? 1 : -1) }), [sortAscending, sortKey, summary.data?.groups])
  const totals = useMemo(() => rows.reduce((value, row) => ({ input: value.input + inputMisses(row), cached: value.cached + Number(row.stats.cached || 0), output: value.output + Number(row.stats.output || 0), requests: value.requests + Number(row.stats.requests || 0), cost: value.cost + Number(row.total_cost || 0) }), { input: 0, cached: 0, output: 0, requests: 0, cost: 0 }), [rows])
  const daily = useMemo(() => analysis.data?.daily ?? [], [analysis.data?.daily]); const streaks = usageStreaks(daily); const peak = Math.max(0, ...daily.map((point) => totalTokens(point)))
  const meter = useMemo(() => usageMeterSnapshot(summary.data, daily, meterRange), [daily, meterRange, summary.data])
  const meterModels = useMemo(() => [...meter.models].sort((left, right) => totalTokens(right.counters) - totalTokens(left.counters) || left.model.localeCompare(right.model, undefined, { numeric: true, sensitivity: 'base' })), [meter.models])
  const setSort = (next: SortKey) => { if (next === sortKey) setSortAscending((value) => !value); else { setSortKey(next); setSortAscending(next === 'model') } }
  const reload = () => { summary.reload(); analysis.reload() }
  const reset = async () => { if (!await confirm({ title: 'Reset metered usage?', message: 'This resets the cluster-wide lifetime meter to zero for every model. Dated history remains available in the 7, 30, and 365 day views.', confirmLabel: 'Reset meter', danger: true })) return; setBusy('reset'); setActionError(undefined); try { await api.usage.reset(); setMeterRange('lifetime'); setNotice('Metered usage reset to zero.'); reload() } catch (reason) { setActionError(reason instanceof Error ? reason.message : 'Could not reset metered usage') } finally { setBusy(undefined) } }
  // member.merge_group holds the destination's resolved group for routed
  // sources; only summary.merge_groups carries the source's own setting.
  const beginEdit = (member: UsageMember) => { setEditing(member); setAlias(member.alias ?? ''); setMergeGroup(summary.data?.merge_groups?.[member.model] ?? ''); setRouteTo(summary.data?.routing_rules?.[member.model] ?? ''); setActionError(undefined) }
  const saveDisplay = async (event: FormEvent) => {
    event.preventDefault(); if (!editing) return
    const destination = routeTo.trim(); const currentRoute = summary.data?.routing_rules?.[editing.model]
    const configuredGroup = summary.data?.merge_groups?.[editing.model] ?? ''
    const nextGroup = mergeGroup.trim()
    setBusy(`alias:${editing.model}`); setActionError(undefined)
    try {
      await api.usage.updateAlias(editing.model, alias.trim() || null, nextGroup === configuredGroup ? undefined : nextGroup || null)
      if (!destination && currentRoute) await api.usage.deleteRoute(editing.model)
      else if (destination && destination !== currentRoute) await api.usage.setRoute(editing.model, destination)
      setEditing(undefined); setNotice(`Updated usage display for ${editing.model}.`); summary.reload()
    } catch (reason) { setActionError(reason instanceof Error ? reason.message : 'Could not update the usage display') } finally { setBusy(undefined) }
  }
  const erase = async (model: string) => { if (!await confirm({ title: `Erase usage for ${model}?`, message: 'This removes all lifetime and hourly usage for this model and cannot be undone.', confirmLabel: 'Erase usage', danger: true })) return; setBusy(`erase:${model}`); setActionError(undefined); try { await api.usage.erase(model); setNotice(`Erased usage for ${model}.`); reload() } catch (reason) { setActionError(reason instanceof Error ? reason.message : 'Could not erase model usage') } finally { setBusy(undefined) } }
  const routeModels = [...new Set([...Object.keys(summary.data?.models ?? {}), ...Object.keys(summary.data?.routing_rules ?? {}), ...Object.values(summary.data?.routing_rules ?? {}), ...(summary.data?.groups ?? []).map((group) => group.route_target).filter((target): target is string => Boolean(target))])].sort()
  const routingRules = Object.entries(summary.data?.routing_rules ?? {}).sort(([left], [right]) => left.localeCompare(right, undefined, { numeric: true, sensitivity: 'base' }))
  const routeOptions = editing ? routeModels.filter((model) => model !== editing.model) : []
  const saveRoute = async (event: FormEvent) => {
    event.preventDefault()
    if (!routeSource || !routeDestination || routeSource === routeDestination) return
    setBusy('route:add'); setActionError(undefined); setRouteError(undefined)
    try {
      await api.usage.setRoute(routeSource, routeDestination)
      setNotice(`Routed ${routeSource} usage to ${routeDestination}.`); setRouteSource(''); setRouteDestination(''); summary.reload()
    } catch (reason) { setRouteError(reason instanceof Error ? reason.message : 'Could not save the routing rule') } finally { setBusy(undefined) }
  }
  const removeRoute = async (source: string) => {
    setBusy(`route:${source}`); setActionError(undefined); setRouteError(undefined)
    try {
      await api.usage.deleteRoute(source)
      setNotice(`Removed the routing rule for ${source}.`)
      if (routeSource === source) setRouteDestination('')
      summary.reload()
    } catch (reason) { setRouteError(reason instanceof Error ? reason.message : 'Could not remove the routing rule') } finally { setBusy(undefined) }
  }

  return <div className="page usage-page"><PageHeader eyebrow="Cluster inference accounting" title="Usage stats" description="Combined token activity and model share from every paired SparkDeck node." actions={<Button onClick={reload}><RefreshCw size={15} /> Refresh</Button>} />
    {actionError && !editing && <p className="inline-error" role="alert">{actionError}</p>}{notice && <p className="inline-success" role="status">{notice}</p>}
    {(summary.loading || analysis.loading) && !summary.data && <LoadingState label="Loading usage stats" />}{summary.error && !summary.data && <ErrorState message={summary.error} onRetry={reload} />}
    {analysis.error && <ErrorState message={`Historical usage: ${analysis.error}`} onRetry={analysis.reload} />}
    {summary.data && <><Panel className="usage-overview-metrics" aria-label="Usage overview"><div><strong>{formatTokens(totals.input + totals.cached + totals.output)}</strong><span>Total tokens</span></div><div><strong>{formatTokens(peak)}</strong><span>Peak day</span></div><div><strong>{streaks.activeDays}</strong><span>Active days</span></div><div><strong>{streaks.current} d</strong><span>Current streak</span></div><div><strong>{streaks.longest} d</strong><span>Longest streak</span></div></Panel>
      <Panel className="usage-meter-panel" aria-label="Token usage meter"><div className="usage-panel-heading usage-meter-heading"><div><h2>Token meter</h2><p>{meterRange === 'lifetime' ? 'Measured since the lifetime meter was last reset' : `Measured over the last ${meterRange} days`}</p></div><Button variant="danger" disabled={busy === 'reset'} onClick={() => void reset()}><RotateCcw size={15} /> {busy === 'reset' ? 'Resetting…' : 'Reset metered usage'}</Button></div>
        <div className="segmented-control usage-meter-range" aria-label="Token meter timeframe"><button aria-pressed={meterRange === 'lifetime'} onClick={() => setMeterRange('lifetime')}>Since reset</button><button aria-pressed={meterRange === 7} onClick={() => setMeterRange(7)}>7 days</button><button aria-pressed={meterRange === 30} onClick={() => setMeterRange(30)}>30 days</button><button aria-pressed={meterRange === 365} onClick={() => setMeterRange(365)}>365 days</button></div>
        <div className="usage-meter-totals"><div><strong>{formatTokens(totalTokens(meter.totals))}</strong><span>Total tokens</span></div><div><strong>{formatTokens(meter.totals.input)}</strong><span>Input tokens</span></div><div><strong>{formatTokens(meter.totals.cached)}</strong><span>Cached input</span></div><div><strong>{formatTokens(meter.totals.output)}</strong><span>Output tokens</span></div><div><strong>{formatTokens(meter.totals.requests)}</strong><span>Requests</span></div></div>
        {!meterModels.length ? <EmptyState title="No measured model usage" description="Models appear here as SparkDeck records calls in this timeframe." /> : <div className="usage-meter-models" role="table" aria-label="Measured usage by model"><div className="usage-meter-model usage-meter-model-header" role="row"><span role="columnheader">Model</span><span role="columnheader">Input</span><span role="columnheader">Output</span><span role="columnheader">Total</span><span role="columnheader">Requests</span></div>{meterModels.map(({ model, counters }) => <div className="usage-meter-model" role="row" key={model}><strong role="cell">{model}</strong><span role="cell" data-label="Input">{formatTokens(counters.input)}</span><span role="cell" data-label="Output">{formatTokens(counters.output)}</span><span role="cell" data-label="Total">{formatTokens(totalTokens(counters))}</span><span role="cell" data-label="Requests">{formatTokens(counters.requests)}</span></div>)}</div>}
      </Panel>
      <Panel className="usage-activity-panel"><div className="usage-panel-heading"><div><h2>Token activity</h2><p>Daily usage over the last year</p></div></div><ActivityHeatmap points={daily} /></Panel>
      <div className="usage-time-heading"><h2>Time range</h2><div className="segmented-control" aria-label="Usage time range"><button aria-pressed={range === 7} onClick={() => setRange(7)}>Last 7 days</button><button aria-pressed={range === 30} onClick={() => setRange(30)}>Last 30 days</button></div></div>
      <Panel className="usage-trend-panel"><div className="usage-panel-heading"><div><h2>Daily token trend</h2><p>Input and output tokens by model</p></div></div><TrendChart points={daily} range={range} /></Panel>
      <Panel className="usage-share-panel"><div className="usage-panel-heading"><div><h2>Model usage</h2><p>Lifetime token share by model or alias</p></div></div><ModelShare rows={rows} /></Panel>
      <div className="section-heading"><div><h2>Detailed accounting</h2><p>Aliases, merge groups, and routing rules affect display only; raw counters remain intact.</p></div></div>
      {!rows.length ? <EmptyState title="No token usage recorded" description="Run a model through any paired SparkDeck node to begin collecting cluster lifetime counters." /> : <Panel className="table-panel usage-table-panel"><div className="responsive-table usage-table" role="table" aria-label="Lifetime model usage"><div className="table-row table-header" role="row">{([['model', 'Model / alias'], ['input', 'Input miss'], ['cached', 'Cache hit'], ['output', 'Output'], ['requests', 'Requests'], ['speed', 'Avg speed'], ['cost', 'Cost']] as Array<[SortKey, string]>).map(([key, label]) => <button role="columnheader" key={key} onClick={() => setSort(key)} aria-sort={sortKey === key ? (sortAscending ? 'ascending' : 'descending') : 'none'}>{label}<span aria-hidden="true">{sortKey === key ? (sortAscending ? <ArrowUp size={12} /> : <ArrowDown size={12} />) : <ArrowUpDown size={12} />}</span></button>)}<span role="columnheader">Details</span></div>
        {rows.map((row) => { const open = expanded.has(row.key); return <div className="usage-row-group" key={row.key}><div className="table-row" role="row"><div role="cell" data-label="Model / alias"><strong>{row.label}</strong><small>{row.merge_group ? `${row.models.length} merged models` : row.route_target ?? row.models[0]}</small></div><div role="cell" data-label="Input miss"><strong>{row.stats.estimated_cached ? '~' : ''}{formatTokens(inputMisses(row))}</strong></div><div role="cell" data-label="Cache hit"><strong>{row.stats.estimated_cached ? '~' : ''}{formatTokens(row.stats.cached)}</strong><small>{row.stats.estimated_cached ? `${formatTokens(row.stats.measured_cached)} measured` : 'Measured reuse'}</small></div><div role="cell" data-label="Output"><strong>{formatTokens(row.stats.output)}</strong><small>{formatDuration(row.stats.gen_time_s)} decode</small></div><div role="cell" data-label="Requests">{formatTokens(row.stats.requests)}</div><div role="cell" data-label="Avg speed"><strong>{row.speed?.tok_s == null ? '--' : `${row.speed.tok_s.toFixed(1)} tok/s`}</strong><small>{row.speed?.legacy ? 'Legacy lifetime average' : row.speed?.tokens ? `Last ${formatTokens(row.speed.tokens)} output` : 'No speed samples'}</small></div><div role="cell" data-label="Cost"><strong>{formatCost(row.total_cost)}</strong><small>{row.cost_estimated ? 'Includes cache estimate' : 'Recorded pricing'}</small></div><div role="cell" data-label="Details" className="row-actions"><Button variant="tertiary" aria-expanded={open} aria-label={`${open ? 'Hide' : 'Show'} details for ${row.label}`} onClick={() => setExpanded((current) => { const next = new Set(current); if (next.has(row.key)) next.delete(row.key); else next.add(row.key); return next })}>{open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}</Button></div></div>
          {open && <div className="usage-members" aria-label={`Models in ${row.label}`}>{row.members.map((member) => <div className="usage-member" key={member.model}><div><strong>{member.alias || member.model}</strong><code>{member.model}</code>{member.routed_to && <small>Displayed under {member.routed_to}</small>}</div><div className="row-actions"><Button variant="tertiary" aria-label={`Edit usage display for ${member.model}`} onClick={() => beginEdit(member)}><Pencil size={14} /> Edit</Button><Button variant="danger" disabled={busy === `erase:${member.model}`} onClick={() => void erase(member.model)}><Trash2 size={14} /> Erase</Button></div></div>)}</div>}</div> })}</div></Panel>}
      <Panel className="usage-routing-panel" aria-labelledby="usage-routing-title"><div className="usage-panel-heading"><div><h2 id="usage-routing-title">Model routing rules</h2><p>Combine one model's accounting into another without changing the underlying counters.</p></div></div>
        <form className="usage-routing-form" aria-label="Add model routing rule" onSubmit={(event) => void saveRoute(event)}><label className="field"><span>Source model</span><select value={routeSource} disabled={busy?.startsWith('route:')} onChange={(event) => { const source = event.target.value; setRouteSource(source); setRouteDestination(summary.data?.routing_rules?.[source] ?? ''); setRouteError(undefined) }}><option value="">Select a model</option>{routeModels.map((model) => <option key={model} value={model}>{model}</option>)}</select></label><label className="field"><span>Destination model</span><select value={routeDestination} onChange={(event) => { setRouteDestination(event.target.value); setRouteError(undefined) }} disabled={!routeSource || busy?.startsWith('route:')}><option value="">Select a model</option>{routeModels.filter((model) => model !== routeSource).map((model) => <option key={model} value={model}>{model}</option>)}</select></label><Button type="submit" variant="primary" disabled={!routeSource || !routeDestination || routeSource === routeDestination || busy?.startsWith('route:')}>{summary.data?.routing_rules?.[routeSource] ? 'Update rule' : 'Add rule'}</Button></form>
        {routeError && <p className="inline-error usage-routing-error" role="alert">{routeError}</p>}
        {routingRules.length ? <ul className="usage-routing-list" aria-label="Current model routing rules">{routingRules.map(([source, destination]) => <li key={source} aria-label={`${source} routes to ${destination}`}><code>{source}</code><ArrowRight size={14} aria-hidden="true" /><code>{destination}</code><Button type="button" variant="tertiary" aria-label={`Remove routing rule for ${source}`} disabled={busy?.startsWith('route:')} onClick={() => void removeRoute(source)}><Trash2 size={14} /> Remove</Button></li>)}</ul> : <p className="usage-routing-empty">No routing rules yet. Add one to combine a source model into a destination model.</p>}
      </Panel>
    </>}
    {editing && <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && setEditing(undefined)}><section className="modal usage-alias-modal" role="dialog" aria-modal="true" aria-labelledby="usage-alias-title"><div className="modal-heading"><div><p className="eyebrow">Display settings</p><h2 id="usage-alias-title">Edit usage model</h2></div><button className="icon-button" onClick={() => setEditing(undefined)} aria-label="Close dialog"><X size={17} /></button></div><code>{editing.model}</code><form onSubmit={(event) => void saveDisplay(event)}><label className="field"><span>Display alias</span><input maxLength={120} value={alias} onChange={(event) => setAlias(event.target.value)} placeholder="Optional friendly name" /></label><label className="field"><span>Merge group</span><input maxLength={120} value={mergeGroup} onChange={(event) => setMergeGroup(event.target.value)} placeholder="Optional group name" /></label><label className="field"><span>Route usage to</span><input maxLength={512} list="usage-route-destinations" value={routeTo} onChange={(event) => setRouteTo(event.target.value)} placeholder="Leave empty to keep this model's own row" /></label><datalist id="usage-route-destinations">{routeOptions.map((model) => <option key={model} value={model} />)}</datalist>{actionError && <p className="inline-error" role="alert">{actionError}</p>}<p>Routing forwards this model's usage totals to another model's row and merge groups combine rows; neither rewrites the underlying counters.</p><div className="modal-actions"><Button type="button" onClick={() => setEditing(undefined)}>Cancel</Button><Button type="submit" variant="primary" disabled={busy === `alias:${editing.model}`}>Save usage display</Button></div></form></section></div>}
    {confirmationDialog}
  </div>
}
