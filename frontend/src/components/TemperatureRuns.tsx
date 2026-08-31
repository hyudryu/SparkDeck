import { useCallback, useEffect, useRef, useState } from 'react'
import { Pencil } from 'lucide-react'
import { api } from '../api/client'
import type { TemperatureRun, TemperatureRunSample } from '../api/types'
import { Button, EmptyState, ErrorState, LoadingState, Panel, Status } from './ui'
import { useResource } from '../hooks/useResource'

// Matches the .chart-color-0..5 palette used by the benchmark charts.
const COLOR_PALETTE = ['#2997ff', '#30d158', '#ff9f0a', '#bf5af2', '#ff375f', '#64d2ff']
const COLOR_STORAGE_KEY = 'temperatureRunColors'
const GAP_SECONDS = 2.5
const STATUS_LABELS: Record<TemperatureRun['status'], string> = {
  armed: 'Waiting for heat',
  recording: 'Recording',
  complete: 'Complete',
  cancelled: 'Cancelled',
  interrupted: 'Interrupted',
}

function truncate(value: string, length: number) {
  return value.length > length ? `${value.slice(0, length - 1)}…` : value
}

function formatRunDuration(seconds?: number | null) {
  const value = Math.max(0, Math.round(Number(seconds) || 0))
  if (value < 60) return `${value}s`
  if (value < 3600) return `${Math.floor(value / 60)}m ${value % 60}s`
  return `${Math.floor(value / 3600)}h ${Math.floor((value % 3600) / 60)}m`
}

function loadStoredColors(): Record<string, string> {
  try {
    const stored: unknown = JSON.parse(localStorage.getItem(COLOR_STORAGE_KEY) || '{}')
    if (!stored || typeof stored !== 'object' || Array.isArray(stored)) return {}
    return Object.fromEntries(
      Object.entries(stored as Record<string, unknown>)
        .filter(([runId, color]) => runId && typeof color === 'string' && /^#[0-9a-f]{6}$/i.test(color)),
    ) as Record<string, string>
  } catch {
    // Invalid or unavailable local storage should not prevent the chart loading.
    return {}
  }
}

function runColor(runId: string, colors: Record<string, string>) {
  const override = colors[runId]
  if (/^#[0-9a-f]{6}$/i.test(override || '')) return override
  let hash = 0
  for (const char of runId) hash = (hash * 31 + char.charCodeAt(0)) >>> 0
  return COLOR_PALETTE[hash % COLOR_PALETTE.length]
}

// Centered moving average; a sampling gap wider than GAP_SECONDS breaks the
// window so values are never averaged across a telemetry outage.
function smoothSamples(samples: TemperatureRunSample[], windowSeconds: number): TemperatureRunSample[] {
  const window = Math.max(0, Number(windowSeconds) || 0)
  if (!samples.length || window <= 0) return samples

  const smoothed = samples.map((sample) => ({ ...sample }))
  const radius = window / 2
  const secondsAt = (index: number) => Number(samples[index]?.elapsed_seconds)
  const finiteValue = (index: number, key: 'cpu_temp_c' | 'gpu_temp_c') => {
    const value = samples[index]?.[key]
    return typeof value === 'number' && Number.isFinite(value) ? value : null
  }

  for (const key of ['cpu_temp_c', 'gpu_temp_c'] as const) {
    let segmentStart = 0
    while (segmentStart < samples.length) {
      while (
        segmentStart < samples.length
        && (!Number.isFinite(secondsAt(segmentStart)) || finiteValue(segmentStart, key) == null)
      ) {
        segmentStart += 1
      }
      if (segmentStart >= samples.length) break

      let segmentEnd = segmentStart + 1
      while (segmentEnd < samples.length) {
        const seconds = secondsAt(segmentEnd)
        const previousSeconds = secondsAt(segmentEnd - 1)
        if (
          !Number.isFinite(seconds)
          || finiteValue(segmentEnd, key) == null
          || !Number.isFinite(previousSeconds)
          || seconds - previousSeconds > GAP_SECONDS
        ) break
        segmentEnd += 1
      }

      let left = segmentStart
      let right = segmentStart
      let sum = 0
      for (let index = segmentStart; index < segmentEnd; index += 1) {
        const center = secondsAt(index)
        const upper = center + radius
        const lower = center - radius
        while (right < segmentEnd && secondsAt(right) <= upper) {
          sum += finiteValue(right, key) ?? 0
          right += 1
        }
        while (left < right && secondsAt(left) < lower) {
          sum -= finiteValue(left, key) ?? 0
          left += 1
        }
        const count = right - left
        if (count > 0) smoothed[index][key] = Math.round((sum / count) * 100) / 100
      }
      segmentStart = segmentEnd
    }
  }
  return smoothed
}

interface ChartSeries {
  id: string
  name: string
  color: string
  cpuPath: string
  gpuPath: string
  legendX: number
  legendY: number
}

interface ChartMarker {
  kind: 'CPU' | 'GPU'
  value: number
  runName: string
  color: string
  x: number
  y: number
  labelX: number
  labelY: number
  textAnchor: 'start' | 'end'
}

interface ChartGeometry {
  left: number
  right: number
  top: number
  bottom: number
  maxSeconds: number
  ticks: { value: number; y: number; label: string }[]
  series: ChartSeries[]
  legendSeries: ChartSeries[]
  legendOverflow: number
  maxMarkers: ChartMarker[]
}

function chartGeometry(runs: TemperatureRun[], colorFor: (runId: string) => string): ChartGeometry {
  // Track extrema incrementally: spreading a long sample history into
  // Math.min/max blows past the engine's argument limit and crashes the pane.
  let minObserved = Infinity
  let maxObserved = -Infinity
  let maxSeconds = 1
  for (const run of runs) {
    for (const sample of run.samples ?? []) {
      maxSeconds = Math.max(maxSeconds, Number(sample.elapsed_seconds) || 0)
      for (const key of ['cpu_temp_c', 'gpu_temp_c'] as const) {
        const rawValue = sample[key]
        if (typeof rawValue === 'number' && Number.isFinite(rawValue)) {
          if (rawValue < minObserved) minObserved = rawValue
          if (rawValue > maxObserved) maxObserved = rawValue
        }
      }
    }
  }

  // Snap the y domain to 10°C steps padded by 5°C, with a 20°C minimum span.
  let minTemp = Number.isFinite(minObserved) ? minObserved : 30
  let maxTemp = Number.isFinite(maxObserved) ? maxObserved : 90
  minTemp = Math.max(0, Math.floor((minTemp - 5) / 10) * 10)
  maxTemp = Math.min(130, Math.ceil((maxTemp + 5) / 10) * 10)
  if (maxTemp - minTemp < 20) maxTemp = Math.min(130, minTemp + 20)
  // The 0–130 clamp must never exclude stored samples: anomalous readings
  // (e.g. 255°C from a disconnected sensor) are valid data, so extend the
  // domain to cover whatever was actually observed.
  if (Number.isFinite(minObserved)) minTemp = Math.min(minTemp, Math.floor((minObserved - 5) / 10) * 10)
  if (Number.isFinite(maxObserved)) maxTemp = Math.max(maxTemp, Math.ceil((maxObserved + 5) / 10) * 10)

  const left = 62
  const right = 878
  const legendColumns = 3
  const legendRows = Math.max(1, Math.ceil(runs.length / legendColumns))
  const bottom = 426
  // Clamp the legend reservation so the plot keeps a usable height (and the
  // temperature scale never inverts) when very many runs are selected.
  const top = Math.min(42 + legendRows * 27, bottom - 160)
  const x = (seconds: number) => left + Math.max(0, Math.min(1, seconds / maxSeconds)) * (right - left)
  const y = (value: number) => bottom - ((value - minTemp) / (maxTemp - minTemp)) * (bottom - top)
  const pathFor = (samples: TemperatureRunSample[] | undefined, key: 'cpu_temp_c' | 'gpu_temp_c') => {
    let path = ''
    let drawing = false
    let previousSeconds: number | null = null
    for (const sample of samples ?? []) {
      const seconds = Number(sample.elapsed_seconds)
      const value = sample[key]
      if (!Number.isFinite(seconds) || typeof value !== 'number' || !Number.isFinite(value)) {
        drawing = false
        previousSeconds = null
        continue
      }
      const gap = previousSeconds != null && seconds - previousSeconds > GAP_SECONDS
      path += `${!drawing || gap ? 'M' : 'L'}${x(seconds).toFixed(1)},${y(value).toFixed(1)} `
      drawing = true
      previousSeconds = seconds
    }
    return path.trim()
  }
  const series = runs.map((run, index) => ({
    id: run.id,
    name: run.name,
    color: colorFor(run.id),
    cpuPath: pathFor(run.samples, 'cpu_temp_c'),
    gpuPath: pathFor(run.samples, 'gpu_temp_c'),
    legendX: left + (index % legendColumns) * 272,
    legendY: 48 + Math.floor(index / legendColumns) * 27,
  }))
  // Only render legend entries that fit above the (clamped) plot top; the
  // remaining runs still plot, and a "+N more" indicator takes the last slot.
  const maxLegendRows = Math.max(1, Math.floor((top - 42) / 27))
  const maxLegendEntries = legendColumns * maxLegendRows
  const legendOverflow = series.length > maxLegendEntries ? series.length - maxLegendEntries + 1 : 0
  const legendSeries = series.slice(0, legendOverflow ? maxLegendEntries - 1 : maxLegendEntries)
  const maximumFor = (key: 'cpu_temp_c' | 'gpu_temp_c', kind: 'CPU' | 'GPU', markerIndex: number): ChartMarker | null => {
    let maximum: { value: number; seconds: number; runId: string; runName: string } | null = null
    for (const run of runs) {
      for (const sample of run.samples ?? []) {
        const seconds = Number(sample.elapsed_seconds)
        const value = sample[key]
        if (
          !Number.isFinite(seconds)
          || typeof value !== 'number'
          || !Number.isFinite(value)
          || (maximum && value <= maximum.value)
        ) continue
        maximum = { value, seconds, runId: run.id, runName: run.name || run.node_name || run.id }
      }
    }
    if (!maximum) return null
    const markerX = x(maximum.seconds)
    const markerY = y(maximum.value)
    const preferBelow = markerY < top + 24 || markerIndex === 1
    const labelY = preferBelow && markerY < bottom - 24 ? markerY + 21 : markerY - 12
    return {
      kind,
      value: maximum.value,
      runName: maximum.runName,
      color: colorFor(maximum.runId),
      x: markerX,
      y: markerY,
      labelX: markerX > right - 190 ? markerX - 10 : markerX + 10,
      labelY,
      textAnchor: markerX > right - 190 ? 'end' : 'start',
    }
  }
  const maxMarkers = [
    maximumFor('cpu_temp_c', 'CPU', 0),
    maximumFor('gpu_temp_c', 'GPU', 1),
  ].filter((marker): marker is ChartMarker => Boolean(marker))
  const tickCount = 5
  const ticks = Array.from({ length: tickCount }, (_, index) => {
    const value = maxTemp - (index * (maxTemp - minTemp) / (tickCount - 1))
    return { value, y: y(value), label: `${Math.round(value)}°C` }
  })
  return { left, right, top, bottom, maxSeconds, ticks, series, legendSeries, legendOverflow, maxMarkers }
}

// The live chart inherits the document theme through CSS classes. A serialized
// standalone SVG has no access to those styles, so the PNG export injects the
// same classes with the theme values resolved to concrete colors.
function exportStyles(surface: string) {
  const root = getComputedStyle(document.documentElement)
  const value = (name: string, fallback: string) => root.getPropertyValue(name).trim() || fallback
  const text = value('--text-tertiary', '#86868b')
  const heading = value('--text-primary', '#1d1d1f')
  const border = value('--border-strong', 'rgba(0, 0, 0, 0.14)')
  return [
    `.temperature-chart-surface{fill:${surface}}`,
    `.chart-grid-line{stroke:${border};stroke-width:1;stroke-dasharray:3 4}`,
    `.temperature-chart-axis{stroke:${border};stroke-width:1.3}`,
    `.chart-axis-label{fill:${text};font-size:10px}`,
    `.chart-axis-title{fill:${text};font-size:9px}`,
    `.temperature-chart-heading-text{fill:${heading};font-size:16px;font-weight:600}`,
    `.temperature-legend-name{fill:${heading};font-size:12px;font-weight:600}`,
    `.temperature-series-line{fill:none;stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round}`,
    `.temperature-max-circle{stroke:${surface}}`,
    `.temperature-max-rect{fill:${surface}}`,
    `.temperature-max-label{fill:${heading};stroke:${surface};stroke-width:4px;stroke-linejoin:round;paint-order:stroke;font-size:11px;font-weight:600}`,
  ].join('\n')
}

function downloadBlob(blob: Blob, extension: string) {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-')
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `temperature-runs-${stamp}.${extension}`
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export function TemperatureRuns() {
  const runs = useResource((signal) => api.temperatureRuns.list(signal))
  const nodes = useResource((signal) => api.nodes.list(signal))
  const [selected, setSelected] = useState<Record<string, boolean>>({})
  const [details, setDetails] = useState<Record<string, TemperatureRun>>({})
  const [targetNode, setTargetNode] = useState('')
  const [targetTemp, setTargetTemp] = useState(60)
  const [triggerMargin, setTriggerMargin] = useState(5)
  const [armBusy, setArmBusy] = useState(false)
  const [cancelBusy, setCancelBusy] = useState(false)
  const [actionError, setActionError] = useState<string>()
  // Background sample polls fail separately from arm/cancel/rename actions so
  // a recovered poll can clear its own alert without clobbering theirs.
  const [detailError, setDetailError] = useState<string>()
  const [renameId, setRenameId] = useState<string>()
  const [renameValue, setRenameValue] = useState('')
  const [renameBusy, setRenameBusy] = useState(false)
  const [smoothingSeconds, setSmoothingSeconds] = useState(0)
  const [chartTitle, setChartTitle] = useState('Temperature Comparison')
  const [colors, setColors] = useState<Record<string, string>>(loadStoredColors)
  const [colorPicker, setColorPicker] = useState<{ runId: string; left: number; top: number }>()
  const colorPickerRef = useRef<HTMLInputElement>(null)
  const chartRef = useRef<SVGSVGElement>(null)
  const chartWrapRef = useRef<HTMLDivElement>(null)
  const detailRequestsRef = useRef(new Set<string>())
  const detailVersionsRef = useRef(new Map<string, number>())

  const state = runs.data
  const runList = state?.runs ?? []
  const activeRun = runList.find((run) => run.id === state?.active_run_id)
  const targetNodes = (nodes.data ?? []).filter((node) => node.online !== false)

  const loadRunDetail = useCallback(async (runId: string, force = false) => {
    if (!force && detailRequestsRef.current.has(runId)) return
    detailRequestsRef.current.add(runId)
    // Only the newest request for a run may write its detail, so a slow
    // pre-completion response cannot overwrite fresher samples.
    const version = (detailVersionsRef.current.get(runId) ?? 0) + 1
    detailVersionsRef.current.set(runId, version)
    try {
      const detail = await api.temperatureRuns.get(runId)
      if (detailVersionsRef.current.get(runId) === version) {
        setDetails((current) => ({ ...current, [runId]: detail }))
      }
      setDetailError(undefined)
    } catch (reason) {
      setDetailError(reason instanceof Error ? `Could not load temperature run: ${reason.message}` : 'Could not load temperature run')
    } finally {
      detailRequestsRef.current.delete(runId)
    }
  }, [])

  // Default the recorder target to the first available node.
  const nodeItems = nodes.data
  useEffect(() => {
    const options = (nodeItems ?? []).filter((node) => node.online !== false)
    if (options.length && !options.some((node) => node.id === targetNode)) {
      setTargetNode(options[0].id)
    }
  }, [nodeItems, targetNode])

  // A newly armed run is always interesting: select it automatically.
  const activeRunId = state?.active_run_id
  useEffect(() => {
    if (!activeRunId) return
    setSelected((current) => current[activeRunId] ? current : { ...current, [activeRunId]: true })
  }, [activeRunId])

  // Lazily fetch samples for every selected run.
  useEffect(() => {
    for (const run of state?.runs ?? []) {
      if (selected[run.id] && !details[run.id]) void loadRunDetail(run.id)
    }
  }, [state, selected, details, loadRunDetail])

  // While a run is armed or recording, poll the list and the active run's samples.
  const polling = activeRun?.status === 'armed' || activeRun?.status === 'recording'
  const reloadRuns = runs.reload
  useEffect(() => {
    if (!polling) return
    const interval = setInterval(() => {
      reloadRuns()
      if (activeRunId) void loadRunDetail(activeRunId)
    }, 2000)
    return () => clearInterval(interval)
  }, [polling, activeRunId, reloadRuns, loadRunDetail])

  // Once the list shows a previously active run has finished, force one final
  // detail refetch: the last in-flight poll may have returned pre-completion
  // samples, and the lazy loader will not refetch a cached detail on its own.
  // This runs even when the run is deselected, so a later reselection never
  // resurrects a partial cache entry.
  const previousActiveRef = useRef<string | null>(null)
  useEffect(() => {
    const previous = previousActiveRef.current
    previousActiveRef.current = activeRunId ?? null
    if (previous && previous !== activeRunId) {
      void loadRunDetail(previous, true)
    }
  }, [activeRunId, loadRunDetail])

  // Chromium anchors its native color panel to the input, so keep the
  // invisible input directly below the legend dot instead of off-screen.
  useEffect(() => {
    if (!colorPicker) return
    const picker = colorPickerRef.current
    if (!picker) return
    picker.value = runColor(colorPicker.runId, colors)
    picker.focus({ preventScroll: true })
    const withPicker = picker as HTMLInputElement & { showPicker?: () => void }
    try {
      if (typeof withPicker.showPicker === 'function') withPicker.showPicker()
      else picker.click()
    } catch {
      // Older browsers can reject showPicker even during a user gesture.
      picker.click()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [colorPicker])

  const selectedRuns = runList
    .filter((run) => selected[run.id])
    .map((run) => details[run.id])
    .filter((run): run is TemperatureRun => Boolean(run))
  const colorFor = (runId: string) => runColor(runId, colors)
  const chartRuns = selectedRuns.map((run) => ({ ...run, samples: smoothSamples(run.samples ?? [], smoothingSeconds) }))
  const geometry = chartGeometry(chartRuns, colorFor)
  const startsAt = (Number(targetTemp || 0) * (1 + Number(triggerMargin || 0) / 100)).toFixed(1)

  const toggleRun = (runId: string) => {
    const next = !selected[runId]
    setSelected((current) => ({ ...current, [runId]: next }))
    if (next && !details[runId]) void loadRunDetail(runId)
  }

  const arm = async () => {
    if (armBusy) return
    setArmBusy(true)
    setActionError(undefined)
    try {
      const run = await api.temperatureRuns.arm({
        node_id: targetNode || 'local',
        target_temp_c: Number(targetTemp) || 0,
        trigger_margin_pct: Number(triggerMargin) || 0,
      })
      setDetails((current) => ({ ...current, [run.id]: run }))
      setSelected((current) => ({ ...current, [run.id]: true }))
      reloadRuns()
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : 'Could not arm the temperature run')
    } finally {
      setArmBusy(false)
    }
  }

  const cancelRun = async () => {
    if (cancelBusy) return
    setCancelBusy(true)
    setActionError(undefined)
    try {
      const run = await api.temperatureRuns.cancel()
      setDetails((current) => ({ ...current, [run.id]: run }))
      reloadRuns()
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : 'Could not cancel the temperature run')
    } finally {
      setCancelBusy(false)
    }
  }

  const saveRename = async (runId: string) => {
    const name = renameValue.trim()
    if (!name || renameBusy) return
    setRenameBusy(true)
    setActionError(undefined)
    try {
      const run = await api.temperatureRuns.rename(runId, name)
      setDetails((current) => ({ ...current, [runId]: run }))
      setRenameId(undefined)
      reloadRuns()
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : 'Could not rename the temperature run')
    } finally {
      setRenameBusy(false)
    }
  }

  const openColorPicker = (runId: string, anchor: Element) => {
    const rect = anchor.getBoundingClientRect()
    setColorPicker({
      runId,
      left: Math.round(Math.max(0, Math.min(window.innerWidth - 2, rect.left + rect.width / 2))),
      top: Math.round(Math.max(0, Math.min(window.innerHeight - 2, rect.bottom + 2))),
    })
  }

  const setRunColor = (color: string) => {
    if (!colorPicker || !/^#[0-9a-f]{6}$/i.test(color)) return
    const next = { ...colors, [colorPicker.runId]: color.toLowerCase() }
    setColors(next)
    try {
      localStorage.setItem(COLOR_STORAGE_KEY, JSON.stringify(next))
    } catch {
      // The selected color still applies for this session when storage is unavailable.
    }
  }

  const exportCSV = () => {
    if (!selectedRuns.length) return
    const cell = (value: unknown) => {
      const text = value == null ? '' : String(value)
      return `"${text.replace(/"/g, '""')}"`
    }
    const rows: unknown[][] = [[
      'run_id', 'run_name', 'node', 'elapsed_seconds', 'cpu_temp_c', 'gpu_temp_c',
    ]]
    for (const run of selectedRuns) {
      for (const sample of run.samples ?? []) {
        rows.push([
          run.id, run.name, run.node_name || run.node_id,
          sample.elapsed_seconds, sample.cpu_temp_c, sample.gpu_temp_c,
        ])
      }
    }
    const csv = rows.map((row) => row.map(cell).join(',')).join('\n')
    downloadBlob(new Blob([csv], { type: 'text/csv;charset=utf-8' }), 'csv')
  }

  const exportPNG = async () => {
    const svg = chartRef.current
    if (!svg || !selectedRuns.length) return
    let svgUrl: string | undefined
    try {
      const surface = chartWrapRef.current
        ? getComputedStyle(chartWrapRef.current).backgroundColor
        : 'transparent'
      const clone = svg.cloneNode(true) as SVGSVGElement
      clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
      clone.setAttribute('width', '1800')
      clone.setAttribute('height', '960')
      const style = document.createElementNS('http://www.w3.org/2000/svg', 'style')
      style.textContent = exportStyles(surface)
      clone.insertBefore(style, clone.firstChild)
      const source = new XMLSerializer().serializeToString(clone)
      svgUrl = URL.createObjectURL(new Blob([source], { type: 'image/svg+xml' }))
      const image = new Image()
      const url = svgUrl
      await new Promise<void>((resolve, reject) => {
        image.onload = () => resolve()
        image.onerror = () => reject(new Error('generated SVG could not be decoded'))
        image.src = url
      })
      const canvas = document.createElement('canvas')
      canvas.width = 1800
      canvas.height = 960
      const context = canvas.getContext('2d')
      if (!context) throw new Error('canvas 2D context unavailable')
      context.drawImage(image, 0, 0, canvas.width, canvas.height)
      const png = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'))
      if (!png) throw new Error('PNG encoder returned no data')
      downloadBlob(png, 'png')
    } catch (reason) {
      setActionError(`PNG export failed: ${reason instanceof Error ? reason.message : String(reason)}`)
    } finally {
      if (svgUrl) URL.revokeObjectURL(svgUrl)
    }
  }

  return (
    <div className="temperature-runs-page">
      <div className="section-heading">
        <div><h2>Temperature runs</h2><p>Overlay CPU and GPU cool-down curves across nodes. Runs align at 0 seconds when the recording threshold is reached.</p></div>
        <div className="temperature-export-actions">
          <Button onClick={exportCSV} disabled={selectedRuns.length === 0}>Export CSV</Button>
          <Button variant="primary" disabled={selectedRuns.length === 0} onClick={() => void exportPNG()}>Export PNG</Button>
        </div>
      </div>
      {runs.loading && <LoadingState label="Loading temperature runs" />}
      {runs.error && <ErrorState message={runs.error} onRetry={reloadRuns} />}
      {state && <>
        <Panel className="temperature-recorder-card">
          <div className="temperature-recorder-controls">
            <label>
              <span>Target node</span>
              <select value={targetNode} disabled={Boolean(activeRun)} onChange={(event) => setTargetNode(event.target.value)}>
                {targetNodes.length === 0 && <option value="local">This node</option>}
                {targetNodes.map((node) => <option key={node.id} value={node.id}>{node.name || node.id}</option>)}
              </select>
            </label>
            <label>
              <span>Stop below</span>
              <span className="temperature-input-suffix">
                <input type="number" min={1} max={120} step={0.1} value={targetTemp} disabled={Boolean(activeRun)} onChange={(event) => setTargetTemp(Number(event.target.value) || 0)} />
                <span>°C</span>
              </span>
            </label>
            <label>
              <span>Start margin</span>
              <span className="temperature-input-suffix">
                <input type="number" min={0} max={100} step={0.5} value={triggerMargin} disabled={Boolean(activeRun)} onChange={(event) => setTriggerMargin(Number(event.target.value) || 0)} />
                <span>%</span>
              </span>
            </label>
            <div className="temperature-trigger-summary">
              <span className="muted">Starts at</span>
              <strong>{startsAt}°C</strong>
              <small>Uses the hotter available CPU/GPU sensor.</small>
            </div>
            <Button variant="primary" disabled={armBusy || Boolean(activeRun)} onClick={() => void arm()}>{armBusy ? 'Arming…' : 'Record'}</Button>
            {activeRun && <Button disabled={cancelBusy} onClick={() => void cancelRun()}>{cancelBusy ? 'Cancelling…' : 'Cancel'}</Button>}
          </div>
          {activeRun && <div className="temperature-active-state">
            <Status status={activeRun.status}>{STATUS_LABELS[activeRun.status] ?? activeRun.status}</Status>
            {activeRun.status === 'armed' && <span>{`Monitoring ${activeRun.node_name}; recording begins at ${activeRun.trigger_temp_c?.toFixed?.(1)}°C.`}</span>}
            {activeRun.status === 'recording' && <span>{`Recording ${activeRun.node_name} every second; it will stop automatically below ${activeRun.target_temp_c?.toFixed?.(1)}°C.`}</span>}
          </div>}
          {actionError && <p className="inline-error" role="alert">{actionError}</p>}
          {detailError && <p className="inline-error" role="alert">{detailError}</p>}
        </Panel>

        <div className="temperature-runs-workspace">
          <Panel className="temperature-run-library">
            <div className="temperature-panel-head">
              <div><h2>Runs</h2><p>Select any number to overlay.</p></div>
              <span className="temperature-run-count">{runList.length}</span>
            </div>
            {runList.length > 0 && <div className="temperature-run-list">
              {runList.map((run) => <div
                className={`temperature-run-entry${selected[run.id] ? ' selected' : ''}`}
                key={run.id}
                role="button"
                tabIndex={0}
                onClick={() => toggleRun(run.id)}
                onKeyDown={(event) => {
                  // Enter inside the nested rename form must submit the form,
                  // not toggle the run's selection.
                  if (event.key !== 'Enter') return
                  if ((event.target as HTMLElement).closest('input, button, form')) return
                  event.preventDefault()
                  toggleRun(run.id)
                }}
              >
                <span className="temperature-run-check">{selected[run.id] ? '✓' : ''}</span>
                <div className="temperature-run-copy">
                  <div className="temperature-run-title-row">
                    <strong>{run.name}</strong>
                    <Status status={run.status}>{STATUS_LABELS[run.status] ?? run.status}</Status>
                  </div>
                  <span className="muted">{`${run.node_name || run.node_id} · ${formatRunDuration(run.duration_seconds)} · ${run.sample_count} samples`}</span>
                  <span className="muted">{`start ≥ ${run.trigger_temp_c}°C · stop < ${run.target_temp_c}°C`}</span>
                </div>
                <Button variant="tertiary" aria-label={`Rename ${run.name}`} onClick={(event) => { event.stopPropagation(); setRenameId(run.id); setRenameValue(run.name || '') }}><Pencil size={14} /></Button>
                {renameId === run.id && <form className="temperature-run-rename" onClick={(event) => event.stopPropagation()} onSubmit={(event) => { event.preventDefault(); void saveRename(run.id) }}>
                  <input maxLength={120} value={renameValue} autoFocus onChange={(event) => setRenameValue(event.target.value)} />
                  <Button variant="primary" type="submit" disabled={renameBusy}>Save</Button>
                  <Button type="button" onClick={() => setRenameId(undefined)}>Cancel</Button>
                </form>}
              </div>)}
            </div>}
            {runList.length === 0 && <EmptyState title="No temperature runs yet" description="Arm a recording above to capture a cool-down curve." />}
          </Panel>

          <Panel className="temperature-run-viewer">
            <div className="temperature-panel-head">
              <div><h2>Graph viewer</h2><p>CPU is solid; GPU is dashed. Every run uses one color.</p></div>
              <div className="temperature-viewer-controls">
                <label className="temperature-chart-title-control">
                  <span>PNG title</span>
                  <input type="text" maxLength={80} value={chartTitle} placeholder="Temperature Comparison" onChange={(event) => setChartTitle(event.target.value)} />
                </label>
                <label title="Centered moving-average window. CSV export always keeps the raw samples.">
                  <span>Smoothing</span>
                  <select value={smoothingSeconds} onChange={(event) => setSmoothingSeconds(Number(event.target.value))}>
                    <option value={0}>Off</option>
                    <option value={3}>3 seconds</option>
                    <option value={5}>5 seconds</option>
                    <option value={10}>10 seconds</option>
                    <option value={30}>30 seconds</option>
                    <option value={60}>60 seconds</option>
                  </select>
                </label>
                <span className="muted">{selectedRuns.length} selected</span>
              </div>
            </div>
            <div className="temperature-run-chart-wrap" ref={chartWrapRef}>
              <svg ref={chartRef} className="temperature-run-chart" viewBox="0 0 900 480" role="img" aria-label="Selected CPU and GPU temperature runs">
                <rect width={900} height={480} rx={8} className="temperature-chart-surface" />
                <text x={450} y={24} textAnchor="middle" className="temperature-chart-heading-text">{chartTitle}</text>
                {geometry.ticks.map((tick) => <g key={tick.value}>
                  <line className="chart-grid-line" x1={geometry.left} y1={tick.y} x2={geometry.right} y2={tick.y} />
                  <line className="temperature-chart-axis" x1={geometry.left - 5} y1={tick.y} x2={geometry.left} y2={tick.y} />
                  <text className="chart-axis-label" x={geometry.left - 10} y={tick.y + 4} textAnchor="end">{tick.label}</text>
                </g>)}
                {geometry.series.map((series) => <g key={series.id}>
                  {series.cpuPath && <path d={series.cpuPath} stroke={series.color} className="temperature-series-line" />}
                  {series.gpuPath && <path d={series.gpuPath} stroke={series.color} strokeDasharray="4 7" className="temperature-series-line" />}
                </g>)}
                {geometry.maxMarkers.map((marker) => <g key={marker.kind}>
                  {marker.kind === 'GPU'
                    ? <rect x={marker.x - 4.5} y={marker.y - 4.5} width={9} height={9} rx={1} transform={`rotate(45 ${marker.x} ${marker.y})`} className="temperature-max-rect" stroke={marker.color} strokeWidth={2.5} />
                    : <circle cx={marker.x} cy={marker.y} r={4.5} fill={marker.color} className="temperature-max-circle" strokeWidth={1.5} />}
                  <text x={marker.labelX} y={marker.labelY} textAnchor={marker.textAnchor} className="temperature-max-label">{`${marker.kind} max ${marker.value.toFixed(1)}°C · ${truncate(marker.runName, 15)}`}</text>
                </g>)}
                {geometry.legendSeries.map((series) => <g key={`legend-${series.id}`} transform={`translate(${series.legendX} ${series.legendY})`}>
                  <circle
                    cx={4} cy={0} r={5} fill={series.color}
                    className="temperature-run-color-dot"
                    role="button"
                    tabIndex={0}
                    aria-label={`Change color for ${series.name || 'temperature run'}`}
                    onClick={(event) => openColorPicker(series.id, event.currentTarget)}
                    onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); openColorPicker(series.id, event.currentTarget) } }}
                  ><title>Change run color</title></circle>
                  <text x={13} y={4} className="temperature-legend-name">{truncate(series.name || '', 18)}</text>
                  <line x1={145} y1={0} x2={164} y2={0} stroke={series.color} strokeWidth={2.5} />
                  <text x={168} y={4} className="chart-axis-label">CPU</text>
                  <line x1={202} y1={0} x2={221} y2={0} stroke={series.color} strokeWidth={2.5} strokeDasharray="4 5" />
                  <text x={225} y={4} className="chart-axis-label">GPU</text>
                </g>)}
                {geometry.legendOverflow > 0 && <text
                  x={geometry.left + (geometry.legendSeries.length % 3) * 272}
                  y={48 + Math.floor(geometry.legendSeries.length / 3) * 27 + 4}
                  className="chart-axis-label"
                >{`+${geometry.legendOverflow} more`}</text>}
                {geometry.series.length === 0 && <text x={450} y={244} textAnchor="middle" className="chart-axis-label">Select one or more runs to graph.</text>}
                <line className="temperature-chart-axis" x1={geometry.left} y1={geometry.top} x2={geometry.left} y2={geometry.bottom} />
                <line className="temperature-chart-axis" x1={geometry.left} y1={geometry.bottom} x2={geometry.right} y2={geometry.bottom} />
                <text className="chart-axis-label" x={geometry.left} y={452} textAnchor="middle">0s</text>
                <text className="chart-axis-label" x={geometry.right} y={452} textAnchor="middle">{formatRunDuration(geometry.maxSeconds)}</text>
                <text className="chart-axis-title" x={450} y={472} textAnchor="middle">Time since recording start</text>
                <text className="chart-axis-title" x={15} y={(geometry.top + geometry.bottom) / 2} textAnchor="middle" transform={`rotate(-90 15 ${(geometry.top + geometry.bottom) / 2})`}>Temperature (°C)</text>
              </svg>
              <input
                ref={colorPickerRef}
                className="temperature-run-color-picker"
                type="color"
                aria-label="Choose temperature run color"
                style={colorPicker ? { left: colorPicker.left, top: colorPicker.top } : undefined}
                onChange={(event) => setRunColor(event.target.value)}
              />
            </div>
          </Panel>
        </div>
      </>}
    </div>
  )
}
