import { useEffect, useMemo, useState } from 'react'
import type { PointerEvent as ReactPointerEvent } from 'react'
import { Fan, Gauge, RefreshCw, Thermometer, Wind } from 'lucide-react'
import { api } from '../api/client'
import type { FanControlNode, FanCurveSettings, FanControlMode } from '../api/types'
import { useResource } from '../hooks/useResource'
import { Button, EmptyState, ErrorState, LoadingState, PageHeader, Panel, Status } from '../components/ui'

const presenceEvent = 'sparkdeck:fan-control-presence-changed'
const pollDelayMs = 2_000
const pendingOverrideMismatchLimit = 2

interface PendingFanOverride {
  enabled: boolean
  lastTelemetryTs: number
  mismatches: number
}

interface CurveDraft {
  curve: FanCurveSettings
  serverSignature: string
  dirty: boolean
  pendingSignature?: string
}

function cloneCurve(curve: FanCurveSettings): FanCurveSettings {
  return { ...curve, curve_points: curve.curve_points.map((point) => [...point]) }
}

function curveSignature(curve: FanCurveSettings): string {
  return JSON.stringify(curve)
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value))
}

function valueOrDash(value: number | null | undefined, suffix = '') {
  return typeof value === 'number' && Number.isFinite(value) ? `${Math.round(value)}${suffix}` : '--'
}

function modeLabel(mode: FanControlMode) {
  if (mode === 'pid') return 'PID'
  return mode.charAt(0).toUpperCase() + mode.slice(1)
}

function ModeSettings({ node }: { node: FanControlNode }) {
  const mode = node.fan.mode
  const settings = node.settings.settings[mode]
  if (!settings) return <p className="muted">No settings were reported for this mode.</p>

  let rows: Array<[string, string]>
  if (mode === 'curve') {
    const curve = settings as FanCurveSettings
    rows = [
      ['Temperature range', `${curve.curve_min_temp} - ${curve.curve_max_temp} °C`],
      ['Minimum fan floor', `${curve.min_floor_pct}%`],
      ['Control points', String(curve.curve_points.length)],
    ]
  } else if (mode === 'pid') {
    const pid = settings as typeof node.settings.settings.pid
    rows = [
      ['Setpoint', `${pid.setpoint} °C`], ['Kp', String(pid.kp)], ['Ki', String(pid.ki)],
      ['Kd', String(pid.kd)], ['Minimum fan floor', `${pid.min_floor_pct}%`],
    ]
  } else if (mode === 'hysteresis') {
    const hysteresis = settings as typeof node.settings.settings.hysteresis
    rows = [['Fan on', `${hysteresis.hyst_on_temp} °C`], ['Fan off', `${hysteresis.hyst_off_temp} °C`]]
  } else {
    const manual = settings as typeof node.settings.settings.manual
    rows = [['Manual duty', `${manual.manual_duty_pct}%`]]
  }

  return <dl className="fan-control-settings-list">
    {rows.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
  </dl>
}

function CurveChart({
  curve,
  node,
  onPointChange,
}: {
  curve: FanCurveSettings
  node: FanControlNode
  onPointChange: (index: number, temperature: number, duty: number) => void
}) {
  const chartId = node.node_id.replace(/[^a-zA-Z0-9_-]/g, '-')
  const [draggingPoint, setDraggingPoint] = useState<number | null>(null)
  const configuredPoints = curve.curve_points.flatMap((point, index) => (
    point.length >= 2 && Number.isFinite(point[0]) && Number.isFinite(point[1])
      ? [{ temperature: point[0], duty: point[1], index }]
      : []
  ))
  const minTemp = Number.isFinite(curve.curve_min_temp) ? curve.curve_min_temp : 0
  const maxTemp = Number.isFinite(curve.curve_max_temp) && curve.curve_max_temp > minTemp ? curve.curve_max_temp : minTemp + 1
  const points = configuredPoints.filter(({ temperature }) => temperature >= minTemp && temperature <= maxTemp)
  const omittedPointCount = configuredPoints.length - points.length
  const x = (temp: number) => 48 + ((temp - minTemp) / (maxTemp - minTemp)) * 432
  const y = (duty: number) => 18 + ((100 - Math.max(0, Math.min(100, duty))) / 100) * 172
  const line = points.map(({ temperature, duty }) => `${x(temperature)},${y(duty)}`).join(' ')
  const liveTemp = node.fan.temp
  const liveDuty = node.fan.duty_pct
  const livePoint = typeof liveTemp === 'number' && typeof liveDuty === 'number'
    && liveTemp >= minTemp && liveTemp <= maxTemp

  const moveFromPointer = (event: ReactPointerEvent<SVGCircleElement>, index: number) => {
    if (draggingPoint !== index) return
    const svg = event.currentTarget.ownerSVGElement
    if (!svg) return
    const bounds = svg.getBoundingClientRect()
    if (bounds.width <= 0 || bounds.height <= 0) return
    const chartX = (event.clientX - bounds.left) * (510 / bounds.width)
    const chartY = (event.clientY - bounds.top) * (225 / bounds.height)
    const temperature = minTemp + ((chartX - 48) / 432) * (maxTemp - minTemp)
    const duty = 100 - ((chartY - 18) / 172) * 100
    onPointChange(index, temperature, duty)
  }

  return <div className="fan-control-chart-wrap">
    <svg className={`fan-control-chart${draggingPoint === null ? '' : ' is-dragging'}`} viewBox="0 0 510 225" role="img" aria-labelledby={`fan-curve-title-${chartId}`} aria-describedby={`fan-curve-desc-${chartId}`}>
      <title id={`fan-curve-title-${chartId}`}>Fan curve for {node.node_name}</title>
      <desc id={`fan-curve-desc-${chartId}`}>Fan duty percentage by temperature. Drag a curve point to edit it, or use the numeric fields below.</desc>
      {[0, 25, 50, 75, 100].map((duty) => <g key={duty}>
        <line className="fan-chart-grid" x1="48" x2="480" y1={y(duty)} y2={y(duty)} />
        <text className="fan-chart-label" x="40" y={y(duty) + 4} textAnchor="end">{duty}%</text>
      </g>)}
      <line className="fan-chart-axis" x1="48" x2="480" y1="190" y2="190" />
      <text className="fan-chart-label" x="48" y="211">{minTemp} °C</text>
      <text className="fan-chart-label" x="480" y="211" textAnchor="end">{maxTemp} °C</text>
      {points.length > 1 && <polyline className="fan-chart-line" points={line} />}
      {points.map(({ temperature, duty, index }) => <g key={index}>
        <circle
          className="fan-chart-point fan-chart-point-editable"
          cx={x(temperature)}
          cy={y(duty)}
          r="6"
          role="button"
          tabIndex={0}
          aria-label={`Move point ${index + 1}, ${temperature} degrees, ${duty} percent`}
          onPointerDown={(event) => {
            event.preventDefault()
            setDraggingPoint(index)
            event.currentTarget.setPointerCapture?.(event.pointerId)
          }}
          onPointerMove={(event) => moveFromPointer(event, index)}
          onPointerUp={(event) => {
            moveFromPointer(event, index)
            event.currentTarget.releasePointerCapture?.(event.pointerId)
            setDraggingPoint(null)
          }}
          onPointerCancel={() => setDraggingPoint(null)}
          onKeyDown={(event) => {
            const step = event.shiftKey ? 5 : 1
            if (event.key === 'ArrowLeft') onPointChange(index, temperature - step, duty)
            else if (event.key === 'ArrowRight') onPointChange(index, temperature + step, duty)
            else if (event.key === 'ArrowDown') onPointChange(index, temperature, duty - step)
            else if (event.key === 'ArrowUp') onPointChange(index, temperature, duty + step)
            else return
            event.preventDefault()
          }}
        />
        <text className="fan-chart-point-label" x={x(temperature)} y={y(duty) - 11} textAnchor="middle">{temperature}° / {duty}%</text>
      </g>)}
      {livePoint && <g>
        <circle className="fan-chart-live-halo" cx={x(liveTemp)} cy={y(liveDuty)} r="8" />
        <circle className="fan-chart-live" cx={x(liveTemp)} cy={y(liveDuty)} r="4" />
      </g>}
    </svg>
    {omittedPointCount > 0 && <p className="muted">{omittedPointCount} configured {omittedPointCount === 1 ? 'point is' : 'points are'} outside this temperature range and not plotted.</p>}
    <table className="fan-curve-point-table">
      <caption>Fan curve points for {node.node_name}</caption>
      <thead><tr><th>Temperature Celsius</th><th>Fan duty percent</th></tr></thead>
      <tbody>{configuredPoints.map(({ temperature, duty, index }) => <tr key={index}>
        <td><input type="number" step="0.5" value={temperature} aria-label={`Point ${index + 1} temperature`} onChange={(event) => {
          const next = Number(event.target.value)
          if (Number.isFinite(next)) onPointChange(index, next, duty)
        }} /></td>
        <td><input type="number" min="0" max="100" step="1" value={duty} aria-label={`Point ${index + 1} fan duty`} onChange={(event) => {
          const next = Number(event.target.value)
          if (Number.isFinite(next)) onPointChange(index, temperature, next)
        }} /></td>
      </tr>)}</tbody>
    </table>
  </div>
}

export function FanControlPage() {
  const resource = useResource((signal) => api.fanControl.get(signal))
  const [selectedNodeId, setSelectedNodeId] = useState('')
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState('')
  const [saveStatus, setSaveStatus] = useState('')
  const [pendingOverrides, setPendingOverrides] = useState<Record<string, PendingFanOverride>>({})
  const [curveDrafts, setCurveDrafts] = useState<Record<string, CurveDraft>>({})

  useEffect(() => {
    if (resource.loading) return
    const timeout = window.setTimeout(resource.reload, pollDelayMs)
    return () => window.clearTimeout(timeout)
  }, [resource.loading, resource.reload])

  useEffect(() => {
    const nodes = resource.data?.nodes ?? []
    if (nodes.length && !nodes.some((node) => node.node_id === selectedNodeId)) setSelectedNodeId(nodes[0].node_id)
  }, [resource.data, selectedNodeId])

  useEffect(() => {
    const overview = resource.data
    if (!overview) return
    setPendingOverrides((current) => {
      const next = { ...current }
      let changed = false
      for (const item of overview.nodes) {
        const pending = next[item.node_id]
        if (!pending) continue
        if (pending.enabled === item.fan.max_speed) {
          delete next[item.node_id]
          changed = true
        } else if (item.fan.ts > pending.lastTelemetryTs) {
          const mismatches = pending.mismatches + 1
          if (mismatches >= pendingOverrideMismatchLimit) {
            delete next[item.node_id]
          } else {
            next[item.node_id] = {
              ...pending,
              lastTelemetryTs: item.fan.ts,
              mismatches,
            }
          }
          changed = true
        }
      }
      return changed ? next : current
    })
  }, [resource.data])

  useEffect(() => {
    const nodes = resource.data?.nodes ?? []
    setCurveDrafts((current) => {
      const next = { ...current }
      let changed = false
      for (const item of nodes) {
        const serverCurve = item.settings.settings.curve
        const signature = curveSignature(serverCurve)
        const existing = next[item.node_id]
        if (!existing) {
          next[item.node_id] = { curve: cloneCurve(serverCurve), serverSignature: signature, dirty: false }
          changed = true
        } else if (existing.pendingSignature) {
          if (existing.pendingSignature === signature) {
            next[item.node_id] = { curve: cloneCurve(serverCurve), serverSignature: signature, dirty: false }
            changed = true
          }
        } else if (!existing.dirty && existing.serverSignature !== signature) {
          next[item.node_id] = { curve: cloneCurve(serverCurve), serverSignature: signature, dirty: false }
          changed = true
        }
      }
      return changed ? next : current
    })
  }, [resource.data])

  const node = useMemo(
    () => resource.data?.nodes.find((item) => item.node_id === selectedNodeId) ?? resource.data?.nodes[0],
    [resource.data, selectedNodeId],
  )

  const refresh = () => {
    resource.reload()
    window.dispatchEvent(new Event(presenceEvent))
  }

  const setMaxSpeed = async (enabled: boolean) => {
    if (!node || saving) return
    setSaving(true)
    setSaveError('')
    setSaveStatus('')
    try {
      await api.fanControl.setMaxSpeed(node.node_id, enabled)
      setPendingOverrides((current) => ({
        ...current,
        [node.node_id]: {
          enabled,
          lastTelemetryTs: node.fan.ts,
          mismatches: 0,
        },
      }))
      setSaveStatus(enabled ? 'Max fan speed enabled.' : 'Automatic fan control enabled.')
      window.dispatchEvent(new Event(presenceEvent))
    } catch (reason) {
      setSaveError(reason instanceof Error ? reason.message : 'Could not update fan control.')
    } finally {
      setSaving(false)
    }
  }

  const curve = node?.settings.settings.curve
  const curveDraft = node && curveDrafts[node.node_id]?.curve
    ? curveDrafts[node.node_id].curve
    : curve
  const curveDirty = node ? Boolean(curveDrafts[node.node_id]?.dirty) : false
  const maxSpeed = node ? pendingOverrides[node.node_id]?.enabled ?? node.fan.max_speed : false

  const updateCurvePoint = (index: number, temperature: number, duty: number) => {
    if (!node || !curve || !curveDraft) return
    const points = curveDraft.curve_points
    if (!points[index]) return
    const minimum = index === 0 ? curveDraft.curve_min_temp : points[index - 1][0] + 0.5
    const maximum = index === points.length - 1 ? curveDraft.curve_max_temp : points[index + 1][0] - 0.5
    const nextTemperature = Math.round(clamp(temperature, minimum, maximum) * 2) / 2
    const nextDuty = Math.round(clamp(duty, 0, 100))
    if (points[index][0] === nextTemperature && points[index][1] === nextDuty) return
    const nextCurve = cloneCurve(curveDraft)
    nextCurve.curve_points[index] = [nextTemperature, nextDuty]
    setCurveDrafts((current) => ({
      ...current,
      [node.node_id]: {
        curve: nextCurve,
        serverSignature: current[node.node_id]?.serverSignature ?? curveSignature(curve),
        dirty: true,
      },
    }))
    setSaveError('')
    setSaveStatus('')
  }

  const resetCurve = () => {
    if (!node || !curve) return
    setCurveDrafts((current) => ({
      ...current,
      [node.node_id]: {
        curve: cloneCurve(curve),
        serverSignature: curveSignature(curve),
        dirty: false,
      },
    }))
    setSaveError('')
    setSaveStatus('')
  }

  const saveCurve = async () => {
    if (!node || !curveDraft || !curveDirty || saving) return
    const submitted = cloneCurve(curveDraft)
    const signature = curveSignature(submitted)
    setSaving(true)
    setSaveError('')
    setSaveStatus('')
    try {
      await api.fanControl.updateSettings(node.node_id, 'curve', submitted, node.fan.mode)
      setCurveDrafts((current) => ({
        ...current,
        [node.node_id]: {
          curve: submitted,
          serverSignature: signature,
          dirty: false,
          pendingSignature: signature,
        },
      }))
      setSaveStatus(node.fan.mode === 'curve' ? 'Fan curve saved.' : 'Fan curve saved and activated.')
      resource.reload()
      window.dispatchEvent(new Event(presenceEvent))
    } catch (reason) {
      setSaveError(reason instanceof Error ? reason.message : 'Could not save the fan curve.')
    } finally {
      setSaving(false)
    }
  }

  return <div className="page fan-control-page">
    <PageHeader
      eyebrow="Node cooling"
      title="Fan Control"
      description="Monitor FanController telemetry, tune each node's curve, and control its fan speed override."
      actions={<Button type="button" onClick={refresh} disabled={resource.loading}><RefreshCw size={15} aria-hidden="true" /> Refresh</Button>}
    />
    {resource.loading && !resource.data && <LoadingState label="Looking for FanController nodes" />}
    {resource.error && !resource.data && <ErrorState message={resource.error} onRetry={refresh} />}
    {resource.error && resource.data && <p className="inline-error" role="status">Live refresh failed: {resource.error}</p>}
    {resource.data && (!resource.data.available || resource.data.nodes.length === 0) && <EmptyState
      title="FanController not detected"
      description="No cluster node is currently publishing a fresh FanController heartbeat. The Fan Control tab will appear automatically when one is detected."
    />}
    {node && <>
      <Panel className="fan-control-toolbar">
        <div>
          <span className="eyebrow">Controller node</span>
          <label htmlFor="fan-controller-node" className="sr-only">FanController node</label>
          <select id="fan-controller-node" value={node.node_id} onChange={(event) => setSelectedNodeId(event.target.value)}>
            {resource.data?.nodes.map((item) => <option key={item.node_id} value={item.node_id}>{item.node_name}{item.local ? ' (local)' : ''}</option>)}
          </select>
        </div>
        <Status status={node.fan.status || 'running'}>{node.fan.status || 'Running'}</Status>
      </Panel>

      <div className="fan-control-metrics">
        <Panel className="fan-control-metric"><Wind size={19} aria-hidden="true" /><span>Fan speed</span><strong>{valueOrDash(node.fan.rpm, ' RPM')}</strong></Panel>
        <Panel className="fan-control-metric"><Gauge size={19} aria-hidden="true" /><span>Fan duty</span><strong>{valueOrDash(node.fan.duty_pct, '%')}</strong></Panel>
        <Panel className="fan-control-metric"><Thermometer size={19} aria-hidden="true" /><span>Control temperature</span><strong>{valueOrDash(node.fan.temp, ' °C')}</strong></Panel>
        <Panel className="fan-control-metric"><Fan size={19} aria-hidden="true" /><span>Active mode</span><strong>{modeLabel(node.fan.mode)}</strong></Panel>
      </div>

      <Panel className="fan-control-override">
        <div>
          <p className="eyebrow">Fan speed override</p>
          <h2>{maxSpeed ? 'Maximum fan speed' : 'Automatic control'}</h2>
          <p>{maxSpeed ? 'FanController is being forced to 100% duty.' : `FanController is following its ${modeLabel(node.fan.mode)} settings.`}</p>
        </div>
        <label className="fan-control-toggle">
          <span>Auto</span>
          <input
            type="checkbox"
            role="switch"
            aria-label="Fan speed override"
            checked={maxSpeed}
            disabled={saving}
            onChange={(event) => void setMaxSpeed(event.target.checked)}
          />
          <span className="fan-control-toggle-track" aria-hidden="true"><span /></span>
          <span>Max fan speed</span>
        </label>
      </Panel>
      {saveError && <p className="inline-error" role="alert">{saveError}</p>}
      {saveStatus && <p className="inline-success" role="status">{saveStatus}</p>}

      <div className="fan-control-details">
        <Panel className="fan-control-curve-panel">
          <div className="fan-control-panel-heading">
            <div><p className="eyebrow">Temperature response</p><h2>Fan curve</h2></div>
            {node.fan.mode !== 'curve' && <span className="badge">Inactive saved curve</span>}
          </div>
          {curveDraft && <>
            <p className="muted fan-curve-help">Drag points on the graph or enter exact values below.</p>
            <CurveChart curve={curveDraft} node={node} onPointChange={updateCurvePoint} />
            <div className="fan-curve-actions">
              <Button type="button" onClick={resetCurve} disabled={!curveDirty || saving}>Reset</Button>
              <Button type="button" onClick={() => void saveCurve()} disabled={!curveDirty || saving}>
                {saving ? 'Saving…' : node.fan.mode === 'curve' ? 'Save curve' : 'Save and activate curve'}
              </Button>
            </div>
          </>}
        </Panel>
        <Panel className="fan-control-settings-panel">
          <p className="eyebrow">Controller settings</p>
          <h2>{modeLabel(node.fan.mode)} mode</h2>
          <ModeSettings node={node} />
          <div className="fan-control-live-details">
            <span>Local temperature <strong>{valueOrDash(node.fan.local_temp, ' °C')}</strong></span>
            <span>Last heartbeat <strong>{new Date(node.fan.ts * 1000).toLocaleTimeString()}</strong></span>
          </div>
        </Panel>
      </div>
    </>}
  </div>
}
