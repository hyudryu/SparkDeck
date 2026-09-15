import { useState } from 'react'
import type { LiveHistoryBucket, LiveHistorySeries } from '../api/types'

const WIDTH = 960
const HEIGHT = 320
const PADDING = { top: 20, right: 66, bottom: 34, left: 62 }
const PLOT_WIDTH = WIDTH - PADDING.left - PADDING.right
const PLOT_HEIGHT = HEIGHT - PADDING.top - PADDING.bottom
const MAX_COLORED_CONCURRENCY = 10

export type HistoryMetric = 'output' | 'thinking' | 'prefill'

/**
 * Concurrency colours.  One, two, and three sessions keep the fixed colours the
 * panel documents; everything up to ten gets a stable generated hue, and any
 * higher concurrency shares the final colour.
 */
export const CONCURRENCY_COLORS = [
  '#30d158', '#ff9f0a', '#2997ff',
  '#bf5af2', '#ff375f', '#64d2ff',
  '#ffd60a', '#ac8e68', '#5e5ce6', '#32d74b',
] as const

export function concurrencyColor(concurrent: number): string {
  const level = Math.max(1, Math.ceil(concurrent))
  return CONCURRENCY_COLORS[Math.min(level, MAX_COLORED_CONCURRENCY) - 1]
}

export function concurrencyLabel(concurrent: number): string {
  const level = Math.max(1, Math.ceil(concurrent))
  return level > MAX_COLORED_CONCURRENCY ? `C${MAX_COLORED_CONCURRENCY}+` : `C${level}`
}

interface HistoryChartProps {
  series: LiveHistorySeries
  /** Trailing window to draw, in seconds. */
  rangeSeconds: number
  /** Sampling interval, which is also the span of one point on the graph. */
  sampleSeconds: number
  metrics: ReadonlySet<HistoryMetric>
  colorByConcurrency: boolean
}

interface HoverPoint {
  index: number
  bucket: LiveHistoryBucket
  x: number
}

function axisMaximum(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return 1
  const magnitude = 10 ** Math.floor(Math.log10(value))
  return Math.ceil(value / magnitude * 2) / 2 * magnitude
}

function compact(value: number): string {
  if (value >= 1_000) return `${(value / 1_000).toFixed(value >= 10_000 ? 0 : 1)}K`
  return value >= 10 ? String(Math.round(value)) : String(Number(value.toFixed(1)))
}

function clockLabel(at: number): string {
  return new Date(at * 1_000).toLocaleTimeString(undefined, {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  })
}

function minuteLabel(at: number): string {
  return new Date(at * 1_000).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
}

function rateLabel(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? `${value.toFixed(1)} tok/s`
    : 'pending'
}

function secondsLabel(value: number): string {
  // The collector publishes hundredths and the displayed rate is computed from
  // that exact value, so rounding to one decimal here would contradict the very
  // rate the number explains: 1,000 tokens over 1.96 s is 510.2 tok/s, and
  // "2.0 s" would read as 500.
  return `${Number(value.toFixed(2))} s`
}

/**
 * The evidence behind one bucket's prompt-processing rate: the prompt tokens it
 * was computed from and the seconds they took.  Only a prefill that completed in
 * the bucket has this -- the engine reports the prompt token count with the first
 * output token, which is what ends the prefill -- so a bucket whose rate is an
 * earlier measurement, or a prefill still running, shows no prompt size at all
 * rather than one that was never measured.
 */
function promptEvidenceLabel(bucket: LiveHistoryBucket): string | undefined {
  const tokens = Number(bucket.prefill_tokens ?? 0)
  const seconds = Number(bucket.prefill_seconds ?? 0)
  if (!(tokens > 0) || !(seconds > 0)) return undefined
  return `${tokens.toLocaleString()} prompt token${tokens === 1 ? '' : 's'} in ${secondsLabel(seconds)}`
}

/**
 * Line style carries the metric, so the three lines stay distinguishable even
 * when colour is encoding concurrency: solid for output, dotted for thinking,
 * dashed for prompt processing.  The dash pattern is applied inline alongside
 * the concurrency colour, because that colour is what the metric CSS class
 * would otherwise be overridden by.
 */
const METRIC_STYLE: Record<HistoryMetric, { label: string; className: string; dash?: string }> = {
  output: { label: 'Token generation', className: 'history-line-output' },
  thinking: { label: 'Thinking', className: 'history-line-thinking', dash: '1 4' },
  prefill: { label: 'Prompt processing', className: 'history-line-prefill', dash: '7 4' },
}

const METRIC_VALUE: Record<HistoryMetric, (bucket: LiveHistoryBucket) => number | null> = {
  output: (bucket) => bucket.output_tok_s,
  thinking: (bucket) => bucket.thinking_tok_s,
  prefill: (bucket) => bucket.prefill_tok_s,
}

function lineStyle(metric: HistoryMetric, color?: string): React.CSSProperties | undefined {
  const dash = METRIC_STYLE[metric].dash
  if (color === undefined && dash === undefined) return undefined
  return { ...(color === undefined ? {} : { stroke: color }), ...(dash === undefined ? {} : { strokeDasharray: dash }) }
}

/**
 * One graph for one serving unit: tokens per second against the trailing hour,
 * split into five-second buckets and coloured by how many sessions were running.
 *
 * Output and thinking share the left axis; prompt processing gets its own right
 * axis because prefill runs one to two orders of magnitude faster and would
 * otherwise flatten the generation lines against the baseline.
 */
export function HistoryChart({
  series, rangeSeconds, sampleSeconds, metrics, colorByConcurrency,
}: HistoryChartProps) {
  const [hover, setHover] = useState<HoverPoint>()

  const buckets = series.buckets.filter((bucket) => bucket.at >= series.last_at - rangeSeconds)
  const visible = buckets.length > 0 ? buckets : series.buckets.slice(-2)
  const windowEnd = visible.length > 0
    ? Math.max(series.last_at, visible[visible.length - 1].at)
    : series.last_at
  const windowStart = windowEnd - rangeSeconds

  const x = (at: number) => PADDING.left
    + Math.min(1, Math.max(0, (at - windowStart) / rangeSeconds)) * PLOT_WIDTH

  const sampleValues = (metric: HistoryMetric) => visible
    .map((bucket) => METRIC_VALUE[metric](bucket))
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value))

  const generationValues = [
    ...(metrics.has('output') ? sampleValues('output') : []),
    ...(metrics.has('thinking') ? sampleValues('thinking') : []),
  ]
  const prefillValues = metrics.has('prefill') ? sampleValues('prefill') : []
  const leftMax = axisMaximum(Math.max(0, ...generationValues))
  const rightMax = axisMaximum(Math.max(0, ...prefillValues))

  const yLeft = (value: number) => PADDING.top + PLOT_HEIGHT - (value / leftMax) * PLOT_HEIGHT
  const yRight = (value: number) => PADDING.top + PLOT_HEIGHT - (value / rightMax) * PLOT_HEIGHT

  const hovered = hover?.bucket
  const hoverIsEstimated = Boolean(hovered && !hovered.prefill_measured && hovered.prefill_tok_s !== null)
  const hoverPrefillEvidence = hovered === undefined ? undefined : promptEvidenceLabel(hovered)

  const pickBucket = (event: React.PointerEvent<SVGSVGElement>) => {
    if (visible.length === 0) return
    const bounds = event.currentTarget.getBoundingClientRect()
    // A layout-less host (and a chart that has not been laid out yet) reports a
    // zero width; without a floor the whole plot collapses to one column and
    // every hover resolves to the oldest bucket.
    const layoutWidth = bounds.width > 0 ? bounds.width : WIDTH
    const pointer = (event.clientX - bounds.left) / layoutWidth * WIDTH
    let closest = visible[0]
    let closestDistance = Number.POSITIVE_INFINITY
    visible.forEach((bucket) => {
      const distance = Math.abs(x(bucket.at) - pointer)
      if (distance < closestDistance) {
        closest = bucket
        closestDistance = distance
      }
    })
    const index = visible.indexOf(closest)
    setHover({ index, bucket: closest, x: x(closest.at) })
  }

  const timeTicks = [0, .25, .5, .75, 1].map((fraction) => windowStart + fraction * rangeSeconds)
  const valueTicks = [0, .25, .5, .75, 1]

  const seriesPath = (metric: HistoryMetric, y: (value: number) => number) => {
    const segments: Array<{ d: string; color: string; key: string }> = []
    let color = ''
    let path = ''
    let previous: { x: number; y: number } | undefined
    visible.forEach((bucket) => {
      const value = METRIC_VALUE[metric](bucket)
      if (value === null || !Number.isFinite(value)) {
        // A gap is a gap: prompt processing only has a rate when the engine
        // measured one, and a straight line across it would invent a rate.
        if (path) segments.push({ d: path, color, key: `${metric}-${segments.length}` })
        path = ''
        previous = undefined
        return
      }
      const point = { x: x(bucket.at), y: y(value) }
      const pointColor = colorByConcurrency ? concurrencyColor(bucket.concurrent) : 'metric'
      if (pointColor !== color) {
        if (path && previous) {
          // The colour changes between buckets, so the new segment starts at the
          // previous point and no part of the line loses its concurrency colour.
          segments.push({ d: path, color, key: `${metric}-${segments.length}` })
          path = `M ${previous.x} ${previous.y}`
        }
        color = pointColor
      }
      path = path ? `${path} L ${point.x} ${point.y}` : `M ${point.x} ${point.y}`
      previous = point
    })
    if (path) segments.push({ d: path, color, key: `${metric}-${segments.length}` })
    return segments
  }

  return (
    <div className="history-chart-wrap">
      <svg
        className="history-chart"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`${series.model} throughput history coloured by concurrent sessions`}
        onPointerMove={pickBucket}
        onPointerLeave={() => setHover(undefined)}
      >
        {valueTicks.map((fraction) => {
          const position = PADDING.top + PLOT_HEIGHT - fraction * PLOT_HEIGHT
          return <g key={`left-${fraction}`}>
            <line className="history-grid-line" x1={PADDING.left} x2={WIDTH - PADDING.right} y1={position} y2={position} />
            <text className="history-axis-label" x={PADDING.left - 8} y={position + 4} textAnchor="end">{compact(leftMax * fraction)}</text>
            {metrics.has('prefill') && (
              <text className="history-axis-label history-axis-right" x={WIDTH - PADDING.right + 8} y={position + 4} textAnchor="start">{compact(rightMax * fraction)}</text>
            )}
          </g>
        })}
        {timeTicks.map((at) => (
          <g key={`time-${at}`}>
            <line className="history-grid-line history-grid-vertical" x1={x(at)} x2={x(at)} y1={PADDING.top} y2={PADDING.top + PLOT_HEIGHT} />
            <text className="history-axis-label" x={x(at)} y={HEIGHT - 12} textAnchor="middle">{minuteLabel(at)}</text>
          </g>
        ))}

        {metrics.has('output') && seriesPath('output', yLeft).map((segment) => (
          <path key={segment.key} className={`history-line ${METRIC_STYLE.output.className}`} style={lineStyle('output', colorByConcurrency ? segment.color : undefined)} d={segment.d} />
        ))}
        {metrics.has('thinking') && seriesPath('thinking', yLeft).map((segment) => (
          <path key={segment.key} className={`history-line ${METRIC_STYLE.thinking.className}`} style={lineStyle('thinking', colorByConcurrency ? segment.color : undefined)} d={segment.d} />
        ))}
        {metrics.has('prefill') && seriesPath('prefill', yRight).map((segment) => (
          <path key={segment.key} className={`history-line ${METRIC_STYLE.prefill.className}`} style={lineStyle('prefill', colorByConcurrency ? segment.color : undefined)} d={segment.d} />
        ))}

        {hovered && (
          <g className="history-cursor">
            <line x1={hover!.x} x2={hover!.x} y1={PADDING.top} y2={PADDING.top + PLOT_HEIGHT} />
            {metrics.has('output') && typeof METRIC_VALUE.output(hovered) === 'number' && (
              <circle cx={hover!.x} cy={yLeft(METRIC_VALUE.output(hovered) as number)} r="3.5" style={{ fill: colorByConcurrency ? concurrencyColor(hovered.concurrent) : undefined }} />
            )}
            {metrics.has('thinking') && typeof METRIC_VALUE.thinking(hovered) === 'number' && (
              <circle cx={hover!.x} cy={yLeft(METRIC_VALUE.thinking(hovered) as number)} r="3.5" style={{ fill: colorByConcurrency ? concurrencyColor(hovered.concurrent) : undefined }} />
            )}
            {metrics.has('prefill') && typeof METRIC_VALUE.prefill(hovered) === 'number' && (
              <circle cx={hover!.x} cy={yRight(METRIC_VALUE.prefill(hovered) as number)} r="3.5" style={{ fill: colorByConcurrency ? concurrencyColor(hovered.concurrent) : undefined }} />
            )}
          </g>
        )}

        <text className="history-axis-title" transform={`translate(16 ${PADDING.top + PLOT_HEIGHT / 2}) rotate(-90)`} textAnchor="middle">Tokens/sec</text>
        {metrics.has('prefill') && (
          <text className="history-axis-title" transform={`translate(${WIDTH - 6} ${PADDING.top + PLOT_HEIGHT / 2}) rotate(-90)`} textAnchor="middle">Prompt tokens/sec</text>
        )}
      </svg>

      {hover && (
        <div
          className="history-tooltip"
          role="status"
          style={{
            left: `calc(${(hover.x / WIDTH) * 100}% + ${hover.x / WIDTH > 0.6 ? -14 : 14}px)`,
            transform: hover.x / WIDTH > 0.6 ? 'translateX(-100%)' : undefined,
          }}
        >
          <strong>Concurrent: {hover.bucket.concurrent.toFixed(hover.bucket.concurrent % 1 === 0 ? 0 : 1)}</strong>
          <span className="history-tooltip-state">
            <i className="history-state-output" />{`${hover.bucket.output_sessions} outputting`}
            <i className="history-state-thinking" />{`${hover.bucket.thinking_sessions} thinking`}
            <i className="history-state-prefill" />{`${hover.bucket.prefill_sessions} prompt processing`}
          </span>
          {metrics.has('thinking') && <span>Thinking: {rateLabel(hover.bucket.thinking_tok_s)}</span>}
          {metrics.has('output') && <span>Output: {rateLabel(hover.bucket.output_tok_s)}</span>}
          {metrics.has('prefill') && (
            <span>
              Prompt processing: {rateLabel(hover.bucket.prefill_tok_s)}
              {hoverIsEstimated && <em> estimated</em>}
            </span>
          )}
          {metrics.has('prefill') && hoverPrefillEvidence !== undefined && (
            <span className="history-tooltip-note">{hoverPrefillEvidence}</span>
          )}
          {metrics.has('prefill') && hover.bucket.prefill_measured && <span className="history-tooltip-note">Measured by the engine</span>}
          <span className="history-tooltip-note">
            {clockLabel(hover.bucket.at)} · {hover.bucket.concurrent_peak} peak
            {hover.bucket.output_peak_tok_s > 0 && ` · ${hover.bucket.output_peak_tok_s.toFixed(0)} tok/s peak`}
          </span>
        </div>
      )}

      <div className="history-chart-legend">
        <span className="history-legend-styles">
          {(Object.keys(METRIC_STYLE) as HistoryMetric[]).filter((metric) => metrics.has(metric)).map((metric) => (
            <span key={metric}>
              <svg className="history-legend-line" viewBox="0 0 18 10" aria-hidden="true">
                <line
                  className={`history-line ${METRIC_STYLE[metric].className}`}
                  x1="1" x2="17" y1="5" y2="5"
                  style={lineStyle(metric, 'currentColor')}
                />
              </svg>
              {METRIC_STYLE[metric].label}
            </span>
          ))}
        </span>
        <span className="history-legend-concurrency">
          {colorByConcurrency
            ? Array.from({ length: MAX_COLORED_CONCURRENCY }, (_, index) => (
              <span key={index}><i style={{ background: CONCURRENCY_COLORS[index] }} />C{index + 1}</span>
            ))
            : null}
        </span>
      </div>
      <p className="history-chart-caption">
        One point every {sampleSeconds} second{sampleSeconds === 1 ? '' : 's'} over the last {Math.round(rangeSeconds / 60)} minutes.
        {colorByConcurrency && ' Line colour is the mean concurrent session count for that point.'}
        {' '}Solid is token generation, dotted is thinking, dashed is prompt processing.
      </p>

      <table className="sr-only">
        <caption>{series.model} throughput history</caption>
        <thead><tr><th>Bucket</th><th>Concurrent</th><th>Thinking tok/s</th><th>Output tok/s</th><th>Prompt processing tok/s</th><th>Prompt tokens</th><th>Prompt processing time</th></tr></thead>
        <tbody>
          {visible.slice(-40).map((bucket) => (
            <tr key={bucket.at}>
              <td>{clockLabel(bucket.at)}</td>
              <td>{bucket.concurrent}</td>
              <td>{bucket.thinking_tok_s}</td>
              <td>{bucket.output_tok_s}</td>
              <td>{bucket.prefill_tok_s === null ? 'pending' : bucket.prefill_tok_s}</td>
              <td>{bucket.prefill_tokens > 0 ? bucket.prefill_tokens : '—'}</td>
              <td>{bucket.prefill_seconds > 0 ? secondsLabel(bucket.prefill_seconds) : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
