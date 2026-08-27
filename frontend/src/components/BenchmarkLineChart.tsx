import type { BenchmarkSeriesPoint } from '../api/types'

const CONCURRENCY = [1, 2, 5, 10] as const
const WIDTH = 760
const HEIGHT = 260
const PADDING = { top: 18, right: 18, bottom: 42, left: 62 }

interface BenchmarkLineChartProps {
  title: string
  metric: 'prompt_tokens_per_second' | 'generation_tokens_per_second'
  points: BenchmarkSeriesPoint[]
}

function compactTokens(value: number) {
  return value >= 1024 ? `${Number((value / 1024).toFixed(value >= 10240 ? 0 : 1))}K` : String(value)
}

function smoothPath(
  points: BenchmarkSeriesPoint[],
  metric: BenchmarkLineChartProps['metric'],
  x: (concurrency: number) => number,
  y: (value: number) => number,
) {
  const coordinates = points.map((point) => ({ x: x(point.concurrency), y: y(point[metric]) }))
  if (!coordinates.length) return ''
  return coordinates.slice(1).reduce((path, point, index) => {
    const previous = coordinates[index]
    const midpoint = (previous.x + point.x) / 2
    return `${path} C ${midpoint} ${previous.y}, ${midpoint} ${point.y}, ${point.x} ${point.y}`
  }, `M ${coordinates[0].x} ${coordinates[0].y}`)
}

export function BenchmarkLineChart({ title, metric, points }: BenchmarkLineChartProps) {
  const windows = [...new Set(points.map((point) => point.context_window_size))].sort((a, b) => a - b)
  const maximum = Math.max(1, ...points.map((point) => point[metric]))
  const chartWidth = WIDTH - PADDING.left - PADDING.right
  const chartHeight = HEIGHT - PADDING.top - PADDING.bottom
  const x = (concurrency: number) => PADDING.left + (CONCURRENCY.indexOf(concurrency as typeof CONCURRENCY[number]) * chartWidth / (CONCURRENCY.length - 1))
  const y = (value: number) => PADDING.top + chartHeight - (value / maximum) * chartHeight
  const ticks = [0, .25, .5, .75, 1].map((fraction) => ({ fraction, value: maximum * fraction }))

  return (
    <section className="benchmark-chart" aria-label={title}>
      <div className="benchmark-chart-heading"><div><h3>{title}</h3><p>Tokens/sec vs concurrency</p></div><div className="benchmark-chart-legend" aria-label="Context window legend">{windows.map((window, index) => <span key={window}><i className={`chart-color-${index % 6}`} />{compactTokens(window)} context</span>)}</div></div>
      {points.length === 0 ? <div className="benchmark-chart-empty">No measured points for this TP size yet.</div> : <>
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-labelledby={`${metric}-chart-title ${metric}-chart-desc`}>
          <title id={`${metric}-chart-title`}>{title}</title>
          <desc id={`${metric}-chart-desc`}>Measured tokens per second at concurrency one, two, five, and ten. Missing measurements are left blank.</desc>
          {ticks.map(({ fraction, value }) => {
            const position = PADDING.top + chartHeight - fraction * chartHeight
            return <g key={fraction}><line className="chart-grid-line" x1={PADDING.left} x2={WIDTH - PADDING.right} y1={position} y2={position} /><text className="chart-axis-label" x={PADDING.left - 10} y={position + 4} textAnchor="end">{Math.round(value).toLocaleString()}</text></g>
          })}
          {CONCURRENCY.map((concurrency) => <g key={concurrency}><line className="chart-grid-line chart-grid-vertical" x1={x(concurrency)} x2={x(concurrency)} y1={PADDING.top} y2={PADDING.top + chartHeight} /><text className="chart-axis-label" x={x(concurrency)} y={HEIGHT - 16} textAnchor="middle">C{concurrency}</text></g>)}
          {windows.map((window, colorIndex) => {
            const values = new Map(points.filter((point) => point.context_window_size === window).map((point) => [point.concurrency, point]))
            const segments: BenchmarkSeriesPoint[][] = []
            let segment: BenchmarkSeriesPoint[] = []
            CONCURRENCY.forEach((concurrency) => {
              const point = values.get(concurrency)
              if (point) segment.push(point)
              else if (segment.length) { segments.push(segment); segment = [] }
            })
            if (segment.length) segments.push(segment)
            return <g className={`chart-series chart-color-${colorIndex % 6}`} key={window}>
              {segments.filter((items) => items.length > 1).map((items, index) => <path key={index} d={smoothPath(items, metric, x, y)} />)}
              {[...values.values()].map((point) => <circle key={point.concurrency} cx={x(point.concurrency)} cy={y(point[metric])} r="4"><title>{compactTokens(window)} context, C{point.concurrency}: {point[metric].toFixed(1)} tok/s from {point.sample_count} run{point.sample_count === 1 ? '' : 's'}</title></circle>)}
            </g>
          })}
          <text className="chart-axis-title" transform={`translate(16 ${PADDING.top + chartHeight / 2}) rotate(-90)`} textAnchor="middle">Tokens/sec</text>
          <text className="chart-axis-title" x={PADDING.left + chartWidth / 2} y={HEIGHT - 1} textAnchor="middle">Concurrency</text>
        </svg>
        <table className="sr-only"><caption>{title} data</caption><thead><tr><th>Context window</th><th>Concurrency</th><th>Tokens per second</th><th>Runs</th></tr></thead><tbody>{points.map((point) => <tr key={`${point.context_window_size}-${point.concurrency}`}><td>{point.context_window_size}</td><td>{point.concurrency}</td><td>{point[metric]}</td><td>{point.sample_count}</td></tr>)}</tbody></table>
      </>}
    </section>
  )
}
