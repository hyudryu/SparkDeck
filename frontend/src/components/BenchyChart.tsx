import type { BenchyResultRow } from '../api/types'

export type BenchyMetricKey =
  | 'pp_tokens_per_second'
  | 'tg_tokens_per_second'
  | 'tg_tokens_per_second_request'

const WIDTH = 760
const HEIGHT = 260
const PADDING = { top: 18, right: 18, bottom: 42, left: 62 }

interface BenchyChartProps {
  title: string
  subtitle: string
  rows: BenchyResultRow[]
  metric: BenchyMetricKey
}

function compactTokens(value: number) {
  return value >= 1024 ? `${Number((value / 1024).toFixed(value >= 10240 ? 0 : 1))}K` : String(value)
}

function smoothPath(coordinates: { x: number; y: number }[]) {
  if (!coordinates.length) return ''
  return coordinates.slice(1).reduce((path, point, index) => {
    const previous = coordinates[index]
    const midpoint = (previous.x + point.x) / 2
    return `${path} C ${midpoint} ${previous.y}, ${midpoint} ${point.y}, ${point.x} ${point.y}`
  }, `M ${coordinates[0].x} ${coordinates[0].y}`)
}

export function BenchyChart({ title, subtitle, rows, metric }: BenchyChartProps) {
  const measured = rows.filter((row) => typeof row[metric] === 'number')
  const promptSizes = [...new Set(measured.map((row) => row.prompt_size ?? 0))].sort((a, b) => a - b)
  const maximum = Math.max(1, ...measured.map((row) => row[metric] as number))
  // A run can sweep several response sizes (and, via the API, context depths)
  // for the same prompt size and concurrency. Each such combination gets its
  // own series so one measurement cannot overwrite another on the chart.
  const responseSizes = [...new Set(measured.map((row) => row.response_size ?? 0))].sort((a, b) => a - b)
  const depths = [...new Set(measured.map((row) => row.context_depth ?? 0))].sort((a, b) => a - b)
  interface Series {
    concurrency: number
    responseSize: number
    depth: number
    label: string
    values: Map<number, BenchyResultRow>
  }
  const seriesById = new Map<string, Series>()
  for (const row of measured) {
    const concurrency = row.concurrency ?? 1
    const responseSize = row.response_size ?? 0
    const depth = row.context_depth ?? 0
    const id = `${concurrency}|${responseSize}|${depth}`
    if (!seriesById.has(id)) {
      seriesById.set(id, {
        concurrency,
        responseSize,
        depth,
        label: `C${concurrency}`
          + (responseSizes.length > 1 ? ` · ${compactTokens(responseSize)} tok` : '')
          + (depths.length > 1 ? ` @ ${compactTokens(depth)} ctx` : ''),
        values: new Map(),
      })
    }
    seriesById.get(id)!.values.set(row.prompt_size ?? 0, row)
  }
  const series = [...seriesById.values()].sort((a, b) =>
    a.concurrency - b.concurrency || a.responseSize - b.responseSize || a.depth - b.depth)
  const chartWidth = WIDTH - PADDING.left - PADDING.right
  const chartHeight = HEIGHT - PADDING.top - PADDING.bottom
  const x = (promptSize: number) =>
    PADDING.left + (promptSizes.indexOf(promptSize) * chartWidth) / Math.max(1, promptSizes.length - 1)
  const y = (value: number) => PADDING.top + chartHeight - (value / maximum) * chartHeight
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((fraction) => ({ fraction, value: maximum * fraction }))

  return (
    <section className="benchmark-chart" aria-label={title}>
      <div className="benchmark-chart-heading">
        <div><h3>{title}</h3><p>{subtitle}</p></div>
        <div className="benchmark-chart-legend" aria-label="Concurrency legend">
          {series.map((item, index) => (
            <span key={`C${item.concurrency}-${item.responseSize}-${item.depth}`}><i className={`chart-color-${index % 6}`} />{item.label}</span>
          ))}
        </div>
      </div>
      {measured.length === 0 || promptSizes.length === 0 ? <div className="benchmark-chart-empty">No measured points for this run yet.</div> : <>
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-labelledby={`${metric}-chart-title ${metric}-chart-desc`}>
          <title id={`${metric}-chart-title`}>{title}</title>
          <desc id={`${metric}-chart-desc`}>{subtitle} One line per concurrency and response-size combination across prompt sizes.</desc>
          {ticks.map(({ fraction, value }) => {
            const position = PADDING.top + chartHeight - fraction * chartHeight
            return <g key={fraction}>
              <line className="chart-grid-line" x1={PADDING.left} x2={WIDTH - PADDING.right} y1={position} y2={position} />
              <text className="chart-axis-label" x={PADDING.left - 10} y={position + 4} textAnchor="end">{Math.round(value).toLocaleString()}</text>
            </g>
          })}
          {promptSizes.map((size) => <g key={size}>
            <line className="chart-grid-line chart-grid-vertical" x1={x(size)} x2={x(size)} y1={PADDING.top} y2={PADDING.top + chartHeight} />
            <text className="chart-axis-label" x={x(size)} y={HEIGHT - 16} textAnchor="middle">{compactTokens(size)}</text>
          </g>)}
          {series.map((item, colorIndex) => {
            const points = promptSizes
              .filter((size) => item.values.has(size))
              .map((size) => ({
                size,
                x: x(size),
                y: y(item.values.get(size)![metric] as number),
                value: item.values.get(size)![metric] as number,
              }))
            const segments: typeof points[] = []
            let segment: typeof points = []
            promptSizes.forEach((size) => {
              const point = points.find((entry) => entry.size === size)
              if (point) segment.push(point)
              else if (segment.length) { segments.push(segment); segment = [] }
            })
            if (segment.length) segments.push(segment)
            return <g className={`chart-series chart-color-${colorIndex % 6}`} key={`C${item.concurrency}-${item.responseSize}-${item.depth}`}>
              {segments.filter((items) => items.length > 1).map((items, index) => (
                <path key={index} d={smoothPath(items.map(({ x, y }) => ({ x, y })))} />
              ))}
              {points.map((point) => <circle key={point.size} cx={point.x} cy={point.y} r="4">
                <title>
                  {compactTokens(point.size)} prompt, {item.label}: {point.value.toFixed(1)} tok/s
                </title>
              </circle>)}
            </g>
          })}
          <text className="chart-axis-title" transform={`translate(16 ${PADDING.top + chartHeight / 2}) rotate(-90)`} textAnchor="middle">Tokens/sec</text>
          <text className="chart-axis-title" x={PADDING.left + chartWidth / 2} y={HEIGHT - 1} textAnchor="middle">Prompt size (tokens)</text>
        </svg>
        <table className="sr-only">
          <caption>{title} data</caption>
          <thead><tr><th>Prompt size</th><th>Output tokens</th><th>Concurrency</th><th>Tokens per second</th></tr></thead>
          <tbody>
            {measured.map((row, index) => <tr key={index}>
              <td>{row.prompt_size}</td><td>{row.response_size}</td><td>{row.concurrency}</td><td>{row[metric]}</td>
            </tr>)}
          </tbody>
        </table>
      </>}
    </section>
  )
}
