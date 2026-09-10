import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { LiveHistoryBucket, LiveHistorySeries } from '../api/types'
import { concurrencyColor, concurrencyLabel } from '../components/HistoryChart'
import { HistoryPage } from './HistoryPage'

function json(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
}

const BASE_AT = 1_700_000_000

function bucket(overrides: Partial<LiveHistoryBucket> = {}): LiveHistoryBucket {
  return {
    at: BASE_AT,
    output_tok_s: 20,
    thinking_tok_s: 0,
    prefill_tok_s: null,
    prefill_measured: false,
    concurrent: 3,
    concurrent_peak: 3,
    output_sessions: 3,
    thinking_sessions: 0,
    prefill_sessions: 0,
    output_active: true,
    thinking_active: false,
    output_peak_tok_s: 22,
    thinking_peak_tok_s: 0,
    prefill_peak_tok_s: null,
    ...overrides,
  }
}

/**
 * A trailing window of five-second buckets, like the collector publishes.  The
 * points are spread across the selected window so a hover resolves to one
 * bucket the way it does with real data.
 */
function trailingBuckets(count: number, last: Partial<LiveHistoryBucket> = {}): LiveHistoryBucket[] {
  return Array.from({ length: count }, (_, index) => bucket({
    at: BASE_AT + index * 5,
    concurrent: 3,
    ...(index === count - 1 ? last : {}),
  }))
}

function series(overrides: Partial<LiveHistorySeries> = {}): LiveHistorySeries {
  const buckets = overrides.buckets ?? trailingBuckets(360)
  return {
    key: 'deployment-a:0',
    group_id: 'deployment-a:0',
    model: 'qwen3-32b',
    deployment_id: 'deployment-a',
    instance_id: 0,
    node_names: ['Node 1', 'Node 2'],
    live_sessions: 3,
    state: { output_sessions: 3, thinking_sessions: 0, prefill_sessions: 0 },
    last_at: buckets[buckets.length - 1].at,
    bucket_seconds: 5,
    ...overrides,
    buckets,
  }
}

function stubHistoryFetch(items: LiveHistorySeries[]) {
  return vi.fn<typeof fetch>().mockImplementation(async (input) => {
    if (String(input).includes('/api/v1/live-history')) {
      return json({
        generated_at: 1_700_000_005, bucket_seconds: 5, range_seconds: 3_600,
        sample_seconds: 1, series: items,
      })
    }
    return json({})
  })
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

/**
 * Aim the pointer at the newest bucket.  The chart scales a 960-unit viewBox to
 * the element width, so the client coordinate is derived rather than guessed;
 * jsdom reports a zero-width element, which the chart floors at its own width.
 */
function lastBucketPointer(chart: Element): number {
  const width = chart.getBoundingClientRect().width || 960
  // The plot's right edge sits at 894 of the 960 viewBox units.
  return (894 / 960) * width
}

describe('HistoryPage', () => {
  it('renders one graph per serving unit with its nodes and live session states', async () => {
    vi.stubGlobal('fetch', stubHistoryFetch([
      series(),
      series({
        key: 'deployment-b:1', group_id: 'deployment-b:1', model: 'llama-70b',
        instance_id: 1, node_names: ['Node 3'], live_sessions: 0,
        state: { output_sessions: 0, thinking_sessions: 0, prefill_sessions: 0 },
      }),
    ]))
    render(<HistoryPage />)

    expect(await screen.findByText('qwen3-32b')).toBeInTheDocument()
    expect(screen.getByText('llama-70b')).toBeInTheDocument()
    expect(screen.getByText('Node 1 + Node 2 · instance 0')).toBeInTheDocument()
    expect(screen.getByText('Node 3 · instance 1')).toBeInTheDocument()
    expect(screen.getByText('3 outputting · 0 thinking · 0 prompt processing')).toBeInTheDocument()
    expect(screen.getByText('3 live')).toBeInTheDocument()
    expect(screen.getByText('No sessions')).toBeInTheDocument()
    // One chart per graph, each with its own accessible name.
    expect(screen.getByLabelText('qwen3-32b throughput history coloured by concurrent sessions')).toBeInTheDocument()
    expect(screen.getByLabelText('llama-70b throughput history coloured by concurrent sessions')).toBeInTheDocument()
  })

  it('shows the concurrent, thinking, output, and prompt processing card on hover', async () => {
    const buckets = trailingBuckets(720, {
      thinking_tok_s: 13, output_tok_s: 45, prefill_tok_s: 900, prefill_measured: true, prefill_sessions: 1,
    })
    vi.stubGlobal('fetch', stubHistoryFetch([series({ buckets })]))
    render(<HistoryPage />)
    const chart = await screen.findByLabelText('qwen3-32b throughput history coloured by concurrent sessions')

    // The newest bucket sits at the right edge of the trailing window.
    fireEvent.pointerMove(chart, { clientX: lastBucketPointer(chart), clientY: 50 })
    const card = await screen.findByRole('status')
    expect(within(card).getByText('Concurrent: 3')).toBeInTheDocument()
    expect(within(card).getByText('Thinking: 13.0 tok/s')).toBeInTheDocument()
    expect(within(card).getByText('Output: 45.0 tok/s')).toBeInTheDocument()
    expect(within(card).getByText('Prompt processing: 900.0 tok/s')).toBeInTheDocument()
    // The state counts share one element with their status dots, so they are
    // matched as substrings rather than as whole-element text.
    expect(within(card).getByText('3 outputting', { exact: false })).toBeInTheDocument()
    expect(within(card).getByText('0 thinking', { exact: false })).toBeInTheDocument()
  })

  it('reports pending prompt processing when the engine has measured no prefill', async () => {
    const buckets = trailingBuckets(720, { output_tok_s: 20, prefill_tok_s: null })
    vi.stubGlobal('fetch', stubHistoryFetch([series({ buckets })]))
    render(<HistoryPage />)
    const chart = await screen.findByLabelText('qwen3-32b throughput history coloured by concurrent sessions')

    fireEvent.pointerMove(chart, { clientX: lastBucketPointer(chart), clientY: 50 })
    const card = await screen.findByRole('status')
    expect(within(card).getByText('Concurrent: 3')).toBeInTheDocument()
    expect(within(card).getByText('Output: 20.0 tok/s')).toBeInTheDocument()
    // Prefill has no rate until the engine reports one; the card must not
    // invent a number for it.
    expect(within(card).getByText('Prompt processing: pending')).toBeInTheDocument()
  })

  it('reports a measured prompt processing rate and labels a live estimate', async () => {
    const buckets = trailingBuckets(720, {
      prefill_tok_s: 900, prefill_measured: true, prefill_sessions: 2,
    })
    // Second-to-last bucket is still prefilling: only a live estimate exists.
    buckets[buckets.length - 2] = bucket({
      at: buckets[buckets.length - 2].at, concurrent: 3,
      prefill_tok_s: 800, prefill_measured: false, prefill_sessions: 1,
    })
    vi.stubGlobal('fetch', stubHistoryFetch([series({ buckets })]))
    render(<HistoryPage />)
    const chart = await screen.findByLabelText('qwen3-32b throughput history coloured by concurrent sessions')

    fireEvent.pointerMove(chart, { clientX: lastBucketPointer(chart), clientY: 50 })
    const card = await screen.findByRole('status')
    expect(within(card).getByText('Prompt processing: 900.0 tok/s')).toBeInTheDocument()
    expect(within(card).getByText('Measured by the engine')).toBeInTheDocument()
    expect(within(card).getByText('2 prompt processing', { exact: false })).toBeInTheDocument()
  })

  it('lists each bucket in the accessible table, including an estimated rate', async () => {
    const buckets = trailingBuckets(4, { prefill_tok_s: 800, prefill_measured: false, thinking_tok_s: 7 })
    vi.stubGlobal('fetch', stubHistoryFetch([series({ buckets })]))
    render(<HistoryPage />)
    await screen.findByText('qwen3-32b')

    const table = screen.getByRole('table', { name: /qwen3-32b throughput history/ })
    expect(within(table).getAllByRole('row')).toHaveLength(buckets.length + 1)
    expect(within(table).getAllByText('800').length).toBeGreaterThan(0)
    expect(within(table).getAllByText('7').length).toBeGreaterThan(0)
  })

  it('lets a metric be turned off without leaving an empty graph', async () => {
    vi.stubGlobal('fetch', stubHistoryFetch([series()]))
    render(<HistoryPage />)
    await screen.findByText('qwen3-32b')

    const generation = screen.getByRole('button', { name: 'Token generation' })
    expect(generation).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(generation)
    expect(generation).toHaveAttribute('aria-pressed', 'false')

    const thinking = screen.getByRole('button', { name: 'Thinking' })
    const prefill = screen.getByRole('button', { name: 'Prompt processing' })
    fireEvent.click(thinking)
    fireEvent.click(prefill)
    // The last remaining line cannot be switched off.
    expect(prefill).toHaveAttribute('aria-pressed', 'true')
  })

  it('switches the concurrency colour legend off in favour of metric colours', async () => {
    vi.stubGlobal('fetch', stubHistoryFetch([series()]))
    render(<HistoryPage />)
    await screen.findByText('qwen3-32b')

    const chart = screen.getByLabelText('qwen3-32b throughput history coloured by concurrent sessions')
    const legend = chart.parentElement?.querySelector('.history-chart-legend') as HTMLElement
    expect(within(legend).getByText('C1')).toBeInTheDocument()
    expect(within(legend).getByText('C10')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Metric' }))
    expect(within(legend).queryByText('C10')).not.toBeInTheDocument()
    // Metric colours name all three lines, including the one that was already
    // labelled in the toolbar.
    expect(within(legend).getByText('Prompt processing')).toBeInTheDocument()
    expect(within(legend).getByText('Token generation')).toBeInTheDocument()
  })

  it('explains how to get history when nothing has run yet', async () => {
    vi.stubGlobal('fetch', stubHistoryFetch([]))
    render(<HistoryPage />)

    expect(await screen.findByText('No throughput history yet')).toBeInTheDocument()
    expect(screen.getByText(/appears within five seconds/)).toBeInTheDocument()
  })

  it('keeps the last snapshot visible when a refresh fails', async () => {
    let calls = 0
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      if (!String(input).includes('/api/v1/live-history')) return json({})
      calls += 1
      if (calls > 1) return new Response('boom', { status: 500 })
      return json({
        generated_at: 1_700_000_005, bucket_seconds: 5, range_seconds: 3_600,
        sample_seconds: 1, series: [series()],
      })
    }))
    render(<HistoryPage />)
    expect(await screen.findByText('qwen3-32b')).toBeInTheDocument()
  })
})

describe('concurrency colours', () => {
  it('maps one, two, and three sessions to the documented colours', () => {
    expect(concurrencyColor(1)).toBe('#30d158')
    expect(concurrencyColor(2)).toBe('#ff9f0a')
    expect(concurrencyColor(3)).toBe('#2997ff')
    expect(concurrencyLabel(2)).toBe('C2')
  })

  it('rounds a fractional bucket mean up so extra pressure is never hidden', () => {
    // A bucket averaging 1.4 sessions still had two sessions in it; C2 is the
    // honest label for the load it carried.
    expect(concurrencyColor(1.4)).toBe('#ff9f0a')
    expect(concurrencyColor(3.4)).toBe('#bf5af2')
  })

  it('gives four through ten distinct colours and clamps above ten', () => {
    const assigned = Array.from({ length: 10 }, (_, index) => concurrencyColor(index + 1))
    expect(new Set(assigned).size).toBe(10)
    expect(concurrencyColor(10)).toBe(assigned[9])
    expect(concurrencyColor(11)).toBe(assigned[9])
    expect(concurrencyColor(64)).toBe(assigned[9])
    expect(concurrencyLabel(11)).toBe('C10+')
  })
})
