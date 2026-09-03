import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { UsageGroup, UsageSummary } from '../api/types'
import { activityHeatmapCalendar, modelShareItems, usageMeterDifference, UsagePage } from './UsagePage'

const summary = {
  models: {
    'org/model': { input: 1500, cached: 500, output: 750, requests: 12, gen_time_s: 25 },
  },
  total: { input: 1500, cached: 500, output: 750, requests: 12 },
  groups: [{
    key: 'model:org/model', label: 'Workload', merge_group: null,
    route_target: 'org/model', models: ['org/model'],
    members: [{ model: 'org/model', alias: 'Workload', merge_group: null, routed_to: null }],
    stats: {
      input: 1500, input_miss: 1000, cached: 500, measured_cached: 500,
      estimated_cached: 0, output: 750, requests: 12, gen_tokens: 750,
      gen_time_s: 25,
    },
    speed: { tokens: 750, active_time_s: 25, tok_s: 30, legacy: false },
    total_cost: 1.25, cost_estimated: false,
  }],
}

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('UsagePage', () => {
  it('shows the top seven model shares and combines the remainder as Others', () => {
    const rows = Array.from({ length: 9 }, (_, index) => {
      const rank = index + 1
      const value = (10 - rank) * 100
      return {
        key: `model:model-${rank}`,
        label: `Model ${rank}`,
        merge_group: null,
        route_target: `model-${rank}`,
        models: [`model-${rank}`],
        members: [],
        stats: { input: value, input_miss: value, cached: 0, measured_cached: 0, estimated_cached: 0, output: 0, requests: 1, gen_tokens: 0, gen_time_s: 0 },
        total_cost: 0,
        cost_estimated: false,
      }
    }) satisfies UsageGroup[]

    expect(modelShareItems([...rows].reverse())).toEqual([
      { key: 'model:model-1', label: 'Model 1', value: 900 },
      { key: 'model:model-2', label: 'Model 2', value: 800 },
      { key: 'model:model-3', label: 'Model 3', value: 700 },
      { key: 'model:model-4', label: 'Model 4', value: 600 },
      { key: 'model:model-5', label: 'Model 5', value: 500 },
      { key: 'model:model-6', label: 'Model 6', value: 400 },
      { key: 'model:model-7', label: 'Model 7', value: 300 },
      { key: 'others', label: 'Others', value: 300 },
    ])
  })

  it('derives heatmap month labels and positions from the rendered dates', () => {
    const calendar = activityHeatmapCalendar([], new Date('2026-01-15T12:00:00Z'))

    expect(calendar.months[0]).toMatchObject({ key: '2025-01', label: 'Jan', column: 1 })
    expect(calendar.months.at(-1)).toMatchObject({ key: '2026-01', label: 'Jan' })
    expect(calendar.months.every((month, index) => index === 0 || month.column > calendar.months[index - 1].column)).toBe(true)
  })

  it('measures only counters added after the interval baseline', () => {
    const current = {
      ...summary,
      models: { 'org/model': { input: 1700, cached: 550, output: 825, requests: 15 } },
      total: { input: 1700, cached: 550, output: 825, requests: 15 },
    }

    const measured = usageMeterDifference(current as UsageSummary, summary as UsageSummary)

    expect(measured.totals).toMatchObject({ input: 200, cached: 50, output: 75, requests: 3 })
    expect(measured.models).toEqual([{ model: 'org/model', counters: { input: 200, cached: 50, output: 75, requests: 3 } }])
    expect(measured.totals.input + measured.totals.output).toBe(275)
  })

  it('shows the token scale legend and a hover tooltip on heatmap cells', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/token-stats/hourly')) return json([])
      if (path.includes('/api/token-stats/daily')) return json([
        { date: '2026-08-26', input: 1800, cached: 500, output: 650, requests: 6, models: { 'org/model': { input: 1800, cached: 500, output: 650, requests: 6 } } },
      ])
      return json(summary)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    const { container } = render(<UsagePage />)

    expect(await screen.findByRole('img', { name: 'Daily token activity for the last year' })).toBeInTheDocument()
    const scale = screen.getByText('Less').closest('.usage-heatmap-scale')
    expect(scale).toHaveTextContent('More')
    expect(scale?.querySelectorAll('.usage-heatmap-cell')).toHaveLength(5)
    const activeCell = container.querySelector('.usage-heatmap-cell.level-4')
    expect(activeCell).not.toBeNull()
    await user.hover(activeCell as HTMLElement)
    const tooltip = container.querySelector('.usage-heatmap-tooltip')
    expect(tooltip).toHaveTextContent('2,450 tokens')
    expect(tooltip).toHaveTextContent('Aug 26, 2026')
    await user.unhover(activeCell as HTMLElement)
    expect(container.querySelector('.usage-heatmap-tooltip')).toBeNull()
  })

  it('restores lifetime model accounting and persisted hourly analysis', async () => {
    const recent = new Date()
    recent.setUTCDate(recent.getUTCDate() - 1)
    const recentDate = recent.toISOString().slice(0, 10)
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/token-stats/hourly')) return json([
        { hour: `${recentDate}T10`, input: 1200, cached: 300, output: 400, requests: 4 },
        { hour: `${recentDate}T11`, input: 600, cached: 200, output: 250, requests: 2 },
      ])
      if (path.includes('/api/token-stats/daily')) return json([
        { date: recentDate, input: 1800, cached: 500, output: 650, requests: 6, models: { 'org/model': { input: 1800, cached: 500, output: 650, requests: 6 } } },
      ])
      return json(summary)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<UsagePage />)

    expect(await screen.findByRole('heading', { name: 'Usage stats' })).toBeInTheDocument()
    expect(screen.getByText('Cluster inference accounting')).toBeInTheDocument()
    expect(screen.getByText('Combined token activity and model share from every paired SparkDeck node.')).toBeInTheDocument()
    expect((await screen.findAllByText('Workload')).length).toBeGreaterThan(0)
    expect(screen.getByLabelText('Usage overview')).toHaveTextContent('2,250')
    expect(screen.getByLabelText('Usage overview')).toHaveTextContent('2,450')
    expect(screen.getByRole('table', { name: 'Lifetime model usage' })).toHaveTextContent('30.0 tok/s')
    expect(screen.getByRole('table', { name: 'Lifetime model usage' })).toHaveTextContent('$1.25')
    expect(screen.getByRole('img', { name: 'Daily token activity for the last year' })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Daily model token trend for the last 30 days' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Last 7 days' }))
    expect(screen.getByRole('img', { name: 'Daily model token trend for the last 7 days' })).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/token-stats/hourly?start='), expect.objectContaining({ signal: expect.any(AbortSignal) }))
    expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/token-stats/daily?start='), expect.objectContaining({ signal: expect.any(AbortSignal) }))
  })

  it('keeps alias and erase controls connected to the existing APIs', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      if (init?.method) return json({ ok: true })
      const path = String(input)
      if (path.includes('/hourly') || path.includes('/daily')) return json([])
      return json(summary)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<UsagePage />)

    await user.click(await screen.findByRole('button', { name: 'Show details for Workload' }))
    await user.click(screen.getByRole('button', { name: 'Edit usage display for org/model' }))
    const alias = screen.getByRole('textbox', { name: 'Display alias' })
    await user.clear(alias)
    await user.type(alias, 'Local coder')
    await user.type(screen.getByRole('textbox', { name: 'Merge group' }), 'Coding')
    await user.click(screen.getByRole('button', { name: 'Save usage display' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/token-stats/alias', expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({ model: 'org/model', alias: 'Local coder', merge_group: 'Coding' }),
    })))

    await user.click(screen.getByRole('button', { name: 'Erase' }))
    await user.click(within(await screen.findByRole('dialog', { name: 'Erase usage for org/model?' })).getByRole('button', { name: 'Erase usage' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/token-stats/org%2Fmodel', expect.objectContaining({ method: 'DELETE' })))
  })

  it('shows persisted deployment pricing and saves only changed values', async () => {
    const pricedSummary = {
      ...summary,
      groups: summary.groups.map((group) => ({
        ...group,
        members: group.members.map((member) => ({
          ...member,
          deployment_id: 'deployment / one',
          pricing: {
            input_cost_per_1m: 1.25,
            cache_cost_per_1m: 0,
            output_cost_per_1m: 2.5,
          },
        })),
      })),
    }
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      if (init?.method) return json({ ok: true })
      if (String(input).includes('/hourly') || String(input).includes('/daily')) return json([])
      return json(pricedSummary)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<UsagePage />)

    await user.click(await screen.findByRole('button', { name: 'Show details for Workload' }))
    const inputPrice = screen.getByRole('spinbutton', { name: 'Input price for org/model' })
    const cachedPrice = screen.getByRole('spinbutton', { name: 'Cached input price for org/model' })
    const outputPrice = screen.getByRole('spinbutton', { name: 'Output price for org/model' })
    const save = screen.getByRole('button', { name: 'Save pricing' })
    expect(inputPrice).toHaveValue(1.25)
    expect(cachedPrice).toHaveValue(0)
    expect(outputPrice).toHaveValue(2.5)
    expect(save).toBeDisabled()

    await user.clear(cachedPrice)
    await user.type(cachedPrice, '0.15')
    expect(save).toBeEnabled()
    await user.click(save)

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/deployments/deployment%20%2F%20one/pricing', expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({
        input_cost_per_1m: 1.25,
        cache_cost_per_1m: 0.15,
        output_cost_per_1m: 2.5,
      }),
    })))
    expect(await screen.findByText('Updated recorded pricing for org/model.')).toBeInTheDocument()
    expect(save).toBeDisabled()
  })

  it('keeps pricing read-only when usage cannot be linked to one deployment', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input) => {
      if (String(input).includes('/hourly') || String(input).includes('/daily')) return json([])
      return json(summary)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<UsagePage />)

    await user.click(await screen.findByRole('button', { name: 'Show details for Workload' }))
    expect(screen.getByText('Pricing unavailable because no unique deployment is linked to this usage key.')).toBeInTheDocument()
    expect(screen.getByRole('spinbutton', { name: 'Input price for org/model' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Save pricing' })).toBeDisabled()
    expect(fetchMock.mock.calls.every(([path]) => !String(path).includes('/token-cost/'))).toBe(true)
  })

  it('starts and stops an interval meter at the bottom of the page', async () => {
    const current = {
      ...summary,
      models: { 'org/model': { input: 1700, cached: 550, output: 825, requests: 15 } },
      total: { input: 1700, cached: 550, output: 825, requests: 15 },
    }
    let syncReads = 0
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.includes('/api/token-stats/hourly')) return json([])
      if (path.includes('/api/token-stats/daily')) return json([])
      if (path === '/api/token-stats/sync') {
        expect(init?.method).toBe('POST')
        syncReads += 1
        return json(syncReads === 1 ? summary : current)
      }
      return json(summary)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<UsagePage />)

    const meter = await screen.findByLabelText('Token usage meter')
    const routing = screen.getByRole('heading', { name: 'Model routing rules' }).closest('.usage-routing-panel')
    expect(routing).not.toBeNull()
    expect((routing as Element).compareDocumentPosition(meter) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(meter).toHaveTextContent('Meter not started')
    expect(within(meter).getByRole('button', { name: 'Start' })).toBeEnabled()
    expect(within(meter).getByRole('button', { name: 'Stop' })).toBeDisabled()

    await user.click(within(meter).getByRole('button', { name: 'Start' }))
    await waitFor(() => expect(within(meter).getByRole('button', { name: 'Stop' })).toBeEnabled())
    expect(meter).toHaveTextContent('Measuring tokens used since you pressed Start')

    await user.click(within(meter).getByRole('button', { name: 'Stop' }))
    await waitFor(() => expect(meter).toHaveTextContent('Measurement stopped'))
    expect(meter).toHaveTextContent('275')
    expect(meter).toHaveTextContent('200')
    expect(meter).toHaveTextContent('75')
    expect(within(meter).getByRole('table', { name: 'Measured usage by model' })).toHaveTextContent('org/model')
    expect(fetchMock.mock.calls.every(([path]) => String(path) !== '/api/token-stats/reset')).toBe(true)
  })

  it('keeps measuring when the final cluster sync fails and allows Stop to retry', async () => {
    let stopAttempts = 0
    const current = {
      ...summary,
      models: { 'org/model': { input: 1600, cached: 525, output: 800, requests: 13 } },
      total: { input: 1600, cached: 525, output: 800, requests: 13 },
    }
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/token-stats/hourly') || path.includes('/api/token-stats/daily')) return json([])
      if (path === '/api/token-stats/sync') {
        if (stopAttempts === 0) {
          stopAttempts += 1
          return json(summary)
        }
        stopAttempts += 1
        return stopAttempts === 2 ? json({ detail: 'Worker: timed out' }, 503) : json(current)
      }
      return json(summary)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<UsagePage />)

    const meter = await screen.findByLabelText('Token usage meter')
    await user.click(within(meter).getByRole('button', { name: 'Start' }))
    await waitFor(() => expect(within(meter).getByRole('button', { name: 'Stop' })).toBeEnabled())
    await user.click(within(meter).getByRole('button', { name: 'Stop' }))

    expect(await within(meter).findByRole('alert')).toHaveTextContent('Worker: timed out')
    expect(meter).toHaveTextContent('Measuring tokens used since you pressed Start')
    expect(within(meter).getByRole('button', { name: 'Stop' })).toBeEnabled()

    await user.click(within(meter).getByRole('button', { name: 'Stop' }))
    await waitFor(() => expect(meter).toHaveTextContent('Measurement stopped'))
    expect(meter).toHaveTextContent('150')
    expect(within(meter).queryByRole('alert')).not.toBeInTheDocument()
  })

  it('leaves the meter stopped when the authoritative baseline sync fails', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/token-stats/hourly') || path.includes('/api/token-stats/daily')) return json([])
      if (path === '/api/token-stats/sync') return json({ detail: 'Worker: timed out' }, 503)
      return json(summary)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<UsagePage />)

    const meter = await screen.findByLabelText('Token usage meter')
    await user.click(within(meter).getByRole('button', { name: 'Start' }))

    expect(await within(meter).findByRole('alert')).toHaveTextContent('Worker: timed out')
    expect(meter).toHaveTextContent('Meter not started')
    expect(within(meter).getByRole('button', { name: 'Start' })).toBeEnabled()
    expect(within(meter).getByRole('button', { name: 'Stop' })).toBeDisabled()
  })

  it('clears a transient polling error after the next successful sample', async () => {
    vi.useFakeTimers()
    let meterStarted = false
    let pollReads = 0
    const current = {
      ...summary,
      models: { 'org/model': { input: 1600, cached: 525, output: 800, requests: 13 } },
      total: { input: 1600, cached: 525, output: 800, requests: 13 },
    }
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/token-stats/hourly') || path.includes('/api/token-stats/daily')) return json([])
      if (path === '/api/token-stats/sync') {
        meterStarted = true
        return json(summary)
      }
      if (path === '/api/token-stats' && meterStarted) {
        pollReads += 1
        if (pollReads === 1) return json({ detail: 'Temporary sample failure' }, 503)
        return json(current)
      }
      return json(summary)
    })
    vi.stubGlobal('fetch', fetchMock)
    render(<UsagePage />)
    await act(async () => { await Promise.resolve() })

    const meter = screen.getByLabelText('Token usage meter')
    await act(async () => {
      fireEvent.click(within(meter).getByRole('button', { name: 'Start' }))
      await Promise.resolve()
    })

    await act(() => vi.advanceTimersByTimeAsync(2_000))
    expect(within(meter).getByRole('alert')).toHaveTextContent('Temporary sample failure')

    await act(() => vi.advanceTimersByTimeAsync(2_000))
    expect(within(meter).queryByRole('alert')).not.toBeInTheDocument()
    expect(meter).toHaveTextContent('150')
  })

  it('waits for a slow token sample to settle before scheduling another poll', async () => {
    vi.useFakeTimers()
    let summaryReads = 0
    let resolveSlowSample!: (response: Response) => void
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/token-stats/hourly') || path.includes('/api/token-stats/daily')) return json([])
      if (path === '/api/token-stats/sync') return json(summary)
      if (path === '/api/token-stats') {
        summaryReads += 1
        if (summaryReads === 2) return new Promise<Response>((resolve) => { resolveSlowSample = resolve })
      }
      return json(summary)
    })
    vi.stubGlobal('fetch', fetchMock)
    render(<UsagePage />)
    await act(async () => { await Promise.resolve() })

    const meter = screen.getByLabelText('Token usage meter')
    await act(async () => {
      fireEvent.click(within(meter).getByRole('button', { name: 'Start' }))
      await Promise.resolve()
    })
    await act(() => vi.advanceTimersByTimeAsync(2_000))
    expect(summaryReads).toBe(2)

    await act(() => vi.advanceTimersByTimeAsync(6_000))
    expect(summaryReads).toBe(2)

    await act(async () => {
      resolveSlowSample(json(summary))
      await Promise.resolve()
    })
    await act(() => vi.advanceTimersByTimeAsync(1_999))
    expect(summaryReads).toBe(2)
    await act(() => vi.advanceTimersByTimeAsync(1))
    expect(summaryReads).toBe(3)
  })

  it('edits and removes usage routing rules from the display dialog', async () => {
    const rules: Record<string, string> = { 'org/model': 'org/target' }
    const routed = {
      ...summary,
      models: { 'org/model': summary.models['org/model'], 'org/target': { input: 900, cached: 100, output: 400, requests: 5 } },
      groups: [{
        key: 'model:org/target', label: 'org/target', merge_group: null,
        route_target: 'org/target', models: ['org/model', 'org/target'],
        members: [
          { model: 'org/model', alias: null, merge_group: null, routed_to: 'org/target' },
          { model: 'org/target', alias: null, merge_group: null, routed_to: null },
        ],
        stats: {
          input: 2400, input_miss: 1900, cached: 600, measured_cached: 600,
          estimated_cached: 0, output: 1150, requests: 17, gen_tokens: 1150,
          gen_time_s: 30,
        },
        speed: null,
        total_cost: 1.5, cost_estimated: false,
      }],
    }
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (init?.method === 'PUT' && path === '/api/token-stats/rules') {
        const body = JSON.parse(String(init?.body)) as { source: string; destination: string }
        rules[body.source] = body.destination
        return json({ ok: true })
      }
      if (init?.method === 'DELETE' && path.startsWith('/api/token-stats/rules/')) {
        delete rules[decodeURIComponent(path.replace('/api/token-stats/rules/', ''))]
        return json({ ok: true })
      }
      if (init?.method) return json({ ok: true })
      if (path.includes('/api/token-stats/hourly') || path.includes('/api/token-stats/daily')) return json([])
      return json({ ...routed, routing_rules: rules })
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<UsagePage />)

    await user.click(await screen.findByRole('button', { name: 'Show details for org/target' }))
    await user.click(screen.getByRole('button', { name: 'Edit usage display for org/model' }))
    expect(screen.getByRole('combobox', { name: 'Route usage to' })).toHaveValue('org/target')
    const options = [...document.querySelectorAll('#usage-route-destinations option')].map((option) => option.getAttribute('value'))
    expect(options).toContain('org/target')
    expect(options).not.toContain('org/model')

    await user.clear(screen.getByRole('combobox', { name: 'Route usage to' }))
    await user.type(screen.getByRole('combobox', { name: 'Route usage to' }), 'org/other')
    await user.click(screen.getByRole('button', { name: 'Save usage display' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/token-stats/rules', expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({ source: 'org/model', destination: 'org/other' }),
    })))
    await screen.findByText('Updated usage display for org/model.')

    await user.click(await screen.findByRole('button', { name: 'Edit usage display for org/model' }))
    expect(screen.getByRole('combobox', { name: 'Route usage to' })).toHaveValue('org/other')
    await user.clear(screen.getByRole('combobox', { name: 'Route usage to' }))
    await user.click(screen.getByRole('button', { name: 'Save usage display' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/token-stats/rules/org%2Fmodel', expect.objectContaining({ method: 'DELETE' })))
  })

  it('adds, updates, and removes rules from the visible routing manager', async () => {
    const rules: Record<string, string> = { 'org/legacy': 'org/target' }
    const routedSummary = {
      ...summary,
      models: {
        'org/model': summary.models['org/model'],
        'org/target': { input: 900, cached: 100, output: 400, requests: 5 },
        'org/legacy': { input: 300, cached: 0, output: 100, requests: 2 },
      },
    }
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (init?.method === 'PUT' && path === '/api/token-stats/rules') {
        const body = JSON.parse(String(init.body)) as { source: string; destination: string }
        rules[body.source] = body.destination
        return json({ ok: true })
      }
      if (init?.method === 'DELETE' && path.startsWith('/api/token-stats/rules/')) {
        delete rules[decodeURIComponent(path.replace('/api/token-stats/rules/', ''))]
        return json({ ok: true })
      }
      if (path.includes('/api/token-stats/hourly') || path.includes('/api/token-stats/daily')) return json([])
      return json({ ...routedSummary, routing_rules: { ...rules } })
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<UsagePage />)

    const form = await screen.findByRole('form', { name: 'Add model routing rule' })
    expect(form).toBeInTheDocument()
    expect(screen.getByLabelText('Current model routing rules')).toHaveTextContent('org/legacy')
    expect(screen.getByLabelText('Current model routing rules')).toHaveTextContent('org/target')

    await user.selectOptions(screen.getByRole('combobox', { name: 'Source model' }), 'org/model')
    await user.selectOptions(screen.getByRole('combobox', { name: 'Destination model' }), 'org/target')
    await user.click(screen.getByRole('button', { name: 'Add rule' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/token-stats/rules', expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({ source: 'org/model', destination: 'org/target' }),
    })))
    expect(await screen.findByRole('button', { name: 'Remove routing rule for org/model' })).toBeInTheDocument()

    await user.selectOptions(screen.getByRole('combobox', { name: 'Source model' }), 'org/legacy')
    expect(screen.getByRole('button', { name: 'Update rule' })).toBeInTheDocument()
    await user.selectOptions(screen.getByRole('combobox', { name: 'Destination model' }), 'org/model')
    await user.click(screen.getByRole('button', { name: 'Update rule' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/token-stats/rules', expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({ source: 'org/legacy', destination: 'org/model' }),
    })))

    await user.click(screen.getByRole('button', { name: 'Remove routing rule for org/legacy' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/token-stats/rules/org%2Flegacy', expect.objectContaining({ method: 'DELETE' })))
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Remove routing rule for org/legacy' })).not.toBeInTheDocument())
  })

  it('keeps a route-only save from overwriting the source merge group', async () => {
    const routed = {
      ...summary,
      models: { 'org/model': summary.models['org/model'], 'org/target': { input: 900, cached: 100, output: 400, requests: 5 } },
      merge_groups: {},
      groups: [{
        key: 'group:Workspace', label: 'Workspace', merge_group: 'Workspace',
        route_target: null, models: ['org/model', 'org/target'],
        members: [
          { model: 'org/model', alias: null, merge_group: 'Workspace', routed_to: 'org/target' },
          { model: 'org/target', alias: null, merge_group: 'Workspace', routed_to: null },
        ],
        stats: {
          input: 2400, input_miss: 1900, cached: 600, measured_cached: 600,
          estimated_cached: 0, output: 1150, requests: 17, gen_tokens: 1150,
          gen_time_s: 30,
        },
        speed: null,
        total_cost: 1.5, cost_estimated: false,
      }],
      routing_rules: { 'org/model': 'org/target' },
    }
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      if (init?.method) return json({ ok: true })
      if (String(input).includes('/api/token-stats/hourly') || String(input).includes('/api/token-stats/daily')) return json([])
      return json(routed)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<UsagePage />)

    await user.click(await screen.findByRole('button', { name: 'Show details for Workspace' }))
    await user.click(screen.getByRole('button', { name: 'Edit usage display for org/model' }))
    expect(screen.getByRole('textbox', { name: 'Merge group' })).toHaveValue('')
    await user.click(screen.getByRole('button', { name: 'Save usage display' }))
    await screen.findByText('Updated usage display for org/model.')

    expect(fetchMock).toHaveBeenCalledWith('/api/token-stats/alias', expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({ model: 'org/model', alias: null }),
    }))
    expect(fetchMock.mock.calls.every(([path]) => !String(path).includes('/api/token-stats/rules'))).toBe(true)
  })

  it('sends an explicit clear when a configured merge group is removed', async () => {
    const grouped = { ...summary, merge_groups: { 'org/model': 'Legacy' } }
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      if (init?.method) return json({ ok: true })
      if (String(input).includes('/api/token-stats/hourly') || String(input).includes('/api/token-stats/daily')) return json([])
      return json(grouped)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<UsagePage />)

    await user.click(await screen.findByRole('button', { name: 'Show details for Workload' }))
    await user.click(screen.getByRole('button', { name: 'Edit usage display for org/model' }))
    expect(screen.getByRole('textbox', { name: 'Merge group' })).toHaveValue('Legacy')
    await user.clear(screen.getByRole('textbox', { name: 'Merge group' }))
    await user.click(screen.getByRole('button', { name: 'Save usage display' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/token-stats/alias', expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({ model: 'org/model', alias: 'Workload', merge_group: null }),
    })))
  })

  it('surfaces routing rule validation errors inside the display dialog', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (init?.method === 'PUT' && path === '/api/token-stats/rules') return json({ detail: 'routing rules cannot contain a cycle' }, 400)
      if (init?.method) return json({ ok: true })
      if (path.includes('/api/token-stats/hourly') || path.includes('/api/token-stats/daily')) return json([])
      return json(summary)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<UsagePage />)

    await user.click(await screen.findByRole('button', { name: 'Show details for Workload' }))
    await user.click(screen.getByRole('button', { name: 'Edit usage display for org/model' }))
    await user.type(screen.getByRole('combobox', { name: 'Route usage to' }), 'org/other')
    await user.click(screen.getByRole('button', { name: 'Save usage display' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('routing rules cannot contain a cycle')
    expect(screen.getByRole('dialog', { name: 'Edit usage model' })).toBeInTheDocument()
  })

  it('surfaces historical analysis failures independently with a retry', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/token-stats/hourly')) return json({ detail: 'history unavailable' }, 503)
      if (path.includes('/api/token-stats/daily')) return json([])
      return json(summary)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()

    render(<UsagePage />)

    expect(await screen.findByRole('alert')).toHaveTextContent('Historical usage: history unavailable')
    expect(screen.getByLabelText('Usage overview')).toHaveTextContent('2,250')
    const callsBeforeRetry = fetchMock.mock.calls.length
    await user.click(screen.getByRole('button', { name: 'Retry' }))
    await waitFor(() => expect(fetchMock.mock.calls.length).toBeGreaterThan(callsBeforeRetry))
  })
})
