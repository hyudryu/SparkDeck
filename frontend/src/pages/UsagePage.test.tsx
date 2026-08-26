import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { activityHeatmapCalendar, UsagePage } from './UsagePage'

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
})

describe('UsagePage', () => {
  it('derives heatmap month labels and positions from the rendered dates', () => {
    const calendar = activityHeatmapCalendar([], new Date('2026-01-15T12:00:00Z'))

    expect(calendar.months[0]).toMatchObject({ key: '2025-01', label: 'Jan', column: 1 })
    expect(calendar.months.at(-1)).toMatchObject({ key: '2026-01', label: 'Jan' })
    expect(calendar.months.every((month, index) => index === 0 || month.column > calendar.months[index - 1].column)).toBe(true)
  })

  it('restores lifetime model accounting and persisted hourly analysis', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/token-stats/hourly')) return json([
        { hour: '2026-08-26T10', input: 1200, cached: 300, output: 400, requests: 4 },
        { hour: '2026-08-26T11', input: 600, cached: 200, output: 250, requests: 2 },
      ])
      if (path.includes('/api/token-stats/daily')) return json([
        { date: '2026-08-26', input: 1800, cached: 500, output: 650, requests: 6, models: { 'org/model': { input: 1800, cached: 500, output: 650, requests: 6 } } },
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

  it('keeps alias, erase, and lifetime reset controls connected to the old APIs', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      if (init?.method) return json({ ok: true })
      const path = String(input)
      if (path.includes('/hourly') || path.includes('/daily')) return json([])
      return json(summary)
    })
    vi.stubGlobal('fetch', fetchMock)
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    const user = userEvent.setup()
    render(<UsagePage />)

    await user.click(await screen.findByRole('button', { name: 'Show details for Workload' }))
    await user.click(screen.getByRole('button', { name: 'Edit alias for org/model' }))
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
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/token-stats/org%2Fmodel', expect.objectContaining({ method: 'DELETE' })))
    await user.click(screen.getByRole('button', { name: 'Reset lifetime' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/token-stats/reset', expect.objectContaining({ method: 'POST' })))
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
