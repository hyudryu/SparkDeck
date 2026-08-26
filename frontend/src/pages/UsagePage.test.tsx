import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { UsagePage } from './UsagePage'

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
  it('restores lifetime model accounting and persisted hourly analysis', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/token-stats/hourly')) return json([
        { hour: '2026-08-26T10', input: 1200, cached: 300, output: 400, requests: 4 },
        { hour: '2026-08-26T11', input: 600, cached: 200, output: 250, requests: 2 },
      ])
      if (path.includes('/api/token-stats/daily')) return json([
        { date: '2026-08-26', input: 1800, cached: 500, output: 650, requests: 6 },
      ])
      return json(summary)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<UsagePage />)

    expect(await screen.findByRole('heading', { name: 'Usage' })).toBeInTheDocument()
    expect(await screen.findByText('Workload')).toBeInTheDocument()
    expect(screen.getByLabelText('Lifetime totals')).toHaveTextContent('1,000')
    expect(screen.getByLabelText('Lifetime totals')).toHaveTextContent('500')
    expect(screen.getByRole('table', { name: 'Lifetime model usage' })).toHaveTextContent('30.0 tok/s')
    expect(screen.getByRole('table', { name: 'Lifetime model usage' })).toHaveTextContent('$1.25')

    await user.click(screen.getByRole('tab', { name: 'Analysis' }))
    expect(await screen.findByRole('img', { name: 'Hourly input and output token activity' })).toBeInTheDocument()
    expect(screen.getByText('Aug 26')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith('/api/token-stats/hourly', expect.objectContaining({ signal: expect.any(AbortSignal) }))
    expect(fetchMock).toHaveBeenCalledWith('/api/token-stats/daily', expect.objectContaining({ signal: expect.any(AbortSignal) }))
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
})
