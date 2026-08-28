import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { FanControlPage } from './FanControlPage'

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

const settings = {
  mode: 'curve' as const,
  settings: {
    curve: { curve_points: [[30, 20], [55, 50], [80, 100]], curve_min_temp: 30, curve_max_temp: 80, min_floor_pct: 20 },
    pid: { setpoint: 60, kp: 2.1, ki: 0.2, kd: 0.1, min_floor_pct: 25 },
    hysteresis: { hyst_on_temp: 65, hyst_off_temp: 50 },
    manual: { manual_duty_pct: 42 },
  },
}

const overview = {
  available: true,
  nodes: [{
    node_id: 'node/1', node_name: 'Rack Spark', local: true,
    fan: { rpm: 4210, duty_byte: 128, duty_pct: 50, temp: 61, local_temp: 58, mode: 'curve' as const, active_settings: settings.settings.curve, status: 'running', max_speed: false, ts: 1787860800 },
    settings,
  }, {
    node_id: 'node-2', node_name: 'Desk Spark', local: false,
    fan: { rpm: 3000, duty_byte: 102, duty_pct: 40, temp: 54, local_temp: 53, mode: 'pid' as const, active_settings: settings.settings.pid, status: 'running', max_speed: true, ts: 1787860800 },
    settings: { ...settings, mode: 'pid' as const },
  }],
}

afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

describe('FanControlPage', () => {
  it('renders live telemetry, the saved curve, and active settings', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(json(overview)))
    render(<FanControlPage />)

    expect(await screen.findByText('Rack Spark (local)')).toBeInTheDocument()
    expect(screen.getByText('4210 RPM')).toBeInTheDocument()
    expect(within(screen.getByText('Fan duty').closest('section')!).getByText('50%')).toBeInTheDocument()
    expect(screen.getByText('61 °C')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Fan curve for Rack Spark' })).toBeInTheDocument()
    expect(screen.getByRole('table', { name: 'Fan curve points for Rack Spark' })).toBeInTheDocument()
    expect(screen.getByText('30 - 80 °C')).toBeInTheDocument()
    expect(screen.getByRole('switch', { name: 'Fan speed override' })).not.toBeChecked()
  })

  it('waits for a slow overview to settle before scheduling the next poll', async () => {
    vi.useFakeTimers()
    let resolveFirst!: (response: Response) => void
    const fetchMock = vi.fn<typeof fetch>()
      .mockImplementationOnce(() => new Promise<Response>((resolve) => { resolveFirst = resolve }))
      .mockImplementation(() => new Promise<Response>(() => undefined))
    vi.stubGlobal('fetch', fetchMock)

    render(<FanControlPage />)
    expect(fetchMock).toHaveBeenCalledTimes(1)

    await act(() => vi.advanceTimersByTimeAsync(6_000))
    expect(fetchMock).toHaveBeenCalledTimes(1)

    await act(async () => {
      resolveFirst(json(overview))
      await Promise.resolve()
    })
    expect(screen.getByText('Rack Spark (local)')).toBeInTheDocument()

    await act(() => vi.advanceTimersByTimeAsync(1_999))
    expect(fetchMock).toHaveBeenCalledTimes(1)
    await act(() => vi.advanceTimersByTimeAsync(1))
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('omits configured curve points outside the displayed temperature range', async () => {
    const curve = { ...settings.settings.curve, curve_points: [[-500, 10], [55, 50], [500, 90]] }
    const rangedOverview = {
      available: true,
      nodes: [{
        ...overview.nodes[0],
        settings: { ...settings, settings: { ...settings.settings, curve } },
      }],
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(json(rangedOverview)))

    render(<FanControlPage />)

    const chart = await screen.findByRole('img', { name: 'Fan curve for Rack Spark' })
    expect(chart.querySelectorAll('.fan-chart-point')).toHaveLength(1)
    expect(within(chart).queryByText('-500Â° / 10%')).not.toBeInTheDocument()
    expect(within(chart).queryByText('500Â° / 90%')).not.toBeInTheDocument()
    expect(screen.getByText('2 configured points are outside this temperature range and not plotted.')).toBeInTheDocument()
    expect(within(screen.getByRole('table', { name: 'Fan curve points for Rack Spark' })).getByRole('spinbutton', { name: 'Point 1 temperature' })).toHaveValue(-500)
  })

  it('drags a curve point and saves the exact curve to the selected node', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (_input, init) => {
      if (init?.method === 'PATCH') return json({
        node_id: 'node/1', mode: 'curve', previous_mode: 'curve',
        active_settings: { ...settings.settings.curve, curve_points: [[30, 20], [60, 70], [80, 100]] },
      })
      return json(overview)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<FanControlPage />)

    const chart = await screen.findByRole('img', { name: 'Fan curve for Rack Spark' })
    vi.spyOn(chart, 'getBoundingClientRect').mockReturnValue({
      x: 0, y: 0, left: 0, top: 0, right: 510, bottom: 225,
      width: 510, height: 225, toJSON: () => ({}),
    })
    const point = screen.getByRole('button', { name: 'Move point 2, 55 degrees, 50 percent' })
    fireEvent.pointerDown(point, { pointerId: 1, clientX: 264, clientY: 104 })
    await waitFor(() => expect(chart).toHaveClass('is-dragging'))
    fireEvent.pointerMove(point, { pointerId: 1, clientX: 307.2, clientY: 69.6 })
    fireEvent.pointerUp(point, { pointerId: 1, clientX: 307.2, clientY: 69.6 })

    expect(screen.getByRole('spinbutton', { name: 'Point 2 temperature' })).toHaveValue(60)
    expect(screen.getByRole('spinbutton', { name: 'Point 2 fan duty' })).toHaveValue(70)
    await user.click(screen.getByRole('button', { name: 'Save curve' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/v1/fan-control/nodes/node%2F1/settings', expect.objectContaining({
      method: 'PATCH',
      body: JSON.stringify({
        mode: 'curve',
        active_settings: {
          ...settings.settings.curve,
          curve_points: [[30, 20], [60, 70], [80, 100]],
        },
        expected_mode: 'curve',
      }),
    })))
    expect(await screen.findByText('Fan curve saved.')).toBeInTheDocument()
  })

  it('keeps an unsaved curve draft when live polling returns older settings', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(json(overview))
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<FanControlPage />)

    const duty = await screen.findByRole('spinbutton', { name: 'Point 2 fan duty' })
    await user.clear(duty)
    await user.type(duty, '65')
    expect(duty).toHaveValue(65)

    await user.click(screen.getByRole('button', { name: 'Refresh' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    expect(screen.getByRole('spinbutton', { name: 'Point 2 fan duty' })).toHaveValue(65)
    expect(screen.getByRole('button', { name: 'Save curve' })).toBeEnabled()
  })

  it('selects another controller node and enables max speed with the exact request', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (_input, init) => {
      if (init?.method === 'PATCH') return json({ node_id: 'node-2', enabled: false })
      return json(overview)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<FanControlPage />)

    const select = await screen.findByRole('combobox', { name: 'FanController node' })
    await user.selectOptions(select, 'node-2')
    expect(screen.getByText('3000 RPM')).toBeInTheDocument()
    expect(screen.getByText('PID mode')).toBeInTheDocument()
    expect(screen.getByText('Inactive saved curve')).toBeInTheDocument()

    const toggle = screen.getByRole('switch', { name: 'Fan speed override' })
    expect(toggle).toBeChecked()
    await user.click(toggle)

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/v1/fan-control/nodes/node-2/max-speed', expect.objectContaining({
      method: 'PATCH', body: JSON.stringify({ enabled: false }),
    })))
    expect(await screen.findByText('Automatic fan control enabled.')).toBeInTheDocument()
  })

  it('keeps the current override when the update fails', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (_input, init) => {
      if (init?.method === 'PATCH') return json({ detail: 'FanController stopped' }, 409)
      return json(overview)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<FanControlPage />)

    const toggle = await screen.findByRole('switch', { name: 'Fan speed override' })
    await user.click(toggle)
    expect(await screen.findByRole('alert')).toHaveTextContent('FanController stopped')
    expect(toggle).not.toBeChecked()
  })

  it('returns to authoritative telemetry after two newer heartbeats disagree', async () => {
    let overviewRequests = 0
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (_input, init) => {
      if (init?.method === 'PATCH') return json({ node_id: 'node/1', enabled: true })
      overviewRequests += 1
      return json({
        available: true,
        nodes: [{
          ...overview.nodes[0],
          fan: { ...overview.nodes[0].fan, max_speed: false, ts: 100 + overviewRequests },
        }],
      })
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<FanControlPage />)

    const toggle = await screen.findByRole('switch', { name: 'Fan speed override' })
    await user.click(toggle)
    await waitFor(() => expect(toggle).toBeChecked())

    await user.click(screen.getByRole('button', { name: 'Refresh' }))
    await waitFor(() => expect(overviewRequests).toBe(2))
    expect(toggle).toBeChecked()

    await user.click(screen.getByRole('button', { name: 'Refresh' }))
    await waitFor(() => expect(overviewRequests).toBe(3))
    await waitFor(() => expect(toggle).not.toBeChecked())
  })

  it('explains why the view is unavailable when no heartbeat is fresh', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(json({ available: false, nodes: [] })))
    render(<FanControlPage />)
    expect(await screen.findByRole('heading', { name: 'FanController not detected' })).toBeInTheDocument()
  })
})
