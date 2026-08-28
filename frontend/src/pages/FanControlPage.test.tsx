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

  it('preserves untouched fractional coordinates for exact numeric edits', async () => {
    const fractionalCurve = {
      ...settings.settings.curve,
      curve_points: [[30, 20], [55.2, 50.4], [80, 100]],
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(json({
      available: true,
      nodes: [{
        ...overview.nodes[0],
        fan: { ...overview.nodes[0].fan, active_settings: fractionalCurve },
        settings: { ...settings, settings: { ...settings.settings, curve: fractionalCurve } },
      }],
    })))
    render(<FanControlPage />)

    const temperature = await screen.findByRole('spinbutton', { name: 'Point 2 temperature' })
    const duty = screen.getByRole('spinbutton', { name: 'Point 2 fan duty' })
    fireEvent.change(duty, { target: { value: '51.7' } })
    expect(temperature).toHaveValue(55.2)
    expect(duty).toHaveValue(51.7)

    fireEvent.change(temperature, { target: { value: '56.3' } })
    expect(temperature).toHaveValue(56.3)
    expect(duty).toHaveValue(51.7)
  })

  it('submits the mode captured when the curve draft first became dirty', async () => {
    let overviewRequests = 0
    const changedNode = {
      ...overview.nodes[0],
      fan: {
        ...overview.nodes[0].fan,
        mode: 'pid' as const,
        active_settings: settings.settings.pid,
        ts: overview.nodes[0].fan.ts + 1,
      },
      settings: { ...settings, mode: 'pid' as const },
    }
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (_input, init) => {
      if (init?.method === 'PATCH') return json({ detail: 'FanController mode changed; refresh and try again' }, 409)
      overviewRequests += 1
      return json(overviewRequests === 1 ? overview : { available: true, nodes: [changedNode, overview.nodes[1]] })
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<FanControlPage />)

    fireEvent.keyDown(await screen.findByRole('button', { name: 'Move point 2, 55 degrees, 50 percent' }), { key: 'ArrowUp' })
    await user.click(screen.getByRole('button', { name: 'Refresh' }))
    await waitFor(() => expect(overviewRequests).toBe(2))
    expect(screen.getByText('PID mode')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Save and activate curve' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/v1/fan-control/nodes/node%2F1/settings', expect.objectContaining({
      method: 'PATCH',
      body: expect.stringContaining('"expected_mode":"curve"'),
    })))
    expect(await screen.findByRole('alert')).toHaveTextContent('FanController mode changed')
  })

  it('disables every curve editor while a save is in flight', async () => {
    let resolvePatch!: (response: Response) => void
    const submittedCurve = {
      ...settings.settings.curve,
      curve_points: [[30, 20], [55, 51], [80, 100]],
    }
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (_input, init) => {
      if (init?.method === 'PATCH') {
        return new Promise<Response>((resolve) => { resolvePatch = resolve })
      }
      return json(overview)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<FanControlPage />)

    const point = await screen.findByRole('button', { name: 'Move point 2, 55 degrees, 50 percent' })
    fireEvent.keyDown(point, { key: 'ArrowUp' })
    expect(screen.getByRole('spinbutton', { name: 'Point 2 fan duty' })).toHaveValue(51)
    await user.click(screen.getByRole('button', { name: 'Save curve' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/v1/fan-control/nodes/node%2F1/settings', expect.anything()))

    const savingPoint = screen.getByRole('button', { name: 'Move point 2, 55 degrees, 51 percent' })
    expect(savingPoint).toHaveAttribute('aria-disabled', 'true')
    expect(screen.getByRole('combobox', { name: 'FanController node' })).toBeDisabled()
    expect(screen.getByRole('spinbutton', { name: 'Point 2 temperature' })).toBeDisabled()
    expect(screen.getByRole('spinbutton', { name: 'Point 2 fan duty' })).toBeDisabled()
    fireEvent.keyDown(savingPoint, { key: 'ArrowUp' })
    expect(screen.getByRole('spinbutton', { name: 'Point 2 fan duty' })).toHaveValue(51)

    await act(async () => {
      resolvePatch(json({
        node_id: 'node/1', mode: 'curve', previous_mode: 'curve', active_settings: submittedCurve,
      }))
      await Promise.resolve()
    })
    expect(await screen.findByText('Fan curve saved.')).toBeInTheDocument()
  })

  it('reports inactive-curve activation only after matching newer telemetry', async () => {
    let overviewRequests = 0
    const submittedCurve = {
      ...settings.settings.curve,
      curve_points: [[30, 20], [55, 51], [80, 100]],
    }
    const transitionalNode = {
      ...overview.nodes[1],
      fan: { ...overview.nodes[1].fan, ts: overview.nodes[1].fan.ts + 1 },
      settings: {
        ...overview.nodes[1].settings,
        settings: { ...overview.nodes[1].settings.settings, curve: submittedCurve },
      },
    }
    const activatedNode = {
      ...transitionalNode,
      fan: {
        ...transitionalNode.fan,
        mode: 'curve' as const,
        active_settings: submittedCurve,
        ts: overview.nodes[1].fan.ts + 2,
      },
      settings: { ...transitionalNode.settings, mode: 'curve' as const },
    }
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (_input, init) => {
      if (init?.method === 'PATCH') return json({
        node_id: 'node-2', mode: 'curve', previous_mode: 'pid', active_settings: submittedCurve,
      })
      overviewRequests += 1
      if (overviewRequests === 1) return json(overview)
      if (overviewRequests === 2) return json({ available: true, nodes: [overview.nodes[0], transitionalNode] })
      return json({ available: true, nodes: [overview.nodes[0], activatedNode] })
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<FanControlPage />)

    await user.selectOptions(await screen.findByRole('combobox', { name: 'FanController node' }), 'node-2')
    fireEvent.keyDown(screen.getByRole('button', { name: 'Move point 2, 55 degrees, 50 percent' }), { key: 'ArrowUp' })
    await user.click(screen.getByRole('button', { name: 'Save and activate curve' }))

    expect(await screen.findByText('Fan curve configuration saved. Waiting for activation telemetry…')).toBeInTheDocument()
    await waitFor(() => expect(overviewRequests).toBe(2))
    expect(screen.queryByText('Fan curve saved and activation confirmed.')).not.toBeInTheDocument()

    await waitFor(() => expect(screen.getByRole('button', { name: 'Refresh' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: 'Refresh' }))
    expect(await screen.findByText('Fan curve saved and activation confirmed.')).toBeInTheDocument()
  })

  it('keeps activation feedback with its originating node', async () => {
    let overviewRequests = 0
    const submittedCurve = {
      ...settings.settings.curve,
      curve_points: [[30, 20], [55, 51], [80, 100]],
    }
    const transitionalNode = {
      ...overview.nodes[1],
      fan: { ...overview.nodes[1].fan, ts: overview.nodes[1].fan.ts + 1 },
      settings: {
        ...overview.nodes[1].settings,
        settings: { ...overview.nodes[1].settings.settings, curve: submittedCurve },
      },
    }
    const activatedNode = {
      ...transitionalNode,
      fan: {
        ...transitionalNode.fan,
        mode: 'curve' as const,
        active_settings: submittedCurve,
        ts: overview.nodes[1].fan.ts + 2,
      },
      settings: { ...transitionalNode.settings, mode: 'curve' as const },
    }
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (_input, init) => {
      if (init?.method === 'PATCH') return json({
        node_id: 'node-2', mode: 'curve', previous_mode: 'pid', active_settings: submittedCurve,
      })
      overviewRequests += 1
      if (overviewRequests === 1) return json(overview)
      if (overviewRequests === 2) return json({ available: true, nodes: [overview.nodes[0], transitionalNode] })
      return json({ available: true, nodes: [overview.nodes[0], activatedNode] })
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<FanControlPage />)

    const select = await screen.findByRole('combobox', { name: 'FanController node' })
    await user.selectOptions(select, 'node-2')
    fireEvent.keyDown(screen.getByRole('button', { name: 'Move point 2, 55 degrees, 50 percent' }), { key: 'ArrowUp' })
    await user.click(screen.getByRole('button', { name: 'Save and activate curve' }))
    expect(await screen.findByText('Fan curve configuration saved. Waiting for activation telemetry…')).toBeInTheDocument()
    await waitFor(() => expect(overviewRequests).toBe(2))

    await user.selectOptions(select, 'node/1')
    expect(screen.queryByText('Fan curve configuration saved. Waiting for activation telemetry…')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Refresh' }))
    await waitFor(() => expect(overviewRequests).toBe(3))
    expect(screen.queryByText('Fan curve saved and activation confirmed.')).not.toBeInTheDocument()

    await user.selectOptions(select, 'node-2')
    expect(screen.getByText('Fan curve saved and activation confirmed.')).toBeInTheDocument()
  })

  it('surfaces newer telemetry that never activates the saved curve', async () => {
    let overviewRequests = 0
    const submittedCurve = {
      ...settings.settings.curve,
      curve_points: [[30, 20], [55, 51], [80, 100]],
    }
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (_input, init) => {
      if (init?.method === 'PATCH') return json({
        node_id: 'node-2', mode: 'curve', previous_mode: 'pid', active_settings: submittedCurve,
      })
      overviewRequests += 1
      if (overviewRequests === 1) return json(overview)
      return json({
        available: true,
        nodes: [overview.nodes[0], {
          ...overview.nodes[1],
          fan: { ...overview.nodes[1].fan, ts: overview.nodes[1].fan.ts + overviewRequests - 1 },
          settings: {
            ...overview.nodes[1].settings,
            settings: { ...overview.nodes[1].settings.settings, curve: submittedCurve },
          },
        }],
      })
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<FanControlPage />)

    await user.selectOptions(await screen.findByRole('combobox', { name: 'FanController node' }), 'node-2')
    fireEvent.keyDown(screen.getByRole('button', { name: 'Move point 2, 55 degrees, 50 percent' }), { key: 'ArrowUp' })
    await user.click(screen.getByRole('button', { name: 'Save and activate curve' }))
    await waitFor(() => expect(overviewRequests).toBe(2))

    for (const expectedRequests of [3, 4]) {
      await waitFor(() => expect(screen.getByRole('button', { name: 'Refresh' })).toBeEnabled())
      await user.click(screen.getByRole('button', { name: 'Refresh' }))
      await waitFor(() => expect(overviewRequests).toBe(expectedRequests))
    }

    expect(await screen.findByRole('alert')).toHaveTextContent('curve mode activation was not confirmed')
  })

  it('allows a full ten-second controller poll before activation times out', async () => {
    vi.useFakeTimers()
    const submittedCurve = {
      ...settings.settings.curve,
      curve_points: [[30, 20], [55, 51], [80, 100]],
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (_input, init) => {
      if (init?.method === 'PATCH') return json({
        node_id: 'node-2', mode: 'curve', previous_mode: 'pid', active_settings: submittedCurve,
      })
      return json(overview)
    }))
    render(<FanControlPage />)
    await act(() => vi.advanceTimersByTimeAsync(0))

    fireEvent.change(screen.getByRole('combobox', { name: 'FanController node' }), { target: { value: 'node-2' } })
    fireEvent.keyDown(screen.getByRole('button', { name: 'Move point 2, 55 degrees, 50 percent' }), { key: 'ArrowUp' })
    fireEvent.click(screen.getByRole('button', { name: 'Save and activate curve' }))
    await act(() => vi.advanceTimersByTimeAsync(0))
    expect(screen.getByText('Fan curve configuration saved. Waiting for activation telemetry…')).toBeInTheDocument()

    await act(() => vi.advanceTimersByTimeAsync(10_000))
    expect(screen.getByText('Fan curve configuration saved. Waiting for activation telemetry…')).toBeInTheDocument()
    await act(() => vi.advanceTimersByTimeAsync(4_999))
    expect(screen.queryByText('curve mode activation timed out', { exact: false })).not.toBeInTheDocument()
    await act(() => vi.advanceTimersByTimeAsync(1))
    expect(screen.getByRole('alert')).toHaveTextContent('curve mode activation timed out')
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

  it('returns to divergent server settings after bounded saved-draft reconciliation', async () => {
    let overviewRequests = 0
    const submittedCurve = {
      ...settings.settings.curve,
      curve_points: [[30, 20], [55, 51], [80, 100]],
    }
    const divergentCurve = {
      ...settings.settings.curve,
      curve_points: [[30, 20], [55, 47], [80, 100]],
    }
    const divergentOverview = {
      available: true,
      nodes: [{
        ...overview.nodes[0],
        fan: { ...overview.nodes[0].fan, active_settings: divergentCurve },
        settings: { ...settings, settings: { ...settings.settings, curve: divergentCurve } },
      }],
    }
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (_input, init) => {
      if (init?.method === 'PATCH') return json({
        node_id: 'node/1', mode: 'curve', previous_mode: 'curve', active_settings: submittedCurve,
      })
      overviewRequests += 1
      return json(overviewRequests === 1 ? { available: true, nodes: [overview.nodes[0]] } : divergentOverview)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<FanControlPage />)

    fireEvent.keyDown(await screen.findByRole('button', { name: 'Move point 2, 55 degrees, 50 percent' }), { key: 'ArrowUp' })
    await user.click(screen.getByRole('button', { name: 'Save curve' }))
    await waitFor(() => expect(overviewRequests).toBe(2))
    expect(screen.getByRole('spinbutton', { name: 'Point 2 fan duty' })).toHaveValue(51)

    for (const expectedRequests of [3, 4]) {
      await user.click(screen.getByRole('button', { name: 'Refresh' }))
      await waitFor(() => expect(overviewRequests).toBe(expectedRequests))
    }
    expect(screen.getByRole('spinbutton', { name: 'Point 2 fan duty' })).toHaveValue(47)
    expect(screen.getByRole('button', { name: 'Save curve' })).toBeDisabled()
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
