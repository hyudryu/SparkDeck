import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
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

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

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

  it('explains why the view is unavailable when no heartbeat is fresh', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(json({ available: false, nodes: [] })))
    render(<FanControlPage />)
    expect(await screen.findByRole('heading', { name: 'FanController not detected' })).toBeInTheDocument()
  })
})
