import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { SwitchPage } from './SwitchPage'

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

const disconnected = {
  detected: false,
  nodes: [{
    node_id: 'local', node_name: 'Main Spark', detected: false, configured: false, connected: false,
    discovery: [{ address: '192.168.88.1', identity: 'Lab Switch', board: 'CRS310', version: '7.20' }],
    health: [], interfaces: [],
  }],
}

const connected = {
  detected: true,
  nodes: [{
    node_id: 'node-2', node_name: 'Rack Spark', detected: true, configured: true, connected: true,
    discovery: [], error: undefined,
    device: { identity: 'Core Switch', 'board-name': 'CRS518', version: '7.20', uptime: '3d04:12:10' },
    health: [{ name: 'temperature', value: '67 °C', type: 'temperature' }, { name: 'fan1-speed', value: 4210, type: 'rpm' }],
    fan_settings: { 'fan-target-temp': '55', 'fan-min-speed-percent': '25', 'cpu-overtemp-check': 'true' },
    fan_capabilities: ['fan-target-temp', 'fan-min-speed-percent', 'cpu-overtemp-check'],
    interfaces: [
      { name: 'sfp-sfpplus1', type: 'ether', running: true, 'rx-byte': 1024, 'tx-byte': 2048 },
      { 'default-name': 'ether2', type: 'ether', running: false, 'rx-byte': 0, 'tx-byte': 0 },
    ],
  }],
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

describe('SwitchPage', () => {
  it('shows the unavailable state and lets a discovered candidate seed onboarding', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(json(disconnected)))
    const user = userEvent.setup()
    render(<SwitchPage />)

    expect(await screen.findByRole('heading', { name: 'Switch not detected' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /Lab Switch/ }))
    expect(screen.getByRole('textbox', { name: /^RouterOS URL/ })).toHaveValue('https://192.168.88.1')
    expect(screen.getByText(/Credentials stay on Main Spark/)).toBeInTheDocument()
  })

  it('saves a RouterOS connection on the selected cluster node', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (_input, init) => {
      if (init?.method === 'PUT') return json(connected.nodes[0])
      return json(disconnected)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<SwitchPage />)

    await screen.findByRole('heading', { name: 'Switch not detected' })
    await user.type(screen.getByRole('textbox', { name: 'Username' }), 'admin')
    await user.type(screen.getByLabelText('Password'), 'secret')
    await user.click(screen.getByRole('button', { name: 'Save connection' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/v1/routeros/nodes/local/connection', expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({ base_url: 'https://192.168.88.1', username: 'admin', password: 'secret', verify_tls: true }),
    })))
  })

  it('renders RouterOS telemetry and updates supported fan settings', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (_input, init) => init?.method === 'PATCH' ? json(connected.nodes[0]) : json(connected))
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<SwitchPage />)

    expect(await screen.findByText('Core Switch')).toBeInTheDocument()
    expect(screen.getByText('67 °C')).toBeInTheDocument()
    expect(screen.getByText('sfp-sfpplus1')).toBeInTheDocument()
    expect(screen.getByText('ether2')).toBeInTheDocument()
    const minimumSpeed = screen.getByRole('spinbutton', { name: /^Minimum fan speed/ })
    expect(minimumSpeed).toHaveAttribute('step', '1')
    fireEvent.change(minimumSpeed, { target: { value: '45' } })
    await user.click(screen.getByRole('button', { name: 'Save fan settings' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/v1/routeros/nodes/node-2/fan-settings', expect.objectContaining({
      method: 'PATCH', body: JSON.stringify({ 'fan-target-temp': '55', 'fan-min-speed-percent': '45', 'cpu-overtemp-check': true }),
    })))
    expect(await screen.findByText('Fan settings saved.')).toBeInTheDocument()
  })
})
