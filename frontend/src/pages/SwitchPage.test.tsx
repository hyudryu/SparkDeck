import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { SwitchPage } from './SwitchPage'

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

const nodes = [
  {
    node_id: 'local', node_name: 'Main Spark', online: true,
    detected: true, configured: false,
    discovery: [{ address: '192.168.88.1', identity: 'Lab Switch', board: 'CRS310', version: '7.20' }],
  },
  {
    node_id: 'node-2', node_name: 'Rack Spark', online: true,
    detected: false, configured: false, discovery: [],
  },
]

const disconnected = {
  detected: true,
  gateway_node_id: 'local',
  nodes,
  gateway: {
    ...nodes[0], connected: false, health: [], interfaces: [],
    network: { rx_bits_per_second: 0, tx_bits_per_second: 0, active_interfaces: 0, total_interfaces: 0 },
    configuration_checks: [
      { id: 'ethernet-gateway', label: 'Ethernet gateway node', status: 'passed', detail: 'Main Spark is online.' },
      { id: 'routeros-authentication', label: 'RouterOS authentication', status: 'warning', detail: 'Enter credentials.' },
    ],
  },
}

const connectedGateway = {
  ...nodes[1], detected: true, configured: true, connected: true,
  base_url: 'https://192.168.88.1', verify_tls: true,
  device: { identity: 'Core Switch', 'board-name': 'CRS518', version: '7.20', uptime: '3d04:12:10' },
  health: [{ name: 'temperature', value: '67 °C', type: 'temperature' }, { name: 'fan1-speed', value: 4210, type: 'rpm' }],
  fan_settings: {
    'fan-target-temp': '55', 'fan-full-speed-temp': '65',
    'fan-min-speed-percent': '25', 'cpu-overtemp-check': 'true',
  },
  fan_capabilities: ['fan-target-temp', 'fan-full-speed-temp', 'fan-min-speed-percent', 'cpu-overtemp-check'],
  network: { rx_bits_per_second: 9_800_000_000, tx_bits_per_second: 8_600_000_000, active_interfaces: 1, total_interfaces: 2 },
  configuration_checks: [
    { id: 'ethernet-gateway', label: 'Ethernet gateway node', status: 'passed', detail: 'Rack Spark is online.' },
    { id: 'routeros-authentication', label: 'RouterOS authentication', status: 'passed', detail: 'Credentials accepted.' },
    { id: 'secure-rest', label: 'Secure REST connection', status: 'passed', detail: 'Certificate verified.' },
  ],
  interfaces: [
    {
      name: 'sfp-sfpplus1', type: 'ether', running: 'true', status: 'link-ok', rate: '10Gbps',
      'full-duplex': 'true', 'rx-bits-per-second': '9800000000', 'tx-bits-per-second': '8600000000',
      'rx-byte': '1024', 'tx-byte': '2048',
    },
    { name: 'ether2', type: 'ether', running: 'false', status: 'no-link', 'rx-bits-per-second': '0', 'tx-bits-per-second': '0' },
  ],
}

const connected = {
  detected: true,
  gateway_node_id: 'node-2',
  nodes: [nodes[0], { ...nodes[1], detected: true, configured: true }],
  gateway: connectedGateway,
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

describe('SwitchPage', () => {
  it('shows one Ethernet gateway selector instead of a console for every node', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(json(disconnected)))
    render(<SwitchPage />)

    const selector = await screen.findByRole('combobox', { name: /^Ethernet-connected node/ })
    expect(selector).toHaveValue('local')
    expect(screen.getByRole('option', { name: 'Main Spark (switch detected)' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'Rack Spark' })).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'Username' })).toBeInTheDocument()
    expect(screen.getByLabelText('Password')).toBeRequired()
    expect(screen.getByText(/Detected Lab Switch at 192.168.88.1/)).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Main Spark' })).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Rack Spark' })).not.toBeInTheDocument()
  })

  it('saves credentials on the node selected as the Ethernet gateway', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (_input, init) => {
      if (init?.method === 'PUT') return json(connectedGateway)
      return json(disconnected)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<SwitchPage />)

    await user.selectOptions(await screen.findByRole('combobox', { name: /^Ethernet-connected node/ }), 'node-2')
    await user.type(screen.getByRole('textbox', { name: 'Username' }), 'admin')
    await user.type(screen.getByLabelText('Password'), 'secret')
    await user.click(screen.getByRole('button', { name: 'Connect and validate' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/v1/routeros/nodes/node-2/connection', expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({ base_url: 'https://192.168.88.1', username: 'admin', password: 'secret', verify_tls: true }),
    })))
  })

  it('keeps the RouterOS address available as an advanced discovery override', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(json(disconnected)))
    const user = userEvent.setup()
    render(<SwitchPage />)

    await screen.findByRole('combobox', { name: /^Ethernet-connected node/ })
    await user.click(screen.getByText('Advanced connection settings'))
    expect(screen.getByDisplayValue('https://192.168.88.1')).toHaveAttribute('type', 'url')
    expect(screen.getByRole('checkbox', { name: /Verify TLS certificate/ })).toBeChecked()
  })

  it('allows every switch discovered from the selected node to be chosen', async () => {
    const multiSwitch = {
      ...disconnected,
      nodes: [{
        ...nodes[0],
        discovery: [
          { address: '192.168.88.1', identity: 'Core Switch' },
          { address: '192.168.88.2', identity: 'Lab Switch' },
        ],
      }, nodes[1]],
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(json(multiSwitch)))
    const user = userEvent.setup()
    render(<SwitchPage />)

    await screen.findByRole('combobox', { name: /^Ethernet-connected node/ })
    await user.click(screen.getByText('Advanced connection settings'))
    await user.selectOptions(
      screen.getByRole('combobox', { name: /Discovered switch/ }),
      'https://192.168.88.2',
    )

    expect(screen.getByDisplayValue('https://192.168.88.2')).toHaveAttribute('type', 'url')
    expect(screen.getByText(/Detected 2 RouterOS switches/)).toBeInTheDocument()
  })

  it('renders configuration validation, a fan curve, and live port speeds', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (_input, init) => init?.method === 'PATCH' ? json(connectedGateway) : json(connected))
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<SwitchPage />)

    expect(await screen.findByText('Core Switch')).toBeInTheDocument()
    expect(screen.getByText('RouterOS authentication')).toBeInTheDocument()
    expect(screen.getAllByText('Passed')).toHaveLength(3)
    expect(screen.getByText('67 °C')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /Fan curve from 25% at 55 degrees to full speed at 65 degrees/ })).toBeInTheDocument()
    expect(screen.getAllByText('9.8 Gbps').length).toBeGreaterThan(0)
    expect(screen.getByText('10Gbps')).toBeInTheDocument()
    expect(screen.getByText('Link up')).toBeInTheDocument()
    expect(screen.getByText('Link down')).toBeInTheDocument()

    const minimumSpeed = screen.getByRole('spinbutton', { name: /^Minimum fan speed/ })
    await user.clear(minimumSpeed)
    await user.type(minimumSpeed, '45')
    await user.click(screen.getByRole('button', { name: 'Save fan curve' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/v1/routeros/nodes/node-2/fan-settings', expect.objectContaining({
      method: 'PATCH',
      body: JSON.stringify({
        'fan-target-temp': '55', 'fan-full-speed-temp': '65',
        'fan-min-speed-percent': '45', 'cpu-overtemp-check': true,
      }),
    })))
    expect(await screen.findByText('Fan settings saved and validation refreshed.')).toBeInTheDocument()
  })

  it('preserves unsaved fan edits when telemetry reloads', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(json(connected))
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<SwitchPage />)

    const minimumSpeed = await screen.findByRole('spinbutton', { name: /^Minimum fan speed/ })
    await user.clear(minimumSpeed)
    await user.type(minimumSpeed, '45')
    await user.click(screen.getByRole('button', { name: 'Refresh' }))
    await waitFor(() => expect(fetchMock.mock.calls.length).toBeGreaterThan(1))

    expect(minimumSpeed).toHaveValue(45)
  })
})
