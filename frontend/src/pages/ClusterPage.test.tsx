import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ClusterPage } from './ClusterPage'

const controllerStatus = {
  role: 'controller',
  node: { id: 'local', name: 'Studio controller', port: 7878, access_urls: ['https://controller.tailnet.ts.net'] },
  controller_reachable: true,
  join_code: 'PAIR-123',
  instructions: ['Connect both nodes to Tailscale.'],
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('ClusterPage', () => {
  it('shows controller access details and joins through the local onboarding API', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (_input, init) => {
      const body = init?.method === 'POST'
        ? { role: 'worker', node: { id: 'worker-2', name: 'Studio Spark', port: 7878, access_urls: ['https://spark-2.tailnet.ts.net'] }, controller_url: 'https://controller.tailnet.ts.net', controller_reachable: true }
        : controllerStatus
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<ClusterPage />)

    expect(await screen.findByText('PAIR-123')).toBeInTheDocument()
    expect(screen.getByText('https://controller.tailnet.ts.net')).toBeInTheDocument()
    expect(screen.getByText(/tailscale serve --bg --https=443 localhost:7878/)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Join this node to another controller' }))
    await user.type(screen.getByRole('textbox', { name: 'Controller Tailscale URL' }), 'https://controller.tailnet.ts.net')
    await user.clear(screen.getByRole('textbox', { name: 'This node’s advertised Tailscale URL' }))
    await user.type(screen.getByRole('textbox', { name: 'This node’s advertised Tailscale URL' }), 'https://spark-2.tailnet.ts.net')
    await user.clear(screen.getByRole('textbox', { name: 'This node’s name' }))
    await user.type(screen.getByRole('textbox', { name: 'This node’s name' }), 'Studio Spark')
    await user.type(screen.getByRole('textbox', { name: 'Pairing code' }), 'PAIR-123')
    await user.click(screen.getByRole('button', { name: 'Join controller' }))

    await waitFor(() => expect(screen.getByText('Worker node')).toBeInTheDocument())
    expect(JSON.parse(String(fetchMock.mock.calls.at(-1)?.[1]?.body))).toEqual({
      controller_url: 'https://controller.tailnet.ts.net',
      join_code: 'PAIR-123',
      advertise_url: 'https://spark-2.tailnet.ts.net',
      name: 'Studio Spark',
    })
    expect(screen.getByText('Joined https://controller.tailnet.ts.net as Studio Spark.')).toHaveAttribute('role', 'status')
  })

  it('keeps the join form open and exposes backend errors', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (_input, init) => {
      if (init?.method === 'POST') return new Response(JSON.stringify({ detail: 'Pairing code expired' }), { status: 400, headers: { 'Content-Type': 'application/json' } })
      return new Response(JSON.stringify(controllerStatus), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }))
    const user = userEvent.setup()
    render(<ClusterPage />)

    await user.click(await screen.findByRole('button', { name: 'Join this node to another controller' }))
    await user.type(screen.getByRole('textbox', { name: 'Controller Tailscale URL' }), 'https://controller.tailnet.ts.net')
    await user.type(screen.getByRole('textbox', { name: 'Pairing code' }), 'EXPIRED')
    await user.click(screen.getByRole('button', { name: 'Join controller' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Pairing code expired')
    expect(screen.getByRole('button', { name: 'Join controller' })).toBeInTheDocument()
  })
})
