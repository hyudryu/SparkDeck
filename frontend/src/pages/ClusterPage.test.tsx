import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ClusterPage } from './ClusterPage'

const controllerStatus = {
  role: 'controller',
  node: { id: 'controller-credential-id', name: 'Studio controller', port: 7878, access_urls: ['https://controller.tailnet.ts.net'] },
  controller_reachable: true,
  join_code: 'PAIR-123',
  instructions: ['Connect both nodes to Tailscale.'],
}

const clusterNodes = {
  items: [
    { id: 'local', name: 'Studio controller', local: true, online: true, docker_ready: true, selectable: true },
    { id: 'spark-2', name: 'Studio Spark', online: false, docker_ready: false, selectable: false },
  ],
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('ClusterPage', () => {
  it('shows controller access details and joins through the local onboarding API', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      if (String(input).endsWith('/api/v1/nodes')) return new Response(JSON.stringify(clusterNodes), { status: 200, headers: { 'Content-Type': 'application/json' } })
      const body = init?.method === 'POST'
        ? { role: 'worker', node: { id: 'worker-2', name: 'Studio Spark', port: 7878, access_urls: ['https://spark-2.tailnet.ts.net'] }, controller_url: 'https://controller.tailnet.ts.net', controller_reachable: true }
        : controllerStatus
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<ClusterPage />)

    expect(await screen.findByText('PAIR-123')).toBeInTheDocument()
    expect(screen.getByText('Controller · Current entry node')).toBeInTheDocument()
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

    await waitFor(() => expect(screen.getByText('Worker node', { selector: '.cluster-summary strong' })).toBeInTheDocument())
    expect(JSON.parse(String(fetchMock.mock.calls.at(-1)?.[1]?.body))).toEqual({
      controller_url: 'https://controller.tailnet.ts.net',
      join_code: 'PAIR-123',
      advertise_url: 'https://spark-2.tailnet.ts.net',
      name: 'Studio Spark',
    })
    expect(screen.getByText('Joined https://controller.tailnet.ts.net as Studio Spark.')).toHaveAttribute('role', 'status')
  })

  it('keeps the join form open and exposes backend errors', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      if (String(input).endsWith('/api/v1/nodes')) return new Response(JSON.stringify(clusterNodes), { status: 200, headers: { 'Content-Type': 'application/json' } })
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

  it('renames any node inline and keeps worker gateway role labels accurate', async () => {
    const workerStatus = {
      role: 'worker',
      node: { id: 'spark-2', name: 'Studio Spark', port: 7878, access_urls: ['https://spark-2.tailnet.ts.net'] },
      controller_url: 'https://controller.tailnet.ts.net',
      controller_reachable: true,
    }
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.endsWith('/api/v1/nodes/spark-2') && init?.method === 'PATCH') {
        return new Response(JSON.stringify({ ...clusterNodes.items[1], name: 'Render Spark' }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path.endsWith('/api/v1/nodes')) return new Response(JSON.stringify(clusterNodes), { status: 200, headers: { 'Content-Type': 'application/json' } })
      return new Response(JSON.stringify(workerStatus), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<ClusterPage />)

    expect(await screen.findByText('Worker node · Current entry node')).toBeInTheDocument()
    expect(screen.getByText('Controller', { selector: '.node-management-state > span' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Edit name for Studio Spark' }))
    const input = screen.getByRole('textbox', { name: 'New name for Studio Spark' })
    await user.clear(input)
    await user.click(screen.getByRole('button', { name: 'Save' }))
    expect(screen.getByRole('alert')).toHaveTextContent('Enter a node name.')
    await user.type(input, '  Render Spark  ')
    await user.click(screen.getByRole('button', { name: 'Save' }))

    expect(await screen.findByText('Renamed Studio Spark to Render Spark.')).toHaveAttribute('role', 'status')
    expect(screen.getByRole('button', { name: 'Edit name for Render Spark' })).toBeInTheDocument()
    expect(JSON.parse(String(fetchMock.mock.calls.find(([url, options]) => String(url).endsWith('/spark-2') && options?.method === 'PATCH')?.[1]?.body))).toEqual({ name: 'Render Spark' })
    expect(screen.getByText('Render Spark', { selector: '.cluster-summary strong' })).toBeInTheDocument()
  })

  it('keeps editing available and announces a rename error', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.endsWith('/api/v1/nodes/local') && init?.method === 'PATCH') return new Response(JSON.stringify({ detail: 'Name is already in use' }), { status: 409, headers: { 'Content-Type': 'application/json' } })
      if (path.endsWith('/api/v1/nodes')) return new Response(JSON.stringify(clusterNodes), { status: 200, headers: { 'Content-Type': 'application/json' } })
      return new Response(JSON.stringify(controllerStatus), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }))
    const user = userEvent.setup()
    render(<ClusterPage />)

    await user.click(await screen.findByRole('button', { name: 'Edit name for Studio controller' }))
    const input = screen.getByRole('textbox', { name: 'New name for Studio controller' })
    await user.clear(input)
    await user.type(input, 'Studio Spark')
    await user.click(screen.getByRole('button', { name: 'Save' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Name is already in use')
    expect(input).toHaveValue('Studio Spark')
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeEnabled()
  })
})
