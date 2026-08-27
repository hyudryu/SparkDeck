import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ClusterPage } from './ClusterPage'

const controllerStatus = {
  role: 'controller',
  node: { id: 'controller-credential-id', name: 'Studio controller', port: 7878, access_urls: ['http://100.64.0.10:7878'] },
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
        ? { role: 'worker', node: { id: 'worker-2', name: 'Studio Spark', port: 7878, access_urls: ['http://100.64.0.11:7878'] }, controller_url: 'http://100.64.0.10:7878', controller_reachable: true }
        : controllerStatus
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<ClusterPage />)

    expect(await screen.findByText('PAIR-123')).toBeInTheDocument()
    expect(screen.getByText('Controller · Current entry node')).toBeInTheDocument()
    expect(screen.getByText('http://100.64.0.10:7878')).toBeInTheDocument()
    expect(screen.getByText('Use a private Tailscale address')).toBeInTheDocument()
    expect(screen.getByText('http://100.x.x.x:7878')).toBeInTheDocument()
    expect(screen.getByText(/Membership is per machine/)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Join this node to another controller' }))
    expect(screen.getByText(/shown by either the controller or an online worker/)).toBeInTheDocument()
    await user.type(screen.getByRole('textbox', { name: 'Existing cluster entry URL' }), 'http://100.64.0.10:7878')
    await user.clear(screen.getByRole('textbox', { name: 'This node’s advertised Tailscale URL' }))
    await user.type(screen.getByRole('textbox', { name: 'This node’s advertised Tailscale URL' }), 'http://100.64.0.11:7878')
    await user.clear(screen.getByRole('textbox', { name: 'This node’s name' }))
    await user.type(screen.getByRole('textbox', { name: 'This node’s name' }), 'Studio Spark')
    await user.type(screen.getByRole('textbox', { name: 'Pairing code' }), 'PAIR-123')
    await user.click(screen.getByRole('button', { name: 'Join controller' }))

    await waitFor(() => expect(screen.getByText('Worker node', { selector: '.cluster-summary strong' })).toBeInTheDocument())
    expect(JSON.parse(String(fetchMock.mock.calls.at(-1)?.[1]?.body))).toEqual({
      controller_url: 'http://100.64.0.10:7878',
      join_code: 'PAIR-123',
      advertise_url: 'http://100.64.0.11:7878',
      name: 'Studio Spark',
    })
    expect(screen.getByText('Joined http://100.64.0.10:7878 as Studio Spark.')).toHaveAttribute('role', 'status')
  })

  it('lets an online worker invite another node into the existing cluster', async () => {
    const workerStatus = {
      role: 'worker',
      node: { id: 'spark-2', name: 'Studio Spark', port: 7878, access_urls: ['https://spark-2.tailnet.ts.net'] },
      controller_url: 'https://controller.tailnet.ts.net',
      controller_node_id: 'controller-credential-id',
      controller_reachable: true,
      join_code: 'PAIR-456',
      instructions: ['Enter this worker entry URL and the one-time pairing code.'],
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      if (String(input).endsWith('/api/v1/nodes')) return new Response(JSON.stringify(clusterNodes), { status: 200, headers: { 'Content-Type': 'application/json' } })
      return new Response(JSON.stringify(workerStatus), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }))

    render(<ClusterPage />)

    expect(await screen.findByRole('heading', { name: 'Add a node through this worker' })).toBeInTheDocument()
    expect(screen.getByText('PAIR-456')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Worker entry URLs' })).toBeInTheDocument()
    expect(screen.getByText('https://spark-2.tailnet.ts.net')).toBeInTheDocument()
    expect(screen.getByText('https://controller.tailnet.ts.net')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Join this node to another controller' })).not.toBeInTheDocument()
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
    await user.type(screen.getByRole('textbox', { name: 'Existing cluster entry URL' }), 'https://controller.tailnet.ts.net')
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

  it('announces a local node rename so the sidebar node chip refreshes', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.endsWith('/api/v1/nodes/local') && init?.method === 'PATCH') return new Response(JSON.stringify({ ...clusterNodes.items[0], name: 'Halo Controller' }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path.endsWith('/api/v1/nodes')) return new Response(JSON.stringify(clusterNodes), { status: 200, headers: { 'Content-Type': 'application/json' } })
      return new Response(JSON.stringify(controllerStatus), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    vi.stubGlobal('fetch', fetchMock)
    const renamedEvents = vi.fn()
    window.addEventListener('sparkdeck:node-name-changed', renamedEvents)
    const user = userEvent.setup()
    render(<ClusterPage />)

    await user.click(await screen.findByRole('button', { name: 'Edit name for Studio controller' }))
    const input = screen.getByRole('textbox', { name: 'New name for Studio controller' })
    await user.clear(input)
    await user.type(input, 'Halo Controller')
    await user.click(screen.getByRole('button', { name: 'Save' }))

    expect(await screen.findByText('Renamed Studio controller to Halo Controller.')).toHaveAttribute('role', 'status')
    expect(renamedEvents).toHaveBeenCalledTimes(1)
    window.removeEventListener('sparkdeck:node-name-changed', renamedEvents)
  })

  it('does not announce worker renames as local node name changes', async () => {
    const workerStatus = {
      role: 'worker',
      node: { id: 'spark-2', name: 'Studio Spark', port: 7878, access_urls: ['https://spark-2.tailnet.ts.net'] },
      controller_url: 'https://controller.tailnet.ts.net',
      controller_reachable: true,
    }
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.endsWith('/api/v1/nodes/spark-2') && init?.method === 'PATCH') return new Response(JSON.stringify({ ...clusterNodes.items[1], name: 'Render Spark' }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path.endsWith('/api/v1/nodes')) return new Response(JSON.stringify(clusterNodes), { status: 200, headers: { 'Content-Type': 'application/json' } })
      return new Response(JSON.stringify(workerStatus), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    vi.stubGlobal('fetch', fetchMock)
    const renamedEvents = vi.fn()
    window.addEventListener('sparkdeck:node-name-changed', renamedEvents)
    const user = userEvent.setup()
    render(<ClusterPage />)

    await user.click(await screen.findByRole('button', { name: 'Edit name for Studio Spark' }))
    const input = screen.getByRole('textbox', { name: 'New name for Studio Spark' })
    await user.clear(input)
    await user.type(input, 'Render Spark')
    await user.click(screen.getByRole('button', { name: 'Save' }))

    expect(await screen.findByText('Renamed Studio Spark to Render Spark.')).toHaveAttribute('role', 'status')
    expect(renamedEvents).not.toHaveBeenCalled()
    window.removeEventListener('sparkdeck:node-name-changed', renamedEvents)
  })

  it('removes an edited worker and force-forgets an offline node with a warning', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.endsWith('/api/v1/nodes/spark-2?force=true') && init?.method === 'DELETE') {
        return new Response(JSON.stringify({ ok: true, node_id: 'spark-2', forced: true }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path.endsWith('/api/v1/nodes')) return new Response(JSON.stringify(clusterNodes), { status: 200, headers: { 'Content-Type': 'application/json' } })
      return new Response(JSON.stringify(controllerStatus), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    vi.stubGlobal('fetch', fetchMock)
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    const user = userEvent.setup()
    render(<ClusterPage />)

    await user.click(await screen.findByRole('button', { name: 'Edit name for Studio Spark' }))
    await user.click(screen.getByRole('button', { name: 'Forget node' }))

    expect(window.confirm).toHaveBeenCalledWith(expect.stringMatching(/cannot notify it.*Leave cluster.*Cached weights stay/is))
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/nodes/spark-2?force=true', expect.objectContaining({ method: 'DELETE' }))
    expect(await screen.findByText(/Forgot offline node Studio Spark/)).toHaveAttribute('role', 'status')
    expect(screen.queryByText('spark-2')).not.toBeInTheDocument()
  })

  it('does not offer removal for the current entry node', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      if (String(input).endsWith('/api/v1/nodes')) return new Response(JSON.stringify(clusterNodes), { status: 200, headers: { 'Content-Type': 'application/json' } })
      return new Response(JSON.stringify(controllerStatus), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }))
    const user = userEvent.setup()
    render(<ClusterPage />)

    await user.click(await screen.findByRole('button', { name: 'Edit name for Studio controller' }))

    expect(screen.queryByRole('button', { name: /Remove node|Forget node/ })).not.toBeInTheDocument()
  })
})
