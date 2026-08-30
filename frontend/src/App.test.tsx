import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { ExplorePage } from './pages/ExplorePage'
import { ModelsPage } from './pages/ModelsPage'

const fetchMock = vi.fn<typeof fetch>()

beforeEach(() => {
  vi.stubGlobal('fetch', fetchMock)
  fetchMock.mockClear()
  fetchMock.mockImplementation(async () => new Response(JSON.stringify({ items: [], total: 0 }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  }))
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  localStorage.clear()
  delete document.documentElement.dataset.theme
})

describe('SparkDeck application shell', () => {
  it('exposes every primary destination in the left navigation', async () => {
    render(<MemoryRouter initialEntries={['/']}><App /></MemoryRouter>)

    expect(screen.getByRole('link', { name: 'SparkDeck home' })).toBeInTheDocument()
    expect(screen.queryByText('Local service')).not.toBeInTheDocument()
    for (const label of ['Dashboard', 'Explore', 'Models', 'Cluster', 'Switch', 'Chat', 'Compare', 'Benchmarks', 'Usage', 'Images', 'Storage', 'Settings', 'Logs']) {
      expect(screen.getByRole('link', { name: label })).toBeInTheDocument()
    }
    expect(screen.getByRole('link', { name: 'Switch' })).toHaveAttribute('aria-disabled', 'true')
    expect(screen.getByRole('link', { name: 'Switch' })).toHaveAttribute('title', 'Switch is not detected')
    expect(screen.queryByRole('link', { name: 'Fan Control' })).not.toBeInTheDocument()
    expect(await screen.findByRole('heading', { name: 'Dashboard' })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: 'System overview' })).toBeInTheDocument()
  })

  it('enables the Switch destination when RouterOS is detected', async () => {
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/routeros/presence')) {
        return new Response(JSON.stringify({ detected: true, nodes: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter initialEntries={['/']}><App /></MemoryRouter>)

    await waitFor(() => expect(screen.getByRole('link', { name: 'Switch' })).toHaveAttribute('href', '/switch'))
    expect(screen.getByRole('link', { name: 'Switch' })).not.toHaveAttribute('aria-disabled')
  })

  it('only shows Fan Control for a non-empty available FanController overview', async () => {
    fetchMock.mockImplementation(async (input) => {
      if (String(input).includes('/api/v1/fan-control')) {
        return new Response(JSON.stringify({ available: true, nodes: [{ node_id: 'fan-node' }] }), {
          status: 200, headers: { 'Content-Type': 'application/json' },
        })
      }
      return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter initialEntries={['/']}><App /></MemoryRouter>)

    await waitFor(() => expect(screen.getByRole('link', { name: 'Fan Control' })).toHaveAttribute('href', '/fan-control'))
  })

  it('shows this node name next to Dashboard in the navigation', async () => {
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/onboarding')) {
        return new Response(JSON.stringify({
          role: 'controller',
          node: { id: 'gx10-node-1', name: 'gx10-node-1', port: 7878, access_urls: ['http://100.64.0.1:7878'] },
          controller_reachable: true,
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter initialEntries={['/']}><App /></MemoryRouter>)

    // The exact accessible-name spacing across the two spans is a jsdom
    // computation quirk, so assert on the chip inside the Dashboard link.
    const dashboard = await screen.findByRole('link', { name: /Dashboard/ })
    expect(dashboard).toHaveAttribute('href', '/')
    expect(within(dashboard).getByText('gx10-node-1')).toBeInTheDocument()
    // Other destinations stay untouched by the node name.
    expect(screen.getByRole('link', { name: 'Explore' })).toHaveAttribute('href', '/explore')
  })

  it('names the chip after the entry node even when a joined worker forwards the controller name', async () => {
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/onboarding')) {
        return new Response(JSON.stringify({
          role: 'worker',
          node: { id: 'spark-2', name: 'gx10-worker-2', port: 7878, access_urls: ['http://100.64.0.11:7878'] },
          controller_url: 'http://100.64.0.10:7878',
          controller_reachable: true,
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      // A joined worker forwards /api/v1/settings to the controller, so any
      // node name in that response identifies the controller, not the node
      // serving the browser.
      if (path.includes('/api/v1/settings')) {
        return new Response(JSON.stringify({ cluster_node_name: 'gx10-controller' }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter initialEntries={['/']}><App /></MemoryRouter>)

    const dashboard = await screen.findByRole('link', { name: /Dashboard/ })
    expect(within(dashboard).getByText('gx10-worker-2')).toBeInTheDocument()
    expect(within(dashboard).queryByText('gx10-controller')).not.toBeInTheDocument()
  })

  it('refreshes the sidebar node name when the entry node is renamed', async () => {
    let nodeName = 'gx10-node-1'
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/onboarding')) {
        return new Response(JSON.stringify({
          role: 'controller',
          node: { id: 'gx10-node-1', name: nodeName, port: 7878, access_urls: ['http://100.64.0.1:7878'] },
          controller_reachable: true,
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter initialEntries={['/']}><App /></MemoryRouter>)
    const dashboard = await screen.findByRole('link', { name: /Dashboard/ })
    expect(within(dashboard).getByText('gx10-node-1')).toBeInTheDocument()

    nodeName = 'renamed-gx10'
    window.dispatchEvent(new Event('sparkdeck:node-name-changed'))
    await waitFor(() => expect(within(screen.getByRole('link', { name: /Dashboard/ })).getByText('renamed-gx10')).toBeInTheDocument())
  })

  it('ignores an older aborted presence failure after a newer refresh succeeds', async () => {
    const presenceRequests: Array<{
      signal?: AbortSignal
      resolve: (response: Response) => void
      reject: (reason: Error) => void
    }> = []
    fetchMock.mockImplementation(async (input, init) => {
      if (!String(input).includes('/api/v1/routeros/presence')) {
        return new Response(JSON.stringify({ items: [], total: 0 }), {
          status: 200, headers: { 'Content-Type': 'application/json' },
        })
      }
      return await new Promise<Response>((resolve, reject) => {
        presenceRequests.push({ signal: init?.signal ?? undefined, resolve, reject })
      })
    })

    render(<MemoryRouter initialEntries={['/']}><App /></MemoryRouter>)
    await waitFor(() => expect(presenceRequests).toHaveLength(1))

    window.dispatchEvent(new Event('sparkdeck:routeros-presence-changed'))
    await waitFor(() => expect(presenceRequests).toHaveLength(2))
    expect(presenceRequests[0].signal?.aborted).toBe(true)

    presenceRequests[1].resolve(new Response(JSON.stringify({ detected: true, nodes: [] }), {
      status: 200, headers: { 'Content-Type': 'application/json' },
    }))
    await waitFor(() => expect(screen.getByRole('link', { name: 'Switch' })).toHaveAttribute('href', '/switch'))

    presenceRequests[0].reject(new Error('late aborted request'))
    await waitFor(() => expect(screen.getByRole('link', { name: 'Switch' })).toHaveAttribute('href', '/switch'))
  })

  it('opens and closes the mobile navigation with accessible controls', async () => {
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/']}><App /></MemoryRouter>)

    const open = screen.getByRole('button', { name: 'Open navigation' })
    expect(screen.queryByRole('button', { name: 'Close navigation' })).not.toBeInTheDocument()
    await user.click(open)
    expect(open).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getAllByRole('button', { name: 'Close navigation' }).length).toBeGreaterThan(0)

    fireEvent.keyDown(document, { key: 'Escape' })
    await waitFor(() => expect(open).toHaveAttribute('aria-expanded', 'false'))
  })

  it('persists the sidebar theme toggle locally and through settings', async () => {
    const user = userEvent.setup()
    let settings = { theme: 'light', hf_token: '', hf_token_configured: false }
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.includes('/api/v1/settings') || path.includes('/api/settings')) {
        if (init?.method === 'PUT' || init?.method === 'POST') {
          settings = JSON.parse(String(init.body)) as typeof settings
        }
        return new Response(JSON.stringify(settings), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    const first = render(<MemoryRouter initialEntries={['/']}><App /></MemoryRouter>)
    const toggle = await screen.findByRole('button', { name: 'Switch to dark mode' })
    await user.click(toggle)

    expect(document.documentElement.dataset.theme).toBe('dark')
    expect(localStorage.getItem('sparkdeck.theme')).toBe('dark')
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/v1/settings', expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({ ...settings, theme: 'dark' }),
    })))
    expect(await screen.findByText('Dark mode saved.')).toBeInTheDocument()

    first.unmount()
    render(<MemoryRouter initialEntries={['/']}><App /></MemoryRouter>)
    expect(await screen.findByRole('button', { name: 'Switch to light mode' })).toBeInTheDocument()
    expect(document.documentElement.dataset.theme).toBe('dark')
  })
})

describe('model discovery', () => {
  it('renders the real catalog compatibility and local deployment shape', async () => {
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      const body = path.includes('/api/v1/catalog/models') ? {
        items: [{
          id: 'org/model', author: 'org', name: 'model', downloads: 1200,
          parameter_count: 7_000_000_000, weight_size_bytes: 14 * 1024 ** 3,
          runtime_compatibility: [{ runtime: 'vllm', supported: true }],
          local_deployment_ids: ['dep-1'], community: null,
        }],
        total: 1,
      } : path.includes('/api/v1/nodes') ? { items: [{
        id: 'local', name: 'Spark', online: true, docker_ready: true, selectable: true,
        stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] },
      }] } : { items: [] }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)

    const user = userEvent.setup()
    const row = await screen.findByRole('button', { name: 'Expand org/model' })
    expect(row).toHaveTextContent('7B')
    expect(row).toHaveTextContent('14 GB')
    expect(screen.queryByLabelText('Compatible runtimes')).not.toBeInTheDocument()
    await user.click(row)
    expect(within(screen.getByLabelText('Compatible runtimes')).getByText('vLLM')).toBeInTheDocument()
    expect(screen.getByText('Local')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Deploy org/model' })).toHaveAttribute('href', '/models?model=org%2Fmodel&runtime=vllm')
  })

  it('sends the search term and runtime filter to the versioned catalog API', async () => {
    const user = userEvent.setup()
    render(<MemoryRouter><ExplorePage /></MemoryRouter>)
    await screen.findByText('No models found')

    await user.type(screen.getByRole('textbox', { name: 'Search models' }), 'Llama 4')
    await user.selectOptions(screen.getByRole('combobox', { name: 'Runtime' }), 'llama.cpp')
    await user.click(screen.getByRole('button', { name: 'Search' }))

    await waitFor(() => {
      expect(fetchMock).toHaveBeenLastCalledWith(
        expect.stringContaining('/api/v1/catalog/models?q=Llama+4&runtime=llama.cpp'),
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      )
    })
  })

  it('debounces Hugging Face searches while typing', async () => {
    const user = userEvent.setup()
    render(<MemoryRouter><ExplorePage /></MemoryRouter>)
    await screen.findByText('No models found')

    await user.type(screen.getByRole('textbox', { name: 'Search models' }), 'Qwen coder')

    await waitFor(() => {
      expect(fetchMock).toHaveBeenLastCalledWith(
        expect.stringContaining('/api/v1/catalog/models?q=Qwen+coder'),
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      )
    }, { timeout: 1500 })
  })

  it('opens deployment with the selected catalog model prefilled', async () => {
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      const body = path.includes('/api/v1/settings')
        ? { default_runtime: 'vllm', default_context_length: 24576 }
        : path.includes('/api/v1/nodes')
        ? { items: [{ id: 'local', name: 'This device', local: true, online: true, docker_ready: true, selectable: true }] }
        : { items: [] }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter initialEntries={['/models?model=org/chosen-model&runtime=sglang']}><ModelsPage /></MemoryRouter>)

    expect(await screen.findByRole('dialog', { name: 'Create deployment' })).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'Model repository or GGUF artifact' })).toHaveValue('org/chosen-model')
    expect(screen.getByRole('textbox', { name: 'Display name' })).toHaveValue('chosen-model')
    await waitFor(() => {
      expect(screen.getByRole('combobox', { name: 'Runtime' })).toHaveValue('sglang')
      expect(screen.getByRole('spinbutton', { name: 'Context length' })).toHaveValue(24576)
    })
    expect(fetchMock.mock.calls.some(([input]) => String(input).includes('/api/v1/settings'))).toBe(true)
  })

  it('applies the selected community GGUF variant to the deployment form and request', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (init?.method === 'POST' && path === '/api/v1/deployments') {
        return new Response(JSON.stringify({
          id: 'qwen-q4', alias: 'Qwen3.8-27B', runtime: 'llama.cpp', kind: 'managed',
          model: {
            repository: 'RadixArk/Qwen3.8-27B', quantization: 'Q4_K_M',
            artifact: 'artifacts/qwen3.8-q4_k_m.gguf',
          },
          status: 'starting',
          settings: {
            context_length: 24576, parallel_slots: 1, gpu_layers: 99,
            quantization: 'Q4_K_M', artifact: 'artifacts/qwen3.8-q4_k_m.gguf',
          },
        }), { status: 201, headers: { 'Content-Type': 'application/json' } })
      }
      const body = path.includes('/api/v1/settings')
        ? { default_runtime: 'sglang', default_context_length: 24576 }
        : path.includes('/api/v1/nodes')
        ? { items: [{ id: 'local', name: 'This device', local: true, online: true, docker_ready: true, selectable: true }] }
        : { items: [] }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter initialEntries={[
      '/models?model=RadixArk%2FQwen3.8-27B&runtime=llama.cpp&quantization=Q4_K_M&artifact=artifacts%2Fqwen3.8-q4_k_m.gguf',
    ]}><ModelsPage /></MemoryRouter>)

    const dialog = await screen.findByRole('dialog', { name: 'Create deployment' })
    expect(within(dialog).getByRole('textbox', { name: 'Model repository or GGUF artifact' })).toHaveValue('RadixArk/Qwen3.8-27B')
    expect(within(dialog).getByRole('textbox', { name: 'Quantization (optional)' })).toHaveValue('Q4_K_M')
    expect(within(dialog).getByRole('textbox', { name: /^GGUF artifact/ })).toHaveValue('artifacts/qwen3.8-q4_k_m.gguf')
    await waitFor(() => {
      expect(within(dialog).getByRole('combobox', { name: 'Runtime' })).toHaveValue('llama.cpp')
      expect(within(dialog).getByRole('spinbutton', { name: 'Context length' })).toHaveValue(24576)
    })

    await user.click(within(dialog).getByRole('button', { name: 'Save deployment' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/deployments',
      expect.objectContaining({ method: 'POST' }),
    ))
    const createRequest = fetchMock.mock.calls.find(([input, init]) => (
      String(input) === '/api/v1/deployments' && init?.method === 'POST'
    ))
    expect(JSON.parse(String(createRequest?.[1]?.body))).toEqual(expect.objectContaining({
      model: 'RadixArk/Qwen3.8-27B',
      runtime: 'llama.cpp',
      quantization: 'Q4_K_M',
      settings: expect.objectContaining({
        quantization: 'Q4_K_M',
        artifact: 'artifacts/qwen3.8-q4_k_m.gguf',
      }),
    }))
  })

  it('turns an aggregate-fit catalog deployment into a valid sharded layout', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (init?.method === 'POST' && path === '/api/v1/deployments') {
        return new Response(JSON.stringify({
          id: 'glm-sharded', alias: 'GLM-5.3-Flash', runtime: 'vllm', kind: 'managed',
          model: { repository: 'zai-org/GLM-5.3-Flash' }, status: 'starting', settings: {},
          node_ids: ['local', 'node-2', 'node-3', 'node-4'],
          selected_nodes: [
            { id: 'local', name: 'Controller' }, { id: 'node-2', name: 'Worker Two' },
            { id: 'node-3', name: 'Worker Three' }, { id: 'node-4', name: 'Worker Four' },
          ],
        }), { status: 201, headers: { 'Content-Type': 'application/json' } })
      }
      const body = path.includes('/api/v1/settings') ? { default_runtime: 'vllm', default_context_length: 8192 }
        : path.includes('/api/v1/nodes') ? { items: [
          { id: 'local', name: 'Controller', local: true, online: true, docker_ready: true, selectable: true },
          { id: 'node-2', name: 'Worker Two', online: true, docker_ready: true, selectable: true },
          { id: 'node-3', name: 'Worker Three', online: true, docker_ready: true, selectable: true },
          { id: 'node-4', name: 'Worker Four', online: true, docker_ready: true, selectable: true },
        ] } : path.includes('/api/v1/onboarding') ? { role: 'controller', node: { id: 'controller', name: 'Controller' } }
          : path.includes('/api/v1/model-cache') ? { nodes: [] }
            : { items: [] }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter initialEntries={['/models?model=zai-org/GLM-5.3-Flash&layout=sharded']}><ModelsPage /></MemoryRouter>)

    const dialog = await screen.findByRole('dialog', { name: 'Create deployment' })
    await waitFor(() => expect(within(dialog).getAllByRole('combobox')).toHaveLength(2))
    const deploymentLayout = within(dialog).getAllByRole('combobox')[1]
    await waitFor(() => expect(deploymentLayout).toHaveValue('sharded'))
    for (const node of ['Controller', 'Worker Two', 'Worker Three', 'Worker Four']) {
      expect(within(dialog).getByRole('checkbox', { name: new RegExp(node) })).toBeChecked()
    }
    expect(within(dialog).getByRole('checkbox', { name: /Controller/ })).toBeEnabled()
    const tensorParallel = within(dialog).getByRole('spinbutton', { name: /^Tensor parallel size/ })
    expect(tensorParallel).toHaveValue(4)
    expect(tensorParallel).toHaveAttribute('readonly')

    await user.selectOptions(deploymentLayout, 'replicated')
    expect(tensorParallel).toHaveValue(1)
    expect(tensorParallel).not.toHaveAttribute('readonly')
    await user.selectOptions(deploymentLayout, 'sharded')
    expect(tensorParallel).toHaveValue(4)

    await user.click(within(dialog).getByRole('button', { name: 'Save deployment' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/deployments',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          model: 'zai-org/GLM-5.3-Flash',
          alias: 'GLM-5.3-Flash',
          runtime: 'vllm',
          kind: 'managed',
          settings: {
            context_length: 8192,
            tensor_parallel_size: 4,
            image: 'nvcr.io/nvidia/vllm:26.03.post1-py3',
            extra_args: [],
          },
          node_ids: ['local', 'node-2', 'node-3', 'node-4'],
          deployment_mode: 'sharded',
        }),
      }),
    ))
  })
})

describe('model deployments', () => {
  it('treats an exited clustered deployment as stopped rather than active launch progress', async () => {
    const timeout = vi.spyOn(window, 'setTimeout')
    let deploymentListCalls = 0
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      const body = path.includes('/api/v1/deployments') ? { items: [{
        id: 'dep-exited', alias: 'Stopped cluster', runtime: 'vllm', kind: 'managed',
        model: { repository: 'org/model' }, status: 'stopped', desired_state: 'stopped',
        launch_phase: 'exited', launch_message: 'Container exited',
        settings: {}, node_ids: ['local'],
      }] } : path.includes('/api/v1/nodes') ? { items: [{
        id: 'local', name: 'This device', local: true, online: true, docker_ready: true, selectable: true,
      }] } : path.includes('/api/v1/model-cache') ? { nodes: [] } : { items: [] }
      if (path.includes('/api/v1/deployments')) deploymentListCalls += 1
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)

    const row = (await screen.findByRole('link', { name: 'Stopped cluster' })).closest('[role="row"]') as HTMLElement
    expect(within(row).getByText('stopped')).toBeInTheDocument()
    expect(within(row).queryByText('Exited')).not.toBeInTheDocument()
    expect(within(row).queryByText('Container exited')).not.toBeInTheDocument()
    expect(within(row).getByRole('button', { name: 'Start' })).toBeEnabled()
    expect(deploymentListCalls).toBe(1)
    expect(timeout).not.toHaveBeenCalledWith(expect.any(Function), 2000)
  })

  it('offers Start when stopped intent outlives a still-running container', async () => {
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      const body = path.includes('/api/v1/deployments') ? { items: [{
        id: 'dep-sticky-stop', alias: 'Stopped intent', runtime: 'vllm', kind: 'managed',
        model: { repository: 'org/model' }, status: 'running', desired_state: 'stopped',
        settings: {}, node_ids: ['local'],
      }] } : path.includes('/api/v1/nodes') ? { items: [{
        id: 'local', name: 'This device', local: true, online: true, docker_ready: true, selectable: true,
      }] } : path.includes('/api/v1/model-cache') ? { nodes: [] } : { items: [] }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)

    expect(await screen.findByRole('button', { name: 'Start' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Stop' })).not.toBeInTheDocument()
  })

  it('deletes a saved recipe without removing deployments or cached weights', async () => {
    const user = userEvent.setup()
    localStorage.setItem('sparkdeck:pinned-recipes', JSON.stringify(['recipe-1']))
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.endsWith('/api/v1/recipes/recipe-1') && init?.method === 'DELETE') {
        return new Response(null, { status: 204 })
      }
      const body = path.includes('/api/v1/recipes') ? { items: [{
        id: 'recipe-1', name: 'Saved cluster', model: 'org/model', engine: 'vllm',
        deployment_mode: 'single', required_node_count: 1, tensor_parallel_size: 1,
        pipeline_parallel_size: 1, node_ids: ['local'], extra_args_count: 0,
      }] } : path.includes('/api/v1/deployments') ? { items: [{
        id: 'dep-1', alias: 'Running model', runtime: 'vllm', kind: 'managed',
        model: { repository: 'org/model' }, status: 'running', settings: {}, node_ids: ['local'],
      }] } : path.includes('/api/v1/nodes') ? { items: [
        { id: 'local', name: 'This device', local: true, online: true, docker_ready: true, selectable: true },
      ] } : path.includes('/api/v1/model-cache') ? { nodes: [] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'org 1' }))
    await user.click(screen.getByRole('button', { name: 'Delete recipe Saved cluster' }))
    const confirmation = await screen.findByRole('dialog', { name: 'Delete recipe Saved cluster?' })
    expect(confirmation).toHaveTextContent('Existing deployments and cached model weights will not be removed.')
    await user.click(within(confirmation).getByRole('button', { name: 'Delete recipe' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/recipes/recipe-1', expect.objectContaining({ method: 'DELETE' }),
    ))
    expect(screen.queryByText('Saved cluster')).not.toBeInTheDocument()
    const runningRow = screen.getByRole('link', { name: 'Running model' }).closest('[role="row"]') as HTMLElement
    expect(within(runningRow).getByRole('link', { name: 'Running model' })).toHaveAttribute('href', '/models/dep-1')
    expect(within(runningRow).getByRole('button', { name: 'Stop' })).toBeEnabled()
    expect(screen.getByRole('status')).toHaveTextContent('Deleted recipe Saved cluster. Existing deployments and cached model weights were left unchanged.')
    expect(localStorage.getItem('sparkdeck:pinned-recipes')).toBe('[]')
  })

  it('shows exact replicated disk usage and deploys legacy saved configurations', async () => {
    const user = userEvent.setup()
    const gib = 1024 ** 3
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.endsWith('/api/v1/storage/transfers/preflight')) return new Response(JSON.stringify({
        enabled: false, model_id: 'org/model', revision: 'main', source: null,
        staging_reserve_bytes: 64 * 1024 ** 2,
        targets: [
          { node_id: 'local', node_name: 'Spark One', has_required_weights: true, eligible: false },
          { node_id: 'node-2', node_name: 'Spark Two', has_required_weights: true, eligible: false },
          { node_id: 'node-3', node_name: 'Spark Three', has_required_weights: false, eligible: false, reason: 'Virtual NAS is disabled' },
        ],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (init?.method === 'POST' && path.includes('/api/v1/recipes/recipe-1/deploy')) return new Response(JSON.stringify({
        id: 'dep-new', alias: 'Saved cluster', runtime: 'vllm', kind: 'managed',
        model: { repository: 'org/model' }, status: 'starting', settings: {}, node_ids: ['local', 'node-2'],
      }), { status: 201, headers: { 'Content-Type': 'application/json' } })
      const body = path.includes('/api/v1/model-cache') ? { nodes: [
        { id: 'local', name: 'Spark One', online: true, total_size: 100 * gib, models: [{ model_id: 'org/model', size_bytes: 2 * gib, revisions: ['main'] }] },
        { id: 'node-2', name: 'Spark Two', online: true, total_size: 100 * gib, models: [{ model_id: 'org/model', size_bytes: 2 * gib, revisions: ['main'] }] },
        { id: 'node-3', name: 'Spark Three', online: true, total_size: 100 * gib, models: [] },
      ] } : path.includes('/api/v1/recipes') ? { items: [{
        id: 'recipe-1', name: 'Saved cluster', model: 'org/model', engine: 'vllm',
        deployment_mode: 'sharded', required_node_count: 2, tensor_parallel_size: 2,
        pipeline_parallel_size: 1, node_ids: ['local', 'node-2'], extra_args_count: 2,
      }] } : path.includes('/api/v1/deployments') ? { items: [{
        id: 'dep-1', alias: 'Running model', runtime: 'vllm', kind: 'managed',
        model: { repository: 'org/model' }, status: 'running', settings: {}, node_ids: ['local', 'node-2'],
      }] } : path.includes('/api/v1/nodes') ? { items: [
        { id: 'local', name: 'Spark One', local: true, online: true, docker_ready: true, selectable: true },
        { id: 'node-2', name: 'Spark Two', online: true, docker_ready: true, selectable: true },
        { id: 'node-3', name: 'Spark Three', online: true, docker_ready: true, selectable: true },
      ] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)

    expect(await screen.findByText('2.0 GB each · 4.0 GB total on 2 nodes')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Recipes' })).toBeInTheDocument()
    await user.click(await screen.findByRole('button', { name: 'org 1' }))
    expect(screen.getByText('Spark One, Spark Two')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Choose nodes & deploy' }))
    const dialog = await screen.findByRole('dialog', { name: 'Deploy Saved cluster' })
    expect(dialog).toHaveTextContent('TP2 requires exactly 2 nodes')
    expect(within(dialog).getByRole('checkbox', { name: /Spark One/ })).toBeChecked()
    const secondNode = within(dialog).getByRole('checkbox', { name: /Spark Two/ })
    expect(secondNode).toBeChecked()
    expect(within(dialog).getByRole('checkbox', { name: /Spark Three/ })).toBeDisabled()
    expect(dialog).toHaveTextContent('Virtual NAS is disabled')

    await user.click(secondNode)
    expect(within(dialog).getByRole('button', { name: 'Prepare selected nodes' })).toBeDisabled()
    expect(dialog).toHaveTextContent('Select exactly 2 nodes to continue')
    await user.click(secondNode)
    await user.click(within(dialog).getByRole('button', { name: 'Deploy on 2 nodes' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/recipes/recipe-1/deploy',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ node_ids: ['local', 'node-2'] }) }),
    ))
    expect(await screen.findByText('Started deployment Saved cluster on This device, Spark Two.')).toBeInTheDocument()
  })

  it('shows recipe nodes while the model-cache inventory is still loading', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/model-cache')) return new Promise<Response>(() => undefined)
      if (path.endsWith('/api/v1/storage/transfers/preflight')) return new Response(JSON.stringify({
        enabled: true,
        model_id: 'org/model',
        revision: 'main',
        source: { node_id: 'local', node_name: 'Spark One', size_bytes: 20 },
        staging_reserve_bytes: 64 * 1024 ** 2,
        targets: [
          { node_id: 'local', node_name: 'Spark One', has_required_weights: true, eligible: false },
        ],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      const body = path.includes('/api/v1/recipes') ? { items: [{
        id: 'recipe-1', name: 'Saved cluster', model: 'org/model', engine: 'vllm',
        deployment_mode: 'single', required_node_count: 1, tensor_parallel_size: 1,
        pipeline_parallel_size: 1, node_ids: ['local'], extra_args_count: 0,
      }] } : path.includes('/api/v1/deployments') ? { items: [] }
        : path.includes('/api/v1/nodes') ? { items: [
          { id: 'local', name: 'Spark One', local: true, online: true, docker_ready: true, selectable: true },
        ] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'org 1' }))
    await user.click(screen.getByRole('button', { name: 'Choose nodes & deploy' }))

    const dialog = await screen.findByRole('dialog', { name: 'Deploy Saved cluster' })
    expect(within(dialog).queryByText('Loading available nodes…')).not.toBeInTheDocument()
    expect(within(dialog).getByRole('radio', { name: /Spark One/ })).toBeChecked()
    expect(await within(dialog).findByRole('button', { name: 'Deploy on 1 node' })).toBeEnabled()
  })

  it('confirms one selected-set preparation and rejects insufficient targets', async () => {
    const user = userEvent.setup()
    const gib = 1024 ** 3
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.endsWith('/api/v1/storage/transfers/preflight')) return new Response(JSON.stringify({
        enabled: true,
        model_id: 'org/model',
        revision: 'main',
        source: { node_id: 'local', node_name: 'Source', size_bytes: 2 * gib },
        staging_reserve_bytes: 64 * 1024 ** 2,
        targets: [
          { node_id: 'local', node_name: 'Source', eligible: false, has_required_weights: true, reason: 'Required model weights are already available', free_bytes: 20 * gib, required_free_bytes: 4 * gib + 64 * 1024 ** 2 },
          { node_id: 'enough', node_name: 'Enough', eligible: true, has_required_weights: false, free_bytes: 10 * gib, required_free_bytes: 4 * gib + 64 * 1024 ** 2 },
          { node_id: 'short', node_name: 'Short', eligible: false, has_required_weights: false, reason: 'Not enough free cache space', free_bytes: 4 * gib, required_free_bytes: 4 * gib + 64 * 1024 ** 2 },
        ],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path.endsWith('/api/v1/recipes/recipe-transfer/prepare/preflight')) return new Response(JSON.stringify({
        enabled: true, model_id: 'org/model', revision: 'main', eligible: true, action: 'transfer',
        node_ids: ['enough'], source: { node_id: 'local', node_name: 'Source', size_bytes: 2 * gib },
        transfer_target_node_ids: ['enough'], targets: [], staging_reserve_bytes: 64 * 1024 ** 2,
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path.endsWith('/api/v1/recipes/recipe-transfer/prepare') && init?.method === 'POST') return new Response(JSON.stringify({
        workflow_id: 'workflow-1',
        job_ids: ['job-1'],
        jobs: [{
          id: 'job-1', model_id: 'org/model', source_node_id: 'local', source_node_name: 'Source',
          target_node_id: 'enough', target_node_name: 'Enough', status: 'queued',
          bytes_total: 2 * gib, bytes_transferred: 0, created_at: 1,
        }],
      }), { status: 202, headers: { 'Content-Type': 'application/json' } })
      if (path.endsWith('/api/v1/storage')) return new Response(JSON.stringify({
        enabled: true, nodes: [], instructions: [], jobs: [{
          id: 'job-1', model_id: 'org/model', source_node_id: 'local', source_node_name: 'Source',
          target_node_id: 'enough', target_node_name: 'Enough', status: 'running',
          bytes_total: 2 * gib, bytes_transferred: gib, created_at: 1,
        }],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      const body = path.includes('/api/v1/model-cache') ? { nodes: [
        { id: 'local', name: 'Source', online: true, free_size: 20 * gib, models: [{ model_id: 'org/model', size_bytes: 2 * gib, revisions: ['main'] }] },
        { id: 'enough', name: 'Enough', online: true, free_size: 10 * gib, models: [] },
        { id: 'short', name: 'Short', online: true, free_size: 4 * gib, models: [] },
      ] } : path.includes('/api/v1/recipes') ? { items: [{
        id: 'recipe-transfer', name: 'Transfer recipe', model: 'org/model', engine: 'vllm',
        deployment_mode: 'single', required_node_count: 1, tensor_parallel_size: 1,
        pipeline_parallel_size: 1, node_ids: ['enough'], extra_args_count: 0,
      }] } : path.includes('/api/v1/deployments') ? { items: [] }
        : path.includes('/api/v1/nodes') ? { items: [
          { id: 'local', name: 'Source', local: true, online: true, docker_ready: true, selectable: true },
          { id: 'enough', name: 'Enough', online: true, docker_ready: true, selectable: true },
          { id: 'short', name: 'Short', online: true, docker_ready: true, selectable: true },
        ] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'org 1' }))
    await user.click(await screen.findByRole('button', { name: 'Choose nodes & deploy' }))
    const dialog = await screen.findByRole('dialog', { name: 'Deploy Transfer recipe' })
    const prepare = await within(dialog).findByRole('button', { name: 'Prepare selected nodes' })

    expect(dialog).toHaveTextContent('Not enough free cache space')
    expect(within(dialog).getByRole('radio', { name: /Short/ })).toBeDisabled()
    await user.click(prepare)
    const confirmation = await screen.findByRole('dialog', { name: 'Prepare model weights?' })
    expect(confirmation).toHaveTextContent('Transfer org/model from Source via Virtual NAS to Enough?')
    await user.click(within(confirmation).getByRole('button', { name: 'Start preparation' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/recipes/recipe-transfer/prepare',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ node_ids: ['enough'] }),
      }),
    ))
    expect(await within(dialog).findByText('Virtual NAS transfer queued to Enough.')).toBeInTheDocument()
  })

  it('queues one Hugging Face seed followed by Virtual NAS fan-out for selected nodes', async () => {
    const user = userEvent.setup()
    const gib = 1024 ** 3
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      const targets = [
        { node_id: 'node-a', node_name: 'Node A', eligible: false, has_required_weights: false, has_model_cache: false, download_eligible: true, transfer_after_download_eligible: true, free_bytes: 20 * gib, download_required_free_bytes: 8 * gib, transfer_after_download_required_free_bytes: 8 * gib },
        { node_id: 'node-b', node_name: 'Node B', eligible: false, has_required_weights: false, has_model_cache: false, download_eligible: true, transfer_after_download_eligible: true, free_bytes: 20 * gib, download_required_free_bytes: 8 * gib, transfer_after_download_required_free_bytes: 8 * gib },
      ]
      if (path.endsWith('/api/v1/storage/transfers/preflight')) return new Response(JSON.stringify({
        enabled: true, model_id: 'org/new-model', revision: 'main', source: null,
        download: { size_bytes: 4 * gib, required_free_bytes: 8 * gib },
        targets, staging_reserve_bytes: 64 * 1024 ** 2,
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path.endsWith('/api/v1/recipes/recipe-seed/prepare/preflight')) return new Response(JSON.stringify({
        enabled: true, model_id: 'org/new-model', revision: 'main', eligible: true, action: 'download',
        node_ids: ['node-a', 'node-b'], download_node_id: 'node-a', transfer_target_node_ids: ['node-b'],
        source: null, download: { size_bytes: 4 * gib, required_free_bytes: 8 * gib }, targets,
        staging_reserve_bytes: 64 * 1024 ** 2,
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path.endsWith('/api/v1/recipes/recipe-seed/prepare') && init?.method === 'POST') return new Response(JSON.stringify({
        workflow_id: 'workflow-seed', job_ids: ['download-1', 'transfer-1'], jobs: [
          { id: 'download-1', kind: 'download', model_id: 'org/new-model', source_node_id: 'huggingface', source_node_name: 'Hugging Face', target_node_id: 'node-a', target_node_name: 'Node A', status: 'queued', bytes_total: 4 * gib, bytes_transferred: 0, created_at: 1 },
          { id: 'transfer-1', kind: 'transfer', model_id: 'org/new-model', source_node_id: 'node-a', source_node_name: 'Node A', target_node_id: 'node-b', target_node_name: 'Node B', status: 'queued', bytes_total: 4 * gib, bytes_transferred: 0, created_at: 1 },
        ],
      }), { status: 202, headers: { 'Content-Type': 'application/json' } })
      if (path.endsWith('/api/v1/storage')) return new Response(JSON.stringify({
        enabled: true, nodes: [], instructions: [], jobs: [
          { id: 'download-1', status: 'running' }, { id: 'transfer-1', status: 'queued' },
        ],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      const body = path.includes('/api/v1/model-cache') ? { nodes: [
        { id: 'node-a', name: 'Node A', online: true, cache_free_size: 20 * gib, models: [] },
        { id: 'node-b', name: 'Node B', online: true, cache_free_size: 20 * gib, models: [] },
      ] } : path.includes('/api/v1/recipes') ? { items: [{
        id: 'recipe-seed', name: 'Seed recipe', model: 'org/new-model', engine: 'vllm',
        deployment_mode: 'replicated', required_node_count: 2, tensor_parallel_size: 1,
        pipeline_parallel_size: 1, node_ids: ['node-a', 'node-b'], extra_args_count: 0,
      }] } : path.includes('/api/v1/deployments') ? { items: [] }
        : path.includes('/api/v1/nodes') ? { items: [
          { id: 'node-a', name: 'Node A', online: true, docker_ready: true, selectable: true },
          { id: 'node-b', name: 'Node B', online: true, docker_ready: true, selectable: true },
        ] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'org 1' }))
    await user.click(await screen.findByRole('button', { name: 'Choose nodes & deploy' }))
    const dialog = await screen.findByRole('dialog', { name: 'Deploy Seed recipe' })
    expect(within(dialog).getByRole('checkbox', { name: /Node A/ })).toBeChecked()
    expect(within(dialog).getByRole('checkbox', { name: /Node B/ })).toBeChecked()
    await user.click(await within(dialog).findByRole('button', { name: 'Prepare selected nodes' }))
    const confirmation = await screen.findByRole('dialog', { name: 'Prepare model weights?' })
    expect(confirmation).toHaveTextContent(
      'Download org/new-model revision main (4.0 GB) from Hugging Face onto Node A, then transfer it via Virtual NAS to Node B?',
    )
    await user.click(within(confirmation).getByRole('button', { name: 'Start preparation' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/recipes/recipe-seed/prepare',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ node_ids: ['node-a', 'node-b'] }) }),
    ))
    expect(await within(dialog).findByText('Hugging Face seed download queued on Node A. Virtual NAS fan-out will follow automatically.')).toBeInTheDocument()
  })

  it('resumes an active recipe preparation and refreshes preflight after failure', async () => {
    const user = userEvent.setup()
    let preflightCalls = 0
    let cacheCalls = 0
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path.endsWith('/api/v1/storage/transfers/preflight')) {
        preflightCalls++
        return new Response(JSON.stringify({
          enabled: true, model_id: 'org/model', revision: 'main', source: null,
          download: { size_bytes: 20, required_free_bytes: 100 }, staging_reserve_bytes: 64,
          targets: [{
            node_id: 'worker', node_name: 'Worker', eligible: false,
            active_job_id: 'job-failed', active_job_status: 'running', active_job_kind: 'download',
            has_required_weights: false, free_bytes: 1000,
          }],
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path.endsWith('/api/v1/storage')) return new Response(JSON.stringify({
        enabled: true, nodes: [], instructions: [], jobs: [{
          id: 'job-failed', model_id: 'org/model', target_node_id: 'worker',
          status: 'failed', error: 'seed download failed', bytes_total: 20,
          bytes_transferred: 0, created_at: 1,
        }],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path.includes('/api/v1/model-cache')) {
        cacheCalls++
        return new Response(JSON.stringify({ nodes: [
          { id: 'worker', name: 'Worker', online: true, cache_free_size: 1000, models: [] },
        ] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      const body = path.includes('/api/v1/recipes') ? { items: [{
        id: 'recipe-resume', name: 'Resume recipe', model: 'org/model', engine: 'vllm',
        deployment_mode: 'single', required_node_count: 1, tensor_parallel_size: 1,
        pipeline_parallel_size: 1, node_ids: ['worker'], extra_args_count: 0,
      }] } : path.includes('/api/v1/deployments') ? { items: [] }
        : path.includes('/api/v1/nodes') ? { items: [
          { id: 'worker', name: 'Worker', online: true, docker_ready: true, selectable: true },
        ] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'org 1' }))
    await user.click(await screen.findByRole('button', { name: 'Choose nodes & deploy' }))
    const dialog = await screen.findByRole('dialog', { name: 'Deploy Resume recipe' })

    expect(await within(dialog).findByRole('alert')).toHaveTextContent('seed download failed')
    await waitFor(() => expect(preflightCalls).toBeGreaterThanOrEqual(2))
    expect(cacheCalls).toBeGreaterThanOrEqual(2)
  })

  it('requires the main cache ref for an unpinned saved configuration', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path.endsWith('/api/v1/storage/transfers/preflight')) return new Response(JSON.stringify({
        enabled: false, model_id: 'org/model', revision: 'main', source: null,
        staging_reserve_bytes: 64 * 1024 ** 2,
        targets: [
          { node_id: 'local', node_name: 'Snapshot only', has_required_weights: false, eligible: false, reason: 'Virtual NAS is disabled' },
          { node_id: 'node-2', node_name: 'Default branch', has_required_weights: true, eligible: false },
        ],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      const body = path.includes('/api/v1/model-cache') ? { nodes: [
        { id: 'local', name: 'Snapshot only', online: true, models: [{ model_id: 'org/model', size_bytes: 20, revisions: ['snapshot-a'] }] },
        { id: 'node-2', name: 'Default branch', online: true, models: [{ model_id: 'org/model', size_bytes: 20, revisions: ['main', 'snapshot-b'] }] },
      ] } : path.includes('/api/v1/recipes') ? { items: [{
        id: 'recipe-main', name: 'Saved main', model: 'org/model', engine: 'vllm',
        deployment_mode: 'single', required_node_count: 1, tensor_parallel_size: 1,
        pipeline_parallel_size: 1, node_ids: ['local'], extra_args_count: 0,
      }] } : path.includes('/api/v1/deployments') ? { items: [] }
        : path.includes('/api/v1/nodes') ? { items: [
          { id: 'local', name: 'Snapshot only', local: true, online: true, docker_ready: true, selectable: true },
          { id: 'node-2', name: 'Default branch', online: true, docker_ready: true, selectable: true },
        ] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'org 1' }))
    await user.click(await screen.findByRole('button', { name: 'Choose nodes & deploy' }))
    const dialog = await screen.findByRole('dialog', { name: 'Deploy Saved main' })

    expect(within(dialog).getByRole('radio', { name: /Snapshot only/ })).toBeDisabled()
    expect(within(dialog).getByRole('radio', { name: /Snapshot only/ })).toBeChecked()
    const defaultBranch = within(dialog).getByRole('radio', { name: /Default branch/ })
    expect(defaultBranch).not.toBeChecked()
    await user.click(defaultBranch)
    expect(defaultBranch).toBeChecked()
  })

  it('allows a saved local model path on the controller without cache inventory', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/model-cache')) {
        return new Response(JSON.stringify({ detail: 'cache unavailable' }), {
          status: 503, headers: { 'Content-Type': 'application/json' },
        })
      }
      const body = path.includes('/api/v1/recipes') ? { items: [{
        id: 'recipe-local', name: 'Local weights', model: '/models/local', engine: 'vllm',
        deployment_mode: 'single', required_node_count: 1, tensor_parallel_size: 1,
        pipeline_parallel_size: 1, node_ids: ['local'], extra_args_count: 0,
      }] } : path.includes('/api/v1/deployments') ? { items: [] }
        : path.includes('/api/v1/nodes') ? { items: [
          { id: 'local', name: 'Controller', local: true, online: true, docker_ready: true, selectable: true },
          { id: 'node-2', name: 'Worker', online: true, docker_ready: true, selectable: true },
        ] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'Other 1' }))
    await user.click(await screen.findByRole('button', { name: 'Choose nodes & deploy' }))
    const dialog = await screen.findByRole('dialog', { name: 'Deploy Local weights' })

    expect(within(dialog).getByRole('radio', { name: /Controller/ })).toBeChecked()
    expect(within(dialog).getByRole('radio', { name: /Worker/ })).toBeDisabled()
    expect(within(dialog).getByRole('button', { name: 'Deploy on 1 node' })).toBeEnabled()
    expect(dialog).not.toHaveTextContent('cache unavailable')
  })

  it('keeps a pending saved deployment open so late failures remain visible', async () => {
    const user = userEvent.setup()
    let resolveDeploy: (response: Response) => void = () => undefined
    const deployResponse = new Promise<Response>((resolve) => { resolveDeploy = resolve })
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.endsWith('/api/v1/storage/transfers/preflight')) return new Response(JSON.stringify({
        enabled: false, model_id: 'org/model', revision: 'main', source: null,
        staging_reserve_bytes: 64 * 1024 ** 2,
        targets: [{ node_id: 'local', node_name: 'Spark One', has_required_weights: true, eligible: false }],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (init?.method === 'POST' && path.includes('/api/v1/recipes/recipe-1/deploy')) return deployResponse
      const body = path.includes('/api/v1/model-cache') ? { nodes: [
        { id: 'local', name: 'Spark One', online: true, models: [{ model_id: 'org/model', size_bytes: 20, revisions: ['main'] }] },
      ] } : path.includes('/api/v1/recipes') ? { items: [{
        id: 'recipe-1', name: 'Saved local', model: 'org/model', engine: 'vllm',
        deployment_mode: 'single', required_node_count: 1, tensor_parallel_size: 1,
        pipeline_parallel_size: 1, node_ids: ['local'], extra_args_count: 0,
      }] } : path.includes('/api/v1/deployments') ? { items: [] }
        : path.includes('/api/v1/nodes') ? { items: [
          { id: 'local', name: 'Spark One', local: true, online: true, docker_ready: true, selectable: true },
        ] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'org 1' }))
    await user.click(await screen.findByRole('button', { name: 'Choose nodes & deploy' }))
    const dialog = await screen.findByRole('dialog', { name: 'Deploy Saved local' })
    await user.click(within(dialog).getByRole('button', { name: 'Deploy on 1 node' }))

    expect(within(dialog).getByRole('button', { name: 'Close dialog' })).toBeDisabled()
    expect(within(dialog).getByRole('button', { name: 'Cancel' })).toBeDisabled()
    fireEvent.mouseDown(dialog.parentElement as HTMLElement)
    expect(screen.getByRole('dialog', { name: 'Deploy Saved local' })).toBeInTheDocument()

    resolveDeploy(new Response(JSON.stringify({ detail: 'launch failed' }), {
      status: 500, headers: { 'Content-Type': 'application/json' },
    }))
    expect(await within(dialog).findByRole('alert')).toHaveTextContent('launch failed')
    expect(within(dialog).getByRole('button', { name: 'Close dialog' })).toBeEnabled()
    expect(within(dialog).getByRole('button', { name: 'Cancel' })).toBeEnabled()
  })

  it('moves an accepted recipe launch into Deployments and refreshes its live phase', async () => {
    const user = userEvent.setup()
    let deploymentListCalls = 0
    let resolveLiveRefresh: (response: Response) => void = () => undefined
    const liveRefresh = new Promise<Response>((resolve) => { resolveLiveRefresh = resolve })
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.endsWith('/api/v1/storage/transfers/preflight')) return new Response(JSON.stringify({
        enabled: false, model_id: 'org/model', revision: 'main', source: null,
        staging_reserve_bytes: 64 * 1024 ** 2,
        targets: [{ node_id: 'local', node_name: 'Spark One', has_required_weights: true, eligible: false }],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (init?.method === 'POST' && path.includes('/api/v1/recipes/recipe-live/deploy')) {
        return new Response(JSON.stringify({
          id: 'dep-live', alias: 'Live recipe', runtime: 'sglang', kind: 'managed',
          model: { repository: 'org/model' }, status: 'launching', settings: {},
          node_ids: ['local'], launch_phase: 'queued', launch_message: 'Launch accepted',
        }), { status: 202, headers: { 'Content-Type': 'application/json' } })
      }
      if (path.includes('/api/v1/deployments')) {
        deploymentListCalls += 1
        if (deploymentListCalls === 2) return liveRefresh
        const deployments = deploymentListCalls === 1 ? [] : [{
          id: 'dep-live', alias: 'Live recipe', runtime: 'sglang', kind: 'managed',
          model: { repository: 'org/model' }, status: 'running', settings: {},
          node_ids: ['local'], launch_phase: 'ready', launch_message: 'SGLang API ready',
        }]
        return new Response(JSON.stringify({ items: deployments }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      const body = path.includes('/api/v1/model-cache') ? { nodes: [
        { id: 'local', name: 'Spark One', online: true, models: [{ model_id: 'org/model', size_bytes: 20, revisions: ['main'] }] },
      ] } : path.includes('/api/v1/recipes') ? { items: [{
        id: 'recipe-live', name: 'Live recipe', model: 'org/model', engine: 'sglang',
        deployment_mode: 'single', required_node_count: 1, tensor_parallel_size: 1,
        pipeline_parallel_size: 1, node_ids: ['local'], extra_args_count: 0,
      }] } : path.includes('/api/v1/nodes') ? { items: [
        { id: 'local', name: 'Spark One', local: true, online: true, docker_ready: true, selectable: true },
      ] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'org 1' }))
    await user.click(await screen.findByRole('button', { name: 'Choose nodes & deploy' }))
    await user.click(within(await screen.findByRole('dialog', { name: 'Deploy Live recipe' })).getByRole('button', { name: 'Deploy on 1 node' }))

    expect(screen.queryByRole('dialog', { name: 'Deploy Live recipe' })).not.toBeInTheDocument()
    const deploymentRow = screen.getByRole('link', { name: 'Live recipe' }).closest('[role="row"]') as HTMLElement
    expect(within(deploymentRow).getByText('launching')).toBeInTheDocument()
    expect(within(deploymentRow).getByText('Queued')).toBeInTheDocument()
    expect(within(deploymentRow).getByText('Launch accepted')).toBeInTheDocument()
    expect(within(deploymentRow).getByRole('button', { name: 'Stop' })).toBeDisabled()
    expect(screen.getByRole('status')).toHaveTextContent('Started deployment Live recipe on This device.')

    await waitFor(() => expect(deploymentListCalls).toBe(2), { timeout: 3500 })
    expect(screen.queryByText('Loading deployments')).not.toBeInTheDocument()
    expect(screen.queryByText('Refreshing deployments…')).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Live recipe' })).toBeInTheDocument()

    resolveLiveRefresh(new Response(JSON.stringify({ items: [{
      id: 'dep-live', alias: 'Live recipe', runtime: 'sglang', kind: 'managed',
      model: { repository: 'org/model' }, status: 'error', settings: {},
      node_ids: ['local'], launch_phase: 'pulling_image', launch_message: 'Downloading Docker image',
    }] }), { status: 200, headers: { 'Content-Type': 'application/json' } }))

    expect(await within(deploymentRow).findByText('Downloading Docker image', {}, { timeout: 3500 })).toBeInTheDocument()
    expect(within(deploymentRow).getByText('Pulling Image')).toBeInTheDocument()
    expect(await within(deploymentRow).findByText('SGLang API ready', {}, { timeout: 3500 })).toBeInTheDocument()
    expect(within(deploymentRow).getByText('Ready')).toBeInTheDocument()
    expect(deploymentListCalls).toBeGreaterThanOrEqual(3)
  })

  it('keeps a live deployment row stable when a background refresh fails', async () => {
    let deploymentListCalls = 0
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/deployments')) {
        deploymentListCalls += 1
        if (deploymentListCalls === 2) {
          return new Response(JSON.stringify({ detail: 'temporary node timeout' }), {
            status: 504, headers: { 'Content-Type': 'application/json' },
          })
        }
        const active = deploymentListCalls === 1
        return new Response(JSON.stringify({ items: [{
          id: 'dep-background', alias: 'Background model', runtime: 'vllm', kind: 'managed',
          model: { repository: 'org/background' }, status: active ? 'starting' : 'running', settings: {},
          node_ids: ['local'], launch_phase: active ? 'loading' : 'ready',
          launch_message: active ? 'Loading weights' : 'vLLM API ready',
        }] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      const body = path.includes('/api/v1/model-cache')
        ? { nodes: [] }
        : path.includes('/api/v1/nodes') || path.includes('/api/v1/recipes')
          ? { items: [] }
          : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    const link = await screen.findByRole('link', { name: 'Background model' })
    const deploymentRow = link.closest('[role="row"]') as HTMLElement

    await waitFor(() => expect(deploymentListCalls).toBe(2), { timeout: 3500 })
    expect(screen.getByRole('link', { name: 'Background model' }).closest('[role="row"]')).toBe(deploymentRow)
    expect(screen.queryByText('temporary node timeout')).not.toBeInTheDocument()
    expect(screen.queryByText('Loading deployments')).not.toBeInTheDocument()
    expect(screen.queryByText('Refreshing deployments…')).not.toBeInTheDocument()

    expect(within(deploymentRow).getByText('Loading weights')).toBeInTheDocument()
  })

  it('keeps an accepted recipe row when an older deployment load finishes afterward', async () => {
    const user = userEvent.setup()
    let deploymentListCalls = 0
    let resolveInitialList: (response: Response) => void = () => undefined
    const initialList = new Promise<Response>((resolve) => { resolveInitialList = resolve })
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.endsWith('/api/v1/storage/transfers/preflight')) return new Response(JSON.stringify({
        enabled: false, model_id: 'org/model', revision: 'main', source: null,
        staging_reserve_bytes: 64 * 1024 ** 2,
        targets: [{ node_id: 'local', node_name: 'Spark One', has_required_weights: true, eligible: false }],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (init?.method === 'POST' && path.includes('/api/v1/recipes/recipe-stale/deploy')) {
        return new Response(JSON.stringify({
          id: 'dep-accepted', alias: 'Accepted recipe', runtime: 'vllm', kind: 'managed',
          model: { repository: 'org/model' }, status: 'starting', settings: {},
          node_ids: ['local'], launch_phase: 'queued', launch_message: 'Launch accepted',
        }), { status: 201, headers: { 'Content-Type': 'application/json' } })
      }
      if (path.includes('/api/v1/deployments')) {
        deploymentListCalls += 1
        if (deploymentListCalls === 1) return initialList
        return new Response(JSON.stringify({ items: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      const body = path.includes('/api/v1/model-cache') ? { nodes: [
        { id: 'local', name: 'Spark One', online: true, models: [{ model_id: 'org/model', size_bytes: 20, revisions: ['main'] }] },
      ] } : path.includes('/api/v1/recipes') ? { items: [{
        id: 'recipe-stale', name: 'Accepted recipe', model: 'org/model', engine: 'vllm',
        deployment_mode: 'single', required_node_count: 1, tensor_parallel_size: 1,
        pipeline_parallel_size: 1, node_ids: ['local'], extra_args_count: 0,
      }] } : path.includes('/api/v1/nodes') ? { items: [
        { id: 'local', name: 'Spark One', local: true, online: true, docker_ready: true, selectable: true },
      ] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'org 1' }))
    await user.click(await screen.findByRole('button', { name: 'Choose nodes & deploy' }))
    await user.click(within(await screen.findByRole('dialog', { name: 'Deploy Accepted recipe' })).getByRole('button', { name: 'Deploy on 1 node' }))

    expect(screen.getByRole('link', { name: 'Accepted recipe' })).toBeInTheDocument()
    resolveInitialList(new Response(JSON.stringify({ items: [] }), {
      status: 200, headers: { 'Content-Type': 'application/json' },
    }))
    await waitFor(() => expect(screen.queryByText('Loading deployments')).not.toBeInTheDocument())
    expect(screen.getByRole('link', { name: 'Accepted recipe' })).toBeInTheDocument()
    expect(screen.getByText('Launch accepted')).toBeInTheDocument()
  })

  it('does not restore an accepted recipe row removed before the first deployment load finishes', async () => {
    const user = userEvent.setup()
    let deploymentListCalls = 0
    let resolveInitialList: (response: Response) => void = () => undefined
    const initialList = new Promise<Response>((resolve) => { resolveInitialList = resolve })
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.endsWith('/api/v1/storage/transfers/preflight')) return new Response(JSON.stringify({
        enabled: false, model_id: 'org/model', revision: 'main', source: null,
        staging_reserve_bytes: 64 * 1024 ** 2,
        targets: [{ node_id: 'local', node_name: 'Spark One', has_required_weights: true, eligible: false }],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (init?.method === 'POST' && path.includes('/api/v1/recipes/recipe-remove/deploy')) {
        return new Response(JSON.stringify({
          id: 'dep-remove', alias: 'Remove accepted', runtime: 'vllm', kind: 'managed',
          model: { repository: 'org/model' }, status: 'starting', settings: {},
          node_ids: ['local'], launch_phase: 'queued', launch_message: 'Launch accepted',
        }), { status: 201, headers: { 'Content-Type': 'application/json' } })
      }
      if (init?.method === 'DELETE' && path.endsWith('/api/v1/deployments/dep-remove')) {
        return new Response(null, { status: 204 })
      }
      if (path.includes('/api/v1/deployments')) {
        deploymentListCalls += 1
        if (deploymentListCalls === 1) return initialList
        return new Response(JSON.stringify({ items: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      const body = path.includes('/api/v1/model-cache') ? { nodes: [
        { id: 'local', name: 'Spark One', online: true, models: [{ model_id: 'org/model', size_bytes: 20, revisions: ['main'] }] },
      ] } : path.includes('/api/v1/recipes') ? { items: [{
        id: 'recipe-remove', name: 'Remove accepted', model: 'org/model', engine: 'vllm',
        deployment_mode: 'single', required_node_count: 1, tensor_parallel_size: 1,
        pipeline_parallel_size: 1, node_ids: ['local'], extra_args_count: 0,
      }] } : path.includes('/api/v1/nodes') ? { items: [
        { id: 'local', name: 'Spark One', local: true, online: true, docker_ready: true, selectable: true },
      ] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'org 1' }))
    await user.click(await screen.findByRole('button', { name: 'Choose nodes & deploy' }))
    await user.click(within(await screen.findByRole('dialog', { name: 'Deploy Remove accepted' })).getByRole('button', { name: 'Deploy on 1 node' }))
    expect(screen.getByRole('link', { name: 'Remove accepted' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Remove Remove accepted' }))
    const confirmation = await screen.findByRole('dialog', { name: 'Remove Remove accepted?' })
    await user.click(within(confirmation).getByRole('button', { name: 'Remove deployment' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/deployments/dep-remove', expect.objectContaining({ method: 'DELETE' }),
    ))
    await waitFor(() => expect(screen.queryByRole('link', { name: 'Remove accepted' })).not.toBeInTheDocument())
    expect(screen.getByText('No deployments yet')).toBeInTheDocument()
    expect(screen.queryByText('Loading deployments')).not.toBeInTheDocument()

    resolveInitialList(new Response(JSON.stringify({ items: [] }), {
      status: 200, headers: { 'Content-Type': 'application/json' },
    }))
    await waitFor(() => expect(screen.queryByText('Loading deployments')).not.toBeInTheDocument())
    expect(screen.queryByRole('link', { name: 'Remove accepted' })).not.toBeInTheDocument()
    expect(screen.queryByText('Launch accepted')).not.toBeInTheDocument()
  })

  it('keeps an accepted recipe row when removal fails before the first deployment load finishes', async () => {
    const user = userEvent.setup()
    let resolveInitialList: (response: Response) => void = () => undefined
    const initialList = new Promise<Response>((resolve) => { resolveInitialList = resolve })
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.endsWith('/api/v1/storage/transfers/preflight')) return new Response(JSON.stringify({
        enabled: false, model_id: 'org/model', revision: 'main', source: null,
        staging_reserve_bytes: 64 * 1024 ** 2,
        targets: [{ node_id: 'local', node_name: 'Spark One', has_required_weights: true, eligible: false }],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (init?.method === 'POST' && path.includes('/api/v1/recipes/recipe-remove-failure/deploy')) {
        return new Response(JSON.stringify({
          id: 'dep-remove-failure', alias: 'Removal failure', runtime: 'vllm', kind: 'managed',
          model: { repository: 'org/model' }, status: 'starting', settings: {},
          node_ids: ['local'], launch_phase: 'queued', launch_message: 'Launch accepted',
        }), { status: 201, headers: { 'Content-Type': 'application/json' } })
      }
      if (init?.method === 'DELETE' && path.endsWith('/api/v1/deployments/dep-remove-failure')) {
        return new Response(JSON.stringify({ detail: 'runtime cleanup failed' }), {
          status: 500, headers: { 'Content-Type': 'application/json' },
        })
      }
      if (path.includes('/api/v1/deployments')) return initialList
      const body = path.includes('/api/v1/model-cache') ? { nodes: [
        { id: 'local', name: 'Spark One', online: true, models: [{ model_id: 'org/model', size_bytes: 20, revisions: ['main'] }] },
      ] } : path.includes('/api/v1/recipes') ? { items: [{
        id: 'recipe-remove-failure', name: 'Removal failure', model: 'org/model', engine: 'vllm',
        deployment_mode: 'single', required_node_count: 1, tensor_parallel_size: 1,
        pipeline_parallel_size: 1, node_ids: ['local'], extra_args_count: 0,
      }] } : path.includes('/api/v1/nodes') ? { items: [
        { id: 'local', name: 'Spark One', local: true, online: true, docker_ready: true, selectable: true },
      ] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'org 1' }))
    await user.click(await screen.findByRole('button', { name: 'Choose nodes & deploy' }))
    await user.click(within(await screen.findByRole('dialog', { name: 'Deploy Removal failure' })).getByRole('button', { name: 'Deploy on 1 node' }))
    expect(screen.getByRole('link', { name: 'Removal failure' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Remove Removal failure' }))
    const confirmation = await screen.findByRole('dialog', { name: 'Remove Removal failure?' })
    await user.click(within(confirmation).getByRole('button', { name: 'Remove deployment' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('runtime cleanup failed')
    expect(screen.getByRole('link', { name: 'Removal failure' })).toBeInTheDocument()

    resolveInitialList(new Response(JSON.stringify({ items: [] }), {
      status: 200, headers: { 'Content-Type': 'application/json' },
    }))
    await waitFor(() => expect(screen.queryByText('Loading deployments')).not.toBeInTheDocument())
    expect(screen.getByRole('link', { name: 'Removal failure' })).toBeInTheDocument()
    expect(screen.getByText('Launch accepted')).toBeInTheDocument()
  })

  it('deduplicates saved node IDs before filling an exact-count selector', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.endsWith('/api/v1/storage/transfers/preflight')) return new Response(JSON.stringify({
        enabled: false, model_id: 'org/model', revision: 'main', source: null,
        staging_reserve_bytes: 64 * 1024 ** 2,
        targets: [
          { node_id: 'local', node_name: 'Spark One', has_required_weights: true, eligible: false },
          { node_id: 'node-2', node_name: 'Spark Two', has_required_weights: true, eligible: false },
        ],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (init?.method === 'POST' && path.includes('/api/v1/recipes/recipe-duplicates/deploy')) {
        return new Response(JSON.stringify({
          id: 'dep-replicas', alias: 'Saved replicas', runtime: 'vllm', kind: 'managed',
          model: { repository: 'org/model' }, status: 'starting', settings: {},
          node_ids: ['local', 'node-2'],
        }), { status: 201, headers: { 'Content-Type': 'application/json' } })
      }
      const body = path.includes('/api/v1/model-cache') ? { nodes: [
        { id: 'local', name: 'Spark One', online: true, models: [{ model_id: 'org/model', size_bytes: 20, revisions: ['main'] }] },
        { id: 'node-2', name: 'Spark Two', online: true, models: [{ model_id: 'org/model', size_bytes: 20, revisions: ['main'] }] },
      ] } : path.includes('/api/v1/recipes') ? { items: [{
        id: 'recipe-duplicates', name: 'Saved replicas', model: 'org/model', engine: 'vllm',
        deployment_mode: 'replicated', required_node_count: 2, tensor_parallel_size: 1,
        pipeline_parallel_size: 1, node_ids: ['local', 'local'], extra_args_count: 0,
      }] } : path.includes('/api/v1/deployments') ? { items: [] }
        : path.includes('/api/v1/nodes') ? { items: [
          { id: 'local', name: 'Spark One', local: true, online: true, docker_ready: true, selectable: true },
          { id: 'node-2', name: 'Spark Two', online: true, docker_ready: true, selectable: true },
        ] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'org 1' }))
    await user.click(await screen.findByRole('button', { name: 'Choose nodes & deploy' }))
    const dialog = await screen.findByRole('dialog', { name: 'Deploy Saved replicas' })

    expect(within(dialog).getByRole('checkbox', { name: /Spark One/ })).toBeChecked()
    expect(within(dialog).getByRole('checkbox', { name: /Spark Two/ })).toBeChecked()
    expect(dialog).toHaveTextContent('This device, Spark Two')
    const deploy = within(dialog).getByRole('button', { name: 'Deploy on 2 nodes' })
    expect(deploy).toBeEnabled()
    await user.click(deploy)
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/recipes/recipe-duplicates/deploy',
      expect.objectContaining({
        method: 'POST', body: JSON.stringify({ node_ids: ['local', 'node-2'] }),
      }),
    ))
  })

  it('does not deploy nodes whose main refs resolve to different commits', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path.endsWith('/api/v1/storage/transfers/preflight')) return new Response(JSON.stringify({
        enabled: true, model_id: 'org/model', revision: 'main', resolved_revision: 'a'.repeat(40),
        source: { node_id: 'local', node_name: 'Spark One', size_bytes: 20 },
        staging_reserve_bytes: 64 * 1024 ** 2,
        download: { size_bytes: 20, required_free_bytes: 1000 },
        targets: [
          { node_id: 'local', node_name: 'Spark One', has_required_weights: true, has_model_cache: true, eligible: false },
          { node_id: 'node-2', node_name: 'Spark Two', has_required_weights: false, has_model_cache: true, eligible: false, download_eligible: true, free_bytes: 2000 },
        ],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      const body = path.includes('/api/v1/model-cache') ? { nodes: [
        { id: 'local', name: 'Spark One', online: true, models: [{ model_id: 'org/model', size_bytes: 20, revisions: ['main', 'a'.repeat(40)] }] },
        { id: 'node-2', name: 'Spark Two', online: true, models: [{ model_id: 'org/model', size_bytes: 20, revisions: ['main', 'b'.repeat(40)] }] },
      ] } : path.includes('/api/v1/recipes') ? { items: [{
        id: 'mixed-main', name: 'Mixed main', model: 'org/model', engine: 'vllm',
        deployment_mode: 'sharded', required_node_count: 2, tensor_parallel_size: 2,
        pipeline_parallel_size: 1, node_ids: ['local', 'node-2'], extra_args_count: 0,
      }] } : path.includes('/api/v1/deployments') ? { items: [] }
        : path.includes('/api/v1/nodes') ? { items: [
          { id: 'local', name: 'Spark One', local: true, online: true, docker_ready: true, selectable: true },
          { id: 'node-2', name: 'Spark Two', online: true, docker_ready: true, selectable: true },
        ] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'org 1' }))
    await user.click(await screen.findByRole('button', { name: 'Choose nodes & deploy' }))
    const dialog = await screen.findByRole('dialog', { name: 'Deploy Mixed main' })

    expect(within(dialog).queryByRole('button', { name: 'Deploy on 2 nodes' })).not.toBeInTheDocument()
    expect(within(dialog).getByRole('button', { name: 'Prepare selected nodes' })).toBeEnabled()
  })

  it('surfaces saved-configuration failures without hiding deployment state', async () => {
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/recipes')) return new Response(JSON.stringify({ detail: 'saved configurations unavailable' }), {
        status: 503,
        headers: { 'Content-Type': 'application/json' },
      })
      const body = path.includes('/api/v1/deployments') ? { items: [] }
        : path.includes('/api/v1/nodes') ? { items: [{ id: 'local', name: 'Spark One', local: true, online: true, docker_ready: true, selectable: true }] }
          : path.includes('/api/v1/model-cache') ? { nodes: [] }
            : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)

    expect(await screen.findByRole('alert')).toHaveTextContent('Saved configurations: saved configurations unavailable')
    expect(screen.getByText('No deployments yet')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument()
  })

  it('retains the deployment context length when switching runtimes', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async () => {
      return new Response(JSON.stringify({ items: [] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'Create deployment' }))

    const contextLength = screen.getByRole('spinbutton', { name: 'Context length' })
    const runtime = screen.getByRole('combobox', { name: 'Runtime' })
    await user.clear(contextLength)
    await user.type(contextLength, '32768')
    expect(contextLength).toHaveValue(32768)

    await user.selectOptions(runtime, 'llama.cpp')
    expect(contextLength).toHaveValue(32768)
    await user.selectOptions(runtime, 'sglang')
    expect(contextLength).toHaveValue(32768)
  })

  it('sends launch arguments when adding a managed model server', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (init?.method === 'POST' && path === '/api/v1/deployments') {
        return new Response(JSON.stringify({
          id: 'dep-new', alias: 'Flagged model', runtime: 'vllm', kind: 'managed',
          model: { repository: 'org/model' }, status: 'starting', settings: {}, node_ids: ['local'],
        }), { status: 201, headers: { 'Content-Type': 'application/json' } })
      }
      const body = path.includes('/api/v1/nodes') ? { items: [
        { id: 'local', name: 'This device', local: true, online: true, docker_ready: true, selectable: true },
      ] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'Create deployment' }))

    await user.type(screen.getByRole('textbox', { name: 'Display name' }), 'Flagged model')
    await user.type(screen.getByRole('textbox', { name: 'Model repository or GGUF artifact' }), 'org/model')
    await user.click(screen.getByRole('button', { name: 'Launch arguments' }))
    await user.type(screen.getByRole('spinbutton', { name: 'GPU memory util' }), '0.85')
    await user.type(screen.getByRole('textbox', { name: 'Extra flags' }), '--served-model-name "My Model" --kv-cache-dtype fp8')
    await user.type(screen.getByRole('textbox', { name: /^Runtime environment variables/ }), 'HF_HUB_OFFLINE=1{enter}NCCL_DEBUG=WARN')
    await user.click(screen.getByRole('button', { name: 'Save deployment' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/deployments',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          model: 'org/model',
          alias: 'Flagged model',
          runtime: 'vllm',
          kind: 'managed',
          settings: {
            context_length: 8192,
            tensor_parallel_size: 1,
            image: 'nvcr.io/nvidia/vllm:26.03.post1-py3',
            extra_args: ['--served-model-name', 'My Model', '--kv-cache-dtype', 'fp8'],
            environment: { HF_HUB_OFFLINE: '1', NCCL_DEBUG: 'WARN' },
            gpu_memory_utilization: 0.85,
          },
          node_ids: ['local'],
          deployment_mode: 'single',
        }),
      }),
    ))
    expect(await screen.findByText('Saved Flagged model. Launch it from the deployments list when ready.')).toBeInTheDocument()
  })

  it('groups saved configurations by company and pins cards within a group', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      const body = path.includes('/api/v1/recipes') ? { items: [
        {
          id: 'recipe-qwen', name: 'Qwen config', model: 'Qwen/Qwen3-32B', engine: 'vllm',
          deployment_mode: 'single', required_node_count: 1, tensor_parallel_size: 1,
          pipeline_parallel_size: 1, node_ids: ['local'], extra_args_count: 0,
        },
        {
          id: 'recipe-ds', name: 'DS config', model: 'deepseek-ai/DeepSeek-V4', engine: 'vllm',
          deployment_mode: 'single', required_node_count: 1, tensor_parallel_size: 1,
          pipeline_parallel_size: 1, node_ids: ['local'], extra_args_count: 0,
        },
      ] } : path.includes('/api/v1/deployments') ? { items: [] }
        : path.includes('/api/v1/nodes') ? { items: [
          { id: 'local', name: 'Spark One', local: true, online: true, docker_ready: true, selectable: true },
        ] } : path.includes('/api/v1/model-cache') ? { nodes: [] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)

    expect(await screen.findByRole('heading', { name: 'deepseek-ai 1' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Qwen 1' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'deepseek-ai 1' })).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByRole('button', { name: 'Pin DS config' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'deepseek-ai 1' }))
    const pin = screen.getByRole('button', { name: 'Pin DS config' })
    expect(pin).toHaveAttribute('aria-pressed', 'false')
    await user.click(pin)
    expect(screen.getByRole('button', { name: 'Unpin DS config' })).toHaveAttribute('aria-pressed', 'true')
    expect(JSON.parse(localStorage.getItem('sparkdeck:pinned-recipes') ?? '[]')).toContain('recipe-ds')
  })

  it('collapses recipe groups by default and toggles card visibility', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      const body = path.includes('/api/v1/recipes') ? { items: [{
        id: 'recipe-1', name: 'Saved cluster', model: 'org/model', engine: 'vllm',
        deployment_mode: 'single', required_node_count: 1, tensor_parallel_size: 1,
        pipeline_parallel_size: 1, node_ids: ['local'], extra_args_count: 2,
      }] } : path.includes('/api/v1/deployments') ? { items: [] }
        : path.includes('/api/v1/nodes') ? { items: [
          { id: 'local', name: 'Spark One', local: true, online: true, docker_ready: true, selectable: true },
        ] } : path.includes('/api/v1/model-cache') ? { nodes: [] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)

    const toggle = await screen.findByRole('button', { name: 'org 1' })
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByRole('button', { name: 'Choose nodes & deploy' })).not.toBeInTheDocument()

    await user.click(toggle)
    expect(screen.getByRole('button', { name: 'org 1' })).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByRole('button', { name: 'Choose nodes & deploy' })).toBeInTheDocument()
    expect(screen.getByText('2 saved')).toBeInTheDocument()

    await user.click(await screen.findByRole('button', { name: 'org 1' }))
    expect(screen.queryByRole('button', { name: 'Choose nodes & deploy' })).not.toBeInTheDocument()
  })

  it('shows deployment logs in a viewer dialog', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/deployments/dep-1/logs')) {
        return new Response(JSON.stringify({ logs: 'INFO model weights loaded\nINFO application started' }), {
          status: 200, headers: { 'Content-Type': 'application/json' },
        })
      }
      const body = path.includes('/api/v1/deployments') ? { items: [
        {
          id: 'dep-1', alias: 'Loud model', runtime: 'vllm', kind: 'managed',
          model: { repository: 'org/model' }, status: 'running', settings: {}, node_ids: ['local'],
        },
        {
          id: 'dep-ext', alias: 'Remote endpoint', runtime: 'vllm', kind: 'external',
          model: { repository: 'org/remote' }, status: 'running', settings: {},
        },
      ] } : path.includes('/api/v1/nodes') ? { items: [
        { id: 'local', name: 'Spark One', local: true, online: true, docker_ready: true, selectable: true },
      ] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)

    await user.click(await screen.findByRole('button', { name: 'Logs for Loud model' }))
    const dialog = await screen.findByRole('dialog', { name: 'Loud model' })
    expect(await within(dialog).findByText(/INFO model weights loaded/)).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/deployments/dep-1/logs?tail=300',
      expect.objectContaining({ headers: expect.anything() }),
    )
    // External endpoints have no managed logs, so they get no log action.
    expect(screen.queryByRole('button', { name: 'Logs for Remote endpoint' })).not.toBeInTheDocument()

    await user.click(within(dialog).getByRole('button', { name: 'Close' }))
    expect(screen.queryByRole('dialog', { name: 'Loud model' })).not.toBeInTheDocument()
  })

  it('asks which nodes to start on and defaults to the weighted ones', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (init?.method === 'POST' && path === '/api/v1/deployments/dep-1/start') {
        return new Response(JSON.stringify({
          id: 'dep-1', alias: 'Sharded model', runtime: 'vllm', kind: 'managed',
          model: { repository: 'org/model' }, status: 'starting', settings: {}, node_ids: ['local', 'node-2'],
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      const body = path.includes('/api/v1/deployments') ? { items: [{
        id: 'dep-1', alias: 'Sharded model', runtime: 'vllm', kind: 'managed',
        model: { repository: 'org/model' }, status: 'stopped', settings: { tensor_parallel_size: 2 }, node_ids: [],
        deployment_mode: 'sharded', required_node_count: 2,
      }] } : path.includes('/api/v1/nodes') ? { items: [
        { id: 'local', name: 'Spark One', local: true, online: true, docker_ready: true, selectable: true },
        { id: 'node-2', name: 'Spark Two', online: true, docker_ready: true, selectable: true },
        { id: 'node-3', name: 'Spark Three', online: true, docker_ready: true, selectable: true },
      ] } : path.includes('/api/v1/model-cache') ? { nodes: [
        { id: 'local', name: 'Spark One', online: true, models: [{ model_id: 'org/model', size_bytes: 2_000_000_000, revisions: ['main'] }] },
        { id: 'node-2', name: 'Spark Two', online: true, models: [{ model_id: 'org/model', size_bytes: 2_000_000_000, revisions: ['main'] }] },
        { id: 'node-3', name: 'Spark Three', online: true, models: [{ model_id: 'org/model', size_bytes: 1_000_000_000, revisions: ['main'], partial: true }] },
      ] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'Start' }))

    const dialog = await screen.findByRole('dialog', { name: 'Start Sharded model' })
    expect(dialog).toHaveTextContent('TP2 requires exactly 2 nodes')
    // Exactly two nodes hold complete weights, so both start selected; a
    // partial cache entry does not count as usable weights.
    expect(within(dialog).getByRole('checkbox', { name: /Spark One/ })).toBeChecked()
    expect(within(dialog).getByRole('checkbox', { name: /Spark Two/ })).toBeChecked()
    expect(within(dialog).getByRole('checkbox', { name: /Spark Three/ })).toBeDisabled()
    await user.click(within(dialog).getByRole('button', { name: 'Launch on 2 nodes' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/deployments/dep-1/start',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ node_ids: ['local', 'node-2'] }) }),
    ))
    expect(await screen.findByText('Starting Sharded model on This device, Spark Two.')).toBeInTheDocument()
  })

  it('defaults a single-node start to the one node holding the weights', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (init?.method === 'POST' && path === '/api/v1/deployments/dep-2/start') {
        return new Response(JSON.stringify({
          id: 'dep-2', alias: 'Solo model', runtime: 'vllm', kind: 'managed',
          model: { repository: 'org/model' }, status: 'starting', settings: {}, node_ids: ['node-2'],
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      const body = path.includes('/api/v1/deployments') ? { items: [{
        id: 'dep-2', alias: 'Solo model', runtime: 'vllm', kind: 'managed',
        model: { repository: 'org/model' }, status: 'stopped', settings: {}, node_ids: [],
        deployment_mode: 'single', required_node_count: 1,
      }] } : path.includes('/api/v1/nodes') ? { items: [
        { id: 'local', name: 'Spark One', local: true, online: true, docker_ready: true, selectable: true },
        { id: 'node-2', name: 'Spark Two', online: true, docker_ready: true, selectable: true },
      ] } : path.includes('/api/v1/model-cache') ? { nodes: [
        { id: 'node-2', name: 'Spark Two', online: true, models: [{ model_id: 'org/model', size_bytes: 2_000_000_000, revisions: ['main'] }] },
      ] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'Start' }))

    const dialog = await screen.findByRole('dialog', { name: 'Start Solo model' })
    expect(within(dialog).getByRole('radio', { name: /Spark Two/ })).toBeChecked()
    expect(within(dialog).getByRole('radio', { name: /Spark One/ })).toBeDisabled()
    await user.click(within(dialog).getByRole('button', { name: 'Launch on 1 node' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/deployments/dep-2/start',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ node_ids: ['node-2'] }) }),
    ))
  })

  it('uses the persisted deployment revision when choosing start nodes', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      const body = path.includes('/api/v1/deployments') ? { items: [{
        id: 'dep-pinned', alias: 'Pinned model', runtime: 'vllm', kind: 'managed',
        model: { repository: 'org/model' }, model_revision: 'release-b',
        status: 'stopped', settings: {}, node_ids: [],
        deployment_mode: 'single', required_node_count: 1,
      }] } : path.includes('/api/v1/nodes') ? { items: [
        { id: 'local', name: 'Default branch', local: true, online: true, docker_ready: true, selectable: true },
        { id: 'node-2', name: 'Pinned revision', online: true, docker_ready: true, selectable: true },
      ] } : path.includes('/api/v1/model-cache') ? { nodes: [
        { id: 'local', name: 'Default branch', online: true, models: [{ model_id: 'org/model', size_bytes: 20, revisions: ['main'] }] },
        { id: 'node-2', name: 'Pinned revision', online: true, models: [{ model_id: 'org/model', size_bytes: 20, revisions: ['release-b'] }] },
      ] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'Start' }))

    const dialog = await screen.findByRole('dialog', { name: 'Start Pinned model' })
    expect(within(dialog).getByRole('radio', { name: /Default branch/ })).toBeDisabled()
    expect(within(dialog).getByRole('radio', { name: /Pinned revision/ })).toBeChecked()
  })

  it('starts controller-owned llama.cpp artifacts without an HF cache entry', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      const body = path.includes('/api/v1/deployments') ? { items: [{
        id: 'dep-llama', alias: 'Local GGUF', runtime: 'llama.cpp', kind: 'managed',
        model: { repository: 'org/llama-artifact', artifact: '/models/local.gguf' }, status: 'stopped', settings: {},
        node_ids: ['local'], deployment_mode: 'single', required_node_count: 1,
      }] } : path.includes('/api/v1/nodes') ? { items: [
        { id: 'local', name: 'Controller', local: true, online: true, docker_ready: true, selectable: true },
        { id: 'node-2', name: 'Worker', online: true, docker_ready: true, selectable: true },
      ] } : path.includes('/api/v1/model-cache') ? { nodes: [] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'Start' }))

    const dialog = await screen.findByRole('dialog', { name: 'Start Local GGUF' })
    expect(within(dialog).getByRole('radio', { name: /Controller/ })).toBeChecked()
    expect(within(dialog).getByRole('radio', { name: /Worker/ })).toBeDisabled()
    expect(within(dialog).getByRole('button', { name: 'Launch on 1 node' })).toBeEnabled()
    expect(dialog).toHaveTextContent('This local artifact can run only on the controller')
  })

  it('launches a saved deployment after transferring weights via Virtual NAS', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      const method = init?.method ?? 'GET'
      if (path === '/api/v1/deployments/dep-saved/prepare/preflight' && method === 'POST') {
        return new Response(JSON.stringify({
          enabled: true, model_id: 'org/model', revision: 'main',
          source: { node_id: 'node-2', node_name: 'Spark Two', size_bytes: 1000 },
          sources: [{ node_id: 'node-2', node_name: 'Spark Two', size_bytes: 1000 }],
          targets: [
            { node_id: 'local', node_name: 'Spark One', eligible: true, has_required_weights: false, free_bytes: 900, required_free_bytes: 500 },
            { node_id: 'node-2', node_name: 'Spark Two', eligible: true, has_required_weights: true },
          ],
          node_ids: ['local', 'node-2'], eligible: true, action: 'transfer',
          download_node_id: null, download_node_ids: [],
          transfer_target_node_ids: ['local'], reason: null,
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/deployments/dep-saved/prepare' && method === 'POST') {
        expect(JSON.parse(String(init?.body))).toEqual({ node_ids: ['local'] })
        return new Response(JSON.stringify({
          workflow_id: 'wf-1', job_ids: ['job-1'],
          jobs: [{ id: 'job-1', model_id: 'org/model', source_node_id: 'node-2', source_node_name: 'Spark Two', target_node_id: 'local', target_node_name: 'Spark One', status: 'queued', bytes_total: 1000, bytes_transferred: 0, created_at: 1 }],
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/storage') {
        return new Response(JSON.stringify({
          enabled: true, nodes: [], instructions: [],
          jobs: [{ id: 'job-1', model_id: 'org/model', source_node_id: 'node-2', source_node_name: 'Spark Two', target_node_id: 'local', target_node_name: 'Spark One', status: 'completed', bytes_total: 1000, bytes_transferred: 1000, created_at: 1 }],
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/deployments/dep-saved/start' && method === 'POST') {
        expect(JSON.parse(String(init?.body))).toEqual({ node_ids: ['local'] })
        return new Response(JSON.stringify({
          id: 'dep-saved', alias: 'Saved model', runtime: 'vllm', kind: 'managed',
          model: { repository: 'org/model' }, status: 'starting', settings: {}, node_ids: ['local'],
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      const body = path.includes('/api/v1/deployments') ? { items: [{
        id: 'dep-saved', alias: 'Saved model', runtime: 'vllm', kind: 'managed',
        model: { repository: 'org/model' }, status: 'saved', settings: {}, node_ids: ['local'],
        deployment_mode: 'single', required_node_count: 1,
      }] } : path.includes('/api/v1/nodes') ? { items: [
        { id: 'local', name: 'Spark One', local: true, online: true, docker_ready: true, selectable: true },
        { id: 'node-2', name: 'Spark Two', online: true, docker_ready: true, selectable: true },
      ] } : path.includes('/api/v1/model-cache') ? { nodes: [
        { id: 'node-2', name: 'Spark Two', online: true, models: [{ model_id: 'org/model', size_bytes: 2_000_000_000, revisions: ['main'] }] },
      ] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'Launch' }))

    const dialog = await screen.findByRole('dialog', { name: 'Launch Saved model' })
    expect(within(dialog).getByRole('radio', { name: /Spark One/ })).toBeChecked()
    expect(within(dialog).getByRole('button', { name: 'Transfer & launch on 1 node' })).toBeEnabled()
    expect(dialog).toHaveTextContent('Weights will be transferred from Spark Two to Spark One via Virtual NAS before launch.')
    await user.click(within(dialog).getByRole('button', { name: 'Transfer & launch on 1 node' }))

    expect(await screen.findByRole('dialog', { name: 'Prepare model weights?' })).toBeInTheDocument()
    await user.click(await screen.findByRole('button', { name: 'Transfer & launch' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/deployments/dep-saved/start',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ node_ids: ['local'] }) }),
    ))
    expect(await screen.findByText('Starting Saved model on This device.')).toBeInTheDocument()
  })

  it('keeps viable nodes launchable when an unrelated node lacks cache space', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (path === '/api/v1/deployments/dep-mixed/prepare/preflight' && (init?.method ?? 'GET') === 'POST') {
        return new Response(JSON.stringify({
          enabled: true, model_id: 'org/model', revision: 'main',
          source: null, sources: [],
          download: { size_bytes: 5_000_000_000, required_free_bytes: 10_000_000_000 },
          targets: [
            { node_id: 'local', node_name: 'Spark One', eligible: true, has_required_weights: false, has_model_cache: false, free_bytes: 900, download_eligible: true },
            { node_id: 'node-2', node_name: 'Spark Two', eligible: false, has_required_weights: false, has_model_cache: false, free_bytes: 100, download_eligible: false, download_reason: 'Not enough free cache space for the Hugging Face download' },
          ],
          node_ids: ['local', 'node-2'], eligible: true, action: 'download',
          download_node_id: 'local', download_node_ids: ['local'],
          transfer_target_node_ids: [], reason: null,
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      const body = path.includes('/api/v1/deployments') ? { items: [{
        id: 'dep-mixed', alias: 'Mixed nodes', runtime: 'vllm', kind: 'managed',
        model: { repository: 'org/model' }, status: 'saved', settings: {}, node_ids: ['local'],
        deployment_mode: 'single', required_node_count: 1,
      }] } : path.includes('/api/v1/nodes') ? { items: [
        { id: 'local', name: 'Spark One', local: true, online: true, docker_ready: true, selectable: true },
        { id: 'node-2', name: 'Spark Two', online: true, docker_ready: true, selectable: true },
      ] } : path.includes('/api/v1/model-cache') ? { nodes: [] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'Launch' }))

    const dialog = await screen.findByRole('dialog', { name: 'Launch Mixed nodes' })
    await waitFor(() => expect(
      within(dialog).getByRole('radio', { name: /Spark Two/ }),
    ).toBeDisabled())
    // Spark One can receive the weights, so the launch stays available even
    // though Spark Two is blocked by its own cache capacity.
    expect(within(dialog).getByRole('radio', { name: /Spark One/ })).toBeChecked()
    expect(within(dialog).getByRole('button', { name: /launch on 1 node/i })).toBeEnabled()
  })

  it('blocks a saved deployment launch when no node has enough free cache space', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (path === '/api/v1/deployments/dep-full/prepare/preflight' && (init?.method ?? 'GET') === 'POST') {
        return new Response(JSON.stringify({
          enabled: true, model_id: 'org/model', revision: 'main',
          source: null, sources: [],
          download: { size_bytes: 5_000_000_000, required_free_bytes: 10_000_000_000 },
          targets: [
            { node_id: 'local', node_name: 'Spark One', eligible: false, has_required_weights: false, has_model_cache: false, free_bytes: 100, download_eligible: false, download_reason: 'Not enough free cache space for the Hugging Face download', transfer_after_download_eligible: false, transfer_after_download_reason: 'Not enough free cache space for Virtual NAS staging' },
          ],
          node_ids: ['local'], eligible: false, action: 'download',
          download_node_id: null, download_node_ids: [],
          transfer_target_node_ids: ['local'],
          reason: 'Not enough free cache space for the Hugging Face download',
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      const body = path.includes('/api/v1/deployments') ? { items: [{
        id: 'dep-full', alias: 'Too big', runtime: 'vllm', kind: 'managed',
        model: { repository: 'org/model' }, status: 'saved', settings: {}, node_ids: ['local'],
        deployment_mode: 'single', required_node_count: 1,
      }] } : path.includes('/api/v1/nodes') ? { items: [
        { id: 'local', name: 'Spark One', local: true, online: true, docker_ready: true, selectable: true },
      ] } : path.includes('/api/v1/model-cache') ? { nodes: [] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'Launch' }))

    const dialog = await screen.findByRole('dialog', { name: 'Launch Too big' })
    await waitFor(() => expect(
      within(dialog).getByRole('radio', { name: /Spark One/ }),
    ).toBeDisabled())
    expect(within(dialog).getByRole('button', { name: /launch on 1 node/i })).toBeDisabled()
    expect(dialog).toHaveTextContent('Only 0 of 1 required node is launchable')
  })

  it('keeps the persisted replicated layout even when tensor parallelism is 1', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      const body = path.includes('/api/v1/deployments') ? { items: [{
        id: 'dep-3', alias: 'Replica model', runtime: 'vllm', kind: 'managed',
        model: { repository: 'org/model' }, status: 'stopped', settings: { tensor_parallel_size: 1 }, node_ids: ['local', 'node-2'],
        deployment_mode: 'replicated', required_node_count: 2,
      }] } : path.includes('/api/v1/nodes') ? { items: [
        { id: 'local', name: 'Spark One', local: true, online: true, docker_ready: true, selectable: true },
        { id: 'node-2', name: 'Spark Two', online: true, docker_ready: true, selectable: true },
        { id: 'node-3', name: 'Spark Three', online: true, docker_ready: true, selectable: true },
      ] } : path.includes('/api/v1/model-cache') ? { nodes: [
        { id: 'local', name: 'Spark One', online: true, models: [{ model_id: 'org/model', size_bytes: 2_000_000_000, revisions: ['main'] }] },
        { id: 'node-2', name: 'Spark Two', online: true, models: [{ model_id: 'org/model', size_bytes: 2_000_000_000, revisions: ['main'] }] },
      ] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'Start' }))

    const dialog = await screen.findByRole('dialog', { name: 'Start Replica model' })
    // A replicated TP1 deployment still runs on its saved two nodes.
    expect(dialog).toHaveTextContent('exactly 2 nodes')
    expect(within(dialog).getByRole('checkbox', { name: /Spark One/ })).toBeChecked()
    expect(within(dialog).getByRole('checkbox', { name: /Spark Two/ })).toBeChecked()
    expect(within(dialog).getByRole('button', { name: 'Launch on 2 nodes' })).toBeEnabled()
  })

  it('sorts deployments by recency or name and renames them inline', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (init?.method === 'PATCH' && path.includes('/api/v1/deployments/dep-1')) {
        return new Response(JSON.stringify({
          id: 'dep-1', alias: 'Renamed', runtime: 'vllm', kind: 'managed',
          model: { repository: 'org/model' }, status: 'running', settings: {},
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      const body = path.includes('/api/v1/deployments') ? { items: [
        {
          id: 'dep-1', alias: 'Zulu', runtime: 'vllm', kind: 'managed',
          model: { repository: 'org/zulu' }, status: 'running', settings: {},
          created_at: '2026-08-25T00:00:00+00:00',
        },
        {
          id: 'dep-2', alias: 'Alpha', runtime: 'vllm', kind: 'managed',
          model: { repository: 'org/alpha' }, status: 'stopped', settings: {},
          created_at: '2026-08-20T00:00:00+00:00',
        },
      ] } : path.includes('/api/v1/recipes') ? { items: [] }
        : path.includes('/api/v1/nodes') ? { items: [
          { id: 'local', name: 'Spark One', local: true, online: true, docker_ready: true, selectable: true },
        ] } : path.includes('/api/v1/model-cache') ? { nodes: [] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)

    // Default: most recently deployed first.
    let rows = await screen.findAllByRole('row')
    expect(within(rows[1]).getByText('Zulu')).toBeInTheDocument()
    expect(within(rows[2]).getByText('Alpha')).toBeInTheDocument()

    await user.selectOptions(screen.getByRole('combobox', { name: 'Sort deployments' }), 'name-asc')
    rows = screen.getAllByRole('row')
    expect(within(rows[1]).getByText('Alpha')).toBeInTheDocument()
    expect(within(rows[2]).getByText('Zulu')).toBeInTheDocument()
    expect(localStorage.getItem('sparkdeck:models-sort')).toBe('name-asc')

    await user.selectOptions(screen.getByRole('combobox', { name: 'Sort deployments' }), 'name-desc')
    rows = screen.getAllByRole('row')
    expect(within(rows[1]).getByText('Zulu')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Rename Zulu' }))
    const input = screen.getByRole('textbox', { name: 'New name for Zulu' })
    await user.clear(input)
    await user.type(input, 'Renamed')
    await user.click(screen.getByRole('button', { name: 'Save name' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/deployments/dep-1',
      expect.objectContaining({ method: 'PATCH', body: JSON.stringify({ alias: 'Renamed' }) }),
    ))
    expect(await screen.findByText('Renamed deployment to Renamed.')).toBeInTheDocument()
  })

  it('edits saved configuration arguments and saves them as launch controls', async () => {
    const user = userEvent.setup()
    const detail = {
      id: 'recipe-args', name: 'Args config', model: 'org/model', engine: 'vllm',
      deployment_mode: 'single', required_node_count: 1, tensor_parallel_size: 1,
      pipeline_parallel_size: 1, node_ids: ['local'], extra_args_count: 6,
      extra_args: [
        '--max-model-len', '32768', '--enable-prefix-caching',
        '--speculative-config', '{"model":"org/draft","num_speculative_tokens":4,"draft_tensor_parallel_size":2}',
        '--default-chat-template-kwargs={"thinking":true,"note":"don\'t drop"}',
      ],
      launch_controls: {
        context_window: 32768, max_concurrency: null, kv_cache_dtype: null,
        thinking_mode: 'default', dspark_num_speculative_tokens: 4,
        max_cudagraph_capture_size: null, max_num_batched_tokens: null,
      },
      gpu_memory_utilization: 0.9, gpu_memory_gb: null,
    }
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (path === '/api/v1/recipes/recipe-args' && init?.method === 'PUT') {
        return new Response(JSON.stringify(detail), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/recipes/recipe-args') {
        return new Response(JSON.stringify(detail), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      const body = path.includes('/api/v1/recipes') ? { items: [{
        id: 'recipe-args', name: 'Args config', model: 'org/model', engine: 'vllm',
        deployment_mode: 'single', required_node_count: 1, tensor_parallel_size: 1,
        pipeline_parallel_size: 1, node_ids: ['local'], extra_args_count: 3,
      }] } : path.includes('/api/v1/deployments') ? { items: [] }
        : path.includes('/api/v1/nodes') ? { items: [
          { id: 'local', name: 'Spark One', local: true, online: true, docker_ready: true, selectable: true },
        ] } : path.includes('/api/v1/model-cache') ? { nodes: [] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)

    await user.click(await screen.findByRole('button', { name: 'org 1' }))
    await user.click(await screen.findByRole('button', { name: 'Arguments' }))
    const contextWindow = await screen.findByRole('spinbutton', { name: 'Context window' })
    expect(contextWindow).toHaveValue(32768)
    expect(screen.getByRole('textbox', { name: 'Other flags' })).toHaveValue(
      `--enable-prefix-caching --speculative-config '{"model":"org/draft","num_speculative_tokens":4,"draft_tensor_parallel_size":2}' '--default-chat-template-kwargs={"thinking":true,"note":"don'\\''t drop"}'`,
    )
    expect(screen.getByRole('spinbutton', { name: 'GPU memory util' })).toHaveValue(0.9)

    await user.clear(contextWindow)
    await user.type(contextWindow, '65536')
    await user.click(screen.getByRole('button', { name: 'Save settings' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/recipes/recipe-args',
      expect.objectContaining({ method: 'PUT' }),
    ))
    const putCall = fetchMock.mock.calls.find(([input, init]) => String(input) === '/api/v1/recipes/recipe-args' && init?.method === 'PUT')
    const payload = JSON.parse(String(putCall?.[1]?.body))
    // Compound JSON flags round-trip verbatim; the backend merges the
    // structured fields into them instead of the editor dropping them.
    expect(payload.extra_args).toEqual([
      '--enable-prefix-caching',
      '--speculative-config',
      '{"model":"org/draft","num_speculative_tokens":4,"draft_tensor_parallel_size":2}',
      '--default-chat-template-kwargs={"thinking":true,"note":"don\'t drop"}',
    ])
    expect(payload.launch_controls.context_window).toBe(65536)
    expect(payload.gpu_memory_utilization).toBe(0.9)
    expect(await screen.findByText('Saved.')).toBeInTheDocument()
  })
})
