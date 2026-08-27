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
    expect(screen.getByRole('link', { name: 'Deploy org/model' })).toHaveAttribute('href', '/models?model=org%2Fmodel')
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
        ? { default_runtime: 'sglang', default_context_length: 24576 }
        : path.includes('/api/v1/nodes')
        ? { items: [{ id: 'local', name: 'This device', local: true, online: true, docker_ready: true, selectable: true }] }
        : { items: [] }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter initialEntries={['/models?model=org/chosen-model']}><ModelsPage /></MemoryRouter>)

    expect(await screen.findByRole('dialog', { name: 'Add a model server' })).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'Model repository or GGUF artifact' })).toHaveValue('org/chosen-model')
    expect(screen.getByRole('textbox', { name: 'Display name' })).toHaveValue('chosen-model')
    await waitFor(() => {
      expect(screen.getByRole('combobox', { name: 'Runtime' })).toHaveValue('sglang')
      expect(screen.getByRole('spinbutton', { name: 'Context length' })).toHaveValue(24576)
    })
    expect(fetchMock.mock.calls.some(([input]) => String(input).includes('/api/v1/settings'))).toBe(true)
  })
})

describe('model deployments', () => {
  it('shows exact replicated disk usage and deploys legacy saved configurations', async () => {
    const user = userEvent.setup()
    const gib = 1024 ** 3
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
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
    expect(dialog).toHaveTextContent('Model weights not cached')

    await user.click(secondNode)
    expect(within(dialog).getByRole('button', { name: 'Deploy on 2 nodes' })).toBeDisabled()
    expect(dialog).toHaveTextContent('Select exactly 2 nodes to continue')
    await user.click(secondNode)
    await user.click(within(dialog).getByRole('button', { name: 'Deploy on 2 nodes' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/recipes/recipe-1/deploy',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ node_ids: ['local', 'node-2'] }) }),
    ))
    expect(await screen.findByText('Deployed saved configuration Saved cluster on This device, Spark Two.')).toBeInTheDocument()
  })

  it('requires the main cache ref for an unpinned saved configuration', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
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
    expect(within(dialog).getByRole('radio', { name: /Default branch/ })).toBeChecked()
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

  it('deduplicates saved node IDs before filling an exact-count selector', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
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
    expect(screen.getByText('No model servers yet')).toBeInTheDocument()
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
    await user.click(await screen.findByRole('button', { name: 'Add model' }))

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
