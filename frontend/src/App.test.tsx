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
    for (const label of ['Dashboard', 'Explore', 'Models', 'Cluster', 'Chat', 'Compare', 'Benchmarks', 'Usage', 'Images', 'Storage', 'Settings', 'Logs']) {
      expect(screen.getByRole('link', { name: label })).toBeInTheDocument()
    }
    expect(await screen.findByRole('heading', { name: 'Dashboard' })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: 'System overview' })).toBeInTheDocument()
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
    let settings = { theme: 'light', default_runtime: 'sglang', default_context_length: 32768, community_api_url: 'https://community.example' }
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
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({
      items: [{
        id: 'org/model', author: 'org', name: 'model', downloads: 1200,
        runtime_compatibility: [{ runtime: 'vllm', supported: true }],
        local_deployment_ids: ['dep-1'], community: null,
      }],
      total: 1,
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)

    expect(await screen.findByRole('heading', { name: 'model' })).toBeInTheDocument()
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
        ? { default_runtime: 'vllm', default_context_length: 8192 }
        : path.includes('/api/v1/nodes')
          ? { items: [{ id: 'local', name: 'This device', local: true, online: true, docker_ready: true, selectable: true }] }
          : { items: [] }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter initialEntries={['/models?model=org/chosen-model']}><ModelsPage /></MemoryRouter>)

    expect(await screen.findByRole('dialog', { name: 'Add a model server' })).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'Model repository or GGUF artifact' })).toHaveValue('org/chosen-model')
    expect(screen.getByRole('textbox', { name: 'Display name' })).toHaveValue('chosen-model')
  })

  it('merges late saved defaults into untouched fields in a catalog deployment', async () => {
    let resolveSettings!: (response: Response) => void
    const settingsResponse = new Promise<Response>((resolve) => { resolveSettings = resolve })
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/settings')) return settingsResponse
      const body = path.includes('/api/v1/nodes')
        ? { items: [{ id: 'local', name: 'This device', local: true, online: true, docker_ready: true, selectable: true }] }
        : { items: [] }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter initialEntries={['/models?model=org/chosen-model']}><ModelsPage /></MemoryRouter>)

    expect(await screen.findByRole('dialog', { name: 'Add a model server' })).toBeInTheDocument()
    resolveSettings(new Response(JSON.stringify({
      default_runtime: 'sglang',
      default_context_length: 32768,
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))

    await waitFor(() => {
      expect(screen.getByRole('combobox', { name: 'Runtime' })).toHaveValue('sglang')
      expect(screen.getByRole('spinbutton', { name: 'Context length' })).toHaveValue(32768)
    })
  })

  it('does not overwrite deployment fields edited before saved defaults load', async () => {
    const user = userEvent.setup()
    let resolveSettings!: (response: Response) => void
    const settingsResponse = new Promise<Response>((resolve) => { resolveSettings = resolve })
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/settings')) return settingsResponse
      const body = path.includes('/api/v1/nodes')
        ? { items: [{ id: 'local', name: 'This device', local: true, online: true, docker_ready: true, selectable: true }] }
        : { items: [] }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    render(<MemoryRouter initialEntries={['/models?model=org/chosen-model']}><ModelsPage /></MemoryRouter>)

    await user.selectOptions(await screen.findByRole('combobox', { name: 'Runtime' }), 'llama.cpp')
    const contextLength = screen.getByRole('spinbutton', { name: 'Context length' })
    resolveSettings(new Response(JSON.stringify({
      default_runtime: 'sglang',
      default_context_length: 32768,
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))

    await waitFor(() => expect(contextLength).toHaveValue(32768))
    expect(screen.getByRole('combobox', { name: 'Runtime' })).toHaveValue('llama.cpp')
  })
})

describe('model deployments', () => {
  it('retains the saved context length when switching runtimes', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      const body = path.includes('/api/v1/settings')
        ? { default_runtime: 'vllm', default_context_length: 32768 }
        : { items: [] }
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    })

    render(<MemoryRouter><ModelsPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'Add model' }))

    const contextLength = screen.getByRole('spinbutton', { name: 'Context length' })
    const runtime = screen.getByRole('combobox', { name: 'Runtime' })
    expect(contextLength).toHaveValue(32768)

    await user.selectOptions(runtime, 'llama.cpp')
    expect(contextLength).toHaveValue(32768)
    await user.selectOptions(runtime, 'sglang')
    expect(contextLength).toHaveValue(32768)
  })
})
