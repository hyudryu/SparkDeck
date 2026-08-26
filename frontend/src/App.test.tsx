import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { ExplorePage } from './pages/ExplorePage'

const fetchMock = vi.fn<typeof fetch>()

beforeEach(() => {
  vi.stubGlobal('fetch', fetchMock)
  fetchMock.mockResolvedValue(new Response(JSON.stringify({ items: [], total: 0 }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  }))
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  localStorage.clear()
})

describe('SparkDeck application shell', () => {
  it('exposes every primary destination in the left navigation', async () => {
    render(<MemoryRouter initialEntries={['/']}><App /></MemoryRouter>)

    expect(screen.getByRole('link', { name: 'SparkDeck home' })).toBeInTheDocument()
    for (const label of ['Explore', 'Models', 'Chat', 'Compare', 'Benchmarks', 'Images', 'Settings', 'Logs']) {
      expect(screen.getByRole('link', { name: label })).toBeInTheDocument()
    }
    expect(await screen.findByText('No models found')).toBeInTheDocument()
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
})
