import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DeploymentPage } from './DeploymentPage'

const fetchMock = vi.fn<typeof fetch>()

const detail = {
  id: 'dep-1', alias: 'Reasoning server', runtime: 'vllm', kind: 'managed',
  model: { repository: 'org/model' }, status: 'stopped', settings: {},
  node_ids: ['local'], deployment_mode: 'single', desired_state: 'stopped',
  editable: true, edit_reason: null,
  extra_args: [
    '--enable-prefix-caching', '--speculative-config',
    '{"method":"ngram","foo":true}', 'C:\\models\\foo', '', "owner's-model",
  ],
  launch_controls: { context_window: 8192, max_concurrency: 4, thinking_mode: 'default' },
  gpu_memory_utilization: 0.9, gpu_memory_gb: null, image: null,
}

beforeEach(() => {
  vi.stubGlobal('fetch', fetchMock)
  fetchMock.mockClear()
  fetchMock.mockImplementation(async (input, init) => {
    const path = String(input)
    if (path === '/api/v1/deployments/dep-1/settings' && init?.method === 'PUT') {
      return new Response(JSON.stringify(detail), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }
    if (path === '/api/v1/deployments/dep-1/start' && init?.method === 'POST') {
      return new Response(JSON.stringify(detail), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }
    return new Response(JSON.stringify(detail), { status: 200, headers: { 'Content-Type': 'application/json' } })
  })
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

function renderPage() {
  return render(<MemoryRouter initialEntries={['/models/dep-1']}><Routes>
    <Route path="/models/:deploymentId" element={<DeploymentPage />} />
    <Route path="/models" element={<h1>Models destination</h1>} />
  </Routes></MemoryRouter>)
}

describe('deployment object page', () => {
  it('shows saved flags and saves before running, then returns to Models', async () => {
    const user = userEvent.setup()
    renderPage()

    expect(await screen.findByRole('heading', { name: 'Reasoning server' })).toBeInTheDocument()
    expect(screen.getByLabelText(/Runtime flags/)).toHaveValue(
      `--enable-prefix-caching --speculative-config '{"method":"ngram","foo":true}' 'C:\\models\\foo' '' 'owner'\\''s-model'`,
    )
    await user.clear(screen.getByLabelText('Max concurrency'))
    await user.type(screen.getByLabelText('Max concurrency'), '6')
    await user.click(screen.getByRole('button', { name: 'Run' }))

    expect(await screen.findByRole('heading', { name: 'Models destination' })).toBeInTheDocument()
    const mutationCalls = fetchMock.mock.calls.filter(([, init]) => init?.method === 'PUT' || init?.method === 'POST')
    expect(mutationCalls.map(([input]) => String(input))).toEqual([
      '/api/v1/deployments/dep-1/settings',
      '/api/v1/deployments/dep-1/start',
    ])
    const saved = JSON.parse(String(mutationCalls[0][1]?.body))
    expect(saved.launch_controls.max_concurrency).toBe(6)
    expect(saved.extra_args).toEqual([
      '--enable-prefix-caching', '--speculative-config',
      '{"method":"ngram","foo":true}', 'C:\\models\\foo', '', "owner's-model",
    ])
  })

  it('keeps a running deployment read-only', async () => {
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({
      ...detail, status: 'running', desired_state: 'running', editable: false,
      edit_reason: 'Stop this deployment before changing its launch settings.',
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    renderPage()

    expect(await screen.findByText('Stop this deployment before changing its launch settings.')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('button', { name: 'Run' })).toBeDisabled())
    expect(screen.getByLabelText(/Runtime flags/)).toBeDisabled()
  })
})
