import { cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { BenchmarkRunner } from './BenchmarkRunner'

const completedRun = {
  id: 'run-1',
  status: 'completed',
  created_at: '2026-08-28T10:00:00Z',
  started_at: '2026-08-28T10:00:00Z',
  finished_at: '2026-08-28T10:04:00Z',
  duration_seconds: 240,
  model: 'unsloth/Qwen3-4B-GGUF',
  model_id: 'unsloth/Qwen3-4B-GGUF',
  quantization: 'Q4_K_M',
  runtime: 'llama.cpp',
  base_url: 'http://127.0.0.1:8080',
  config: {
    model_id: 'unsloth/Qwen3-4B-GGUF',
    prompt_sizes: [2048],
    response_sizes: [128],
    concurrency_levels: [1, 2],
    context_depths: [0],
    runs: 3,
    warmup_runs: 1,
    enable_prefix_caching: true,
    exact_tg: false,
  },
  benchy_version: '0.1.2',
  result_count: 2,
  csv_filename: 'results.csv',
  results: [
    {
      prompt_size: 2048, response_size: 128, context_depth: 0, concurrency: 1,
      is_context_prefill_phase: false,
      pp_tokens_per_second: 1800.5, pp_tokens_per_second_std: 12.1,
      pp_tokens_per_second_request: 1800.5, pp_tokens_per_second_request_std: 12.1,
      tg_tokens_per_second: 42.5, tg_tokens_per_second_std: 0.4,
      tg_tokens_per_second_request: 42.5, tg_tokens_per_second_request_std: 0.4,
      peak_tg_tokens_per_second: 45.0, peak_tg_tokens_per_second_request: 45.0,
      ttfr_ms: 480.2, est_ppt_ms: 1100.0, e2e_ttft_ms: 505.0,
    },
    {
      prompt_size: 2048, response_size: 128, context_depth: 0, concurrency: 2,
      is_context_prefill_phase: false,
      pp_tokens_per_second: 3200.0, pp_tokens_per_second_std: 15.0,
      pp_tokens_per_second_request: 1600.0, pp_tokens_per_second_request_std: 8.0,
      tg_tokens_per_second: 60.0, tg_tokens_per_second_std: 0.8,
      tg_tokens_per_second_request: 30.0, tg_tokens_per_second_request_std: 0.5,
      peak_tg_tokens_per_second: 63.0, peak_tg_tokens_per_second_request: 31.5,
      ttfr_ms: 700.0, est_ppt_ms: 640.0, e2e_ttft_ms: 720.0,
    },
  ],
  report: { benchy_version: '0.1.2', latency_mode: 'api', latency_ms: 12.5, prefix_caching_enabled: false },
}

function stubFetch(overrides: {
  status?: { installed: boolean; version?: string }
  runs?: unknown[]
  runDetail?: unknown
  startError?: number
  modelsError?: number
} = {}) {
  const state = {
    status: overrides.status ?? { installed: true, version: '0.1.2' },
    runs: 'runs' in overrides ? overrides.runs : [completedRun],
    runDetail: overrides.runDetail ?? completedRun,
  }
  const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
    const path = String(input)
    let body: unknown
    if (path.endsWith('/api/v1/benchmark-runner/status')) body = { ...state.status, active_run_id: null }
    else if (path.endsWith('/api/v1/benchmark-runner/install')) {
      state.status = { installed: true, version: '0.1.2' }
      body = { installed: true, version: '0.1.2', launch_mode: 'python_module', path_on_host: false }
    }
    else if (path.endsWith('/api/v1/benchmark-runner/models')) {
      if (overrides.modelsError) {
        return new Response(JSON.stringify({ detail: 'Model discovery failed' }), {
          status: overrides.modelsError, headers: { 'Content-Type': 'application/json' },
        })
      }
      body = { items: [{
        id: 'unsloth/Qwen3-4B-GGUF', label: 'Qwen3-4B-GGUF', runtime: 'llama.cpp',
        deployment_id: null, model: 'unsloth/Qwen3-4B-GGUF',
        quantization: 'Q4_K_M', base_url: 'http://127.0.0.1:8080',
      }] }
    }
    else if (path.endsWith('/api/v1/benchmark-runner/runs') && init?.method === 'POST') {
      if (overrides.startError) {
        return new Response(JSON.stringify({ detail: 'llama-benchy is not installed' }), {
          status: overrides.startError, headers: { 'Content-Type': 'application/json' },
        })
      }
      body = { ...completedRun, id: 'run-2', status: 'running', results: [] }
    }
    else if (path.endsWith('/api/v1/benchmark-runner/runs')) body = { items: state.runs }
    else if (path.includes('/api/v1/benchmark-runner/runs/')) body = state.runDetail
    else body = {}
    return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
  })
  vi.stubGlobal('fetch', fetchMock)
  return state
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

function renderPage() {
  return render(<MemoryRouter><BenchmarkRunner /></MemoryRouter>)
}

describe('BenchmarkRunner', () => {
  it('offers a one-button install when llama-benchy is missing', async () => {
    const state = stubFetch({ status: { installed: false }, runs: [] })
    const user = userEvent.setup()
    renderPage()

    expect(await screen.findByText(/Not installed\./)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Start benchmark' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /Install llama-benchy/ }))

    expect(await screen.findByText(/Installed · v0\.1\.2/)).toBeInTheDocument()
    expect(state.status.installed).toBe(true)
  })

  it('starts a run from the served model list and shows history', async () => {
    stubFetch()
    const user = userEvent.setup()
    renderPage()

    await screen.findByRole('option', { name: /Qwen3-4B-GGUF · Q4_K_M/ })
    expect(screen.getByLabelText(/Served model/).closest('.runner-config-panel')).not.toBeNull()
    expect(within(screen.getByLabelText(/Served model/)).getByRole('option', { name: /Qwen3-4B-GGUF · Q4_K_M/ })).toBeInTheDocument()
    expect(screen.getByLabelText(/Concurrency levels/)).toHaveValue('1, 2, 5, 10')
    expect(screen.getByLabelText(/Context depths/)).toHaveValue('0, 4096, 8192, 16384, 32768, 65535, 100000')
    expect(screen.getByLabelText(/Enable prefix caching/)).toBeChecked()

    await user.clear(screen.getByLabelText(/Concurrency levels/))
    await user.type(screen.getByLabelText(/Concurrency levels/), '1, 2, 4')
    await user.click(screen.getByRole('button', { name: 'Start benchmark' }))

    const startCall = vi.mocked(globalThis.fetch).mock.calls.find(
      ([input, init]) => String(input).endsWith('/api/v1/benchmark-runner/runs') && init?.method === 'POST',
    )
    expect(startCall).toBeTruthy()
    const body = JSON.parse(String(startCall![1]?.body ?? '{}'))
    expect(body.model_id).toBe('unsloth/Qwen3-4B-GGUF')
    expect(body.concurrency_levels).toEqual([1, 2, 4])
    expect(body.prompt_sizes).toEqual([2048])
    expect(body.response_sizes).toEqual([128])
    expect(body.context_depths).toEqual([0, 4096, 8192, 16384, 32768, 65535, 100000])
    expect(body.enable_prefix_caching).toBe(true)

    const history = await screen.findByRole('table', { name: 'Benchmark run history' })
    expect(within(history).getAllByText('Q4_K_M').length).toBeGreaterThan(0)
  })

  it('reports model discovery failures instead of claiming no models are served', async () => {
    stubFetch({ modelsError: 503 })
    renderPage()

    expect(await screen.findByRole('alert')).toHaveTextContent('Could not load served models: Model discovery failed')
    expect(screen.getByLabelText(/Served model/)).toHaveDisplayValue('Served models unavailable')
    expect(screen.queryByRole('option', { name: 'No models currently served' })).not.toBeInTheDocument()
  })

  it('rejects empty configuration with a inline error instead of a request', async () => {
    stubFetch()
    const user = userEvent.setup()
    renderPage()

    await screen.findByLabelText(/Served model/)
    await user.clear(screen.getByLabelText(/Concurrency levels/))
    await user.click(screen.getByRole('button', { name: 'Start benchmark' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Enter at least one concurrency level.')
    const startCalls = vi.mocked(globalThis.fetch).mock.calls.filter(
      ([input, init]) => String(input).endsWith('/api/v1/benchmark-runner/runs') && init?.method === 'POST',
    )
    expect(startCalls).toHaveLength(0)
  })

  it('rejects malformed numeric input instead of truncating it', async () => {
    stubFetch()
    const user = userEvent.setup()
    renderPage()

    await screen.findByRole('option', { name: /Qwen3-4B-GGUF · Q4_K_M/ })
    await user.clear(screen.getByLabelText(/Prompt sizes/))
    await user.type(screen.getByLabelText(/Prompt sizes/), '1.5')
    await user.click(screen.getByRole('button', { name: 'Start benchmark' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Invalid values: 1.5')
    const startCalls = vi.mocked(globalThis.fetch).mock.calls.filter(
      ([input, init]) => String(input).endsWith('/api/v1/benchmark-runner/runs') && init?.method === 'POST',
    )
    expect(startCalls).toHaveLength(0)
  })

  it('charts prompt processing, generation, and per-output speed for a selected run', async () => {
    stubFetch()
    const user = userEvent.setup()
    renderPage()

    const history = await screen.findByRole('table', { name: 'Benchmark run history' })
    await user.click(within(history).getByText('2,048 → 128 tok · C1 C2'))

    expect(await screen.findByRole('region', { name: /Processing speed \(generation\)/ })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: /Prompt processing speed/ })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: /Tokens per output/ })).toBeInTheDocument()
    expect(screen.getAllByText('C1').length).toBeGreaterThan(0)
    expect(screen.getAllByText('C2').length).toBeGreaterThan(0)
    // One prompt size per series renders measured points; lines appear with two or more.
    expect(document.querySelectorAll('.chart-series circle').length).toBeGreaterThanOrEqual(4)

    const measurements = screen.getByRole('table', { name: 'Run measurements' })
    expect(within(measurements).getAllByText('42.5 tok/s').length).toBeGreaterThan(0)
    expect(within(measurements).getAllByText('30.0 tok/s').length).toBeGreaterThan(0)

    const csv = screen.getByRole('link', { name: 'Download CSV' })
    expect(csv).toHaveAttribute('href', '/api/v1/benchmark-runner/runs/run-1/csv')
  })

  it('keeps response sizes as separate chart series instead of overwriting points', async () => {
    const multiResponse = {
      ...completedRun,
      results: [
        { ...completedRun.results[0], response_size: 64, tg_tokens_per_second: 44.0, tg_tokens_per_second_request: 44.0 },
        { ...completedRun.results[0], response_size: 128 },
      ],
    }
    stubFetch({ runDetail: multiResponse })
    const user = userEvent.setup()
    renderPage()

    const history = await screen.findByRole('table', { name: 'Benchmark run history' })
    await user.click(within(history).getByText('2,048 → 128 tok · C1 C2'))

    expect(await screen.findAllByText('C1 · 64 tok').then((items) => items.length)).toBeGreaterThan(0)
    expect(screen.getAllByText('C1 · 128 tok').length).toBeGreaterThan(0)
    // 3 charts × 2 series = 6 measured points, none overwritten.
    expect(document.querySelectorAll('.chart-series circle').length).toBe(6)
  })
})
