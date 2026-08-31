import { cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { BenchmarksPage } from './BenchmarksPage'

const communityAccess = vi.hoisted(() => ({ signedIn: true, sharingEnabled: true, loading: false, enabled: true, reload: vi.fn() }))

vi.mock('../hooks/useCommunityAccess', () => ({
  communityAccessHint: (signedIn: boolean) => signedIn
    ? 'Enable telemetry under Settings → Community Features to see community data.'
    : 'Sign in under Settings → Community Features to see community data.',
  useCommunityAccess: () => communityAccess,
}))

beforeEach(() => {
  Object.assign(communityAccess, { signedIn: true, sharingEnabled: true, loading: false, enabled: true })
  communityAccess.reload.mockClear()
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('BenchmarksPage community privacy', () => {
  it('switches between the Speed and Temp panes', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      let body: unknown
      if (path.endsWith('/api/temperature-runs')) body = { runs: [], active_run_id: null }
      else if (path.endsWith('/api/v1/nodes')) body = { items: [] }
      else if (path.endsWith('/api/v1/benchmark-models')) body = { items: [] }
      else if (path.includes('/api/v1/benchmark-history/models')) body = { items: [] }
      else if (path.endsWith('/api/v1/community/aggregates')) body = { items: [], availability: 'available', evidence_policy: { minimum_samples: 10, exact_match_dimensions: [], metric: 'inference_tokens_per_second' } }
      else body = { consent: true, pairing: { status: 'paired' }, upload_configured: true, outbox: {} }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }))
    const user = userEvent.setup()

    render(<MemoryRouter><BenchmarksPage /></MemoryRouter>)

    expect(await screen.findByText('Benchmark runs by model')).toBeInTheDocument()
    expect(screen.queryByText('Temperature runs')).not.toBeInTheDocument()

    await user.click(screen.getByRole('tab', { name: 'Temp' }))
    expect(await screen.findByText('Temperature runs')).toBeInTheDocument()
    expect(screen.getByText('No temperature runs yet')).toBeInTheDocument()
    expect(screen.queryByText('Benchmark runs by model')).not.toBeInTheDocument()

    await user.click(screen.getByRole('tab', { name: 'Speed' }))
    expect(await screen.findByText('Benchmark runs by model')).toBeInTheDocument()
    expect(screen.queryByText('Temperature runs')).not.toBeInTheDocument()
  })

  it('opens per-model C1/C2/C5/C10 charts and filters by TP size', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      let body: unknown
      if (path.endsWith('/api/v1/benchmark-models')) body = { items: [{
        model_id: 'nvidia/Qwen3.5-35B', run_count: 8,
        best_prompt_tokens_per_second: 5900, best_generation_tokens_per_second: 174,
        context_windows: [4096, 16384], tensor_parallel_sizes: [1, 2],
        latest_at: '2026-08-27T12:00:00Z',
      }] }
      else if (path.includes('/api/v1/benchmark-models/')) body = {
        model_id: 'nvidia/Qwen3.5-35B', points: [
          { context_window_size: 4096, concurrency: 1, tensor_parallel_size: 1, prompt_tokens_per_second: 5900, generation_tokens_per_second: 98, sample_count: 2 },
          { context_window_size: 4096, concurrency: 2, tensor_parallel_size: 1, prompt_tokens_per_second: 5800, generation_tokens_per_second: 160, sample_count: 2 },
          { context_window_size: 4096, concurrency: 5, tensor_parallel_size: 1, prompt_tokens_per_second: 5000, generation_tokens_per_second: 174, sample_count: 2 },
          { context_window_size: 4096, concurrency: 10, tensor_parallel_size: 1, prompt_tokens_per_second: 4800, generation_tokens_per_second: 150, sample_count: 2 },
          { context_window_size: 16384, concurrency: 1, tensor_parallel_size: 2, prompt_tokens_per_second: 4100, generation_tokens_per_second: 90, sample_count: 1 },
        ],
      }
      else if (path.includes('/api/v1/benchmark-history/models')) body = { items: [] }
      else if (path.endsWith('/api/v1/community/aggregates')) body = { items: [], availability: 'available', evidence_policy: { minimum_samples: 10, exact_match_dimensions: [], metric: 'inference_tokens_per_second' } }
      else body = { consent: true, pairing: { status: 'paired' }, upload_configured: true, outbox: {} }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }))
    const user = userEvent.setup()

    render(<MemoryRouter><BenchmarksPage /></MemoryRouter>)

    const model = await screen.findByRole('button', { name: /nvidia\/Qwen3.5-35B/ })
    expect(model).toHaveTextContent('5900.0 tok/s')
    await user.click(model)

    const dialog = await screen.findByRole('dialog', { name: 'nvidia/Qwen3.5-35B' })
    expect(within(dialog).getByRole('img', { name: /^Prompt throughput/ })).toBeInTheDocument()
    expect(within(dialog).getByRole('img', { name: /^Text generation throughput/ })).toBeInTheDocument()
    expect(within(dialog).getAllByText('C1').length).toBeGreaterThan(0)
    expect(within(dialog).getAllByText('C10').length).toBeGreaterThan(0)
    expect(dialog.querySelectorAll('.chart-series path').length).toBeGreaterThan(0)
    expect(dialog.querySelector('polyline')).not.toBeInTheDocument()
    await user.click(within(dialog).getByRole('tab', { name: 'TP 2' }))
    expect(within(dialog).getAllByText('16K context').length).toBeGreaterThan(0)
    expect(within(dialog).queryByText('4K context')).not.toBeInTheDocument()
  })

  it('never renders a previous model detail while the next model loads or fails', async () => {
    let resolveModelB: (response: Response) => void = () => undefined
    const modelBResponse = new Promise<Response>((resolve) => { resolveModelB = resolve })
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      let body: unknown
      if (path.endsWith('/api/v1/benchmark-models')) body = { items: [
        {
          model_id: 'org/model-a', run_count: 1,
          best_prompt_tokens_per_second: 100, best_generation_tokens_per_second: 20,
          context_windows: [4096], tensor_parallel_sizes: [1],
          latest_at: '2026-08-27T12:00:00Z',
        },
        {
          model_id: 'org/model-b', run_count: 1,
          best_prompt_tokens_per_second: 200, best_generation_tokens_per_second: 40,
          context_windows: [16384], tensor_parallel_sizes: [2],
          latest_at: '2026-08-27T13:00:00Z',
        },
      ] }
      else if (path.endsWith('/api/v1/benchmark-models/org%2Fmodel-a')) body = {
        model_id: 'org/model-a', points: [{
          context_window_size: 4096, concurrency: 1, tensor_parallel_size: 1,
          prompt_tokens_per_second: 100, generation_tokens_per_second: 20, sample_count: 1,
        }],
      }
      else if (path.endsWith('/api/v1/benchmark-models/org%2Fmodel-b')) return modelBResponse
      else if (path.includes('/api/v1/benchmark-history/models')) body = { items: [] }
      else if (path.endsWith('/api/v1/community/aggregates')) body = {
        items: [], availability: 'available',
        evidence_policy: { minimum_samples: 10, exact_match_dimensions: [], metric: 'inference_tokens_per_second' },
      }
      else body = { consent: true, pairing: { status: 'paired' }, upload_configured: true, outbox: {} }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }))
    const user = userEvent.setup()

    render(<MemoryRouter><BenchmarksPage /></MemoryRouter>)

    await user.click(await screen.findByRole('button', { name: /org\/model-a/ }))
    const modelADialog = await screen.findByRole('dialog', { name: 'org/model-a' })
    expect(within(modelADialog).getAllByText('4K context').length).toBeGreaterThan(0)

    await user.click(screen.getByRole('button', { name: /org\/model-b/ }))
    const modelBDialog = screen.getByRole('dialog', { name: 'org/model-b' })
    expect(within(modelBDialog).getByRole('status')).toHaveTextContent('Loading model benchmark')
    expect(within(modelBDialog).queryByText('4K context')).not.toBeInTheDocument()

    resolveModelB(new Response(JSON.stringify({ detail: 'Model B unavailable' }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' },
    }))

    expect(await within(modelBDialog).findByRole('alert')).toHaveTextContent('Model B unavailable')
    expect(within(modelBDialog).queryByText('4K context')).not.toBeInTheDocument()
  })

  it('discloses the exact shared fields and renders only contract-safe estimates', async () => {
    let consent = false
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      let body: unknown
      if (path.includes('/api/v1/benchmark-history/models')) body = { items: [] }
      else if (path.endsWith('/api/v1/community/aggregates')) body = {
        items: [
          {
            model_id: 'org/model', quantization: 'NVFP4', prompt_tokens_bucket: 1000,
            inference_tokens_per_second: 42.5, sample_count: 12, unique_cluster_count: 5,
          },
          {
            model_id: 'org/model', quantization: 'Q4_K_M', prompt_tokens_bucket: 1000,
            inference_tokens_per_second: 31.2, sample_count: 8, unique_cluster_count: 3,
          },
        ],
        availability: 'available',
        evidence_policy: { minimum_samples: 10, exact_match_dimensions: ['model_id', 'quantization', 'prompt_tokens_bucket'], metric: 'inference_tokens_per_second' },
      }
      else if (path.endsWith('/api/v1/community/consent') && init?.method === 'PUT') {
        consent = (JSON.parse(String(init.body)) as { enabled: boolean }).enabled
        body = {}
      } else body = { consent, pairing: { status: 'paired' }, upload_configured: true, outbox: { pending: 0, synced: 2 } }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<MemoryRouter><BenchmarksPage /></MemoryRouter>)

    expect(await screen.findByText('42.5 tok/s')).toBeInTheDocument()
    expect(screen.getByText('31.2 tok/s')).toBeInTheDocument()
    expect(screen.getAllByText('1,000 tokens')).toHaveLength(2)
    expect(screen.getByText('NVFP4')).toBeInTheDocument()
    expect(screen.getByText('Q4_K_M')).toBeInTheDocument()
    expect(screen.getByText('12 contributors')).toBeInTheDocument()
    expect(screen.getByText(/matched only on model name, quantization, and prompt-length bucket/)).toBeInTheDocument()
    expect(screen.queryByText('vLLM')).not.toBeInTheDocument()
    expect(screen.queryByText(/hardware class/i)).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Review & enable' }))
    const dialog = screen.getByRole('dialog', { name: 'Enable community sharing?' })
    expect(dialog).toHaveTextContent('Benchmark JSON: canonical model identifier, quantization, prompt-length/context-occupancy bucket, measured inference tok/s')
    expect(dialog).toHaveTextContent('only samples captured after you enable sharing can be uploaded; existing benchmark history stays local')
    expect(dialog).toHaveTextContent('stable opaque telemetry cluster ID')
    expect(dialog).toHaveTextContent('contains no account ID, hostname, node name, or endpoint alias')
    expect(dialog).toHaveTextContent('Never in benchmark JSON: prompts or outputs')
    expect(dialog).not.toHaveTextContent('tensor-parallel (TP) size')
    expect(dialog).toHaveTextContent('ordinary authenticated request and network metadata')
    await user.click(within(dialog).getByRole('button', { name: /I understand, enable sharing/ }))
    expect(await screen.findByText('Sharing enabled')).toBeInTheDocument()
    expect(communityAccess.reload).toHaveBeenCalled()
  })

  it('refreshes model summaries and local aggregates after deleting a contributing benchmark', async () => {
    let deleted = false
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      let body: unknown
      if (path.endsWith('/api/v1/benchmark-models')) {
        body = { items: deleted ? [] : [{
          model_id: 'org/model', run_count: 1,
          best_prompt_tokens_per_second: 100, best_generation_tokens_per_second: 42.5,
          context_windows: [8192], tensor_parallel_sizes: [1],
          latest_at: '2026-08-26T00:00:00Z',
        }] }
      } else if (path.includes('/api/v1/benchmark-history/models')) {
        if (init?.method === 'DELETE') {
          deleted = true
          body = { ok: true }
        } else {
          body = {
            items: deleted ? [] : [{
              id: 'sample-1', created_at: '2026-08-26T00:00:00Z',
              model: { repository: 'org/model' }, runtime: 'vllm',
              configuration: { context_length: 8192 }, latency_ms: 100,
              generation_tokens_per_second: 42.5, eligible_for_community: true,
            }],
            total: deleted ? 0 : 1, limit: 100, offset: 0,
          }
        }
      } else if (path.endsWith('/api/v1/community/aggregates')) {
        body = {
          items: deleted ? [] : [{
            model_id: 'org/model', quantization: 'NVFP4', prompt_tokens_bucket: 1000,
            inference_tokens_per_second: 42.5, sample_count: 1, unique_cluster_count: 1,
          }],
          availability: deleted ? 'not_configured' : 'local',
          evidence_policy: { minimum_samples: 10, exact_match_dimensions: ['model_id', 'quantization', 'prompt_tokens_bucket'], metric: 'inference_tokens_per_second' },
        }
      } else {
        body = { consent: false, pairing: { status: 'not_paired' }, outbox: {} }
      }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }))
    const user = userEvent.setup()

    render(<MemoryRouter><BenchmarksPage /></MemoryRouter>)

    const modelList = await screen.findByLabelText('Benchmarked models')
    const explorerModel = within(modelList).getByRole('button', { name: /org\/model/ })
    expect(explorerModel).toBeInTheDocument()
    expect((await screen.findAllByText('42.5 tok/s')).length).toBeGreaterThan(0)
    await user.click(screen.getByRole('button', { name: 'Delete all benchmarks for org/model' }))

    expect(await screen.findByText('No coordinated benchmark runs yet')).toBeInTheDocument()
    expect(explorerModel).not.toBeInTheDocument()
    expect(await screen.findByText('No community estimates yet')).toBeInTheDocument()
    expect(screen.queryByText('42.5 tok/s')).not.toBeInTheDocument()
  })

  it('locks community estimates behind sign-in and telemetry opt-in but keeps the consent path usable', async () => {
    Object.assign(communityAccess, { signedIn: true, sharingEnabled: false, enabled: false })
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      let body: unknown
      if (path.includes('/api/v1/benchmark-history/models')) {
        body = { items: [], total: 0, limit: 100, offset: 0 }
      } else if (path.endsWith('/api/v1/community/aggregates')) {
        body = {
          items: [{
            model_id: 'org/model', quantization: 'NVFP4', prompt_tokens_bucket: 1000,
            inference_tokens_per_second: 42.5, sample_count: 12, unique_cluster_count: 5,
          }],
          availability: 'available',
          evidence_policy: { minimum_samples: 10, exact_match_dimensions: ['model_id', 'quantization', 'prompt_tokens_bucket'], metric: 'inference_tokens_per_second' },
        }
      } else {
        body = { consent: false, pairing: { status: 'not_paired' }, outbox: {} }
      }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<MemoryRouter><BenchmarksPage /></MemoryRouter>)

    expect(await screen.findByText('Community estimates are locked')).toBeInTheDocument()
    expect(screen.getByText('Enable telemetry under Settings → Community Features to see community data.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Review sharing' })).toBeInTheDocument()
    expect(screen.queryByText('42.5 tok/s')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Review & enable' })).toBeInTheDocument()
    expect(fetchMock.mock.calls.some(([input]) => (
      String(input).endsWith('/api/v1/community/aggregates')
    ))).toBe(false)
  })

  it('renders local history rows with unmeasured speed or TTFT as placeholders', async () => {
    // Proxied captures can be recorded without timing data (ttft_ms and
    // generation_tps stay NULL in storage), and the wire payload then carries
    // JSON null instead of a number.
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      let body: unknown
      if (path.endsWith('/api/v1/benchmark-models')) body = { items: [] }
      else if (path.includes('/api/v1/benchmark-history/models')) body = {
        items: [{
          id: 'sample-unmeasured', created_at: '2026-08-26T00:00:00Z',
          model: { repository: 'org/model' }, runtime: 'vllm',
          configuration: {}, latency_ms: 100, ttft_ms: null,
          generation_tokens_per_second: null, eligible_for_community: false,
          sample_count: 2,
        }],
      }
      else if (path.endsWith('/api/v1/community/aggregates')) body = {
        items: [], availability: 'not_configured',
        evidence_policy: { minimum_samples: 10, exact_match_dimensions: [], metric: 'inference_tokens_per_second' },
      }
      else body = { consent: false, pairing: { status: 'not_paired' }, outbox: {} }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }))

    render(<MemoryRouter><BenchmarksPage /></MemoryRouter>)

    const history = await screen.findByRole('table', { name: 'Local benchmark history' })
    expect(within(history).getAllByText('org/model')).toHaveLength(1)
    expect(within(history).getByText(/2 saved results/)).toBeInTheDocument()
    expect(within(history).queryByText('local-model')).not.toBeInTheDocument()
    expect(within(history).getAllByText('—').length).toBeGreaterThanOrEqual(2)
  })

  it('surfaces failed worker consent withdrawal and lets the user retry it', async () => {
    let consent = true
    let withdrawalAttempts = 0
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      let body: unknown
      if (path.includes('/api/v1/benchmark-history/models')) {
        body = { items: [], total: 0, limit: 100, offset: 0 }
      } else if (path.endsWith('/api/v1/community/aggregates')) {
        body = {
          items: [], availability: 'available',
          evidence_policy: { minimum_samples: 10, exact_match_dimensions: [], metric: 'inference_tokens_per_second' },
        }
      } else if (path.endsWith('/api/v1/community/consent') && init?.method === 'PUT') {
        consent = false
        withdrawalAttempts += 1
        body = {
          cluster: {
            applied: ['Spark Two'], conflicts: [],
            errors: withdrawalAttempts === 1 ? ['Spark Three: unreachable'] : [],
          },
        }
      } else {
        body = {
          consent, pairing: { status: 'paired' }, upload_configured: true,
          outbox: { pending: 0, synced: 0, failed: 0 },
        }
      }
      return new Response(JSON.stringify(body), {
        status: 200, headers: { 'Content-Type': 'application/json' },
      })
    }))
    const user = userEvent.setup()

    render(<MemoryRouter><BenchmarksPage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'Turn off' }))

    expect(await screen.findByText('Sharing off')).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent('Spark Three: unreachable')
    const retry = screen.getByRole('button', { name: 'Retry turn off everywhere' })
    await user.click(retry)

    expect(withdrawalAttempts).toBe(2)
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
})
