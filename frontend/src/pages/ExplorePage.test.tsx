import { cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ExplorePage } from './ExplorePage'

const communityAccess = vi.hoisted(() => ({ signedIn: true, sharingEnabled: true, loading: false, enabled: true, reload: () => undefined }))

vi.mock('../hooks/useCommunityAccess', () => ({
  communityAccessHint: (signedIn: boolean) => signedIn
    ? 'Enable telemetry under Settings → Community Features to see community data.'
    : 'Sign in under Settings → Community Features to see community data.',
  useCommunityAccess: () => communityAccess,
}))

const gib = 1024 ** 3

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

beforeEach(() => {
  Object.assign(communityAccess, { signedIn: true, sharingEnabled: true, loading: false, enabled: true })
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('ExplorePage model rows', () => {
  it('reveals community evidence only after expansion and describes other SparkDeck users', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({
        items: [{
          id: 'org/model', author: 'org', name: 'model', downloads: 1000, likes: 12,
          parameter_count: 7_000_000_000, weight_size_bytes: 14 * gib,
          runtime_compatibility: [],
          community: { model_id: 'org/model', context_window_size: 8192, inference_tokens_per_second: 31.25, sample_count: 14 },
        }],
        total: 1,
      })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [{
        id: 'local', name: 'Spark', online: true, docker_ready: true, selectable: true,
        stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] },
      }] })
      return json({ items: [], availability: 'not_configured', evidence_policy: {} })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)

    const row = await screen.findByRole('button', { name: 'Expand org/model' })
    expect(row).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByText('Sampled from other SparkDeck users')).not.toBeInTheDocument()
    expect(screen.queryByText(/No estimate is available/)).not.toBeInTheDocument()

    await user.click(row)

    const estimate = screen.getByLabelText('Community inference-speed estimate for org/model')
    expect(within(estimate).getByText('Sampled from other SparkDeck users')).toBeInTheDocument()
    expect(within(estimate).getByText('31.3 tok/s')).toBeInTheDocument()
    expect(within(estimate).getByText(/8,192-token context window/)).toBeInTheDocument()
    expect(within(estimate).getByText(/estimate, not a guarantee/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Deploy org/model' })).toHaveAttribute('href', '/models?model=org%2Fmodel')
  })

  it('color codes cluster fit and filters fitting models largest first', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({ items: [
        { id: 'org/easy', name: 'easy', parameter_count: 100_000_000_000, weight_size_bytes: 200 * gib, downloads: 30, likes: 3, runtime_compatibility: [], community: { model_id: 'org/easy', context_window_size: 8192, inference_tokens_per_second: 40, sample_count: 10 } },
        { id: 'zai-org/GLM-5.3-Flash', name: 'GLM-5.3-Flash', parameter_count: 321_300_000_000, weight_size_bytes: 306 * gib, downloads: 20, likes: 2, runtime_compatibility: [] },
        { id: 'org/tight', name: 'tight', parameter_count: 400_000_000_000, weight_size_bytes: 400 * gib, downloads: 15, likes: 1, runtime_compatibility: [] },
        { id: 'org/large', name: 'large', parameter_count: 600_000_000_000, weight_size_bytes: 600 * gib, downloads: 10, likes: 1, runtime_compatibility: [] },
        { id: 'org/unknown', name: 'unknown', downloads: 5, likes: 0, runtime_compatibility: [] },
      ], total: 5 })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [
        { id: 'local', name: 'Spark One', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] } },
        { id: 'node-2', name: 'Spark Two', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] } },
        { id: 'node-3', name: 'Spark Three', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] } },
        { id: 'node-4', name: 'Spark Four', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] } },
        { id: 'offline', name: 'Offline', online: false, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 500 * 1024 }] } },
      ] })
      return json({ items: [], availability: 'not_configured', evidence_policy: {} })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)

    expect(await screen.findByRole('button', { name: 'Expand org/easy' })).toBeInTheDocument()
    expect(screen.getByText('200 GB').closest('.catalog-model-size')).toHaveClass('fit-easy')
    expect(screen.getByText('306 GB').closest('.catalog-model-size')).toHaveClass('fit-easy')
    expect(screen.getByText('400 GB').closest('.catalog-model-size')).toHaveClass('fit-tight')
    expect(screen.getByText('600 GB').closest('.catalog-model-size')).toHaveClass('fit-no-fit')
    expect(screen.getByText('512 GB aggregate sharded memory across 4 measured nodes')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Expand zai-org/GLM-5.3-Flash' }))
    const fitDetails = screen.getByText(/Fit assumes a sharded deployment/)
    expect(fitDetails).toHaveTextContent('512 GB aggregate memory across 4 measured nodes')
    expect(fitDetails).toHaveTextContent('replicated deployments still require the full model weights')
    expect(screen.getByRole('link', { name: 'Deploy zai-org/GLM-5.3-Flash' })).toHaveAttribute(
      'href', '/models?model=zai-org%2FGLM-5.3-Flash&layout=sharded',
    )

    await user.click(screen.getByRole('checkbox', { name: /Only what fits/ }))

    const visibleRows = screen.getAllByRole('button', { name: /^(Expand|Collapse) (org|zai-org)\// })
    expect(visibleRows.map((row) => row.getAttribute('aria-label'))).toEqual([
      'Expand org/tight',
      'Collapse zai-org/GLM-5.3-Flash',
      'Expand org/easy',
    ])
    expect(screen.queryByRole('button', { name: 'Expand org/large' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Expand org/unknown' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('checkbox', { name: /Only with community data/ }))
    expect(screen.getByRole('button', { name: 'Expand org/easy' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Expand org/tight' })).not.toBeInTheDocument()
  })

  it('does not pool worker memory when the controller cannot join a sharded deployment', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({ items: [{
        id: 'org/worker-pool-only', name: 'worker-pool-only', weight_size_bytes: 200 * gib,
        downloads: 1, likes: 0, runtime_compatibility: [],
      }], total: 1 })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [
        { id: 'local', name: 'Controller', local: true, online: true, docker_ready: false, selectable: false, stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] } },
        { id: 'node-2', name: 'Worker Two', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] } },
        { id: 'node-3', name: 'Worker Three', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] } },
      ] })
      return json({ items: [], availability: 'not_configured', evidence_policy: {} })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)

    const expand = await screen.findByRole('button', { name: 'Expand org/worker-pool-only' })
    expect(screen.getByText('200 GB').closest('.catalog-model-size')).toHaveClass('fit-no-fit')
    expect(screen.getByText('128 GB largest per-node memory across 2 measured nodes')).toBeInTheDocument()
    await user.click(expand)
    expect(screen.getByRole('link', { name: 'Deploy org/worker-pool-only' })).toHaveAttribute(
      'href', '/models?model=org%2Fworker-pool-only',
    )

    await user.click(screen.getByRole('checkbox', { name: /Only what fits/ }))
    expect(screen.queryByRole('button', { name: 'Expand org/worker-pool-only' })).not.toBeInTheDocument()
  })

  it('lists aggregated benchmark models in the community tab without claiming live tracking', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({ detail: 'Hugging Face unavailable' }, 503)
      if (path.endsWith('/api/v1/nodes')) return json({ items: [] })
      return json({
        items: [{ model_id: 'community/model', context_window_size: 32768, inference_tokens_per_second: 18.5, sample_count: 22 }],
        availability: 'local',
        evidence_policy: {},
      })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)
    await user.click(await screen.findByRole('tab', { name: 'Community Run Models' }))

    expect(await screen.findByRole('button', { name: 'Expand community/model (unknown)' })).toBeInTheDocument()
    expect(screen.queryByText('Hugging Face unavailable')).not.toBeInTheDocument()
    expect(screen.getByText('Based on aggregated benchmark samples—not live session tracking.')).toBeInTheDocument()
    expect(screen.getByRole('checkbox', { name: /Only with community data/ })).toBeChecked()
    await user.click(screen.getByRole('button', { name: 'Expand community/model (unknown)' }))
    expect(screen.getByText('Aggregated from benchmarks on this controller')).toBeInTheDocument()
    expect(screen.queryByText('Sampled from other SparkDeck users')).not.toBeInTheDocument()
  })

  it('pages large community result sets instead of rendering every model at once', async () => {
    const user = userEvent.setup()
    const items = Array.from({ length: 120 }, (_, index) => ({
      model_id: `community/model-${index}`,
      context_window_size: 8192,
      inference_tokens_per_second: 20 + index,
      sample_count: 10,
    }))
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({ items: [], total: 0 })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [] })
      return json({ items, availability: 'available', evidence_policy: {} })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)
    await user.click(await screen.findByRole('tab', { name: 'Community Run Models' }))

    expect(await screen.findAllByRole('button', { name: /^Expand community\/model-/ })).toHaveLength(50)
    const loadMore = screen.getByRole('button', { name: 'Load more community models (70 remaining)' })
    await user.click(loadMore)

    expect(screen.getAllByRole('button', { name: /^Expand community\/model-/ })).toHaveLength(100)
    expect(screen.getByRole('button', { name: 'Load more community models (20 remaining)' })).toBeInTheDocument()
  })

  it('keeps quantization variants separate and shows community-specific columns and artifacts', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models?')) return json({ items: [{
        id: 'RadixArk/Qwen3.8-27B', name: 'Qwen3.8-27B', parameter_count: 27_000_000_000,
        weight_size_bytes: 16 * gib, downloads: 9000, likes: 400, runtime_compatibility: [],
      }], total: 1 })
      if (path.includes('/api/v1/catalog/models/')) return json({ model: {
        id: 'RadixArk/Qwen3.8-27B', name: 'Qwen3.8-27B', runtime_compatibility: [],
        quantizations: [
          { name: 'NVFP4', weight_size_bytes: 16 * gib, files: [{ filename: 'model-nvfp4.safetensors', size_bytes: 16 * gib }] },
          { name: 'Q4_K_M', weight_size_bytes: 15 * gib, files: [{ filename: 'qwen3.8-q4_k_m.gguf', size_bytes: 15 * gib }] },
        ],
      }, aggregates: [] })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [] })
      return json({
        items: [
          { model_id: 'RadixArk/Qwen3.8-27B', quantization: 'NVFP4', context_window_size: 8192, inference_tokens_per_second: 52.4, sample_count: 30, unique_cluster_count: 7, parameter_count: 27_000_000_000, weight_size_bytes: 16 * gib },
          { model_id: 'RadixArk/Qwen3.8-27B', quantization: 'Q4_K_M', context_window_size: 8192, inference_tokens_per_second: 31.2, sample_count: 20, unique_cluster_count: 4, parameter_count: 27_000_000_000, weight_size_bytes: 15 * gib },
        ],
        availability: 'available', evidence_policy: {},
      })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)
    await user.click(await screen.findByRole('tab', { name: 'Community Run Models' }))

    const header = document.querySelector('.catalog-model-header')
    expect(header).toHaveTextContent('Output speed')
    expect(header).toHaveTextContent('Unique clusters')
    expect(screen.queryByText('Downloads')).not.toBeInTheDocument()
    expect(screen.queryByText('Likes')).not.toBeInTheDocument()
    const nvfp4 = await screen.findByRole('button', { name: 'Expand RadixArk/Qwen3.8-27B (NVFP4)' })
    const gguf = screen.getByRole('button', { name: 'Expand RadixArk/Qwen3.8-27B (Q4_K_M)' })
    expect(within(nvfp4).getByText('52.4 tok/s')).toBeInTheDocument()
    expect(within(nvfp4).getByText('7')).toBeInTheDocument()
    expect(within(gguf).getByText('31.2 tok/s')).toBeInTheDocument()
    expect(within(gguf).getByText('4')).toBeInTheDocument()

    await user.click(nvfp4)
    expect(await screen.findByText('Available quantizations and artifacts')).toBeInTheDocument()
    expect(screen.getByText('model-nvfp4.safetensors')).toBeInTheDocument()
    expect(screen.getByText('qwen3.8-q4_k_m.gguf')).toBeInTheDocument()
  })

  it('locks community features behind sign-in and telemetry opt-in', async () => {
    Object.assign(communityAccess, { signedIn: false, sharingEnabled: false, enabled: false })
    const user = userEvent.setup()
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({
        items: [{
          id: 'org/model', author: 'org', name: 'model', downloads: 1000, likes: 12,
          parameter_count: 7_000_000_000, weight_size_bytes: 14 * gib,
          runtime_compatibility: [],
          community: { model_id: 'org/model', context_window_size: 8192, inference_tokens_per_second: 31.25, sample_count: 14 },
        }],
        total: 1,
      })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [] })
      return json({ items: [], availability: 'not_configured', evidence_policy: {} })
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)

    const tab = await screen.findByRole('tab', { name: 'Community Run Models' })
    expect(tab).toBeDisabled()
    expect(tab).toHaveAttribute('title', 'Sign in under Settings → Community Features to see community data.')
    expect(screen.getByRole('checkbox', { name: /Only with community data/ })).toBeDisabled()

    await user.click(await screen.findByRole('button', { name: 'Expand org/model' }))

    expect(screen.queryByLabelText('Community inference-speed estimate for org/model')).not.toBeInTheDocument()
    expect(fetchMock.mock.calls.some(([input]) => String(input).endsWith('/api/v1/community/aggregates'))).toBe(false)
  })

  it('does not request hosted aggregates for a signed-in user who opted out', async () => {
    Object.assign(communityAccess, { signedIn: true, sharingEnabled: false, enabled: false })
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({ items: [], total: 0 })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [] })
      throw new Error(`Unexpected request: ${path}`)
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)

    const tab = await screen.findByRole('tab', { name: 'Community Run Models' })
    expect(tab).toBeDisabled()
    expect(tab).toHaveAttribute('title', 'Enable telemetry under Settings → Community Features to see community data.')
    expect(fetchMock.mock.calls.some(([input]) => String(input).endsWith('/api/v1/community/aggregates'))).toBe(false)
  })
})
