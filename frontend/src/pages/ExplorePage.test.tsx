import { cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ExplorePage } from './ExplorePage'

const gib = 1024 ** 3

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

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
        { id: 'org/easy', name: 'easy', parameter_count: 40_000_000_000, weight_size_bytes: 80 * gib, downloads: 30, likes: 3, runtime_compatibility: [], community: { model_id: 'org/easy', context_window_size: 8192, inference_tokens_per_second: 40, sample_count: 10 } },
        { id: 'org/tight', name: 'tight', parameter_count: 60_000_000_000, weight_size_bytes: 120 * gib, downloads: 20, likes: 2, runtime_compatibility: [] },
        { id: 'org/large', name: 'large', parameter_count: 100_000_000_000, weight_size_bytes: 200 * gib, downloads: 10, likes: 1, runtime_compatibility: [] },
        { id: 'org/unknown', name: 'unknown', downloads: 5, likes: 0, runtime_compatibility: [] },
      ], total: 4 })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [
        { id: 'local', name: 'Spark One', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] } },
        { id: 'node-2', name: 'Spark Two', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] } },
      ] })
      return json({ items: [], availability: 'not_configured', evidence_policy: {} })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)

    expect(await screen.findByRole('button', { name: 'Expand org/easy' })).toBeInTheDocument()
    expect(screen.getByText('80 GB').closest('.catalog-model-size')).toHaveClass('fit-easy')
    expect(screen.getByText('120 GB').closest('.catalog-model-size')).toHaveClass('fit-tight')
    expect(screen.getByText('200 GB').closest('.catalog-model-size')).toHaveClass('fit-no-fit')
    expect(screen.getByText('128 GB largest per-node memory across 2 measured nodes')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Expand org/large' }))
    expect(screen.getByText(/128 GB on the largest measured node/)).toBeInTheDocument()
    expect(screen.getByText(/every replica must hold the full model weights/)).toBeInTheDocument()

    await user.click(screen.getByRole('checkbox', { name: /Only what fits/ }))

    const visibleRows = screen.getAllByRole('button', { name: /^Expand org\// })
    expect(visibleRows.map((row) => row.getAttribute('aria-label'))).toEqual([
      'Expand org/tight',
      'Expand org/easy',
    ])
    expect(screen.queryByRole('button', { name: 'Expand org/large' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Expand org/unknown' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('checkbox', { name: /Only with community data/ }))
    expect(screen.getByRole('button', { name: 'Expand org/easy' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Expand org/tight' })).not.toBeInTheDocument()
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

    expect(await screen.findByRole('button', { name: 'Expand community/model' })).toBeInTheDocument()
    expect(screen.queryByText('Hugging Face unavailable')).not.toBeInTheDocument()
    expect(screen.getByText('Based on aggregated benchmark samples—not live session tracking.')).toBeInTheDocument()
    expect(screen.getByRole('checkbox', { name: /Only with community data/ })).toBeChecked()
    await user.click(screen.getByRole('button', { name: 'Expand community/model' }))
    expect(screen.getByText('Aggregated from benchmarks on this controller')).toBeInTheDocument()
    expect(screen.queryByText('Sampled from other SparkDeck users')).not.toBeInTheDocument()
  })
})
