import { cleanup, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ExplorePage } from './ExplorePage'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('ExplorePage community estimates', () => {
  it('pairs an inference-speed estimate with its context and avoids runtime or hardware claims', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({
      items: [{
        id: 'org/model', author: 'org', downloads: 1000, likes: 12, runtime_compatibility: [],
        community: { model_id: 'org/model', context_window_size: 8192, inference_tokens_per_second: 31.25, sample_count: 14 },
      }],
      total: 1,
      next_cursor: null,
    }), { status: 200, headers: { 'Content-Type': 'application/json' } })))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)

    const estimate = await screen.findByLabelText('Community inference-speed estimate for org/model')
    expect(within(estimate).getByText('31.3 tok/s')).toBeInTheDocument()
    expect(within(estimate).getByText(/8,192-token context window/)).toBeInTheDocument()
    expect(within(estimate).getByText(/estimate, not a guarantee/)).toBeInTheDocument()
    expect(within(estimate).queryByText(/hardware|runtime/i)).not.toBeInTheDocument()
  })
})
