import { cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { BenchmarksPage } from './BenchmarksPage'

const communityAccess = vi.hoisted(() => ({ signedIn: true, sharingEnabled: true, loading: false, enabled: true, reload: vi.fn() }))

vi.mock('../hooks/useCommunityAccess', () => ({
  COMMUNITY_ACCESS_HINT: 'Sign in and enable telemetry in Settings → Community Features to see community data.',
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
  it('discloses the exact shared fields and renders only contract-safe estimates', async () => {
    let consent = false
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      let body: unknown
      if (path.includes('/api/v1/benchmarks')) body = { items: [], total: 0, limit: 100, offset: 0 }
      else if (path.endsWith('/api/v1/community/aggregates')) body = {
        items: [{ model_id: 'org/model', context_window_size: 16384, inference_tokens_per_second: 42.5, sample_count: 12 }],
        availability: 'available',
        evidence_policy: { minimum_samples: 10, exact_match_dimensions: ['model_id', 'context_window_size'], metric: 'inference_tokens_per_second' },
      }
      else if (path.endsWith('/api/v1/community/consent') && init?.method === 'PUT') {
        consent = (JSON.parse(String(init.body)) as { enabled: boolean }).enabled
        body = {}
      } else body = { consent, pairing: { status: 'paired' }, outbox: { pending: 0, synced: 2 } }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<MemoryRouter><BenchmarksPage /></MemoryRouter>)

    expect(await screen.findByText('42.5 tok/s')).toBeInTheDocument()
    expect(screen.getByText('16,384 tokens')).toBeInTheDocument()
    expect(screen.getByText('12 samples')).toBeInTheDocument()
    expect(screen.getByText(/matched only on model name and context window/)).toBeInTheDocument()
    expect(screen.queryByText('vLLM')).not.toBeInTheDocument()
    expect(screen.queryByText(/hardware class/i)).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Review & enable' }))
    const dialog = screen.getByRole('dialog', { name: 'Enable community sharing?' })
    expect(dialog).toHaveTextContent('Only shared: model name, context window size, and measured inference tok/s.')
    expect(dialog).toHaveTextContent('Not shared: prompts or outputs, runtime, revision, quantization, hardware, settings, host or network identity, or paths.')
    await user.click(within(dialog).getByRole('button', { name: /I understand, enable sharing/ }))
    expect(await screen.findByText('Sharing enabled')).toBeInTheDocument()
    expect(communityAccess.reload).toHaveBeenCalled()
  })

  it('refreshes local aggregates after deleting a contributing benchmark', async () => {
    let deleted = false
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      let body: unknown
      if (path.includes('/api/v1/benchmarks')) {
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
          items: deleted ? [] : [{ model_id: 'org/model', context_window_size: 8192, inference_tokens_per_second: 42.5, sample_count: 1 }],
          availability: deleted ? 'not_configured' : 'local',
          evidence_policy: { minimum_samples: 10, exact_match_dimensions: ['model_id', 'context_window_size'], metric: 'inference_tokens_per_second' },
        }
      } else {
        body = { consent: false, pairing: { status: 'not_paired' }, outbox: {} }
      }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }))
    const user = userEvent.setup()

    render(<MemoryRouter><BenchmarksPage /></MemoryRouter>)

    expect((await screen.findAllByText('42.5 tok/s')).length).toBeGreaterThan(0)
    await user.click(screen.getByRole('button', { name: 'Delete benchmark for org/model' }))

    expect(await screen.findByText('No community estimates yet')).toBeInTheDocument()
    expect(screen.queryByText('42.5 tok/s')).not.toBeInTheDocument()
  })

  it('locks community estimates behind sign-in and telemetry opt-in but keeps the consent path usable', async () => {
    Object.assign(communityAccess, { signedIn: false, sharingEnabled: false, enabled: false })
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      let body: unknown
      if (path.includes('/api/v1/benchmarks')) {
        body = { items: [], total: 0, limit: 100, offset: 0 }
      } else if (path.endsWith('/api/v1/community/aggregates')) {
        body = {
          items: [{ model_id: 'org/model', context_window_size: 16384, inference_tokens_per_second: 42.5, sample_count: 12 }],
          availability: 'available',
          evidence_policy: { minimum_samples: 10, exact_match_dimensions: ['model_id', 'context_window_size'], metric: 'inference_tokens_per_second' },
        }
      } else {
        body = { consent: false, pairing: { status: 'not_paired' }, outbox: {} }
      }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }))

    render(<MemoryRouter><BenchmarksPage /></MemoryRouter>)

    expect(await screen.findByText('Community estimates are locked')).toBeInTheDocument()
    expect(screen.getByText('Sign in and enable telemetry in Settings → Community Features to see community data.')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open community settings' })).toHaveAttribute('href', '/settings')
    expect(screen.queryByText('42.5 tok/s')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Review & enable' })).toBeInTheDocument()
  })
})
