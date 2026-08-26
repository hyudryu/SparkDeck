import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from './client'

afterEach(() => vi.restoreAllMocks())

describe('API client adapters', () => {
  it('normalizes deployment wire records for the UI', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({
      items: [{
        id: 'dep-1',
        alias: 'reasoner',
        runtime: 'llama.cpp',
        kind: 'managed',
        model: { repository: 'org/model', quantization: 'Q4_K_M' },
        status: 'running',
        settings: { context_length: 4096, parallel_slots: 2 },
      }],
    }), { status: 200, headers: { 'Content-Type': 'application/json' } })))

    await expect(api.deployments.list()).resolves.toEqual([
      expect.objectContaining({
        id: 'dep-1',
        model_id: 'org/model',
        runtime: 'llama.cpp',
        managed: true,
        settings: expect.objectContaining({ quantization: 'Q4_K_M', parallel_slots: 2 }),
      }),
    ])
  })

  it('uses the backend consent wire shape', async () => {
    const fetchMock = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ consent: true, pairing: { status: 'paired' }, outbox: { pending: 2, synced: 7 } }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)

    const status = await api.benchmarks.setConsent(true)
    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))).toEqual({ enabled: true })
    expect(status).toEqual(expect.objectContaining({ sharing_enabled: true, account_paired: true, pending_count: 2, synced_count: 7 }))
  })

  it('normalizes the refreshed deployment returned by an action', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({
      id: 'dep-1', alias: 'reasoner', runtime: 'vllm', kind: 'managed',
      model: { repository: 'org/model' }, status: 'running', settings: {},
    }), { status: 200, headers: { 'Content-Type': 'application/json' } })))

    await expect(api.deployments.action('dep-1', 'start')).resolves.toEqual(
      expect.objectContaining({ id: 'dep-1', model_id: 'org/model', status: 'running' }),
    )
  })
})
