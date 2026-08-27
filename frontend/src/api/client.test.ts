import { afterEach, describe, expect, it, vi } from 'vitest'
import { api, setAuthTokenProvider } from './client'

afterEach(() => {
  setAuthTokenProvider(undefined)
  vi.restoreAllMocks()
})

describe('API client adapters', () => {
  it('awaits an async bearer provider before protected community requests', async () => {
    let resolveToken: ((token: string) => void) | undefined
    setAuthTokenProvider(() => new Promise<string>((resolve) => {
      resolveToken = resolve
    }))
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({
      items: [], availability: 'available',
      evidence_policy: { minimum_samples: 10, exact_match_dimensions: [], metric: 'inference_tokens_per_second' },
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)

    const request = api.benchmarks.aggregates()
    await Promise.resolve()
    expect(fetchMock).not.toHaveBeenCalled()
    resolveToken?.('refreshed-token')
    await request

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/community/aggregates',
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: 'Bearer refreshed-token' }),
      }),
    )
  })

  it('keeps consent withdrawal independent from Cognito token refresh', async () => {
    const tokenProvider = vi.fn<() => Promise<string>>().mockRejectedValue(new Error('Cognito unavailable'))
    setAuthTokenProvider(tokenProvider)
    const fetchMock = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        consent: false, pairing: { status: 'paired' }, outbox: {},
      }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.benchmarks.setConsent(false)).resolves.toEqual(expect.objectContaining({
      sharing_enabled: false,
      account_paired: true,
    }))

    expect(tokenProvider).not.toHaveBeenCalled()
    expect(fetchMock.mock.calls.map(([path]) => path)).toEqual([
      '/api/v1/community/consent',
      '/api/v1/community/sync',
    ])
    expect(fetchMock.mock.calls.every(([, init]) => (
      !('Authorization' in ((init?.headers ?? {}) as Record<string, string>))
    ))).toBe(true)
  })

  it('uses bearer lookup only for the protected unpair operation', async () => {
    const tokenProvider = vi.fn().mockResolvedValue('current-token')
    setAuthTokenProvider(tokenProvider)
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({
      pairing: { status: 'not_paired' }, cluster: { applied: [], conflicts: [], errors: [] },
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)

    await api.community.unpair()

    expect(tokenProvider).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/community/pair', expect.objectContaining({
      method: 'DELETE',
      headers: expect.objectContaining({ Authorization: 'Bearer current-token' }),
    }))
  })
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

  it('preserves aggregate evidence dimensions without adding device or runtime fields', async () => {
    const response = {
      items: [{ model_id: 'org/model', context_window_size: 8192, inference_tokens_per_second: 28.5, sample_count: 11 }],
      availability: 'available',
      evidence_policy: { minimum_samples: 10, exact_match_dimensions: ['model_id', 'context_window_size'], metric: 'inference_tokens_per_second' },
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify(response), { status: 200, headers: { 'Content-Type': 'application/json' } })))

    await expect(api.benchmarks.aggregates()).resolves.toEqual(response)
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

  it('sends selected node IDs and mode when creating a deployment', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({
      id: 'dep-2', alias: 'distributed', runtime: 'vllm', kind: 'managed',
      model: { repository: 'org/model' }, status: 'starting', settings: {},
      node_ids: ['local', 'spark-2'],
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)

    await api.deployments.create({
      alias: 'distributed', model_id: 'org/model', runtime: 'vllm', managed: true,
      settings: { context_length: 16384 }, node_ids: ['local', 'spark-2'], deployment_mode: 'replicated',
    })

    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))).toEqual(expect.objectContaining({
      node_ids: ['local', 'spark-2'],
      deployment_mode: 'replicated',
    }))
  })

  it('keeps llama.cpp local in the UI without sending cluster target fields', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({
      id: 'dep-llama', alias: 'local-gguf', runtime: 'llama.cpp', kind: 'managed',
      model: { repository: 'models/local.gguf' }, status: 'registered', settings: {},
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)

    await api.deployments.create({
      alias: 'local-gguf', model_id: 'models/local.gguf', runtime: 'llama.cpp', managed: true,
      settings: { context_length: 8192, gpu_layers: 99 }, node_ids: ['local'], deployment_mode: 'single',
    })

    const body = JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body)) as Record<string, unknown>
    expect(body).not.toHaveProperty('node_ids')
    expect(body).not.toHaveProperty('selected_nodes')
    expect(body).not.toHaveProperty('deployment_mode')
  })

  it('sends selected node IDs when pulling an image', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({
      ok: true, image: 'vllm/vllm-openai:v1', node_ids: ['local', 'spark-2'], results: [],
    }), { status: 201, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)

    await api.images.pull('vllm/vllm-openai:v1', ['local', 'spark-2'])

    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))).toEqual({
      image: 'vllm/vllm-openai:v1',
      node_ids: ['local', 'spark-2'],
    })
  })

  it('encodes the node ID and sends only the validated name when renaming', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({
      id: 'spark/2', name: 'Render Spark', online: true,
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.nodes.rename('spark/2', { name: 'Render Spark' })).resolves.toEqual(
      expect.objectContaining({ id: 'spark/2', name: 'Render Spark' }),
    )
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/nodes/spark%2F2', expect.objectContaining({
      method: 'PATCH',
      body: JSON.stringify({ name: 'Render Spark' }),
    }))
  })

  it('preserves remote image availability and defaults legacy inventory to local', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({ items: [
      { id: 'remote-image', repository: 'org/remote', node_ids: ['spark-2'], selected_nodes: [{ id: 'spark-2', name: 'Studio Spark' }] },
      { id: 'legacy-image', repository: 'org/legacy' },
    ] }), { status: 200, headers: { 'Content-Type': 'application/json' } })))

    await expect(api.images.list()).resolves.toEqual([
      expect.objectContaining({ id: 'remote-image', node_ids: ['spark-2'], selected_nodes: [{ id: 'spark-2', name: 'Studio Spark' }] }),
      expect.objectContaining({ id: 'legacy-image', node_ids: ['local'] }),
    ])
  })

  it('parses legacy log levels and defaults unstructured lines to info', async () => {
    const fetchMock = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(new Response('{}', { status: 404, statusText: 'Not Found', headers: { 'Content-Type': 'application/json' } }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        logs: [
          '12:34:56 WARNING  sparkdeck.worker  request throttled',
          'plain legacy message',
        ],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.logs.list()).resolves.toEqual([
      {
        timestamp: '12:34:56',
        level: 'warning',
        message: 'sparkdeck.worker  request throttled',
      },
      { level: 'info', message: 'plain legacy message' },
    ])
  })

  it('loads persisted lifetime and hourly usage from the existing controller APIs', async () => {
    const fetchMock = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(new Response(JSON.stringify({ models: {}, groups: [], total: { input: 0, output: 0, cached: 0, requests: 0 } }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
      .mockResolvedValueOnce(new Response(JSON.stringify([{ hour: '2026-08-26T10', input: 10, output: 4, cached: 3, requests: 1 }]), { status: 200, headers: { 'Content-Type': 'application/json' } }))
      .mockResolvedValueOnce(new Response(JSON.stringify([{ date: '2026-08-26', input: 10, output: 4, cached: 3, requests: 1 }]), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)

    await api.usage.get()
    await api.usage.analysis('2026-08-01', '2026-08-26')

    expect(fetchMock.mock.calls.map(([path]) => String(path))).toEqual([
      '/api/token-stats',
      '/api/token-stats/hourly?start=2026-08-01&end=2026-08-26',
      '/api/token-stats/daily?start=2026-08-01&end=2026-08-26',
    ])
  })
})
