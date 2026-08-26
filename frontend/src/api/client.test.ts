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
})
