import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from './client'
import type { ChatStreamUpdate } from './types'

afterEach(() => {
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('API client adapters', () => {
  it('streams split reasoning, output, usage, and per-response metrics', async () => {
    const encoder = new TextEncoder()
    const payload = [
      'data: {"choices":[{"delta":{"reasoning":"Inspecting the request"}}],"usage":{"prompt_tokens":10,"completion_tokens":1}}\r\n\r\n',
      'data: {"choices":[{"delta":{"content":"Hello from the model"}}]}\n\n',
      'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":2}}}\n\n',
      'data: [DONE]\n\n',
    ].join('')
    const bytes = encoder.encode(payload)
    const response = new Response(new ReadableStream({
      start(controller) {
        controller.enqueue(bytes.slice(0, 17))
        controller.enqueue(bytes.slice(17, 79))
        controller.enqueue(bytes.slice(79))
      },
      cancel() {
        cancelled = true
      },
    }), { status: 200, headers: { 'Content-Type': 'text/event-stream' } })
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(response)
    vi.stubGlobal('fetch', fetchMock)
    const times = [0, 250, 250, 500, 1000, 1000]
    vi.spyOn(performance, 'now').mockImplementation(() => times.shift() ?? 1000)
    let cancelled = false
    const updates: ChatStreamUpdate[] = []

    const result = await api.chatStream('reasoner', [{ role: 'user', content: 'Hello' }], {
      onUpdate: (update) => updates.push(update),
    })

    expect(fetchMock).toHaveBeenCalledWith('/v1/chat/completions', expect.objectContaining({
      method: 'POST',
      headers: expect.objectContaining({ Accept: 'text/event-stream' }),
    }))
    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))).toEqual(expect.objectContaining({
      model: 'reasoner', stream: true, stream_options: { include_usage: true },
    }))
    expect(updates).toEqual(expect.arrayContaining([
      expect.objectContaining({ reasoning: 'Inspecting the request' }),
      expect.objectContaining({ content: 'Hello from the model' }),
    ]))
    expect(updates[0]?.metrics?.prompt_tokens_per_second).toBeUndefined()
    expect(cancelled).toBe(false)
    expect(result.message.content).toBe('Hello from the model')
    expect(result.reasoning).toBe('Inspecting the request')
    expect(result.metrics).toEqual(expect.objectContaining({
      prompt_tokens_per_second: 32,
      ttft_ms: 250,
      prompt_tokens: 10,
      completion_tokens: 4,
    }))
    expect(result.metrics.output_tokens_per_second).toBeCloseTo(5.33, 1)
    await response.body?.cancel()
    expect(cancelled).toBe(true)
  })

  it('surfaces an error sent inside an otherwise successful SSE response', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(new Response(
      'data: {"error":{"message":"GPU worker stopped"}}\n\ndata: [DONE]\n\n',
      { status: 200, headers: { 'Content-Type': 'text/event-stream' } },
    )))

    await expect(api.chatStream('reasoner', [])).rejects.toThrow('GPU worker stopped')
  })

  it('rejects non-SSE and truncated successful responses', async () => {
    const fetchMock = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(new Response('{"choices":[]}', {
        status: 200, headers: { 'Content-Type': 'application/json' },
      }))
      .mockResolvedValueOnce(new Response(
        'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
        { status: 200, headers: { 'Content-Type': 'text/event-stream' } },
      ))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.chatStream('reasoner', [])).rejects.toThrow('non-streaming response')
    await expect(api.chatStream('reasoner', [])).rejects.toThrow('ended unexpectedly')
  })

  it('does not reuse mutable API responses from the browser cache', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({ items: [] }), {
      status: 200, headers: { 'Content-Type': 'application/json' },
    }))
    vi.stubGlobal('fetch', fetchMock)

    await api.nodes.list()

    expect(fetchMock).toHaveBeenCalledWith('/api/v1/nodes', expect.objectContaining({
      cache: 'no-store',
    }))
  })

  it('turns an indefinitely pending API call into a retryable timeout error', async () => {
    vi.useFakeTimers()
    const fetchMock = vi.fn<typeof fetch>().mockImplementation((_input, init) => new Promise((_resolve, reject) => {
      init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true })
    }))
    vi.stubGlobal('fetch', fetchMock)

    const result = expect(api.deployments.list()).rejects.toMatchObject({
      status: 408,
      message: 'The request timed out. Check the node connection and retry.',
    })
    await vi.advanceTimersByTimeAsync(30_000)
    await result
  })

  it('keeps the timeout active while reading a JSON response body', async () => {
    vi.useFakeTimers()
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (_input, init) => ({
      ok: true,
      status: 200,
      headers: new Headers({ 'Content-Type': 'application/json' }),
      json: () => new Promise((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true })
      }),
    }) as Response)
    vi.stubGlobal('fetch', fetchMock)

    const result = expect(api.deployments.list()).rejects.toMatchObject({ status: 408 })
    await vi.advanceTimersByTimeAsync(30_000)
    await result
  })

  it('forwards caller cancellation while reading a JSON response body', async () => {
    const caller = new AbortController()
    let readingBody = false
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (_input, init) => ({
      ok: true,
      status: 200,
      headers: new Headers({ 'Content-Type': 'application/json' }),
      json: () => new Promise((_resolve, reject) => {
        readingBody = true
        init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true })
      }),
    }) as Response)
    vi.stubGlobal('fetch', fetchMock)

    const result = api.nodes.list(caller.signal)
    await vi.waitFor(() => expect(readingBody).toBe(true))
    caller.abort()

    await expect(result).rejects.toMatchObject({ name: 'AbortError' })
  })

  it('does not time out a long-running deployment launch', async () => {
    vi.useFakeTimers()
    let finishRequest: ((response: Response) => void) | undefined
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(() => new Promise((resolve) => {
      finishRequest = resolve
    }))
    vi.stubGlobal('fetch', fetchMock)

    const result = api.deployments.create({
      alias: 'slow-model', model_id: 'org/model', runtime: 'vllm', managed: true,
      settings: { context_length: 16_384 }, node_ids: ['local'], deployment_mode: 'single',
    })
    await vi.advanceTimersByTimeAsync(60_000)

    expect(fetchMock.mock.calls[0]?.[1]?.signal?.aborted).toBe(false)
    finishRequest?.(new Response(JSON.stringify({
      id: 'dep-slow', alias: 'slow-model', runtime: 'vllm', kind: 'managed',
      model: { repository: 'org/model' }, status: 'running', settings: {},
    }), { status: 201, headers: { 'Content-Type': 'application/json' } }))
    await expect(result).resolves.toEqual(expect.objectContaining({ id: 'dep-slow', status: 'running' }))
  })

  it('does not time out a long-running image pull', async () => {
    vi.useFakeTimers()
    let finishRequest: ((response: Response) => void) | undefined
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(() => new Promise((resolve) => {
      finishRequest = resolve
    }))
    vi.stubGlobal('fetch', fetchMock)

    const result = api.images.pull('vllm/vllm-openai:v1', ['local'])
    await vi.advanceTimersByTimeAsync(60_000)

    expect(fetchMock.mock.calls[0]?.[1]?.signal?.aborted).toBe(false)
    finishRequest?.(new Response(JSON.stringify({
      ok: true, image: 'vllm/vllm-openai:v1', node_ids: ['local'], results: [],
    }), { status: 201, headers: { 'Content-Type': 'application/json' } }))
    await expect(result).resolves.toEqual(expect.objectContaining({ ok: true, node_ids: ['local'] }))
  })

  it.each([
    [
      'deployment stop',
      121_000,
      () => api.deployments.action('dep-slow', 'stop'),
      () => new Response(JSON.stringify({
        id: 'dep-slow', alias: 'slow-model', runtime: 'vllm', kind: 'managed',
        model: { repository: 'org/model' }, status: 'stopped', settings: {},
      }), { status: 200, headers: { 'Content-Type': 'application/json' } }),
    ],
    [
      'deployment removal',
      121_000,
      () => api.deployments.action('dep-slow', 'remove'),
      () => new Response(null, { status: 204 }),
    ],
    [
      'image removal',
      121_000,
      () => api.images.remove('sha256:slow'),
      () => new Response(null, { status: 204 }),
    ],
    [
      'RouterOS fan update',
      46_000,
      () => api.routeros.updateFanSettings('spark-2', { mode: 'curve' }),
      () => new Response(JSON.stringify({ node_id: 'spark-2' }), {
        status: 200, headers: { 'Content-Type': 'application/json' },
      }),
    ],
    [
      'storage model removal',
      601_000,
      () => api.storage.removeModel('spark-2', 'org/model'),
      () => new Response(null, { status: 204 }),
    ],
  ] as const)('does not time out a backend-owned %s', async (_name, elapsedMs, run, response) => {
    vi.useFakeTimers()
    let finishRequest: ((value: Response) => void) | undefined
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(() => new Promise((resolve) => {
      finishRequest = resolve
    }))
    vi.stubGlobal('fetch', fetchMock)

    const result = run()
    await vi.advanceTimersByTimeAsync(elapsedMs)

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock.mock.calls[0]?.[1]?.signal?.aborted).toBe(false)
    finishRequest?.(response())
    await result
  })

  it('does not time out a long-running non-streaming chat completion', async () => {
    vi.useFakeTimers()
    let finishRequest: ((response: Response) => void) | undefined
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(() => new Promise((resolve) => {
      finishRequest = resolve
    }))
    vi.stubGlobal('fetch', fetchMock)

    const result = api.chat('slow-model', [{ role: 'user', content: 'Think carefully.' }])
    await vi.advanceTimersByTimeAsync(60_000)

    expect(fetchMock.mock.calls[0]?.[1]?.signal?.aborted).toBe(false)
    finishRequest?.(new Response(JSON.stringify({
      id: 'chat-1', model: 'slow-model',
      choices: [{ index: 0, message: { role: 'assistant', content: 'Done.' }, finish_reason: 'stop' }],
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    await expect(result).resolves.toEqual(expect.objectContaining({ id: 'chat-1' }))
  })

  it('updates a node fan override with an encoded ID and boolean body', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({ node_id: 'node/1', enabled: true }), {
      status: 200, headers: { 'Content-Type': 'application/json' },
    }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.fanControl.setMaxSpeed('node/1', true)).resolves.toEqual({ node_id: 'node/1', enabled: true })
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/fan-control/nodes/node%2F1/max-speed', expect.objectContaining({
      method: 'PATCH', body: JSON.stringify({ enabled: true }),
    }))
  })

  it('updates fan settings on the selected node with optimistic mode matching', async () => {
    const curve = {
      curve_points: [[30, 20], [60, 55], [80, 100]],
      curve_min_temp: 30,
      curve_max_temp: 80,
      min_floor_pct: 20,
    }
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({
      node_id: 'node/1', mode: 'curve', previous_mode: 'pid', active_settings: curve,
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)

    await api.fanControl.updateSettings('node/1', 'curve', curve, 'pid')

    expect(fetchMock).toHaveBeenCalledWith('/api/v1/fan-control/nodes/node%2F1/settings', expect.objectContaining({
      method: 'PATCH',
      body: JSON.stringify({ mode: 'curve', active_settings: curve, expected_mode: 'pid' }),
    }))
  })

  it('restores sanitized community state from the node session', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({
      status: 'signed-in', email: 'user@example.com', token_invalid: false,
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.community.session()).resolves.toEqual({
      status: 'signed-in', email: 'user@example.com', token_invalid: false,
    })

    expect(fetchMock).toHaveBeenCalledWith('/api/v1/community/session', expect.objectContaining({
      headers: expect.not.objectContaining({ Authorization: expect.anything() }),
    }))
  })

  it('keeps consent withdrawal independent from Cognito token refresh', async () => {
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

    expect(fetchMock.mock.calls.map(([path]) => path)).toEqual([
      '/api/v1/community/consent',
      '/api/v1/community/sync',
    ])
    expect(fetchMock.mock.calls.every(([, init]) => (
      !('Authorization' in ((init?.headers ?? {}) as Record<string, string>))
    ))).toBe(true)
  })

  it('uses fresh Cognito account proof for destructive unpairing', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({
      pairing: { status: 'not_paired' }, cluster: { applied: [], conflicts: [], errors: [] },
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)

    await api.community.unpair('reauthenticated-id-token')

    expect(fetchMock).toHaveBeenCalledWith('/api/v1/community/pair', expect.objectContaining({
      method: 'DELETE',
      headers: expect.objectContaining({ Authorization: 'Bearer reauthenticated-id-token' }),
    }))
  })

  it('renews the node session once when an aggregate cookie expires', async () => {
    const aggregate = {
      items: [], availability: 'available',
      evidence_policy: { minimum_samples: 10, exact_match_dimensions: [], metric: 'inference_tokens_per_second' },
    }
    const fetchMock = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'session expired' }), {
        status: 401, headers: { 'Content-Type': 'application/json' },
      }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        status: 'signed-in', email: 'user@example.com',
      }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
      .mockResolvedValueOnce(new Response(JSON.stringify(aggregate), {
        status: 200, headers: { 'Content-Type': 'application/json' },
      }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.benchmarks.aggregates()).resolves.toEqual(aggregate)

    expect(fetchMock.mock.calls.map(([path]) => path)).toEqual([
      '/api/v1/community/aggregates',
      '/api/v1/community/session',
      '/api/v1/community/aggregates',
    ])
  })
  it('normalizes deployment wire records for the UI', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({
      items: [{
        id: 'dep-1',
        alias: 'reasoner',
        runtime: 'llama.cpp',
        kind: 'managed',
        model: { repository: 'org/model', quantization: 'Q4_K_M' },
        status: 'starting',
        launch_phase: 'pulling_image',
        launch_message: 'Downloading Docker image',
        settings: { context_length: 4096, parallel_slots: 2 },
      }],
    }), { status: 200, headers: { 'Content-Type': 'application/json' } })))

    await expect(api.deployments.list()).resolves.toEqual([
      expect.objectContaining({
        id: 'dep-1',
        model_id: 'org/model',
        runtime: 'llama.cpp',
        managed: true,
        status: 'starting',
        launch_phase: 'pulling_image',
        launch_message: 'Downloading Docker image',
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
