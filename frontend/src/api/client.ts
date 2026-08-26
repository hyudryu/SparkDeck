import type {
  AppSettings,
  BenchmarkAggregate,
  BenchmarkSample,
  CatalogResponse,
  ChatCompletionResponse,
  ChatMessage,
  ContainerImage,
  CreateDeploymentInput,
  Deployment,
  LogEntry,
  SyncStatus,
} from './types'
import type { RuntimeKind } from './types'

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...init?.headers,
    },
  })
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`
    try {
      const body = (await response.json()) as { detail?: string; message?: string }
      message = body.detail ?? body.message ?? message
    } catch {
      // The status text is still useful for non-JSON failures.
    }
    throw new ApiError(message, response.status)
  }
  if (response.status === 204) return undefined as T
  if (!response.headers.get('content-type')?.includes('application/json')) return undefined as T
  return response.json() as Promise<T>
}

async function requestWithFallback<T>(primary: string, fallback: string, init?: RequestInit) {
  try {
    return await request<T>(primary, init)
  } catch (error) {
    if (!(error instanceof ApiError) || error.status !== 404) throw error
    return request<T>(fallback, init)
  }
}

interface WireDeployment {
  id: string
  alias: string
  runtime: RuntimeKind
  kind: 'managed' | 'external'
  model: { repository: string; revision?: string; artifact?: string; quantization?: string }
  status: Deployment['status']
  container_name?: string
  settings?: Deployment['settings']
  base_url_set?: boolean
  port?: number
}

interface WireBenchmark {
  id: string
  created_at: string
  deployment_id?: string
  model: { repository: string; revision?: string; artifact?: string; quantization?: string }
  runtime: RuntimeKind
  hardware?: { hardware_class?: string; gpu_model?: string }
  configuration?: Record<string, unknown>
  input_tokens: number
  output_tokens: number
  latency_ms: number
  ttft_ms?: number
  generation_tokens_per_second?: number
  cold_start?: boolean
  eligible_for_community: boolean
  sync_state?: BenchmarkSample['sync_state']
}

function deploymentFromWire(item: WireDeployment): Deployment {
  return {
    id: item.id,
    alias: item.alias,
    model_id: item.model.repository,
    model_revision: item.model.revision,
    runtime: item.runtime,
    status: item.status,
    managed: item.kind === 'managed',
    settings: { ...item.settings, port: item.port, quantization: item.model.quantization },
  }
}

function queryString(values: Record<string, string | number | undefined>) {
  const params = new URLSearchParams()
  Object.entries(values).forEach(([key, value]) => {
    if (value !== undefined && value !== '') params.set(key, String(value))
  })
  const query = params.toString()
  return query ? `?${query}` : ''
}

export const api = {
  catalog: {
    search: (query = '', runtime?: string, cursor?: string, signal?: AbortSignal) =>
      request<CatalogResponse>(
        `/api/v1/catalog/models${queryString({ q: query, runtime, cursor, limit: 24 })}`,
        { signal },
      ),
    model: (id: string, signal?: AbortSignal) =>
      request<{ model: CatalogResponse['items'][number]; aggregates: BenchmarkAggregate[] }>(
        `/api/v1/catalog/models/${encodeURIComponent(id)}`,
        { signal },
      ),
  },
  deployments: {
    list: async (signal?: AbortSignal) => {
      const data = await request<{ items: WireDeployment[] }>('/api/v1/deployments', { signal })
      return data.items.map(deploymentFromWire)
    },
    create: async (input: CreateDeploymentInput) => {
      const data = await request<WireDeployment>('/api/v1/deployments', {
        method: 'POST',
        body: JSON.stringify({
          model: input.model_id,
          alias: input.alias,
          runtime: input.runtime,
          kind: input.managed ? 'managed' : 'external',
          base_url: input.endpoint_url || undefined,
          api_key: input.api_key || undefined,
          settings: input.settings,
          quantization: input.settings.quantization,
        }),
      })
      return deploymentFromWire(data)
    },
    action: async (id: string, action: 'start' | 'stop' | 'remove') => {
      if (action === 'remove') {
        return request<void>(`/api/v1/deployments/${encodeURIComponent(id)}`, { method: 'DELETE' })
      }
      const data = await request<WireDeployment>(`/api/v1/deployments/${encodeURIComponent(id)}/${action}`, { method: 'POST' })
      return deploymentFromWire(data)
    },
  },
  chat: (model: string, messages: ChatMessage[], signal?: AbortSignal) =>
    request<ChatCompletionResponse>('/v1/chat/completions', {
      method: 'POST',
      body: JSON.stringify({ model, messages, stream: false }),
      signal,
    }),
  benchmarks: {
    list: async (signal?: AbortSignal): Promise<BenchmarkSample[]> => {
      const data = await request<{ items: WireBenchmark[] }>('/api/v1/benchmarks?limit=100&offset=0', { signal })
      return data.items.map((item) => ({
        id: item.id,
        deployment_id: item.deployment_id,
        model_id: item.model.repository,
        model_revision: item.model.revision,
        runtime: item.runtime,
        quantization: item.model.quantization,
        hardware_class: item.hardware?.hardware_class ?? item.hardware?.gpu_model,
        input_tokens: item.input_tokens,
        output_tokens: item.output_tokens,
        latency_ms: item.latency_ms,
        ttft_ms: item.ttft_ms,
        tokens_per_second: item.generation_tokens_per_second,
        cold_start: item.cold_start,
        upload_eligible: item.eligible_for_community,
        sync_state: item.sync_state ?? 'local',
        created_at: item.created_at,
      }))
    },
    aggregates: async (signal?: AbortSignal): Promise<BenchmarkAggregate[]> => {
      const data = await request<{ items: BenchmarkAggregate[] }>('/api/v1/community/aggregates', { signal })
      return data.items
    },
    syncStatus: async (signal?: AbortSignal): Promise<SyncStatus> => {
      const data = await request<{ consent: boolean; pairing?: { status?: string }; outbox?: Record<string, number> }>('/api/v1/community/sync', { signal })
      return {
        sharing_enabled: data.consent,
        account_paired: data.pairing?.status === 'paired',
        pending_count: (data.outbox?.pending ?? 0) + (data.outbox?.waiting_for_account ?? 0),
        synced_count: data.outbox?.synced ?? 0,
        failed_count: data.outbox?.failed ?? 0,
      }
    },
    setConsent: async (sharing_enabled: boolean) => {
      await request<unknown>('/api/v1/community/consent', {
        method: 'PUT',
        body: JSON.stringify({ enabled: sharing_enabled }),
      })
      return api.benchmarks.syncStatus()
    },
    retry: async () => {
      await request<unknown>('/api/v1/community/retry', { method: 'POST' })
      return api.benchmarks.syncStatus()
    },
    deleteLocal: (id: string) =>
      request<void>(`/api/v1/benchmarks/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  },
  images: {
    list: async (signal?: AbortSignal): Promise<ContainerImage[]> => {
      const data = await requestWithFallback<ContainerImage[] | { items?: ContainerImage[]; images?: ContainerImage[] }>('/api/v1/images', '/api/images', {
        signal,
      })
      const images = Array.isArray(data) ? data : (data.items ?? data.images ?? [])
      return images.map((item) => {
        const firstTag = item.tags?.[0]
        const separator = firstTag?.lastIndexOf(':') ?? -1
        return {
          ...item,
          repository: item.repository ?? (separator > 0 ? firstTag?.slice(0, separator) : firstTag),
          tag: item.tag ?? (separator > 0 ? firstTag?.slice(separator + 1) : undefined),
          created_at: item.created_at ?? item.created,
          runtimes: item.runtimes ?? (item.is_vllm ? ['vllm'] : undefined),
        }
      })
    },
    pull: (image: string) =>
      requestWithFallback<void>('/api/v1/images/pull', '/api/images/pull', { method: 'POST', body: JSON.stringify({ image }) }),
    remove: (id: string) =>
      requestWithFallback<void>(`/api/v1/images/${encodeURIComponent(id)}`, `/api/images/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  },
  settings: {
    get: (signal?: AbortSignal) => requestWithFallback<AppSettings>('/api/v1/settings', '/api/settings', { signal }),
    update: async (settings: AppSettings) => {
      try {
        return await request<AppSettings>('/api/v1/settings', { method: 'PUT', body: JSON.stringify(settings) })
      } catch (error) {
        if (!(error instanceof ApiError) || error.status !== 404) throw error
        return request<AppSettings>('/api/settings', { method: 'POST', body: JSON.stringify(settings) })
      }
    },
  },
  logs: {
    list: async (signal?: AbortSignal): Promise<LogEntry[]> => {
      const data = await requestWithFallback<LogEntry[] | { entries?: LogEntry[]; logs?: string[] }>(
        '/api/v1/logs',
        '/api/server-logs',
        { signal },
      )
      if (Array.isArray(data)) return data
      if (data.entries) return data.entries
      return (data.logs ?? []).map((message) => ({ message }))
    },
  },
}
