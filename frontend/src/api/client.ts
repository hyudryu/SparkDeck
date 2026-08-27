import type {
  AppSettings,
  BenchmarkAggregate,
  CommunityAggregatesResponse,
  BenchmarkSample,
  BenchmarkModelDetail,
  BenchmarkModelSummary,
  CatalogResponse,
  ChatCompletionResponse,
  ChatMessage,
  ContainerImage,
  CreateDeploymentInput,
  Deployment,
  DashboardData,
  SystemStats,
  AdmissionStats,
  LogEntry,
  SyncStatus,
  CommunityClusterSync,
  CommunityPairResponse,
  CommunityAuthConfig,
  CommunitySession,
  NodeInventoryItem,
  RenameNodeInput,
  ImagePullResult,
  OnboardingStatus,
  JoinClusterInput,
  CreateStorageTransferInput,
  StorageState,
  StorageTransferPreflight,
  StorageTransferResult,
  RecipePreparationPlan,
  RecipePreparationResult,
  ModelCacheState,
  SavedConfiguration,
  SavedConfigurationDetail,
  RecipeUpdateInput,
  UsageAnalysis,
  UsageSummary,
  SystemUpdateOverview,
  SystemUpdateJob,
  RouterOSConnectionInput,
  RouterOSNodeOverview,
  RouterOSOverview,
  RouterOSPresence,
} from './types'
import type { RuntimeKind } from './types'

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body?: unknown,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    credentials: 'same-origin',
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...init?.headers,
    },
  })
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`
    let body: { detail?: unknown; message?: string } | undefined
    try {
      body = (await response.json()) as { detail?: unknown; message?: string }
      if (typeof body.detail === 'string') message = body.detail
      else if (body.message) message = body.message
    } catch {
      // The status text is still useful for non-JSON failures.
    }
    throw new ApiError(message, response.status, body)
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

function normalizeLegacyLog(message: string): LogEntry {
  const match = message.match(/^(?:(\d{2}:\d{2}:\d{2})\s+)?(debug|info|warn(?:ing)?|error|critical|fatal)\b\s*(.*)$/i)
  if (!match) return { level: 'info', message }

  const [, timestamp, rawLevel, remainder] = match
  const level = rawLevel.toLowerCase() === 'warn' ? 'warning' : rawLevel.toLowerCase()
  return {
    ...(timestamp ? { timestamp } : {}),
    level,
    message: remainder || message,
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
  node_ids?: string[]
  selected_nodes?: Deployment['selected_nodes']
  created_at?: string
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
    node_ids: item.node_ids,
    selected_nodes: item.selected_nodes,
    created_at: item.created_at,
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
  dashboard: {
    load: async (signal?: AbortSignal): Promise<DashboardData> => {
      const nodeInventory = api.nodes.list(signal).catch(() => [])
      const [stats, admission, deployments, sync, nodes] = await Promise.all([
        request<SystemStats>('/api/stats', { signal }),
        request<Record<string, AdmissionStats>>('/api/inference-queue', { signal }),
        api.deployments.list(signal),
        api.benchmarks.syncStatus(signal),
        nodeInventory,
      ])
      return { stats, admission, deployments, sync, nodes }
    },
  },
  catalog: {
    search: (query = '', runtime?: string, cursor?: string, signal?: AbortSignal) =>
      request<CatalogResponse>(
        `/api/v1/catalog/models${queryString({ q: query, runtime, cursor, limit: 100 })}`,
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
          node_ids: input.managed && input.runtime !== 'llama.cpp' ? input.node_ids : undefined,
          deployment_mode: input.managed && input.runtime !== 'llama.cpp' ? input.deployment_mode : undefined,
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
    logs: async (id: string, tail = 300) => {
      const data = await request<{ logs: string }>(`/api/v1/deployments/${encodeURIComponent(id)}/logs?tail=${tail}`)
      return data.logs
    },
    rename: async (id: string, alias: string) => {
      const data = await request<WireDeployment>(`/api/v1/deployments/${encodeURIComponent(id)}`, {
        method: 'PATCH',
        body: JSON.stringify({ alias }),
      })
      return deploymentFromWire(data)
    },
  },
  recipes: {
    list: async (signal?: AbortSignal) => {
      const data = await request<{ items: SavedConfiguration[] }>('/api/v1/recipes', { signal })
      return data.items
    },
    get: (id: string, signal?: AbortSignal) =>
      request<SavedConfigurationDetail>(`/api/v1/recipes/${encodeURIComponent(id)}`, { signal }),
    update: (id: string, input: RecipeUpdateInput) =>
      request<SavedConfigurationDetail>(`/api/v1/recipes/${encodeURIComponent(id)}`, {
        method: 'PUT',
        body: JSON.stringify(input),
      }),
    remove: (id: string) => request<void>(
      `/api/v1/recipes/${encodeURIComponent(id)}`,
      { method: 'DELETE' },
    ),
    deploy: (id: string, nodeIds: string[]) => request<WireDeployment>(
      `/api/v1/recipes/${encodeURIComponent(id)}/deploy`,
      { method: 'POST', body: JSON.stringify({ node_ids: nodeIds }) },
    ).then(deploymentFromWire),
  },
  nodes: {
    list: async (signal?: AbortSignal): Promise<NodeInventoryItem[]> => {
      const data = await request<{ items: NodeInventoryItem[] }>('/api/v1/nodes', { signal })
      return data.items
    },
    rename: (id: string, input: RenameNodeInput) => request<NodeInventoryItem>(`/api/v1/nodes/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      body: JSON.stringify(input),
    }),
    remove: (id: string, force = false) => request<{ ok: boolean; node_id: string; forced: boolean }>(
      `/api/v1/nodes/${encodeURIComponent(id)}${force ? '?force=true' : ''}`,
      { method: 'DELETE' },
    ),
  },
  routeros: {
    presence: (signal?: AbortSignal) => request<RouterOSPresence>('/api/v1/routeros/presence', { signal }),
    get: (signal?: AbortSignal) => request<RouterOSOverview>('/api/v1/routeros', { signal }),
    connect: (nodeId: string, input: RouterOSConnectionInput) => request<RouterOSNodeOverview>(
      `/api/v1/routeros/nodes/${encodeURIComponent(nodeId)}/connection`,
      { method: 'PUT', body: JSON.stringify(input) },
    ),
    disconnect: (nodeId: string) => request<void>(
      `/api/v1/routeros/nodes/${encodeURIComponent(nodeId)}/connection`,
      { method: 'DELETE' },
    ),
    updateFanSettings: (nodeId: string, settings: Record<string, unknown>) => request<RouterOSNodeOverview>(
      `/api/v1/routeros/nodes/${encodeURIComponent(nodeId)}/fan-settings`,
      { method: 'PATCH', body: JSON.stringify(settings) },
    ),
  },
  onboarding: {
    get: (signal?: AbortSignal) => request<OnboardingStatus>('/api/v1/onboarding', { signal }),
    join: (input: JoinClusterInput) => request<OnboardingStatus>('/api/v1/onboarding/join', {
      method: 'POST',
      body: JSON.stringify(input),
    }),
    leave: () => request<OnboardingStatus>('/api/v1/onboarding/leave', {
      method: 'POST',
      body: JSON.stringify({}),
    }),
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
    models: async (signal?: AbortSignal): Promise<BenchmarkModelSummary[]> => {
      const data = await request<{ items: BenchmarkModelSummary[] }>('/api/v1/benchmark-models', { signal })
      return data.items
    },
    model: (modelId: string, signal?: AbortSignal): Promise<BenchmarkModelDetail> =>
      request<BenchmarkModelDetail>(`/api/v1/benchmark-models/${encodeURIComponent(modelId)}`, { signal }),
    aggregates: async (signal?: AbortSignal): Promise<CommunityAggregatesResponse> => {
      try {
        return await request<CommunityAggregatesResponse>('/api/v1/community/aggregates', { signal })
      } catch (error) {
        if (!(error instanceof ApiError) || error.status !== 401) throw error
        const session = await request<CommunitySession>('/api/v1/community/session', { signal })
        if (session.status !== 'signed-in') throw error
        return request<CommunityAggregatesResponse>('/api/v1/community/aggregates', { signal })
      }
    },
    syncStatus: async (signal?: AbortSignal): Promise<SyncStatus> => {
      const data = await request<{ consent: boolean; pairing?: { status?: string; token_invalid?: boolean }; outbox?: Record<string, number>; upload_configured?: boolean }>('/api/v1/community/sync', { signal })
      return {
        sharing_enabled: data.consent,
        account_paired: data.pairing?.status === 'paired',
        token_invalid: Boolean(data.pairing?.token_invalid),
        upload_configured: Boolean(data.upload_configured),
        pending_count: (data.outbox?.pending ?? 0) + (data.outbox?.waiting_for_account ?? 0),
        synced_count: data.outbox?.synced ?? 0,
        failed_count: data.outbox?.failed ?? 0,
      }
    },
    setConsent: async (sharing_enabled: boolean) => {
      const result = await request<{ cluster?: CommunityClusterSync }>('/api/v1/community/consent', {
        method: 'PUT',
        body: JSON.stringify({ enabled: sharing_enabled }),
      })
      return {
        ...await api.benchmarks.syncStatus(),
        cluster_errors: result.cluster?.errors ?? [],
      }
    },
    retry: async () => {
      await request<unknown>('/api/v1/community/retry', { method: 'POST' })
      return api.benchmarks.syncStatus()
    },
    deleteLocal: (id: string) =>
      request<void>(`/api/v1/benchmarks/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  },
  community: {
    authConfig: () => request<CommunityAuthConfig>('/api/v1/community/auth-config'),
    session: () => request<CommunitySession>('/api/v1/community/session'),
    pair: (idToken: string, refreshToken?: string) => request<CommunityPairResponse>('/api/v1/community/pair', {
      method: 'POST',
      body: JSON.stringify({
        id_token: idToken,
        ...(refreshToken ? { refresh_token: refreshToken } : {}),
      }),
    }),
    unpair: (idToken: string) => request<CommunityPairResponse>('/api/v1/community/pair', {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${idToken}` },
    }),
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
          node_ids: item.node_ids ?? item.selected_nodes?.map((node) => node.id) ?? ['local'],
        }
      })
    },
    pull: async (image: string, nodeIds?: string[]): Promise<ImagePullResult> => {
      const result = await requestWithFallback<ImagePullResult>('/api/v1/images/pull', '/api/images/pull', {
        method: 'POST',
        body: JSON.stringify({ image, node_ids: nodeIds }),
      })
      return result ?? { ok: true, image, node_ids: nodeIds ?? ['local'], results: [] }
    },
    remove: (id: string) =>
      requestWithFallback<void>(`/api/v1/images/${encodeURIComponent(id)}`, `/api/images/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  },
  storage: {
    get: (signal?: AbortSignal) => request<StorageState>('/api/v1/storage', { signal }),
    setEnabled: (enabled: boolean) => request<{ enabled: boolean }>('/api/v1/storage/settings', {
      method: 'PUT',
      body: JSON.stringify({ enabled }),
    }),
    preflight: (modelId: string, revision = 'main', signal?: AbortSignal) => request<StorageTransferPreflight>('/api/v1/storage/transfers/preflight', {
      method: 'POST',
      body: JSON.stringify({ model_id: modelId, revision }),
      signal,
    }),
    transfer: (input: CreateStorageTransferInput) => request<StorageTransferResult>('/api/v1/storage/transfers', {
      method: 'POST',
      body: JSON.stringify(input),
    }),
    preparationPreflight: (recipeId: string, nodeIds: string[]) => request<RecipePreparationPlan>(`/api/v1/recipes/${encodeURIComponent(recipeId)}/prepare/preflight`, {
      method: 'POST',
      body: JSON.stringify({ node_ids: nodeIds }),
    }),
    prepareRecipe: (recipeId: string, nodeIds: string[]) => request<RecipePreparationResult>(`/api/v1/recipes/${encodeURIComponent(recipeId)}/prepare`, {
      method: 'POST',
      body: JSON.stringify({ node_ids: nodeIds }),
    }),
    cancel: (id: string) => request<void>(`/api/v1/storage/transfers/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),
    removeModel: (nodeId: string, modelId: string) => request<void>(
      `/api/v1/storage/nodes/${encodeURIComponent(nodeId)}/models/${encodeURIComponent(modelId)}`,
      { method: 'DELETE' },
    ),
  },
  modelCache: {
    get: (signal?: AbortSignal) => request<ModelCacheState>('/api/v1/model-cache', { signal }),
  },
  usage: {
    get: (signal?: AbortSignal) => request<UsageSummary>('/api/token-stats', { signal }),
    analysis: async (start = '', end = '', signal?: AbortSignal): Promise<UsageAnalysis> => {
      const query = queryString({ start, end })
      const [hourly, daily] = await Promise.all([
        request<UsageAnalysis['hourly']>(`/api/token-stats/hourly${query}`, { signal }),
        request<UsageAnalysis['daily']>(`/api/token-stats/daily${query}`, { signal }),
      ])
      return { hourly, daily }
    },
    reset: () => request<UsageSummary>('/api/token-stats/reset', { method: 'POST' }),
    updateAlias: (model: string, alias: string | null, merge_group: string | null) =>
      request<void>('/api/token-stats/alias', {
        method: 'PUT',
        body: JSON.stringify({ model, alias, merge_group }),
      }),
    erase: (model: string) => request<void>(
      `/api/token-stats/${encodeURIComponent(model)}`,
      { method: 'DELETE' },
    ),
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
    clearHfToken: () => request<AppSettings>('/api/v1/settings/hf-token', { method: 'DELETE' }),
  },
  updates: {
    overview: (signal?: AbortSignal) => request<SystemUpdateOverview>('/api/v1/system-update', { signal }),
    start: (revision: string) => request<SystemUpdateJob>('/api/v1/system-update', {
      method: 'POST',
      body: JSON.stringify({ confirm: 'update-entire-cluster', revision }),
    }),
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
      return (data.logs ?? []).map(normalizeLegacyLog)
    },
  },
}
