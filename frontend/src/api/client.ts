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
  ChatResponseMetrics,
  ChatStreamResult,
  ChatStreamUpdate,
  ChatUsage,
  ContainerImage,
  CreateDeploymentInput,
  Deployment,
  DeploymentDetail,
  DeploymentLaunchControls,
  DeploymentUpdateInput,
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
  FanControlOverview,
  FanControlMode,
  FanCurveSettings,
  FanHysteresisSettings,
  FanManualSettings,
  FanMaxSpeedResult,
  FanPidSettings,
  FanSettingsUpdateResult,
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

const REQUEST_TIMEOUT_MS = 30_000
// Long-running mutations and non-streaming inference already have
// backend-owned limits. Keep their browser connection alive so the server can
// return the authoritative result instead of inviting a duplicate retry after
// 30 seconds while the original state change is still completing.
const NO_REQUEST_TIMEOUT: null = null

async function request<T>(
  path: string,
  init?: RequestInit,
  timeoutMs: number | null = REQUEST_TIMEOUT_MS,
): Promise<T> {
  const controller = new AbortController()
  const callerSignal = init?.signal
  let timedOut = false
  const forwardAbort = () => controller.abort(callerSignal?.reason)
  if (callerSignal?.aborted) forwardAbort()
  else callerSignal?.addEventListener('abort', forwardAbort, { once: true })
  const timeout = timeoutMs === null ? undefined : globalThis.setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)

  try {
    const response = await fetch(path, {
      credentials: 'same-origin',
      cache: 'no-store',
      ...init,
      signal: controller.signal,
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
      } catch (error) {
        // Abort still needs to propagate while the response body is being read.
        if (controller.signal.aborted) throw error
        // The status text is still useful for non-JSON failures.
      }
      throw new ApiError(message, response.status, body)
    }
    if (response.status === 204) return undefined as T
    if (!response.headers.get('content-type')?.includes('application/json')) return undefined as T
    return await response.json() as T
  } catch (error) {
    if (timedOut) {
      throw new ApiError('The request timed out. Check the node connection and retry.', 408)
    }
    throw error
  } finally {
    if (timeout !== undefined) globalThis.clearTimeout(timeout)
    callerSignal?.removeEventListener('abort', forwardAbort)
  }
}

async function requestWithFallback<T>(
  primary: string,
  fallback: string,
  init?: RequestInit,
  timeoutMs: number | null = REQUEST_TIMEOUT_MS,
) {
  try {
    return await request<T>(primary, init, timeoutMs)
  } catch (error) {
    if (!(error instanceof ApiError) || error.status !== 404) throw error
    return request<T>(fallback, init, timeoutMs)
  }
}

export interface ChatStreamOptions {
  signal?: AbortSignal
  onUpdate?: (update: ChatStreamUpdate) => void
}

function positiveNumber(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : undefined
}

async function streamChat(model: string, messages: ChatMessage[], options: ChatStreamOptions = {}): Promise<ChatStreamResult> {
  const startedAt = performance.now()
  const response = await fetch('/v1/chat/completions', {
    method: 'POST',
    credentials: 'same-origin',
    cache: 'no-store',
    headers: { Accept: 'text/event-stream', 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, messages, stream: true, stream_options: { include_usage: true } }),
    signal: options.signal,
  })
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`
    try {
      const body = await response.json() as { detail?: unknown; message?: string }
      if (typeof body.detail === 'string') message = body.detail
      else if (body.message) message = body.message
    } catch {
      // Keep the HTTP status for non-JSON failures.
    }
    throw new ApiError(message, response.status)
  }
  if (!response.headers.get('content-type')?.includes('text/event-stream')) {
    throw new Error('The model returned a non-streaming response')
  }
  if (!response.body) throw new Error('The model returned an empty response stream')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let eventData: string[] = []
  let content = ''
  let reasoning = ''
  let usage: ChatUsage | undefined
  let firstTokenAt: number | undefined
  let nativePromptRate: number | undefined
  let nativeOutputRate: number | undefined
  let doneReceived = false
  let terminalSeen = false

  const metricsAt = (now: number): ChatResponseMetrics => {
    const ttftMs = firstTokenAt === undefined ? undefined : Math.max(0, firstTokenAt - startedAt)
    const promptTokens = positiveNumber(usage?.prompt_tokens)
    const completionTokens = positiveNumber(usage?.completion_tokens)
    const rawCachedTokens = usage?.prompt_tokens_details?.cached_tokens
    const cachedTokens = typeof rawCachedTokens === 'number' && Number.isFinite(rawCachedTokens) && rawCachedTokens >= 0
      ? rawCachedTokens
      : undefined
    const processedPromptTokens = promptTokens === undefined || cachedTokens === undefined
      ? undefined
      : Math.max(0, promptTokens - cachedTokens)
    const generationMs = firstTokenAt === undefined ? undefined : now - firstTokenAt
    return {
      prompt_tokens_per_second: nativePromptRate ?? (
        processedPromptTokens && ttftMs ? processedPromptTokens / (ttftMs / 1000) : undefined
      ),
      ttft_ms: ttftMs,
      output_tokens_per_second: nativeOutputRate ?? (
        completionTokens !== undefined && completionTokens >= 2 && generationMs !== undefined && generationMs >= 50
          ? completionTokens / (generationMs / 1000)
          : undefined
      ),
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
    }
  }

  const publish = (update: Omit<ChatStreamUpdate, 'metrics'> = {}) => {
    options.onUpdate?.({ ...update, metrics: metricsAt(performance.now()) })
  }

  const consumeData = (data: string) => {
    if (!data) return
    if (data === '[DONE]') {
      doneReceived = true
      terminalSeen = true
      return
    }
    let payload: Record<string, unknown>
    try {
      payload = JSON.parse(data) as Record<string, unknown>
    } catch {
      throw new Error('The model returned malformed stream data')
    }
    const streamError = payload.error
    if (streamError) {
      const message = typeof streamError === 'string'
        ? streamError
        : String((streamError as { message?: unknown }).message ?? 'The model stream failed')
      throw new Error(message)
    }
    const choices = Array.isArray(payload.choices) ? payload.choices : []
    const choice = choices[0] as { delta?: Record<string, unknown>; finish_reason?: unknown } | undefined
    if (choice?.finish_reason !== undefined && choice.finish_reason !== null) terminalSeen = true
    const delta = choice?.delta
    const contentDelta = typeof delta?.content === 'string' ? delta.content : ''
    const reasoningDelta = typeof delta?.reasoning_content === 'string'
      ? delta.reasoning_content
      : typeof delta?.reasoning === 'string' ? delta.reasoning : ''
    if ((contentDelta || reasoningDelta) && firstTokenAt === undefined) firstTokenAt = performance.now()
    content += contentDelta
    reasoning += reasoningDelta
    if (payload.usage && typeof payload.usage === 'object') usage = payload.usage as ChatUsage
    const timings = payload.timings && typeof payload.timings === 'object'
      ? payload.timings as Record<string, unknown>
      : undefined
    nativePromptRate = positiveNumber(timings?.prompt_per_second) ?? nativePromptRate
    nativeOutputRate = positiveNumber(timings?.predicted_per_second) ?? nativeOutputRate
    if (contentDelta || reasoningDelta || payload.usage || timings) {
      publish({ content: contentDelta || undefined, reasoning: reasoningDelta || undefined })
    }
  }

  const consumeLine = (rawLine: string) => {
    const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine
    if (line === '') {
      if (eventData.length) consumeData(eventData.join('\n'))
      eventData = []
    } else if (line.startsWith('data:')) {
      eventData.push(line.slice(5).trimStart())
    }
  }

  try {
    while (true) {
      const { value, done } = await reader.read()
      buffer += decoder.decode(value, { stream: !done })
      let newline = buffer.indexOf('\n')
      while (newline >= 0 && !doneReceived) {
        consumeLine(buffer.slice(0, newline))
        buffer = buffer.slice(newline + 1)
        newline = buffer.indexOf('\n')
      }
      if (doneReceived) {
        break
      }
      if (done) break
    }
    if (!doneReceived) {
      if (buffer) consumeLine(buffer)
      if (eventData.length) consumeData(eventData.join('\n'))
    }
    if (!terminalSeen) throw new Error('The model response stream ended unexpectedly')
  } catch (error) {
    await reader.cancel().catch(() => undefined)
    throw error
  } finally {
    reader.releaseLock()
  }

  return {
    message: { role: 'assistant', content },
    reasoning,
    usage,
    metrics: metricsAt(performance.now()),
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
  deployment_mode?: string
  required_node_count?: number
  model_revision?: string
  created_at?: string
  desired_state?: 'running' | 'stopped'
  launch_phase?: string
  launch_message?: string
}

interface WireDeploymentDetail extends WireDeployment {
  editable: boolean
  edit_reason?: string | null
  desired_state: 'running' | 'stopped'
  extra_args: string[]
  launch_controls: DeploymentLaunchControls
  gpu_memory_utilization?: number | null
  gpu_memory_gb?: number | null
  image?: string | null
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
    model_revision: item.model_revision ?? item.model.revision,
    runtime: item.runtime,
    status: item.status,
    managed: item.kind === 'managed',
    settings: { ...item.settings, port: item.port, quantization: item.model.quantization },
    deployment_mode: item.deployment_mode,
    required_node_count: item.required_node_count,
    node_ids: item.node_ids,
    selected_nodes: item.selected_nodes,
    created_at: item.created_at,
    desired_state: item.desired_state,
    launch_phase: item.launch_phase,
    launch_message: item.launch_message,
  }
}

function deploymentDetailFromWire(item: WireDeploymentDetail): DeploymentDetail {
  return {
    ...deploymentFromWire(item),
    editable: item.editable,
    edit_reason: item.edit_reason,
    desired_state: item.desired_state,
    extra_args: item.extra_args ?? [],
    launch_controls: item.launch_controls ?? {},
    gpu_memory_utilization: item.gpu_memory_utilization,
    gpu_memory_gb: item.gpu_memory_gb,
    image: item.image ?? undefined,
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
    details: (id: string, signal?: AbortSignal) =>
      request<{ model: CatalogResponse['items'][number]; aggregates: BenchmarkAggregate[] }>(
        `/api/v1/catalog/models/${encodeURIComponent(id)}`,
        { signal },
      ),
    model: (id: string, signal?: AbortSignal) => api.catalog.details(id, signal),
  },
  deployments: {
    list: async (signal?: AbortSignal) => {
      const data = await request<{ items: WireDeployment[] }>('/api/v1/deployments', { signal })
      return data.items.map(deploymentFromWire)
    },
    get: async (id: string, signal?: AbortSignal) => {
      const data = await request<WireDeploymentDetail>(`/api/v1/deployments/${encodeURIComponent(id)}`, { signal })
      return deploymentDetailFromWire(data)
    },
    update: async (id: string, input: DeploymentUpdateInput) => {
      const data = await request<WireDeploymentDetail>(`/api/v1/deployments/${encodeURIComponent(id)}/settings`, {
        method: 'PUT',
        body: JSON.stringify(input),
      })
      return deploymentDetailFromWire(data)
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
      }, NO_REQUEST_TIMEOUT)
      return deploymentFromWire(data)
    },
    action: async (id: string, action: 'start' | 'stop' | 'remove', nodeIds?: string[], additionalNodeIds?: string[]) => {
      if (action === 'remove') {
        return request<void>(
          `/api/v1/deployments/${encodeURIComponent(id)}`,
          { method: 'DELETE' },
          NO_REQUEST_TIMEOUT,
        )
      }
      const payload = action === 'start' && additionalNodeIds?.length
        ? { additional_node_ids: additionalNodeIds }
        : action === 'start' && nodeIds?.length ? { node_ids: nodeIds } : undefined
      const data = await request<WireDeployment>(`/api/v1/deployments/${encodeURIComponent(id)}/${action}`, {
        method: 'POST',
        body: payload ? JSON.stringify(payload) : undefined,
      }, NO_REQUEST_TIMEOUT)
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
      NO_REQUEST_TIMEOUT,
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
    setDashboardHidden: (id: string, hidden_from_dashboard: boolean) => request<Pick<NodeInventoryItem, 'id' | 'hidden_from_dashboard'>>(`/api/v1/nodes/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      body: JSON.stringify({ hidden_from_dashboard }),
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
      NO_REQUEST_TIMEOUT,
    ),
  },
  fanControl: {
    get: (signal?: AbortSignal) => request<FanControlOverview>('/api/v1/fan-control', { signal }),
    setMaxSpeed: (nodeId: string, enabled: boolean) => request<FanMaxSpeedResult>(
      `/api/v1/fan-control/nodes/${encodeURIComponent(nodeId)}/max-speed`,
      { method: 'PATCH', body: JSON.stringify({ enabled }) },
    ),
    updateSettings: (
      nodeId: string,
      mode: FanControlMode,
      activeSettings: FanCurveSettings | FanPidSettings | FanHysteresisSettings | FanManualSettings,
      expectedMode: FanControlMode,
    ) => request<FanSettingsUpdateResult>(
      `/api/v1/fan-control/nodes/${encodeURIComponent(nodeId)}/settings`,
      { method: 'PATCH', body: JSON.stringify({ mode, active_settings: activeSettings, expected_mode: expectedMode }) },
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
    }, NO_REQUEST_TIMEOUT),
  chatStream: streamChat,
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
      }, NO_REQUEST_TIMEOUT)
      return result ?? { ok: true, image, node_ids: nodeIds ?? ['local'], results: [] }
    },
    remove: (id: string) =>
      requestWithFallback<void>(
        `/api/v1/images/${encodeURIComponent(id)}`,
        `/api/images/${encodeURIComponent(id)}`,
        { method: 'DELETE' },
        NO_REQUEST_TIMEOUT,
      ),
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
    finishDownload: (nodeId: string, modelId: string, revision?: string) => request<StorageTransferResult>(
      `/api/v1/storage/nodes/${encodeURIComponent(nodeId)}/models/${encodeURIComponent(modelId)}/download`,
      {
        method: 'POST',
        body: JSON.stringify(revision ? { revision } : {}),
      },
      NO_REQUEST_TIMEOUT,
    ),
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
      NO_REQUEST_TIMEOUT,
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
    updateAlias: (model: string, alias: string | null, merge_group?: string | null) =>
      request<void>('/api/token-stats/alias', {
        method: 'PUT',
        // An undefined merge_group is dropped from the JSON body, which tells
        // the backend to leave the stored merge group untouched.
        body: JSON.stringify({ model, alias, merge_group }),
      }),
    setRoute: (source: string, destination: string) =>
      request<void>('/api/token-stats/rules', {
        method: 'PUT',
        body: JSON.stringify({ source, destination }),
      }),
    deleteRoute: (source: string) => request<void>(
      `/api/token-stats/rules/${encodeURIComponent(source)}`,
      { method: 'DELETE' },
    ),
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
