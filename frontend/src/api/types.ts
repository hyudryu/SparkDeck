export type RuntimeKind = 'vllm' | 'llama.cpp' | 'sglang'
export type DeploymentStatus = 'registered' | 'running' | 'starting' | 'stopped' | 'error' | 'unknown'

export interface RuntimeCompatibility {
  runtime: RuntimeKind
  supported: boolean
  reason?: string
}

export interface BenchmarkAggregate {
  model_id: string
  context_window_size: number
  inference_tokens_per_second: number
  sample_count: number
}

export interface CommunityEvidencePolicy {
  minimum_samples: number
  exact_match_dimensions: Array<'model_id' | 'context_window_size'>
  metric: 'inference_tokens_per_second'
}

export interface CommunityAggregatesResponse {
  items: BenchmarkAggregate[]
  availability: string
  evidence_policy: CommunityEvidencePolicy
}

export interface CatalogModel {
  id: string
  author?: string
  name?: string
  revision?: string
  downloads?: number
  likes?: number
  parameter_count?: number
  tags?: string[]
  runtime_compatibility?: RuntimeCompatibility[]
  local_deployment_ids?: string[]
  community?: BenchmarkAggregate | null
}

export interface CatalogResponse {
  items: CatalogModel[]
  total?: number
  next_cursor?: string | null
}

export interface DeploymentSettings {
  context_length?: number
  tensor_parallel_size?: number
  data_parallel_size?: number
  pipeline_parallel_size?: number
  gpu_layers?: number
  parallel_slots?: number
  gpu_split?: string
  quantization?: string
  dtype?: string
  port?: number
  extra_args?: string[]
}

export interface Deployment {
  id: string
  alias: string
  model_id: string
  model_revision?: string
  runtime: RuntimeKind
  status: DeploymentStatus
  managed: boolean
  endpoint_url?: string
  runtime_version?: string
  image?: string
  settings: DeploymentSettings
  last_error?: string
  created_at?: string
  updated_at?: string
  node_ids?: string[]
  selected_nodes?: NodeSummary[]
}

export interface CreateDeploymentInput {
  alias: string
  model_id: string
  runtime: RuntimeKind
  endpoint_url?: string
  api_key?: string
  managed: boolean
  settings: DeploymentSettings
  node_ids?: string[]
  deployment_mode?: 'single' | 'replicated' | 'sharded'
}

export interface NodeSummary {
  id: string
  name: string
  local?: boolean
}

export interface NodeInventoryItem extends NodeSummary {
  online?: boolean
  docker_ready?: boolean
  fabric_ready?: boolean
  selectable?: boolean
  stats?: SystemStats
  disk?: { total?: number; total_bytes?: number; free?: number; free_bytes?: number }
}

export interface RenameNodeInput {
  name: string
}

export interface OnboardingStatus {
  role: 'controller' | 'worker'
  node: { id: string; name: string; port: number; access_urls: string[] }
  controller_url?: string
  controller_reachable: boolean
  join_code?: string
  instructions?: string[]
}

export interface JoinClusterInput {
  controller_url: string
  join_code: string
  advertise_url: string
  name: string
}

export interface BenchmarkSample {
  id: string
  deployment_id?: string
  model_id: string
  model_revision?: string
  runtime: RuntimeKind
  quantization?: string
  hardware_class?: string
  input_tokens?: number
  output_tokens?: number
  latency_ms: number
  ttft_ms?: number
  tokens_per_second?: number
  cold_start?: boolean
  upload_eligible?: boolean
  sync_state?: 'local' | 'pending' | 'synced' | 'failed' | 'waiting_for_account'
  created_at: string
}

export interface SyncStatus {
  sharing_enabled: boolean
  account_paired: boolean
  pending_count: number
  synced_count: number
  failed_count: number
  last_sync_at?: string | null
  last_error?: string | null
}

export interface ContainerImage {
  id: string
  repository?: string
  tag?: string
  size?: number
  created_at?: string
  runtimes?: RuntimeKind[]
  in_use?: boolean
  tags?: string[]
  created?: string
  is_vllm?: boolean
  node_ids?: string[]
  selected_nodes?: NodeSummary[]
}

export interface ImagePullResult {
  ok: boolean
  image: string
  node_ids: string[]
  selected_nodes?: NodeSummary[]
  results?: Array<{ node_id: string; node_name: string; ok: boolean; error?: string }>
}

export interface AppSettings {
  theme?: 'system' | 'light' | 'dark'
  hf_token?: string
  hf_token_configured?: boolean
  community_api_url?: string
  telemetry_interval_seconds?: number
  [key: string]: unknown
}

export interface LogEntry {
  timestamp?: string
  level?: string
  source?: string
  message: string
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface ChatCompletionResponse {
  choices: Array<{ message: ChatMessage }>
  usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number }
}

export interface GpuStats {
  index: number
  name?: string
  util?: number | null
  mem_used_mib?: number | null
  mem_total_mib?: number | null
  temp?: number | null
  error?: string
}

export interface ActiveRequestStats {
  connections: number
  queued?: number
  decoded_tokens?: number
  thinking_tok_s?: number
  output_tok_s?: number
  pp_tok_s?: number | null
  admission_limit?: number
}

export interface SystemStats {
  cpu_pct?: number | null
  cpu_temp_c?: number | null
  mem?: { total?: number; used?: number; available?: number; pct?: number }
  gpus?: GpuStats[]
  active_requests?: Record<string, ActiveRequestStats>
  ts?: number
}

export interface AdmissionStats {
  model?: string
  limit?: number | null
  effective_limit?: number
  running: number
  queued: number
  oldest_wait_seconds?: number
}

export interface DashboardData {
  stats: SystemStats
  admission: Record<string, AdmissionStats>
  deployments: Deployment[]
  sync: SyncStatus
  nodes: NodeInventoryItem[]
}

export interface StorageModel {
  model_id: string
  size_bytes: number
  last_modified?: string
  revision?: string
  revisions?: string[]
  file_count?: number
}

export interface StorageNode {
  id: string
  name: string
  online: boolean
  total_size: number
  models: StorageModel[]
}

export interface StorageTransferJob {
  id: string
  model_id: string
  source_node_id: string
  source_node_name: string
  target_node_id: string
  target_node_name: string
  status: string
  bytes_total: number
  bytes_transferred: number
  progress?: number
  created_at: string | number
  started_at?: string | number
  completed_at?: string | number
  error?: string
}

export interface StorageState {
  enabled: boolean
  nodes: StorageNode[]
  jobs: StorageTransferJob[]
  instructions: string[]
}

export interface ModelCacheState {
  nodes: StorageNode[]
}

export interface SavedConfiguration {
  id: string
  name: string
  model: string
  engine: 'vllm' | 'sglang'
  image?: string | null
  extra_args_count?: number
  gpu_memory_utilization?: number | null
  gpu_memory_gb?: number | null
  sg_tp_size?: number | null
  sg_context_length?: number | null
  sg_max_running_requests?: number | null
  sg_mem_fraction?: number | null
  sg_image?: string | null
  deployment_mode: string
  required_node_count: number
  tensor_parallel_size: number
  pipeline_parallel_size: number
  model_revision?: string | null
  node_ids: string[]
  supported?: boolean
  error?: string
  launch?: { phase?: string; image?: string }
}

export interface CreateStorageTransferInput {
  model_id: string
  source_node_id: string
  target_node_ids: string[]
}

export interface UsageCounters {
  input: number
  output: number
  cached: number
  requests: number
  input_miss?: number
  measured_cached?: number
  estimated_cached?: number
  gen_tokens?: number
  gen_time_s?: number
}

export interface UsageMember {
  model: string
  alias?: string | null
  merge_group?: string | null
  routed_to?: string | null
}

export interface UsageGroup {
  key: string
  label: string
  merge_group?: string | null
  route_target?: string | null
  models: string[]
  members: UsageMember[]
  stats: UsageCounters
  speed?: {
    tokens?: number
    active_time_s?: number
    tok_s?: number | null
    legacy?: boolean
  }
  total_cost: number
  cost_estimated?: boolean
}

export interface UsageSummary {
  models: Record<string, UsageCounters>
  groups: UsageGroup[]
  total: UsageCounters
}

export interface HourlyUsagePoint {
  hour: string
  input: number
  output: number
  cached: number
  requests: number
}

export interface DailyUsagePoint {
  date: string
  input: number
  output: number
  cached: number
  requests: number
  models?: Record<string, UsageCounters>
}

export interface UsageAnalysis {
  hourly: HourlyUsagePoint[]
  daily: DailyUsagePoint[]
}
