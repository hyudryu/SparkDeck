export type RuntimeKind = 'vllm' | 'llama.cpp' | 'sglang'
export type DeploymentStatus = 'registered' | 'running' | 'starting' | 'stopped' | 'error' | 'unknown'

export interface RuntimeCompatibility {
  runtime: RuntimeKind
  supported: boolean
  reason?: string
}

export interface BenchmarkAggregate {
  model_id: string
  model_revision?: string
  runtime: RuntimeKind
  quantization?: string
  hardware_class?: string
  context_length?: number
  median_tokens_per_second?: number
  p25_tokens_per_second?: number
  p75_tokens_per_second?: number
  median_ttft_ms?: number
  sample_count: number
  distinct_device_count: number
  community_proven?: boolean
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
  huggingface_token_configured?: boolean
  default_runtime?: RuntimeKind
  default_context_length?: number
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
}

export interface StorageModel {
  model_id: string
  size_bytes: number
  last_modified?: string
  revision?: string
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

export interface CreateStorageTransferInput {
  model_id: string
  source_node_id: string
  target_node_ids: string[]
}
