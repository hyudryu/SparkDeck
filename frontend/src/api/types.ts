export type RuntimeKind = 'vllm' | 'llama.cpp' | 'sglang'
export type DeploymentStatus = 'registered' | 'launching' | 'running' | 'ready' | 'starting' | 'stopped' | 'degraded' | 'error' | 'unknown'

export interface RuntimeCompatibility {
  runtime: RuntimeKind
  supported: boolean
  reason?: string
}

export interface BenchmarkAggregate {
  model_id: string
  quantization: string
  prompt_tokens_bucket: number
  inference_tokens_per_second: number
  sample_count: number
  unique_cluster_count: number
  parameter_count?: number | null
  weight_size_bytes?: number | null
}

export interface CommunityEvidencePolicy {
  minimum_samples: number
  exact_match_dimensions: Array<'model_id' | 'quantization' | 'prompt_tokens_bucket'>
  metric: 'inference_tokens_per_second'
}

export interface CommunityAggregatesResponse {
  items: BenchmarkAggregate[]
  availability: 'available' | 'local' | 'not_configured' | 'ok' | 'unavailable'
  evidence_policy: CommunityEvidencePolicy
}

export interface CatalogModel {
  id: string
  author?: string
  name?: string
  revision?: string
  downloads?: number
  likes?: number
  parameter_count?: number | null
  weight_size_bytes?: number | null
  weight_size_source?: 'safetensors' | 'gguf' | null
  tags?: string[]
  runtime_compatibility?: RuntimeCompatibility[]
  local_deployment_ids?: string[]
  community?: BenchmarkAggregate | null
  quantizations?: Array<{
    name: string
    files: Array<{ filename: string; size_bytes?: number | null }>
    weight_size_bytes?: number | null
  }>
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
  artifact?: string
  dtype?: string
  gpu_memory_utilization?: number
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
  deployment_mode?: string
  required_node_count?: number
  last_error?: string
  created_at?: string
  updated_at?: string
  node_ids?: string[]
  selected_nodes?: NodeSummary[]
  desired_state?: 'running' | 'stopped'
  launch_phase?: string
  launch_message?: string
}

export interface DeploymentLaunchControls {
  context_window?: number | null
  max_concurrency?: number | null
  kv_cache_dtype?: string | null
  thinking_mode?: string | null
  dspark_num_speculative_tokens?: number | null
  max_cudagraph_capture_size?: number | null
  max_num_batched_tokens?: number | null
}

export interface DeploymentDetail extends Deployment {
  editable: boolean
  edit_reason?: string | null
  desired_state: 'running' | 'stopped'
  extra_args: string[]
  launch_controls: DeploymentLaunchControls
  gpu_memory_utilization?: number | null
  gpu_memory_gb?: number | null
  sg_tp_size?: number | null
  sg_mem_fraction?: number | null
  image?: string
}

export interface DeploymentUpdateInput {
  extra_args: string[]
  launch_controls: DeploymentLaunchControls
  gpu_memory_utilization?: number | null
  gpu_memory_gb?: number | null
  sg_tp_size?: number | null
  sg_mem_fraction?: number | null
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
  hidden_from_dashboard?: boolean
  docker_ready?: boolean
  status_message?: string | null
  fabric_ready?: boolean
  selectable?: boolean
  // Rename results only: whether the new name reached the node itself
  // ("pending" means an offline worker still answers with its old name).
  name_sync?: 'local' | 'synchronized' | 'pending'
  stats?: SystemStats
  disk?: { total?: number; total_bytes?: number; free?: number; free_bytes?: number }
}

export interface RouterOSPresenceNode {
  node_id: string
  node_name: string
  detected: boolean
  configured: boolean
  connected?: boolean
}

export interface RouterOSPresence {
  detected: boolean
  nodes: RouterOSPresenceNode[]
}

export interface RouterOSDiscoveryCandidate {
  address: string
  identity?: string
  platform?: string
  version?: string
  board?: string
  mac?: string
}

export interface RouterOSHealthItem {
  name: string
  value: unknown
  type?: string
}

export interface RouterOSNodeOverview extends RouterOSPresenceNode {
  connected: boolean
  discovery?: RouterOSDiscoveryCandidate[]
  base_url?: string
  username?: string
  verify_tls?: boolean
  error?: string
  device?: Record<string, unknown>
  health: RouterOSHealthItem[] | null
  fan_settings?: Record<string, unknown> | null
  fan_capabilities?: string[] | null
  interfaces: Array<Record<string, unknown>>
}

export interface RouterOSOverview {
  detected: boolean
  nodes: RouterOSNodeOverview[]
}

export interface RouterOSConnectionInput {
  base_url: string
  username: string
  password: string
  verify_tls: boolean
}

export type FanControlMode = 'curve' | 'pid' | 'hysteresis' | 'manual'

export interface FanCurveSettings {
  curve_points: number[][]
  curve_min_temp: number
  curve_max_temp: number
  min_floor_pct: number
}

export interface FanPidSettings {
  setpoint: number
  kp: number
  ki: number
  kd: number
  min_floor_pct: number
}

export interface FanHysteresisSettings {
  hyst_on_temp: number
  hyst_off_temp: number
}

export interface FanManualSettings {
  manual_duty_pct: number
}

export interface FanControlSettings {
  mode: FanControlMode
  settings: {
    curve: FanCurveSettings
    pid: FanPidSettings
    hysteresis: FanHysteresisSettings
    manual: FanManualSettings
  }
}

export interface FanSettingsUpdateResult {
  node_id: string
  mode: FanControlMode
  previous_mode: FanControlMode
  active_settings: FanCurveSettings | FanPidSettings | FanHysteresisSettings | FanManualSettings
}

export interface FanControlState {
  rpm?: number | null
  duty_byte?: number | null
  duty_pct?: number | null
  temp?: number | null
  local_temp?: number | null
  temperature_override?: Record<string, unknown> | null
  temperature_override_active?: boolean
  mode: FanControlMode
  active_settings?: Record<string, unknown>
  status?: string | null
  max_speed: boolean
  ts: number
}

export interface FanControlNode {
  node_id: string
  node_name: string
  local: boolean
  fan: FanControlState
  settings: FanControlSettings
}

export interface FanControlOverview {
  available: boolean
  nodes: FanControlNode[]
}

export interface FanMaxSpeedResult {
  node_id: string
  enabled: boolean
}

export interface RenameNodeInput {
  name: string
}

export interface OnboardingStatus {
  role: 'controller' | 'worker'
  node: { id: string; name: string; port: number; access_urls: string[] }
  controller_url?: string
  controller_node_id?: string
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

export interface BenchmarkModelSummary {
  model_id: string
  run_count: number
  best_prompt_tokens_per_second: number
  best_generation_tokens_per_second: number
  context_windows: number[]
  tensor_parallel_sizes: number[]
  latest_at: string
}

export interface BenchmarkSeriesPoint {
  context_window_size: number
  concurrency: 1 | 2 | 5 | 10
  tensor_parallel_size: number
  prompt_tokens_per_second: number
  generation_tokens_per_second: number
  sample_count: number
}

export interface BenchmarkModelDetail {
  model_id: string
  points: BenchmarkSeriesPoint[]
}

export interface DevicePairing {
  status: 'paired' | 'not_paired'
  sub?: string
  email?: string
}

export interface CommunityClusterSync {
  applied: string[]
  conflicts: { node: string; email?: string }[]
  errors: string[]
}

export interface CommunityPairResponse {
  pairing: DevicePairing
  cluster?: CommunityClusterSync
}

export interface CommunityAuthConfig {
  idp_endpoint: string
  client_id: string
}

export interface CommunitySession {
  status: 'signed-in' | 'signed-out' | 'reauth-required'
  email?: string
  token_invalid?: boolean
}

export interface SyncStatus {
  sharing_enabled: boolean
  account_paired: boolean
  /** The stored refresh token was rejected; the account must sign in again. */
  token_invalid: boolean
  upload_configured: boolean
  pending_count: number
  synced_count: number
  failed_count: number
  last_sync_at?: string | null
  last_error?: string | null
  /** Joined nodes that did not apply the latest consent change. */
  cluster_errors?: string[]
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
  // Retained for backward compatibility with settings saved before these
  // controls were removed from the Settings page.
  default_runtime?: RuntimeKind
  default_context_length?: number
  hf_token?: string
  hf_token_configured?: boolean
  telemetry_interval_seconds?: number
  [key: string]: unknown
}

export interface SystemUpdateRelease {
  tag: string
  revision?: string
  name: string
  url?: string
  published_at?: string
  prerelease?: boolean
}

export interface SystemUpdateNode {
  id: string
  name: string
  local: boolean
  online: boolean
  current_revision?: string
  phase?: string
  error?: string
  blockers: string[]
}

export interface SystemUpdateJob {
  id: string
  active: boolean
  phase: string
  message?: string
  error?: string
  target_branch: 'main'
  target_revision: string
  nodes: SystemUpdateNode[]
}

export interface SystemUpdateOverview {
  repository: string
  current_revision?: string
  target?: {
    branch: 'main'
    revision: string
    url?: string
  } | null
  up_to_date?: boolean
  can_update: boolean
  blockers: string[]
  nodes: SystemUpdateNode[]
  job?: SystemUpdateJob | null
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
  usage?: ChatUsage
}

export interface ChatUsage {
  prompt_tokens?: number
  completion_tokens?: number
  total_tokens?: number
  prompt_tokens_details?: { cached_tokens?: number }
}

export interface ChatResponseMetrics {
  prompt_tokens_per_second?: number
  ttft_ms?: number
  output_tokens_per_second?: number
  prompt_tokens?: number
  completion_tokens?: number
}

export interface ChatStreamUpdate {
  content?: string
  reasoning?: string
  metrics?: ChatResponseMetrics
}

export interface ChatStreamResult {
  message: ChatMessage
  reasoning: string
  usage?: ChatUsage
  metrics: ChatResponseMetrics
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
  cpu_logical_count?: number | null
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
  partial?: boolean
  has_partial_download?: boolean
  partial_size_bytes?: number
  last_modified?: string
  revision?: string
  revisions?: string[]
  file_count?: number
}

export interface StorageNode {
  id: string
  name: string
  online: boolean
  total_size?: number | null
  cache_free_size?: number | null
  free_size?: number | null
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
  kind?: 'download' | 'transfer'
  revision?: string
  depends_on_job_id?: string | null
  workflow_id?: string | null
  workflow_node_ids?: string[]
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

export interface StorageTransferPreflightTarget {
  node_id: string
  node_name: string
  eligible: boolean
  reason?: string | null
  free_bytes?: number | null
  required_free_bytes?: number | null
  active_job_id?: string | null
  active_job_status?: string | null
  active_job_kind?: string | null
  has_preparation_conflict?: boolean
  preparation_conflict_reason?: string | null
  has_required_weights?: boolean
  has_model_cache?: boolean
  download_eligible?: boolean
  download_reason?: string | null
  download_required_free_bytes?: number | null
  transfer_after_download_eligible?: boolean
  transfer_after_download_reason?: string | null
  transfer_after_download_required_free_bytes?: number | null
}

export interface StorageTransferPreflight {
  enabled: boolean
  model_id: string
  revision: string
  resolved_revision?: string | null
  source?: { node_id: string; node_name: string; size_bytes: number } | null
  sources?: { node_id: string; node_name: string; size_bytes: number }[]
  download?: { size_bytes: number; required_free_bytes: number } | null
  download_error?: string | null
  targets: StorageTransferPreflightTarget[]
  staging_reserve_bytes: number
}

export interface RecipePreparationPlan extends StorageTransferPreflight {
  node_ids: string[]
  eligible: boolean
  action: 'ready' | 'transfer' | 'download'
  download_node_id?: string | null
  download_node_ids?: string[]
  transfer_target_node_ids: string[]
  reason?: string | null
}

export interface StorageTransferResult {
  job_ids: string[]
  jobs: StorageTransferJob[]
}

export interface RecipePreparationResult extends StorageTransferResult {
  workflow_id?: string | null
  plan?: RecipePreparationPlan
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

export interface LaunchControls {
  context_window?: number | null
  max_concurrency?: number | null
  kv_cache_dtype?: string | null
  thinking_mode?: string | null
  dspark_num_speculative_tokens?: number | null
  max_cudagraph_capture_size?: number | null
  max_num_batched_tokens?: number | null
}

export interface SavedConfigurationDetail extends SavedConfiguration {
  extra_args: string[]
  launch_controls: LaunchControls
}

export interface RecipeUpdateInput {
  name?: string
  extra_args?: string[]
  launch_controls?: LaunchControls
  gpu_memory_utilization?: number | null
  gpu_memory_gb?: number | null
  sg_tp_size?: number | null
  sg_context_length?: number | null
  sg_max_running_requests?: number | null
  sg_mem_fraction?: number | null
  sg_image?: string | null
}

export interface CreateStorageTransferInput {
  model_id: string
  source_node_id: string
  target_node_ids: string[]
  revision?: string
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
  routing_rules?: Record<string, string>
  merge_groups?: Record<string, string>
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

export interface BenchyStatus {
  installed: boolean
  version?: string | null
  launch_mode?: 'path' | 'python_module' | null
  path_on_host: boolean
  active_run_id?: string | null
}

export interface BenchyServedModel {
  id: string
  label: string
  runtime?: string
  deployment_id?: string | null
  model: string
  quantization?: string | null
  base_url: string
}

export interface BenchyRunConfig {
  model_id: string
  prompt_sizes: number[]
  response_sizes: number[]
  concurrency_levels: number[]
  context_depths: number[]
  runs: number
  warmup_runs: number
  exact_tg: boolean
}

export interface BenchyRunProgress {
  requests_done: number
  requests_failed: number
  current?: {
    prompt_size?: number
    response_size?: number
    context_depth?: number
    concurrency?: number
  } | null
}

export type BenchyRunStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface BenchyResultRow {
  prompt_size?: number
  response_size?: number
  context_depth?: number
  concurrency?: number
  is_context_prefill_phase?: boolean
  pp_tokens_per_second?: number | null
  pp_tokens_per_second_std?: number | null
  pp_tokens_per_second_request?: number | null
  pp_tokens_per_second_request_std?: number | null
  tg_tokens_per_second?: number | null
  tg_tokens_per_second_std?: number | null
  tg_tokens_per_second_request?: number | null
  tg_tokens_per_second_request_std?: number | null
  peak_tg_tokens_per_second?: number | null
  peak_tg_tokens_per_second_request?: number | null
  ttfr_ms?: number | null
  est_ppt_ms?: number | null
  e2e_ttft_ms?: number | null
}

export interface BenchyRunSummary {
  id: string
  status: BenchyRunStatus
  created_at: string
  started_at?: string | null
  finished_at?: string | null
  duration_seconds?: number | null
  model: string
  model_id: string
  quantization?: string | null
  runtime?: string | null
  base_url?: string
  deployment_id?: string | null
  config: BenchyRunConfig
  benchy_version?: string | null
  error?: string | null
  result_count?: number
  progress?: BenchyRunProgress
}

export interface BenchyRunDetail extends BenchyRunSummary {
  results: BenchyResultRow[]
  csv_filename?: string | null
  report?: {
    benchy_version?: string | null
    latency_mode?: string | null
    latency_ms?: number | null
    prefix_caching_enabled?: boolean | null
  } | null
}
