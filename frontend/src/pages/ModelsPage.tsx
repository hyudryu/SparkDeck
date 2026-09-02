import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import { ArrowDownToLine, Bookmark, Check, ChevronDown, ChevronRight, Copy, FolderPlus, HardDrive, Pencil, Play, Plus, ScrollText, Server, Settings2, Trash2, UploadCloud, X } from 'lucide-react'
import { Link, useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import type { AppSettings, CreateDeploymentInput, Deployment, DeploymentLogsResponse, RecipeUpdateInput, RuntimeKind, SavedConfiguration, SavedConfigurationDetail, StorageTransferPreflightTarget } from '../api/types'
import { KvCacheDtypeSelect } from '../components/KvCacheDtypeSelect'
import { Button, EmptyState, ErrorState, LoadingState, PageHeader, Panel, RuntimeMark, SplitButton, Status, Tooltip } from '../components/ui'
import { useConfirmDialog } from '../components/useConfirmDialog'
import { isNodeSelectable, NodeSelector, selectedNodeLabel } from '../components/NodeSelector'
import { useResource } from '../hooks/useResource'
import { formatEnvironment, parseEnvironment } from '../utils/environment'
import { formatBytes } from '../utils/format'
import { artifactFilesDownloaded, ggufArtifactOptions, type GgufArtifactOption, type GgufQuantization } from '../utils/gguf'

const initialForm: CreateDeploymentInput = {
  alias: '',
  model_id: '',
  runtime: 'vllm',
  managed: true,
  endpoint_url: '',
  settings: {
    context_length: 8192,
    tensor_parallel_size: 1,
    image: 'nvcr.io/nvidia/vllm:26.03.post1-py3',
  },
  node_ids: ['local'],
  deployment_mode: 'single',
}

const isRuntimeKind = (value: unknown): value is RuntimeKind => value === 'vllm' || value === 'llama.cpp' || value === 'sglang'

const EMPTY_QUANTIZATIONS: GgufQuantization[] = []
const EMPTY_FILE_SETS: ReadonlyArray<ReadonlySet<string>> = []

// Sentinel option in the GGUF artifact dropdown: switching to it reveals
// the text field so a controller-local artifact path can be entered.
const MANUAL_ARTIFACT_OPTION = '__enter-local-path__'

// Dropdown option labels state the download size and whether the weights
// already sit in the cluster cache, so the badge travels with the text.
const quantizationOptionLabel = (variant: GgufQuantization, downloaded: boolean) => (
  `${variant.name}${variant.weight_size_bytes ? ` · ${formatBytes(variant.weight_size_bytes)}` : ''}${downloaded ? ' · ✓ Downloaded' : ''}`
)
const artifactOptionLabel = (option: GgufArtifactOption, downloaded: boolean) => (
  `${option.filename}${option.weightSize ? ` · ${formatBytes(option.weightSize)}` : ''}${downloaded ? ' · ✓ Downloaded' : ''}`
)
const quantizationFilesDownloaded = (
  variant: GgufQuantization,
  cachedFileSets: ReadonlyArray<ReadonlySet<string>>,
) => [variant.files, ...(variant.artifacts ?? []).map((artifact) => artifact.files)]
  .some((files) => artifactFilesDownloaded(files, cachedFileSets))

function deploymentDefaults(settings?: AppSettings, localNodeId = 'local'): CreateDeploymentInput {
  const runtime = isRuntimeKind(settings?.default_runtime) ? settings.default_runtime : initialForm.runtime
  const savedContextLength = settings?.default_context_length
  const contextLength = typeof savedContextLength === 'number'
    && Number.isInteger(savedContextLength)
    && savedContextLength >= 256
    ? savedContextLength
    : initialForm.settings.context_length
  const vllmImage = typeof settings?.vllm_image === 'string' && settings.vllm_image.trim()
    ? settings.vllm_image.trim()
    : initialForm.settings.image
  return {
    ...initialForm,
    runtime,
    settings: runtime === 'llama.cpp'
      ? { context_length: contextLength, parallel_slots: 1, gpu_layers: 99 }
      : {
        context_length: contextLength,
        tensor_parallel_size: 1,
        image: runtime === 'vllm' ? vllmImage : undefined,
      },
    node_ids: [localNodeId],
  }
}

const isLocalModelPath = (model: string) => model.startsWith('/') || model.startsWith('~')

// A GGUF artifact is controller-local when it names a file on this machine
// instead of a path inside the shared Hugging Face cache.
const isLocalArtifact = (artifact?: string) => Boolean(
  artifact && (/^\//.test(artifact) || artifact.startsWith('~') || /^[A-Za-z]:[\\/]/.test(artifact)),
)

const isControllerArtifact = (deployment: Deployment) => (
  isLocalModelPath(deployment.model_id)
  || (deployment.runtime === 'llama.cpp' && isLocalArtifact(deployment.settings.artifact))
)

// "Target nodes" alone is ambiguous: a selection can mean one model split
// across the nodes (tensor parallelism) or one complete model per node
// (parallel instances). Every picker states which one it means.
const layoutLegend = (mode: string | undefined, nodeCount: number) => {
  if (mode === 'sharded') return 'Target nodes · tensor parallelism (one model split across nodes)'
  if (mode === 'replicated' || nodeCount > 1) return 'Target nodes · parallel instances (a full model copy per node)'
  return 'Target node · single instance'
}

const layoutHelp = (mode: string | undefined) => {
  if (mode === 'sharded') {
    return 'Tensor parallelism: the selected nodes cooperate on one model, so the node count and tensor parallel size must match.'
  }
  if (mode === 'replicated') {
    return 'Parallel instances: every selected node runs its own complete copy of the model.'
  }
  return 'The deployment runs on one node.'
}

const deploymentTargetLayout = (deployment: Deployment) => {
  if (deployment.deployment_mode === 'sharded') {
    const tp = deployment.settings.tensor_parallel_size
    return tp && tp > 1 ? ` · tensor parallel (TP${tp})` : ' · tensor parallel'
  }
  if (deployment.deployment_mode === 'replicated') return ' · parallel instances'
  return ''
}

export function recipePreparationRequiredBytes(
  option: StorageTransferPreflightTarget | undefined,
  hasExactSource: boolean,
) {
  if (!option) return undefined
  if (hasExactSource) {
    return option.has_model_cache && !option.has_required_weights
      ? option.download_required_free_bytes
      : option.required_free_bytes
  }
  return Math.max(
    option.download_required_free_bytes ?? 0,
    option.transfer_after_download_required_free_bytes ?? 0,
  )
}

const PIN_STORAGE_KEY = 'sparkdeck:pinned-recipes'
const SORT_STORAGE_KEY = 'sparkdeck:models-sort'

type SortMode = 'recent' | 'name-asc' | 'name-desc'

const ACTIVE_DEPLOYMENT_STATUSES = new Set<Deployment['status']>(['launching', 'starting', 'stopping'])
const STOPPABLE_DEPLOYMENT_STATUSES = new Set<Deployment['status']>(['launching', 'starting', 'running', 'ready'])
const PRE_CONTAINER_LAUNCH_PHASES = new Set(['queued', 'preparing', 'checking_image', 'pulling_image', 'creating_container'])
const FINISHED_LAUNCH_PHASES = new Set(['ready', 'error', 'failed', 'stopped', 'exited'])

const deploymentNeedsPoll = (deployment: Deployment) => (
  ACTIVE_DEPLOYMENT_STATUSES.has(deployment.status)
  || Boolean(deployment.launch_phase && !FINISHED_LAUNCH_PHASES.has(deployment.launch_phase))
)

const showLaunchDetails = (deployment: Deployment) => !(
  deployment.status === 'stopped' && deployment.launch_phase === 'exited'
)

const formatLaunchPhase = (phase: string) => phase
  .replaceAll('_', ' ')
  .replace(/\b\w/g, (character) => character.toUpperCase())

const formatInferenceAge = (timestamp: number, now: number) => {
  const seconds = Math.max(0, Math.round(now - timestamp))
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

const deploymentTimestampMs = (timestamp: string | number | undefined) => {
  if (timestamp === undefined) return 0
  const value = new Date(typeof timestamp === 'number' ? timestamp * 1000 : timestamp).getTime()
  return Number.isNaN(value) ? 0 : value
}

export const formatDeploymentTimestamp = (timestamp: string | number, now = Date.now() / 1000) => {
  const timestampMs = deploymentTimestampMs(timestamp)
  if (!timestampMs) return undefined
  const date = new Date(timestampMs)
  if (Number.isNaN(date.getTime())) return undefined
  const ageSeconds = Math.max(0, Math.floor(now - date.getTime() / 1000))
  if (ageSeconds < 60) return 'just now'
  const minutes = Math.floor(ageSeconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  if (days <= 7) return `${days}d ago`
  return date.toLocaleDateString([], { dateStyle: 'medium' })
}

const deploymentConcurrency = (deployment: Deployment) => (
  deployment.settings.max_concurrency
  ?? deployment.settings.max_running_requests
  ?? deployment.settings.parallel_slots
)

const isDiscoveredExternal = (deployment: Deployment) => (
  !deployment.managed && deployment.id.startsWith('container:')
)

const canPromoteDiscovered = (deployment: Deployment) => (
  isDiscoveredExternal(deployment) && deployment.promotable !== false
)

// Scalar flags the structured argument editor manages; everything else in a
// saved configuration's extra args is shown verbatim in the "Other flags"
// field. Compound JSON flags (--speculative-config,
// --default-chat-template-kwargs) intentionally stay in "Other flags" so a
// save round-trips their full payloads — the backend merges the structured
// speculative-token count / thinking mode into the JSON without discarding
// unrelated properties.
const CONTROLLED_FLAGS = new Set([
  '--max-model-len', '--max-model-length', '--max-num-seqs', '--kv-cache-dtype',
  '--context-length', '--max-running-requests', '--max-cudagraph-capture-size',
  '--max-num-batched-tokens',
])

const shellQuote = (arg: string) => (/[\s"']/.test(arg) ? `'${arg.replace(/'/g, `'\\''`)}'` : arg)

// Shell-aware whitespace splitter: respects single/double quotes, strips them.
function shellSplit(input: string): string[] {
  const out: string[] = []
  let cur = ''
  let quote: string | null = null
  let inWord = false
  for (let i = 0; i < input.length; i++) {
    const ch = input[i]
    if (quote) {
      if (ch === quote) quote = null
      else cur += ch
      inWord = true
    } else if (ch === '"' || ch === "'") {
      quote = ch
      inWord = true
    } else if (/\s/.test(ch)) {
      if (inWord) { out.push(cur); cur = ''; inWord = false }
    } else if (ch === '\\' && i + 1 < input.length) {
      cur += input[i + 1]
      i++
      inWord = true
    } else {
      cur += ch
      inWord = true
    }
  }
  if (inWord) out.push(cur)
  return out
}

// Extra args minus the flags the structured editor controls, shell-quoted for display.
function remainingArgs(args: string[]): string {
  const out: string[] = []
  for (let i = 0; i < args.length; i++) {
    const token = args[i]
    const flag = token.split('=')[0]
    if (CONTROLLED_FLAGS.has(flag)) {
      if (!token.includes('=') && i + 1 < args.length && !args[i + 1].startsWith('-')) i++
      continue
    }
    out.push(token)
  }
  return out.map(shellQuote).join(' ')
}

const companyOf = (recipe: SavedConfiguration) => {
  const first = (recipe.model || recipe.name || '').split('/')[0]?.trim()
  return first || 'Other'
}

type ArgsForm = Record<string, string>

type ArgsEditorState = {
  open: boolean
  loading: boolean
  saving: boolean
  saved: boolean
  error?: string
  form: ArgsForm
}

const SPECULATIVE_METHODS = ['dspark', 'dflash', 'draft_model', 'eagle3', 'mtp', 'ngram', 'ngram_gpu', 'suffix']
const DRAFT_SAMPLE_METHODS = ['greedy', 'probabilistic']

const seedArgsForm = (detail: SavedConfigurationDetail): ArgsForm => {
  const controls = detail.launch_controls ?? {}
  return {
    context_window: controls.context_window?.toString() ?? '',
    max_concurrency: controls.max_concurrency?.toString() ?? '',
    kv_cache_dtype: controls.kv_cache_dtype ?? '',
    thinking_mode: controls.thinking_mode ?? 'default',
    speculative_method: controls.speculative_method ?? '',
    draft_sample_method: controls.draft_sample_method ?? '',
    dspark_num_speculative_tokens: controls.dspark_num_speculative_tokens?.toString() ?? '',
    max_cudagraph_capture_size: controls.max_cudagraph_capture_size?.toString() ?? '',
    max_num_batched_tokens: controls.max_num_batched_tokens?.toString() ?? '',
    gpu_memory_utilization: detail.gpu_memory_utilization?.toString() ?? '',
    gpu_memory_gb: detail.gpu_memory_gb?.toString() ?? '',
    sg_tp_size: detail.sg_tp_size?.toString() ?? '',
    sg_mem_fraction: detail.sg_mem_fraction?.toString() ?? '',
    remaining_flags: remainingArgs(detail.extra_args ?? []),
  }
}

const readPinned = (): string[] => {
  try {
    const parsed = JSON.parse(localStorage.getItem(PIN_STORAGE_KEY) ?? '[]')
    return Array.isArray(parsed) ? parsed.filter((id) => typeof id === 'string') : []
  } catch {
    return []
  }
}

const readSortMode = (): SortMode => {
  const value = localStorage.getItem(SORT_STORAGE_KEY)
  return value === 'name-asc' || value === 'name-desc' ? value : 'recent'
}

type RecipeTrackedJob = {
  jobId: string
  targetId: string
  modelId: string
  recipeId: string
}

const recipeJobKey = (recipeId: string, modelId: string, targetId: string) => `${recipeId}\u0000${modelId}\u0000${targetId}`

export function ModelsPage() {
  const { confirm, confirmationDialog } = useConfirmDialog()
  const [searchParams, setSearchParams] = useSearchParams()
  // Tick so the relative "last inference" age on running rows stays fresh.
  const [now, setNow] = useState(() => Date.now() / 1000)
  const [recipeDeployment, setRecipeDeployment] = useState<{ recipe: SavedConfiguration; nodeIds: string[] }>()
  const [recipeSeedNodeId, setRecipeSeedNodeId] = useState<string>()
  const acceptedDeployments = useRef(new Map<string, Deployment>())
  const resource = useResource(async (signal) => {
    const deployments = await api.deployments.list(signal)
    const accepted = acceptedDeployments.current
    if (!accepted.size) return deployments
    const loadedIds = new Set(deployments.map((deployment) => deployment.id))
    loadedIds.forEach((id) => accepted.delete(id))
    return [...accepted.values(), ...deployments]
  })
  const nodes = useResource((signal) => api.nodes.list(signal))
  const onboarding = useResource((signal) => api.onboarding.get(signal))
  const appSettings = useResource((signal) => api.settings.get(signal))
  const modelCache = useResource((signal) => api.modelCache.get(signal))
  const recipes = useResource((signal) => api.recipes.list(signal))
  const transferPreflight = useResource(
    (signal) => {
      if (!recipeDeployment) throw new Error('No recipe selected')
      return api.storage.preflight(
        recipeDeployment.recipe.model,
        recipeDeployment.recipe.model_revision ?? 'main',
        signal,
      )
    },
    [
      recipeDeployment?.recipe.id,
      recipeDeployment?.recipe.model,
      recipeDeployment?.recipe.model_revision,
    ],
    Boolean(recipeDeployment && !isLocalModelPath(recipeDeployment.recipe.model)),
  )
  const reloadModelCache = modelCache.reload
  const reloadTransferPreflight = transferPreflight.reload
  const [creating, setCreating] = useState(false)
  const [form, setForm] = useState<CreateDeploymentInput>(initialForm)
  const [busy, setBusy] = useState<string>()
  const [formError, setFormError] = useState<string>()
  const [actionError, setActionError] = useState<string>()
  const [actionNotice, setActionNotice] = useState<string>()
  const [recipeError, setRecipeError] = useState<string>()
  const [recipeTransferNotice, setRecipeTransferNotice] = useState<string>()
  const [recipeTransferJobs, setRecipeTransferJobs] = useState<Record<string, RecipeTrackedJob>>({})
  const [sortMode, setSortMode] = useState<SortMode>(readSortMode)
  const [pinned, setPinned] = useState<string[]>(readPinned)
  const [expandedGroups, setExpandedGroups] = useState<string[]>([])
  const [renaming, setRenaming] = useState<{ id: string; value: string }>()
  const logRequestRef = useRef(0)
  const [logViewer, setLogViewer] = useState<Deployment>()
  const [logData, setLogData] = useState<DeploymentLogsResponse>()
  const [selectedLogNodeId, setSelectedLogNodeId] = useState<string>()
  const [logLoading, setLogLoading] = useState(false)
  const [logError, setLogError] = useState<string>()
  const [logTailing, setLogTailing] = useState(false)
  const logPanelRef = useRef<HTMLDivElement>(null)
  const [startSelection, setStartSelection] = useState<{ deployment: Deployment; nodeIds: string[] }>()
  const [startError, setStartError] = useState<string>()
  const [startNotice, setStartNotice] = useState<string>()
  const [editingDeployment, setEditingDeployment] = useState<Deployment>()
  const [launchSeed, setLaunchSeed] = useState<string>()
  // Per-node preparation plan for launching a saved deployment: which nodes
  // hold the weights, which can receive them, and which are blocked (for
  // example by free disk space on the model-cache volume).
  const startPreflight = useResource(
    (signal) => {
      if (!startSelection || startSelection.deployment.status !== 'saved') {
        throw new Error('No saved deployment selected')
      }
      const selectableIds = (nodes.data ?? []).filter(isNodeSelectable).map((node) => node.id)
      if (!selectableIds.length) throw new Error('No selectable nodes')
      return api.deployments.preparePreflight(startSelection.deployment.id, selectableIds, signal)
    },
    [startSelection?.deployment.id],
    Boolean(
      startSelection
      && startSelection.deployment.status === 'saved'
      && !isControllerArtifact(startSelection.deployment)
      && nodes.data?.length,
    ),
  )
  const [additionalLaunch, setAdditionalLaunch] = useState<{ deployment: Deployment; currentIds: string[]; additionalIds: string[] }>()
  const [additionalError, setAdditionalError] = useState<string>()
  const [argsEditors, setArgsEditors] = useState<Record<string, ArgsEditorState>>({})
  const [launchArgsOpen, setLaunchArgsOpen] = useState(false)
  const [extraFlags, setExtraFlags] = useState('')
  const [gpuMemoryUtil, setGpuMemoryUtil] = useState('')
  const [runtimeEnvironment, setRuntimeEnvironment] = useState('')
  const defaultsApplied = useRef(false)
  const runtimeTouched = useRef(false)
  const contextLengthTouched = useRef(false)
  const catalogShardedLayout = useRef(false)
  // Provenance refs for the deployment creator: which quantization/artifact
  // values were derived from a repository listing rather than typed by
  // hand, plus the last model id seen for invalidating them. (The retained
  // catalog listing's provenance lives in state, see catalogResponseQuery
  // below.)
  const linkedQuantizationRef = useRef<string | undefined>(undefined)
  const linkedArtifactRef = useRef<string | undefined>(undefined)
  const previousModelIdRef = useRef('')
  const reloadDeployments = resource.reload
  const recoveredStalledInitialLoad = useRef(false)

  useEffect(() => {
    if (!resource.loading || resource.data || resource.error || recoveredStalledInitialLoad.current) return
    // A stale browser connection can leave the first request pending even
    // while a fresh request succeeds. Recover once without shortening the
    // full timeout contract for the replacement request.
    const timer = window.setTimeout(() => {
      recoveredStalledInitialLoad.current = true
      reloadDeployments()
    }, 10_000)
    return () => window.clearTimeout(timer)
  }, [resource.data, resource.error, resource.loading, reloadDeployments])

  useEffect(() => {
    const timer = window.setInterval(() => {
      setNow(Date.now() / 1000)
      // Running rows otherwise stop polling, which would freeze the
      // last-inference age at whatever the page first loaded.
      if (resource.data?.some((deployment) => deployment.status === 'running')) {
        reloadDeployments()
      }
    }, 30_000)
    return () => window.clearInterval(timer)
  }, [resource.data, reloadDeployments])

  useEffect(() => {
    if (resource.loading || !resource.data?.some(deploymentNeedsPoll)) return
    const timer = window.setTimeout(reloadDeployments, 2000)
    return () => window.clearTimeout(timer)
  }, [resource.data, resource.loading, reloadDeployments])

  useEffect(() => {
    if (!recipeDeployment) return
    const { recipe } = recipeDeployment
    const active = (transferPreflight.data?.targets ?? [])
      .filter((target) => target.active_job_id && (target.active_job_status === 'queued' || target.active_job_status === 'running'))
    if (!active.length) return
    setRecipeTransferJobs((current) => {
      const next = { ...current }
      let changed = false
      for (const target of active) {
        const key = recipeJobKey(recipe.id, recipe.model, target.node_id)
        if (next[key]?.jobId !== target.active_job_id) {
          next[key] = {
            jobId: target.active_job_id as string,
            targetId: target.node_id,
            modelId: recipe.model,
            recipeId: recipe.id,
          }
          changed = true
        }
      }
      return changed ? next : current
    })
  }, [recipeDeployment, transferPreflight.data])

  useEffect(() => {
    const tracked = Object.entries(recipeTransferJobs)
    if (!tracked.length) return
    let disposed = false
    let timer: number | undefined
    const poll = async () => {
      try {
        const state = await api.storage.get()
        if (disposed) return
        const completed: RecipeTrackedJob[] = []
        const finalKeys: string[] = []
        const failedMessages: string[] = []
        let active = false
        for (const [key, trackedJob] of tracked) {
          const job = state.jobs.find((item) => item.id === trackedJob.jobId)
          if (!job || job.status === 'queued' || job.status === 'running') {
            active = true
          } else if (job.status === 'completed') {
            completed.push(trackedJob)
            finalKeys.push(key)
          } else {
            finalKeys.push(key)
            if (
              trackedJob.recipeId === recipeDeployment?.recipe.id
              && trackedJob.modelId === recipeDeployment.recipe.model
            ) failedMessages.push(job.error || `Model preparation ${job.status}`)
          }
        }
        if (finalKeys.length) {
          const finalKeySet = new Set(finalKeys)
          setRecipeTransferJobs((current) => Object.fromEntries(
            Object.entries(current).filter(([key]) => !finalKeySet.has(key)),
          ))
          reloadModelCache()
          reloadTransferPreflight()
        }
        const completedCurrent = completed.filter((item) => (
          item.recipeId === recipeDeployment?.recipe.id
          && item.modelId === recipeDeployment.recipe.model
        ))
        if (completedCurrent.length) {
          const names = completedCurrent.map((item) => nodes.data?.find((node) => node.id === item.targetId)?.name ?? item.targetId)
          setRecipeTransferNotice(`Weights transferred to ${names.join(', ')}. The node is now available for deployment.`)
        }
        if (failedMessages.length) setRecipeError(failedMessages.join(' '))
        if (active) timer = window.setTimeout(() => void poll(), 3000)
      } catch (reason) {
        if (!disposed) {
          setRecipeError(reason instanceof Error ? reason.message : 'Could not check model transfer status')
          timer = window.setTimeout(() => void poll(), 3000)
        }
      }
    }
    void poll()
    return () => {
      disposed = true
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [recipeDeployment, recipeTransferJobs, nodes.data, reloadModelCache, reloadTransferPreflight])

  useEffect(() => {
    const modelId = searchParams.get('model')?.trim()
    if (!modelId) return
    const sharded = searchParams.get('layout') === 'sharded'
    const requestedRuntimeValue = searchParams.get('runtime')?.trim()
    const requestedRuntime = isRuntimeKind(requestedRuntimeValue) ? requestedRuntimeValue : undefined
    const quantization = searchParams.get('quantization')?.trim()
    const artifact = searchParams.get('artifact')?.trim()
    const ggufArtifact = Boolean(artifact?.toLocaleLowerCase().endsWith('.gguf'))
    catalogShardedLayout.current = sharded
    if (ggufArtifact || requestedRuntime) runtimeTouched.current = true
    setForm((current) => ({
      ...current,
      model_id: modelId,
      alias: current.alias || modelId.split('/').at(-1) || modelId,
      runtime: ggufArtifact ? 'llama.cpp' : requestedRuntime ?? current.runtime,
      deployment_mode: ggufArtifact ? 'single' : sharded ? 'sharded' : current.deployment_mode,
      settings: ggufArtifact
        ? {
          context_length: current.settings.context_length,
          parallel_slots: current.settings.parallel_slots ?? 1,
          gpu_layers: current.settings.gpu_layers ?? 99,
          quantization: quantization || undefined,
          artifact,
        }
        : {
          ...current.settings,
          quantization: quantization || current.settings.quantization,
        },
    }))
    // A deep link names repository artifacts, so record them as
    // listing-derived for the stale-selection cleanup above, and treat the
    // model id as already observed so the linked picks are not wiped.
    if (ggufArtifact && artifact) {
      linkedArtifactRef.current = artifact
      linkedQuantizationRef.current = quantization || undefined
    }
    previousModelIdRef.current = modelId
    setCreating(true)
    setSearchParams({}, { replace: true })
  }, [searchParams, setSearchParams])

  useEffect(() => {
    const inventory = nodes.data
    if (!inventory?.length) return
    setForm((current) => {
      const local = inventory.find((node) => node.local)
      const available = (current.node_ids ?? []).filter((id) => inventory.some((node) => node.id === id && isNodeSelectable(node)))
      const fallback = local && isNodeSelectable(local) ? local : inventory.find(isNodeSelectable)
      const requestedSharded = current.deployment_mode === 'sharded' || catalogShardedLayout.current
      const shardedCandidates = available.length > 1
        ? available
        : inventory.filter(isNodeSelectable).map((node) => node.id)
      const nodeIds = requestedSharded
        ? shardedCandidates
        : available.length ? available : fallback ? [fallback.id] : []
      const sharded = requestedSharded && nodeIds.length > 1
      const deploymentMode = sharded && nodeIds.length > 1 ? 'sharded' : nodeIds.length > 1 ? 'replicated' : 'single'
      catalogShardedLayout.current = false
      return {
        ...current,
        node_ids: nodeIds,
        deployment_mode: deploymentMode,
        settings: current.runtime === 'llama.cpp' ? current.settings : {
          ...current.settings,
          tensor_parallel_size: deploymentMode === 'sharded' ? nodeIds.length : 1,
        },
      }
    })
  }, [nodes.data])

  const localNodeId = nodes.data?.find((node) => node.local)?.id
  const shardedAvailable = (nodes.data?.filter(isNodeSelectable).length ?? 0) > 1

  useEffect(() => {
    if (!appSettings.data || defaultsApplied.current) return
    defaultsApplied.current = true
    setForm((current) => {
      const defaults = deploymentDefaults(appSettings.data, localNodeId ?? current.node_ids?.[0] ?? 'local')
      const runtime = runtimeTouched.current ? current.runtime : defaults.runtime
      const contextLength = contextLengthTouched.current
        ? current.settings.context_length
        : defaults.settings.context_length
      const nodeIds = runtime === 'llama.cpp'
        ? [localNodeId ?? current.node_ids?.[0] ?? 'local']
        : current.node_ids
      const deploymentMode = runtime !== 'llama.cpp'
        && current.deployment_mode === 'sharded'
        && (nodeIds?.length ?? 0) > 1
        ? 'sharded'
        : (nodeIds?.length ?? 0) > 1 ? 'replicated' : 'single'
      return {
        ...current,
        runtime,
        node_ids: nodeIds,
        deployment_mode: deploymentMode,
        settings: runtime === 'llama.cpp'
          ? {
            context_length: contextLength,
            parallel_slots: current.settings.parallel_slots ?? 1,
            gpu_layers: current.settings.gpu_layers ?? 99,
            quantization: current.settings.quantization,
            artifact: current.settings.artifact,
          }
          : {
            context_length: contextLength,
            tensor_parallel_size: deploymentMode === 'sharded' ? nodeIds?.length ?? 1 : current.settings.tensor_parallel_size ?? 1,
            quantization: current.settings.quantization,
            image: runtime === 'vllm' ? current.settings.image ?? defaults.settings.image : undefined,
          },
      }
    })
  }, [appSettings.data, localNodeId])

  const selectionReady = !nodes.loading && !nodes.error && (form.node_ids?.length ?? 0) > 0
    && (form.node_ids ?? []).every((id) => nodes.data?.some((node) => node.id === id && isNodeSelectable(node)))
    && (form.deployment_mode !== 'sharded' || (
      (form.node_ids?.length ?? 0) > 1
      && form.settings.tensor_parallel_size === (form.node_ids?.length ?? 0)
    ))
  const localLabel = onboarding.data?.role === 'worker' ? 'Controller' : 'This device'

  // Models already present on any node's Virtual NAS cache, for the
  // "already have it" picker in the deployment creator. Externally managed
  // bundles (for example installed ComfyUI weights) live outside the cache
  // and cannot back or transfer a runtime deployment, so only models with
  // a normal cache entry are offered.
  const cachedModels = useMemo(() => {
    const byModel = new Map<string, { modelId: string; nodeCount: number; sizeBytes: number }>()
    const externallyManaged = new Set<string>()
    for (const node of modelCache.data?.nodes ?? []) {
      for (const model of node.models) {
        if (model.partial) continue
        if (model.externally_managed || model.transferable === false) {
          externallyManaged.add(model.model_id)
          continue
        }
        const entry = byModel.get(model.model_id)
          ?? { modelId: model.model_id, nodeCount: 0, sizeBytes: 0 }
        entry.nodeCount += 1
        entry.sizeBytes = Math.max(entry.sizeBytes, model.size_bytes ?? 0)
        byModel.set(model.model_id, entry)
      }
    }
    return [...byModel.values()]
      .filter((entry) => !externallyManaged.has(entry.modelId) || byModel.has(entry.modelId))
      .sort((a, b) => a.modelId.localeCompare(b.modelId))
  }, [modelCache.data])

  // Which of the creator's selected nodes already hold the chosen model.
  // Externally managed bundles cannot seed a runtime deployment, so they
  // do not count as cached weights here either.
  const createModelCacheInfo = useMemo(() => {
    if (!form.managed || !form.model_id || isLocalModelPath(form.model_id)) return undefined
    const cachedIds = new Set((modelCache.data?.nodes ?? [])
      .filter((node) => node.models.some((model) => !model.partial
        && !model.externally_managed
        && model.transferable !== false
        && model.model_id === form.model_id))
      .map((node) => node.id))
    const nameOf = (id: string) => id === 'local'
      ? localLabel
      : (nodes.data?.find((node) => node.id === id)?.name ?? id)
    const selected = form.node_ids ?? []
    return {
      cached: selected.filter((id) => cachedIds.has(id)).map(nameOf),
      missing: selected.filter((id) => !cachedIds.has(id)).map(nameOf),
    }
  }, [form.managed, form.model_id, form.node_ids, modelCache.data, nodes.data, localLabel])

  // Cached snapshot files per model, keyed by revision commit and kept
  // per node, with each repository's main ref target. Downloaded marks
  // must compare the revision a listing resolved to — never the union of
  // every cached snapshot — and per node, because a sharded artifact is
  // only usable from a single node that holds all of its files.
  const cachedModelSnapshots = useMemo(() => {
    const byModel = new Map<string, {
      mainSha?: string
      nodeFileSets: Array<Map<string, Set<string>>>
    }>()
    for (const node of modelCache.data?.nodes ?? []) {
      for (const model of node.models) {
        if (!model.snapshot_files) continue
        let entry = byModel.get(model.model_id)
        if (!entry) {
          entry = { mainSha: model.revision_refs?.main, nodeFileSets: [] }
          byModel.set(model.model_id, entry)
        }
        if (!entry.mainSha && model.revision_refs?.main) entry.mainSha = model.revision_refs.main
        const revisions = new Map<string, Set<string>>()
        for (const [revision, files] of Object.entries(model.snapshot_files)) {
          if (!files.length) continue
          revisions.set(revision, new Set(files))
        }
        entry.nodeFileSets.push(revisions)
      }
    }
    return byModel
  }, [modelCache.data])

  // Repository file listing for the model id being configured, so the
  // creator offers the repository's real quantizations and GGUF artifacts
  // instead of free-text guesses. Debounced because the id is typed by hand.
  const [catalogModelQuery, setCatalogModelQuery] = useState('')
  // When the repository lists GGUF artifacts, the artifact field becomes a
  // dropdown; this flag brings back the text field so a controller-local
  // path can still be entered.
  const [artifactManualEntry, setArtifactManualEntry] = useState(false)
  useEffect(() => {
    if (!creating) setArtifactManualEntry(false)
  }, [creating])
  useEffect(() => {
    const modelId = form.model_id.trim()
    if (!creating || !modelId.includes('/') || isLocalModelPath(modelId)) {
      setCatalogModelQuery('')
      return
    }
    const timer = window.setTimeout(() => setCatalogModelQuery(modelId), 350)
    return () => window.clearTimeout(timer)
  }, [creating, form.model_id])
  // Provenance for the retained listing: useResource keeps the previous
  // response while a new request is in flight or after it fails, so every
  // consumer must be able to tell which query the data answers.
  const [catalogResponseQuery, setCatalogResponseQuery] = useState<string>()
  const createCatalogModel = useResource(
    async (signal) => {
      const data = await api.catalog.details(catalogModelQuery, signal)
      setCatalogResponseQuery(catalogModelQuery)
      return data
    },
    [catalogModelQuery],
    Boolean(catalogModelQuery),
  )
  const trimmedModelId = form.model_id.trim()
  const createCatalogData = createCatalogModel.data?.model
  // Trust the listing only while it answers the model id currently in the
  // form: an exact id match, or a listing resolved for this very query
  // (the Hub follows repository renames and answers with the canonical id).
  const createCatalogUsable = Boolean(
    trimmedModelId
    && createCatalogData?.id
    && (createCatalogData.id === trimmedModelId || catalogResponseQuery === trimmedModelId),
  )

  // Adopt the canonical repository id after a rename redirect — but only
  // from the response produced by the current query; retained data from an
  // earlier repository must never canonicalize the field. The previous id
  // is synced so the model-change cleanup treats a verified rename of the
  // same repository differently from a switch to a different repository.
  useEffect(() => {
    const canonicalId = createCatalogModel.data?.model?.id
    if (!canonicalId || !catalogModelQuery || canonicalId === catalogModelQuery) return
    if (catalogResponseQuery !== catalogModelQuery) return
    previousModelIdRef.current = canonicalId
    setForm((current) => (
      current.model_id.trim() === catalogModelQuery ? { ...current, model_id: canonicalId } : current
    ))
  }, [createCatalogModel.data, catalogModelQuery, catalogResponseQuery])

  const createQuantizations = createCatalogUsable
    ? (createCatalogData?.quantizations ?? EMPTY_QUANTIZATIONS)
    : EMPTY_QUANTIZATIONS
  const createArtifactOptions = useMemo(
    () => ggufArtifactOptions(createQuantizations),
    [createQuantizations],
  )
  // Provenance for quantization/artifact values the UI derived from a
  // listing: an empty successful listing must be able to clear them while
  // genuinely manual (or local-path) input survives.
  const createCachedFileSets = useMemo(() => {
    const entry = cachedModelSnapshots.get(form.model_id)
    if (!entry) return EMPTY_FILE_SETS
    // When the listing resolved a commit, only that exact revision's cached
    // files count — an older snapshot must not mark current files as
    // present. The node's own main ref is the fallback for cache-only data
    // with no resolved revision (for example when the Hub is unreachable).
    const resolvedRevision = createCatalogUsable ? createCatalogData?.revision : undefined
    const revision = resolvedRevision ?? entry.mainSha
    if (!revision) return EMPTY_FILE_SETS
    const sets: Array<Set<string>> = []
    for (const revisions of entry.nodeFileSets) {
      const files = revisions.get(revision)
      if (files?.size) sets.push(files)
    }
    return sets
  }, [cachedModelSnapshots, createCatalogData, createCatalogUsable, form.model_id])
  const createQuantizationsByAvailability = useMemo(() => (
    createQuantizations
      .map((variant, index) => ({
        variant,
        index,
        downloaded: quantizationFilesDownloaded(variant, createCachedFileSets),
      }))
      .sort((left, right) => (
        Number(right.downloaded) - Number(left.downloaded)
        || left.index - right.index
      ))
      .map(({ variant }) => variant)
  ), [createCachedFileSets, createQuantizations])

  // Drop quantization and artifact selections that the current listing does
  // not offer (picked from a previous repository, or renamed away
  // upstream): they would fail at launch while looking valid in the form.
  // Free-text quantization for vLLM/SGLang and local artifact paths are not
  // repository-derived and stay untouched.
  useEffect(() => {
    if (!createCatalogUsable) return
    setForm((current) => {
      if (current.runtime !== 'llama.cpp') return current
      const quantization = current.settings.quantization
      const artifact = current.settings.artifact
      const staleQuantization = quantization && (
        (createQuantizations.length > 0 && !createQuantizations.some((variant) => variant.name === quantization))
        || (createQuantizations.length === 0 && linkedQuantizationRef.current === quantization)
      )
      const staleArtifact = artifact && !isLocalArtifact(artifact) && (
        (createArtifactOptions.length > 0 && !createArtifactOptions.some((option) => option.filename === artifact))
        || (createArtifactOptions.length === 0 && linkedArtifactRef.current === artifact)
      )
      if (!staleQuantization && !staleArtifact) return current
      if (staleQuantization) linkedQuantizationRef.current = undefined
      if (staleArtifact) linkedArtifactRef.current = undefined
      return {
        ...current,
        settings: {
          ...current.settings,
          quantization: staleQuantization ? undefined : quantization,
          artifact: staleArtifact ? undefined : artifact,
        },
      }
    })
  }, [createCatalogUsable, createQuantizations, createArtifactOptions])

  // Changing the model id invalidates listing-derived selections right
  // away — the replacement lookup may be slow or fail entirely, and the
  // old repository's artifact must not survive in the fallback text fields
  // until (or regardless of whether) the new listing arrives. The previous
  // id is compared against the form value inside the updater so a deep
  // link setting model and artifacts together is not treated as a change.
  useEffect(() => {
    setForm((current) => {
      const currentModelId = current.model_id.trim()
      if (previousModelIdRef.current === currentModelId) return current
      previousModelIdRef.current = currentModelId
      if (current.runtime !== 'llama.cpp') return current
      const quantization = current.settings.quantization
      const artifact = current.settings.artifact
      const staleQuantization = Boolean(quantization) && linkedQuantizationRef.current === quantization
      const staleArtifact = Boolean(artifact) && !isLocalArtifact(artifact) && linkedArtifactRef.current === artifact
      if (!staleQuantization && !staleArtifact) return current
      linkedQuantizationRef.current = undefined
      linkedArtifactRef.current = undefined
      return {
        ...current,
        settings: {
          ...current.settings,
          quantization: staleQuantization ? undefined : quantization,
          artifact: staleArtifact ? undefined : artifact,
        },
      }
    })
  }, [trimmedModelId])

  const updateCreateQuantization = (name: string) => setForm((current) => {
    if (current.runtime !== 'llama.cpp') {
      return { ...current, settings: { ...current.settings, quantization: name || undefined } }
    }
    const variant = createQuantizations.find((entry) => entry.name === name)
    const linkedArtifact = Boolean(current.settings.artifact
      && createArtifactOptions.some((option) => option.filename === current.settings.artifact))
    // The artifact carries the actual weights for llama.cpp, so keep it in
    // lockstep with the quantization pick: a variant selects its artifact,
    // and clearing the quantization clears a previously linked artifact so
    // a compatible file must be chosen again. Manually entered artifacts
    // are left untouched.
    const variantArtifacts = variant ? ggufArtifactOptions([variant]) : []
    const variantArtifact = (
      variantArtifacts.find((option) => artifactFilesDownloaded(option.files, createCachedFileSets))
      ?? variantArtifacts[0]
    )?.filename
    linkedQuantizationRef.current = variant?.name
    linkedArtifactRef.current = variantArtifact ?? (linkedArtifact ? undefined : current.settings.artifact)
    const artifact = variantArtifact ?? (linkedArtifact ? undefined : current.settings.artifact)
    return {
      ...current,
      settings: { ...current.settings, quantization: name || undefined, artifact },
    }
  })

  const updateCreateArtifact = (filename: string) => {
    const option = createArtifactOptions.find((entry) => entry.filename === filename)
    linkedArtifactRef.current = option?.filename
    if (option) linkedQuantizationRef.current = option.quantization
    setForm((current) => ({
      ...current,
      settings: {
        ...current.settings,
        artifact: filename || undefined,
        ...(option ? { quantization: option.quantization } : {}),
      },
    }))
  }

  const updateManualArtifact = (artifact: string) => {
    linkedArtifactRef.current = undefined
    setForm((current) => ({ ...current, settings: { ...current.settings, artifact: artifact || undefined } }))
  }

  // Fresh form sessions (a new creator, or an editor loading saved
  // settings) start without any listing-derived provenance.
  const resetSelectionProvenance = (modelId?: string) => {
    linkedQuantizationRef.current = undefined
    linkedArtifactRef.current = undefined
    previousModelIdRef.current = modelId ?? ''
    setArtifactManualEntry(false)
  }

  const updateNodeSelection = (selectedIds: string[]) => setForm((current) => {
    const sharded = current.deployment_mode === 'sharded'
    const nodeIds = selectedIds
    const deploymentMode = nodeIds.length > 1 ? (sharded ? 'sharded' : 'replicated') : 'single'
    return {
      ...current,
      node_ids: nodeIds,
      deployment_mode: deploymentMode,
      settings: current.runtime === 'llama.cpp' ? current.settings : {
        ...current.settings,
        tensor_parallel_size: deploymentMode === 'sharded' ? nodeIds.length : 1,
      },
    }
  })

  const updateDeploymentMode = (mode: 'replicated' | 'sharded') => setForm((current) => {
    const selectedIds = current.node_ids ?? []
    const nodeIds = selectedIds
    return {
      ...current,
      node_ids: nodeIds,
      deployment_mode: mode,
      settings: { ...current.settings, tensor_parallel_size: mode === 'sharded' ? nodeIds.length : 1 },
    }
  })

  const openCreator = () => {
    // Canceling an edit leaves the shared form populated with the edited
    // deployment; a fresh creator must never inherit those values.
    setEditingDeployment(undefined)
    setForm(deploymentDefaults(appSettings.data, localNodeId ?? 'local'))
    setExtraFlags('')
    setGpuMemoryUtil('')
    setRuntimeEnvironment('')
    resetSelectionProvenance()
    setFormError(undefined)
    setCreating(true)
  }

  const openEditor = (deployment: Deployment) => {
    // Reuse the creation form to edit a saved deployment before launch.
    setEditingDeployment(deployment)
    setExtraFlags((deployment.settings.extra_args ?? []).map(shellQuote).join(' '))
    setGpuMemoryUtil(deployment.settings.gpu_memory_utilization?.toString() ?? '')
    setRuntimeEnvironment(formatEnvironment(deployment.settings.environment))
    setForm({
      alias: deployment.alias,
      model_id: deployment.model_id,
      runtime: deployment.runtime,
      managed: true,
      endpoint_url: '',
      settings: {
        ...deployment.settings,
        image: deployment.settings.image
          ?? deployment.image
          ?? (deployment.runtime === 'vllm' ? deploymentDefaults(appSettings.data, localNodeId ?? 'local').settings.image : undefined),
        extra_args: deployment.settings.extra_args ?? [],
      },
      node_ids: deployment.node_ids?.length ? deployment.node_ids : (localNodeId ? [localNodeId] : []),
      deployment_mode: (deployment.deployment_mode as CreateDeploymentInput['deployment_mode']) ?? 'single',
    })
    // The saved settings were not derived from the current listing, so a
    // stale provenance from a previous creator session must not treat them
    // as leftovers and clear them before display.
    resetSelectionProvenance(deployment.model_id)
    setFormError(undefined)
    setCreating(true)
  }

  const create = async (event: FormEvent) => {
    event.preventDefault()
    const editing = editingDeployment
    setBusy(editing ? 'edit' : 'create')
    setFormError(undefined)
    setActionNotice(undefined)
    try {
      const settings = {
        ...form.settings,
        extra_args: form.managed ? shellSplit(extraFlags) : [],
        environment: form.managed && form.runtime === 'vllm'
          ? (runtimeEnvironment.trim() || editing
              ? parseEnvironment(runtimeEnvironment)
              : undefined)
          : undefined,
      }
      const utilization = Number(gpuMemoryUtil)
      if (form.managed && form.runtime !== 'llama.cpp' && gpuMemoryUtil.trim() && Number.isFinite(utilization)) {
        settings.gpu_memory_utilization = utilization
      }
      if (editing) {
        // Saved deployments are editable bookmarks: the same form updates the
        // recorded runtime settings and node preferences without launching.
        // Runtime and model are fixed once saved. The alias travels in the
        // same request so settings + rename succeed or fail as one save.
        await api.deployments.update(editing.id, {
          alias: form.alias,
          image: form.runtime === 'vllm' ? settings.image ?? null : undefined,
          context_length: settings.context_length ?? null,
          tensor_parallel_size: settings.tensor_parallel_size ?? null,
          parallel_slots: settings.parallel_slots ?? null,
          gpu_layers: settings.gpu_layers ?? null,
          quantization: settings.quantization ?? null,
          artifact: settings.artifact ?? null,
          extra_args: settings.extra_args ?? [],
          environment: settings.environment,
          gpu_memory_utilization: settings.gpu_memory_utilization ?? null,
          node_ids: form.node_ids,
          deployment_mode: form.deployment_mode,
        })
        setActionNotice(`Updated ${form.alias}. Launch it from the deployments list when ready.`)
      } else {
        const deployment = await api.deployments.create({ ...form, settings })
        setActionNotice(`Saved ${deployment.alias}. Launch it from the deployments list when ready.`)
      }
      setCreating(false)
      setEditingDeployment(undefined)
      runtimeTouched.current = false
      contextLengthTouched.current = false
      setLaunchArgsOpen(false)
      setExtraFlags('')
      setGpuMemoryUtil('')
      setRuntimeEnvironment('')
      setForm(deploymentDefaults(appSettings.data, localNodeId ?? 'local'))
      resource.reload()
    } catch (reason) {
      setFormError(reason instanceof Error ? reason.message : 'Could not save deployment')
    } finally {
      setBusy(undefined)
    }
  }

  const act = async (deployment: Deployment, action: 'start' | 'stop' | 'remove') => {
    setBusy(deployment.id)
    setActionError(undefined)
    try {
      if (action === 'stop') {
        // The stop request awaits every rank; show the transition at once
        // instead of leaving the row looking runnable until it resolves.
        resource.setData((current) => current?.map((item) => (
          item.id === deployment.id ? { ...item, status: 'stopping' } : item
        )))
      }
      await api.deployments.action(deployment.id, action)
      if (action === 'remove') {
        acceptedDeployments.current.delete(deployment.id)
        resource.setData((current) => current?.filter((item) => item.id !== deployment.id))
      }
      resource.reload()
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : 'Could not update deployment')
      if (action === 'stop') resource.reload()
    } finally {
      setBusy(undefined)
    }
  }

  const cloneDeployment = async (deployment: Deployment) => {
    setBusy(deployment.id)
    setActionError(undefined)
    setActionNotice(undefined)
    try {
      const cloned = await api.deployments.clone(deployment.id)
      // Keep the accepted row visible if an older list request completes
      // before the clone appears in the backend's probed deployment list.
      acceptedDeployments.current.set(cloned.id, cloned)
      resource.setData((current) => [
        cloned,
        ...(current ?? []).filter((item) => item.id !== cloned.id),
      ])
      setActionNotice(`Cloned ${deployment.alias} as ${cloned.alias}.`)
      // External clone responses are initially "registered"; the list probe
      // resolves them to their live running/error state.
      resource.reload()
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : 'Could not clone deployment')
    } finally {
      setBusy(undefined)
    }
  }

  const importContainerRecipe = async (deployment: Deployment) => {
    const containerName = deployment.id.startsWith('container:')
      ? decodeURIComponent(deployment.id.slice('container:'.length))
      : deployment.id
    setBusy(deployment.id)
    setActionError(undefined)
    setActionNotice(undefined)
    try {
      const recipe = await api.recipes.importFromContainer(containerName)
      setActionNotice(`Imported ${recipe.name || recipe.model} into Recipes.`)
      recipes.reload()
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : 'Could not import the container as a recipe')
    } finally {
      setBusy(undefined)
    }
  }

  // Poll Virtual NAS jobs to completion. Returns the first failure message,
  // or null once every job finished successfully.
  const waitForPreparationJobs = async (jobIds: string[]) => {
    const pending = new Set(jobIds)
    const total = jobIds.length
    while (pending.size) {
      const state = await api.storage.get()
      for (const job of state.jobs) {
        if (!pending.has(job.id)) continue
        if (job.status === 'completed') pending.delete(job.id)
        else if (job.status !== 'queued' && job.status !== 'running') {
          return job.error || `Model preparation ${job.status}`
        }
      }
      if (pending.size) {
        setStartNotice(
          `Transferring model weights via Virtual NAS… ${total - pending.size}/${total} complete`,
        )
        await new Promise((resolve) => setTimeout(resolve, 3000))
      }
    }
    reloadModelCache()
    return null
  }

  const confirmStart = async () => {
    if (!startSelection) return
    const { deployment, nodeIds } = startSelection
    setBusy(deployment.id)
    setStartError(undefined)
    setStartNotice(undefined)
    try {
      if (deployment.status === 'saved' && !isControllerArtifact(deployment)) {
        // First launch of a saved deployment: route missing weights to the
        // selected nodes through Virtual NAS before starting. Controller-local
        // artifacts never go through preparation; the node holds them already.
        const plan = await api.deployments.preparePreflight(deployment.id, nodeIds)
        if (!plan.eligible) {
          throw new Error(plan.reason || 'The selected nodes cannot be prepared for launch')
        }
        if (plan.action !== 'ready') {
          const downloadNodeIds = plan.download_node_ids?.length
            ? plan.download_node_ids
            : plan.download_node_id ? [plan.download_node_id] : []
          const nameOf = (id: string) => nodes.data?.find((node) => node.id === id)?.name ?? id
          const targetNames = plan.transfer_target_node_ids.map(nameOf)
          const message = plan.action === 'download'
            ? `${deployment.model_id} is not cached on ${downloadNodeIds.map(nameOf).join(', ')}. Download it from Hugging Face (${formatBytes(plan.download?.size_bytes ?? 0)})${targetNames.length ? ` and transfer it via Virtual NAS to ${targetNames.join(', ')}` : ''}?`
            : `Transfer ${deployment.model_id} from ${plan.source?.node_name ?? 'a cluster node'} via Virtual NAS to ${targetNames.join(', ')}?`
          if (!await confirm({
            title: 'Prepare model weights?',
            message: `${message}\n\nCapacity was verified on each selected node's model-cache volume.`,
            confirmLabel: 'Transfer & launch',
          })) return
          const result = await api.deployments.prepare(
            deployment.id,
            nodeIds,
            launchSeed && nodeIds.includes(launchSeed) ? launchSeed : undefined,
          )
          if (result.jobs.length) {
            setStartNotice('Preparing model weights…')
            const failure = await waitForPreparationJobs(result.jobs.map((job) => job.id))
            if (failure) throw new Error(failure)
          }
        }
      }
      const promote = canPromoteDiscovered(deployment)
      await api.deployments.action(deployment.id, 'start', nodeIds, undefined, promote)
      setActionNotice(`${promote ? 'Converting' : 'Starting'} ${deployment.alias} on ${selectedNodeLabel(nodes.data ?? [], nodeIds, localLabel)}.`)
      setStartSelection(undefined)
      resource.reload()
    } catch (reason) {
      // Render inside the dialog: the page-level alert sits behind the
      // modal backdrop where the user cannot see it.
      setStartError(reason instanceof Error ? reason.message : 'Could not start deployment')
    } finally {
      setBusy(undefined)
      setStartNotice(undefined)
    }
  }

  // The backend derives the persisted layout contract (replicated saved-node
  // count, TP for sharded, single otherwise); unknown layouts start on one.
  const deploymentRequiredNodes = (deployment: Deployment) => deployment.required_node_count ?? 1

  const deploymentWeightedNodes = (deployment: Deployment) => {
    if (isControllerArtifact(deployment)) {
      return new Set(localNodeId ? [localNodeId] : [])
    }
    return new Set((modelCache.data?.nodes ?? [])
      .filter((node) => node.models.some((model) => !model.partial && model.model_id === deployment.model_id
        && model.revisions?.includes(deployment.model_revision ?? 'main')))
      .map((node) => node.id))
  }

  const openStartPicker = (deployment: Deployment) => {
    const required = deploymentRequiredNodes(deployment)
    // Saved node preferences are the default selection even before weights
    // exist; the launch flow can prepare missing nodes via Virtual NAS.
    const selectableIds = (nodes.data ?? []).filter(isNodeSelectable).map((node) => node.id)
    const saved = (deployment.node_ids ?? []).filter((id) => selectableIds.includes(id))
    let nodeIds = saved.slice(0, required)
    if (nodeIds.length < required) {
      const candidates = deployment.managed
        ? nodes.data?.filter((node) => deploymentWeightedNodes(deployment).has(node.id) && isNodeSelectable(node)) ?? []
        : nodes.data?.filter(isNodeSelectable) ?? []
      nodeIds = [...new Set([...nodeIds, ...candidates.map((node) => node.id)])].slice(0, required)
    }
    setStartError(undefined)
    setStartNotice(undefined)
    setLaunchSeed(undefined)
    setStartSelection({ deployment, nodeIds })
  }

  const deploymentRunningNodeNames = (deployment: Deployment) => {
    const ids = deployment.node_ids ?? deployment.selected_nodes?.map((node) => node.id) ?? []
    if (!ids.length) {
      // Managed standalone deployments (llama.cpp, legacy single containers)
      // always run on the controller, matching the Target column fallback.
      // External endpoints are not bound to a node.
      return deployment.managed ? [localLabel] : []
    }
    return ids.map((id) => id === 'local'
      ? localLabel
      : (nodes.data?.find((node) => node.id === id)?.name ?? id))
  }

  // Growing the replica set requires a Manager cluster: standalone and legacy
  // containers have no persisted layout and no cluster to grow, controller-
  // local artifacts cannot leave the controller, and sharded layouts have a
  // fixed tensor-parallel node count.
  const supportsAdditionalNodes = (deployment: Deployment) => (
    deployment.deployment_mode !== undefined
    && !isControllerArtifact(deployment)
    && deployment.deployment_mode !== 'sharded'
  )

  const openAdditionalPicker = (deployment: Deployment) => {
    const currentIds = deployment.node_ids ?? deployment.selected_nodes?.map((node) => node.id) ?? []
    setAdditionalError(undefined)
    setAdditionalLaunch({ deployment, currentIds, additionalIds: [] })
  }

  const confirmAdditionalLaunch = async () => {
    if (!additionalLaunch) return
    const { deployment, additionalIds } = additionalLaunch
    setBusy(deployment.id)
    setAdditionalError(undefined)
    try {
      await api.deployments.action(deployment.id, 'start', undefined, additionalIds)
      setActionNotice(`Launching ${deployment.alias} on ${selectedNodeLabel(nodes.data ?? [], additionalIds, localLabel)} too. Existing replicas restart during the relaunch.`)
      setAdditionalLaunch(undefined)
      resource.reload()
    } catch (reason) {
      // Render inside the dialog: the page-level alert sits behind the
      // modal backdrop where the user cannot see it.
      setAdditionalError(reason instanceof Error ? reason.message : 'Could not launch on additional nodes')
    } finally {
      setBusy(undefined)
    }
  }

  const saveRename = async () => {
    if (!renaming) return
    const alias = renaming.value.trim()
    if (!alias) return
    setBusy(renaming.id)
    setActionError(undefined)
    try {
      const updated = await api.deployments.rename(renaming.id, alias)
      setRenaming(undefined)
      setActionNotice(`Renamed deployment to ${updated.alias}.`)
      resource.reload()
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : 'Could not rename deployment')
    } finally {
      setBusy(undefined)
    }
  }

  const loadLogs = useCallback(async (id: string) => {
    // A slower request for a previously viewed deployment must never
    // overwrite the currently displayed one.
    const requestId = ++logRequestRef.current
    setLogLoading(true)
    setLogError(undefined)
    try {
      const logs = await api.deployments.logs(id)
      if (logRequestRef.current !== requestId) return
      setLogData(logs)
      const members = [...(logs.members ?? [])].sort((left, right) => (left.rank ?? Number.MAX_SAFE_INTEGER) - (right.rank ?? Number.MAX_SAFE_INTEGER))
      setSelectedLogNodeId((current) => (
        current && members.some((member) => member.node_id === current)
          ? current
          : members[0]?.node_id
      ))
    } catch (reason) {
      if (logRequestRef.current !== requestId) return
      setLogError(reason instanceof Error ? reason.message : 'Could not load logs')
    } finally {
      if (logRequestRef.current === requestId) setLogLoading(false)
    }
  }, [])

  const openLogs = (deployment: Deployment) => {
    setLogViewer(deployment)
    setLogData(undefined)
    setSelectedLogNodeId(undefined)
    setLogError(undefined)
    setLogTailing(false)
    void loadLogs(deployment.id)
  }

  const closeLogs = () => {
    setLogViewer(undefined)
    setLogTailing(false)
  }

  // Tailing refreshes the open viewer on an interval and pins the panel to
  // the newest output, like `logs --follow`. The next refresh is scheduled
  // only after the current one settles so a slow log request cannot pile up
  // overlapping requests that discard each other as stale.
  useEffect(() => {
    if (!logTailing || !logViewer) return
    let cancelled = false
    let timer: number | undefined
    const schedule = () => {
      timer = window.setTimeout(() => {
        void loadLogs(logViewer.id).finally(() => {
          if (!cancelled) schedule()
        })
      }, 2000)
    }
    schedule()
    return () => {
      cancelled = true
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [logTailing, logViewer, loadLogs])

  useEffect(() => {
    if (!logTailing) return
    const panel = logPanelRef.current
    if (panel) panel.scrollTop = panel.scrollHeight
  }, [logData, logTailing])

  const logMembers = [...(logData?.members ?? [])].sort((left, right) => (
    (left.rank ?? Number.MAX_SAFE_INTEGER) - (right.rank ?? Number.MAX_SAFE_INTEGER)
  ))
  const selectedLogMember = logMembers.find((member) => member.node_id === selectedLogNodeId) ?? logMembers[0]
  const logNodeLabel = (nodeId: string, nodeName?: string) => (
    nodeName
    ?? logViewer?.selected_nodes?.find((node) => node.id === nodeId)?.name
    ?? (nodeId === 'local' ? localLabel : nodeId)
  )
  const selectLogTabFromKeyboard = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    let nextIndex: number | undefined
    if (event.key === 'ArrowRight') nextIndex = (index + 1) % logMembers.length
    else if (event.key === 'ArrowLeft') nextIndex = (index - 1 + logMembers.length) % logMembers.length
    else if (event.key === 'Home') nextIndex = 0
    else if (event.key === 'End') nextIndex = logMembers.length - 1
    if (nextIndex === undefined) return
    event.preventDefault()
    setSelectedLogNodeId(logMembers[nextIndex].node_id)
    event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>('[role="tab"]')[nextIndex]?.focus()
  }

  const changeSortMode = (mode: SortMode) => {
    setSortMode(mode)
    localStorage.setItem(SORT_STORAGE_KEY, mode)
  }

  const togglePin = (recipeId: string) => {
    setPinned((current) => {
      const next = current.includes(recipeId)
        ? current.filter((id) => id !== recipeId)
        : [...current, recipeId]
      localStorage.setItem(PIN_STORAGE_KEY, JSON.stringify(next))
      return next
    })
  }

  const deleteRecipe = async (recipe: SavedConfiguration) => {
    const label = recipe.name || recipe.model
    if (!await confirm({
      title: `Delete recipe ${label}?`,
      message: 'Existing deployments and cached model weights will not be removed.',
      confirmLabel: 'Delete recipe',
      danger: true,
    })) return
    const busyKey = `recipe-delete:${recipe.id}`
    setBusy(busyKey)
    setActionError(undefined)
    setActionNotice(undefined)
    try {
      await api.recipes.remove(recipe.id)
      recipes.setData((current) => current?.filter((item) => item.id !== recipe.id))
      setPinned((current) => {
        const next = current.filter((id) => id !== recipe.id)
        localStorage.setItem(PIN_STORAGE_KEY, JSON.stringify(next))
        return next
      })
      setArgsEditors((current) => {
        const next = { ...current }
        delete next[recipe.id]
        return next
      })
      if (recipeDeployment?.recipe.id === recipe.id) setRecipeDeployment(undefined)
      setActionNotice(`Deleted recipe ${label}. Existing deployments and cached model weights were left unchanged.`)
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : 'Could not delete recipe')
    } finally {
      setBusy(undefined)
    }
  }

  const toggleGroup = (company: string) => {
    setExpandedGroups((current) => current.includes(company)
      ? current.filter((name) => name !== company)
      : [...current, company])
  }

  const sortedDeployments = useMemo(() => {
    const items = [...(resource.data ?? [])]
    if (sortMode === 'name-asc' || sortMode === 'name-desc') {
      items.sort((a, b) => a.alias.localeCompare(b.alias, undefined, { sensitivity: 'base' }))
      if (sortMode === 'name-desc') items.reverse()
    } else {
      // Stable sort: never-run deployments stay after deployments with an
      // observed launch time while preserving their existing relative order.
      items.sort((a, b) => (
        deploymentTimestampMs(b.last_deployed_at)
        - deploymentTimestampMs(a.last_deployed_at)
      ))
    }
    return items
  }, [resource.data, sortMode])

  const recipeGroups = useMemo(() => {
    const groups = new Map<string, SavedConfiguration[]>()
    for (const recipe of recipes.data ?? []) {
      const company = companyOf(recipe)
      groups.set(company, [...(groups.get(company) ?? []), recipe])
    }
    const pinnedSet = new Set(pinned)
    return [...groups.entries()]
      .sort(([a], [b]) => a.localeCompare(b, undefined, { sensitivity: 'base' }))
      .map(([company, items]): [string, SavedConfiguration[]] => [company, [...items].sort((a, b) => {
        const pinDelta = Number(pinnedSet.has(b.id)) - Number(pinnedSet.has(a.id))
        if (pinDelta) return pinDelta
        return (a.name || a.model).localeCompare(b.name || b.model, undefined, { sensitivity: 'base' })
      })])
  }, [recipes.data, pinned])

  const setArgsEditor = (recipeId: string, next: Partial<ArgsEditorState>) => {
    setArgsEditors((current) => {
      const previous: ArgsEditorState = current[recipeId] ?? {
        open: false, loading: false, saving: false, saved: false, form: {},
      }
      return { ...current, [recipeId]: { ...previous, ...next } }
    })
  }

  const toggleArgsEditor = async (recipe: SavedConfiguration) => {
    const current = argsEditors[recipe.id]
    if (current?.open) {
      setArgsEditor(recipe.id, { open: false })
      return
    }
    if (current && Object.keys(current.form).length) {
      setArgsEditor(recipe.id, { open: true })
      return
    }
    setArgsEditor(recipe.id, { open: true, loading: true, error: undefined })
    try {
      const detail = await api.recipes.get(recipe.id)
      setArgsEditor(recipe.id, { loading: false, form: seedArgsForm(detail) })
    } catch (reason) {
      setArgsEditor(recipe.id, {
        loading: false,
        error: reason instanceof Error ? reason.message : 'Could not load arguments',
      })
    }
  }

  const saveArgs = async (recipe: SavedConfiguration) => {
    const editor = argsEditors[recipe.id]
    if (!editor) return
    const numeric = (value: string) => {
      const trimmed = (value ?? '').trim()
      if (!trimmed) return null
      const parsed = Number(trimmed)
      return Number.isFinite(parsed) ? parsed : null
    }
    const editorForm = editor.form
    const sgTpSize = numeric(editorForm.sg_tp_size)
    const sgMemFraction = numeric(editorForm.sg_mem_fraction)
    if (sgTpSize !== null && (!Number.isInteger(sgTpSize) || sgTpSize < 1)) {
      setArgsEditor(recipe.id, { error: 'TP size must be a positive integer' })
      return
    }
    if (sgMemFraction !== null && (sgMemFraction <= 0 || sgMemFraction > 1)) {
      setArgsEditor(recipe.id, { error: 'Mem fraction must be between 0 and 1' })
      return
    }
    const payload: RecipeUpdateInput = {
      extra_args: shellSplit(editorForm.remaining_flags ?? ''),
      launch_controls: {
        context_window: numeric(editorForm.context_window),
        max_concurrency: numeric(editorForm.max_concurrency),
        kv_cache_dtype: editorForm.kv_cache_dtype?.trim() || null,
        thinking_mode: editorForm.thinking_mode || 'default',
        speculative_method: editorForm.speculative_method || null,
        draft_sample_method: editorForm.draft_sample_method || null,
        dspark_num_speculative_tokens: numeric(editorForm.dspark_num_speculative_tokens),
        max_cudagraph_capture_size: numeric(editorForm.max_cudagraph_capture_size),
        max_num_batched_tokens: numeric(editorForm.max_num_batched_tokens),
      },
      gpu_memory_utilization: numeric(editorForm.gpu_memory_utilization),
      gpu_memory_gb: numeric(editorForm.gpu_memory_gb),
      sg_tp_size: sgTpSize,
      sg_mem_fraction: sgMemFraction,
    }
    setArgsEditor(recipe.id, { saving: true, error: undefined, saved: false })
    try {
      await api.recipes.update(recipe.id, payload)
      setArgsEditor(recipe.id, { saving: false, saved: true })
      setTimeout(() => setArgsEditor(recipe.id, { saved: false }), 2500)
      recipes.reload()
    } catch (reason) {
      setArgsEditor(recipe.id, {
        saving: false,
        error: reason instanceof Error ? reason.message : 'Could not save arguments',
      })
    }
  }

  const nodesWithWeights = (recipe: SavedConfiguration) => {
    if (isLocalModelPath(recipe.model)) {
      return new Set(localNodeId ? [localNodeId] : [])
    }
    return new Set((modelCache.data?.nodes ?? [])
      .filter((node) => node.models.some((model) => !model.partial && model.model_id === recipe.model
        && model.revisions?.includes(recipe.model_revision ?? 'main')))
      .map((node) => node.id))
  }

  const openRecipeDeployment = (recipe: SavedConfiguration) => {
    const weighted = nodesWithWeights(recipe)
    const eligible = (nodes.data ?? []).filter(isNodeSelectable)
    const preferred = [...new Set(recipe.node_ids)]
      .filter((id) => eligible.some((node) => node.id === id))
    const weightedFallback = eligible.filter((node) => weighted.has(node.id)).map((node) => node.id)
    const nodeIds = [...preferred, ...weightedFallback, ...eligible.map((node) => node.id)]
      .filter((id, index, values) => values.indexOf(id) === index)
      .slice(0, recipe.required_node_count)
    setRecipeError(undefined)
    setRecipeTransferNotice(undefined)
    setRecipeSeedNodeId(undefined)
    setRecipeDeployment({ recipe, nodeIds })
  }

  const prepareRecipeWeights = async () => {
    if (!recipeDeployment) return
    const { recipe, nodeIds } = recipeDeployment
    const revision = recipe.model_revision ?? 'main'
    setBusy(`recipe-prepare:${recipe.id}`)
    setRecipeError(undefined)
    try {
      const seedNodeId = recipeSeedNodeId && nodeIds.includes(recipeSeedNodeId)
        ? recipeSeedNodeId
        : undefined
      const plan = await api.storage.preparationPreflight(recipe.id, nodeIds, seedNodeId)
      reloadTransferPreflight()
      if (!plan.eligible) throw new Error(plan.reason || 'The selected nodes are no longer eligible')
      if (plan.action === 'ready') {
        reloadModelCache()
        return
      }
      const downloadNodeIds = plan.download_node_ids?.length
        ? plan.download_node_ids
        : plan.download_node_id ? [plan.download_node_id] : []
      const downloadNames = downloadNodeIds.map((id) => (
        nodes.data?.find((node) => node.id === id)?.name ?? id
      ))
      const sourceName = plan.action === 'download'
        ? downloadNames[0]
        : plan.source?.node_name
      const targetNames = plan.transfer_target_node_ids.map((id) => (
        nodes.data?.find((node) => node.id === id)?.name ?? id
      ))
      const fanoutSource = plan.source?.node_name
      const message = plan.action === 'download'
        ? `Download ${recipe.model} revision ${revision} (${formatBytes(plan.download?.size_bytes ?? 0)}) from Hugging Face onto ${downloadNames.join(', ')}${targetNames.length ? fanoutSource ? `, then transfer it from ${fanoutSource} via Virtual NAS to ${targetNames.join(', ')}` : `, then transfer it via Virtual NAS to ${targetNames.join(', ')}` : ''}?`
        : `Transfer ${recipe.model} from ${sourceName} via Virtual NAS to ${targetNames.join(', ')}?`
      if (!await confirm({
        title: 'Prepare model weights?',
        message: `${message}\n\nCapacity was verified on each selected node's model-cache volume.`,
        confirmLabel: 'Start preparation',
      })) return
      setRecipeTransferNotice(undefined)
      const result = await api.storage.prepareRecipe(recipe.id, nodeIds, seedNodeId)
      if (!result.jobs.length) {
        reloadModelCache()
        return
      }
      setRecipeTransferJobs((current) => {
        const next = { ...current }
        for (const job of result.jobs) {
          const key = recipeJobKey(recipe.id, recipe.model, job.target_node_id)
          next[key] = {
            jobId: job.id,
            targetId: job.target_node_id,
            modelId: recipe.model,
            recipeId: recipe.id,
          }
        }
        return next
      })
      setRecipeTransferNotice(
        plan.action === 'download'
          ? `Hugging Face ${downloadNames.length > 1 ? 'downloads' : 'seed download'} queued on ${downloadNames.join(', ')}.${targetNames.length ? ' Virtual NAS fan-out will follow automatically.' : ''}`
          : `Virtual NAS transfer queued to ${targetNames.join(', ')}.`,
      )
      reloadTransferPreflight()
    } catch (reason) {
      setRecipeError(reason instanceof Error ? reason.message : 'Could not prepare model weights')
      reloadTransferPreflight()
    } finally {
      setBusy(undefined)
    }
  }

  const deployRecipe = async () => {
    if (!recipeDeployment) return
    const { recipe, nodeIds } = recipeDeployment
    setBusy(`recipe:${recipe.id}`)
    setActionError(undefined)
    setActionNotice(undefined)
    setRecipeError(undefined)
    try {
      const deployment = await api.recipes.deploy(recipe.id, nodeIds)
      const selected = selectedNodeLabel(nodes.data ?? [], nodeIds, localLabel)
      setRecipeDeployment(undefined)
      acceptedDeployments.current.set(deployment.id, deployment)
      resource.setData((current) => [
        deployment,
        ...(current ?? []).filter((item) => item.id !== deployment.id),
      ])
      setActionNotice(`Started deployment ${deployment.alias} on ${selected}.`)
    } catch (reason) {
      setRecipeError(reason instanceof Error ? reason.message : 'Could not deploy saved configuration')
    } finally {
      setBusy(undefined)
    }
  }

  const modelStorage = (deployment: Deployment) => {
    const locations = (modelCache.data?.nodes ?? []).flatMap((node) =>
      node.models
        .filter((model) => !model.partial && model.model_id === deployment.model_id)
        .map((model) => ({ ...model, nodeId: node.id, nodeName: node.name })),
    )
    if (!locations.length) return 'Disk size unavailable'
    const total = locations.reduce((sum, location) => sum + location.size_bytes, 0)
    let base: string
    if (locations.length === 1) {
      base = `${formatBytes(total)} on ${locations[0].nodeName}`
    } else {
      const perCopy = locations.every((location) => location.size_bytes === locations[0].size_bytes)
        ? `${formatBytes(locations[0].size_bytes)} each · `
        : ''
      base = `${perCopy}${formatBytes(total)} total on ${locations.length} nodes`
    }
    // Multi-node deployments: call out selected nodes whose weights were not
    // found in the cache inventory instead of silently omitting them.
    const cachedIds = new Set(locations.map((location) => location.nodeId))
    const expectedIds = deployment.selected_nodes?.map((node) => node.id) ?? deployment.node_ids ?? []
    const missing = expectedIds
      .filter((id) => !cachedIds.has(id))
      .map((id) => id === 'local' ? localLabel : (nodes.data?.find((node) => node.id === id)?.name ?? id))
    return missing.length ? `${base} · not cached on ${missing.join(', ')}` : base
  }

  const updateRuntime = (runtime: RuntimeKind) => {
    runtimeTouched.current = true
    setForm((current) => {
      const contextLength = current.settings.context_length ?? 8192
      const nodeIds = current.node_ids?.length ? current.node_ids : (localNodeId ? [localNodeId] : [])
      // Sharded layouts keep their vLLM/SGLang-only constraints; Llama server
      // always runs complete replicas.
      const sharded = runtime !== 'llama.cpp'
        && current.deployment_mode === 'sharded'
        && (nodeIds?.length ?? 0) > 1
      const deploymentMode = sharded ? 'sharded' : (nodeIds?.length ?? 0) > 1 ? 'replicated' : 'single'
      return {
        ...current,
        runtime,
        node_ids: nodeIds,
        deployment_mode: deploymentMode,
        settings: runtime === 'llama.cpp'
          ? {
            context_length: contextLength,
            parallel_slots: 1,
            gpu_layers: 99,
            quantization: current.settings.quantization,
            artifact: current.settings.artifact,
          }
          : {
            context_length: contextLength,
            tensor_parallel_size: deploymentMode === 'sharded' ? nodeIds?.length ?? 1 : 1,
            quantization: current.settings.quantization,
            image: runtime === 'vllm'
              ? current.settings.image ?? deploymentDefaults(appSettings.data, localNodeId ?? 'local').settings.image
              : undefined,
          },
      }
    })
  }

  return (
    <div className="page">
      <PageHeader
        eyebrow="Local runtimes"
        title="Models"
        description="Manage model servers across vLLM, SGLang, and Llama server from one place."
        actions={<Button variant="primary" onClick={openCreator}><Plus size={16} /> Create deployment</Button>}
      />
      {resource.loading && !resource.data && <LoadingState label="Loading deployments" />}
      {resource.error && !resource.data && <ErrorState message={resource.error} onRetry={resource.reload} />}
      {recipes.error && <ErrorState message={`Saved configurations: ${recipes.error}`} onRetry={recipes.reload} />}
      {actionError && <p className="form-error" role="alert">{actionError}</p>}
      {actionNotice && <p className="inline-success" role="status">{actionNotice}</p>}
      {resource.data?.length === 0 && (
        <EmptyState title="No deployments yet" description="Save a deployment for any runtime, then launch it on the nodes you choose." action={<Button variant="primary" onClick={openCreator}>Create your first deployment</Button>} />
      )}
      {resource.data && resource.data.length > 0 && (
        <section className="deployments" aria-labelledby="deployments-title">
          <div className="section-heading">
            <div><h2 id="deployments-title">Deployments</h2></div>
            <label className="sort-field"><span>Sort by</span>
              <span className="sort-control">
                <select value={sortMode} onChange={(event) => changeSortMode(event.target.value as SortMode)} aria-label="Sort deployments">
                  <option value="recent">Most recently deployed</option>
                  <option value="name-asc">Name A–Z</option>
                  <option value="name-desc">Name Z–A</option>
                </select>
                <ChevronDown size={15} aria-hidden="true" />
              </span>
            </label>
          </div>
          <Panel className="table-panel">
            <div className="responsive-table deployments-table" role="table" aria-label="Model deployments">
              <div className="table-row table-header" role="row">
                <span role="columnheader">Model</span><span role="columnheader">Runtime</span><span role="columnheader">Configuration</span><span role="columnheader">Target</span><span role="columnheader">Status</span><span role="columnheader">Actions</span>
              </div>
              {sortedDeployments.map((deployment) => (
                <div className="table-row" role="row" key={deployment.id} tabIndex={0}>
                  <div role="cell" data-label="Model">
                    {renaming?.id === deployment.id ? (
                      <span className="rename-row">
                        <input
                          className="rename-input"
                          autoFocus
                          value={renaming.value}
                          aria-label={`New name for ${deployment.alias}`}
                          onChange={(event) => setRenaming({ id: deployment.id, value: event.target.value })}
                          onKeyDown={(event) => {
                            if (event.key === 'Enter') void saveRename()
                            if (event.key === 'Escape') setRenaming(undefined)
                          }}
                        />
                        <Button variant="tertiary" disabled={busy === deployment.id} aria-label="Save name" onClick={() => void saveRename()}><Check size={15} /></Button>
                        <Button variant="tertiary" aria-label="Cancel rename" onClick={() => setRenaming(undefined)}><X size={15} /></Button>
                      </span>
                    ) : (
                      <strong><Link to={`/models/${encodeURIComponent(deployment.id)}`}>{deployment.alias}</Link></strong>
                    )}
                    <small>{deployment.model_id}</small>
                    <small className="model-disk-usage"><HardDrive size={12} /> {modelStorage(deployment)}</small>
                  </div>
                  <div role="cell" data-label="Runtime"><RuntimeMark runtime={deployment.runtime} /><small>{deployment.runtime_version ?? (deployment.managed ? 'Managed' : 'External')}</small></div>
                  <div role="cell" data-label="Configuration"><span>{deployment.settings.context_length?.toLocaleString() ?? '—'} CTX</span><small>{deploymentConcurrency(deployment)?.toLocaleString() ?? '—'} concurrent · {deployment.settings.quantization ?? 'Default precision'}</small></div>
                  <div role="cell" data-label="Target"><span>{deployment.selected_nodes?.map((node, index) => `${node.id === 'local' ? localLabel : node.name}${deployment.selected_nodes!.length > 1 && index === 0 ? ' (primary)' : ''}`).join(', ') || deployment.node_ids?.map((id, index) => `${id === 'local' ? localLabel : id}${deployment.node_ids!.length > 1 && index === 0 ? ' (primary)' : ''}`).join(', ') || localLabel}{deploymentTargetLayout(deployment)}</span></div>
                  <div role="cell" data-label="Status" className="deployment-status-cell">
                    <span aria-live="polite">
                      {STOPPABLE_DEPLOYMENT_STATUSES.has(deployment.status) && deploymentRunningNodeNames(deployment).length > 0
                        ? <Tooltip label={<><strong>{deployment.status === 'starting' || deployment.status === 'launching' ? 'Starting on' : 'Running on'}</strong><span>{deploymentRunningNodeNames(deployment).join(', ')}</span></>}><Status status={deployment.status} /></Tooltip>
                        : <Status status={deployment.status} />}
                      {deployment.status !== 'running' && (
                        <>
                          {showLaunchDetails(deployment) && deployment.launch_phase && <small className="deployment-launch-phase">{formatLaunchPhase(deployment.launch_phase)}</small>}
                          {showLaunchDetails(deployment) && deployment.launch_message && <small className="deployment-launch-message">{deployment.launch_message}</small>}
                        </>
                      )}
                    </span>
                    {deployment.status === 'running' && deployment.last_used_at !== undefined && (
                      <small className="deployment-launch-message">
                        {deployment.last_used_at
                          ? `Last inference ${formatInferenceAge(deployment.last_used_at, now)}`
                          : 'No inference yet'}
                      </small>
                    )}
                    {deployment.status === 'stopped' && deployment.last_deployed_at && formatDeploymentTimestamp(deployment.last_deployed_at, now) && (
                      <small className="deployment-launch-message">
                        Last deployed {formatDeploymentTimestamp(deployment.last_deployed_at, now)}
                      </small>
                    )}
                  </div>
                  <div role="cell" data-label="Actions" className="row-actions">
                    {(deployment.managed || deployment.controllable) && (deployment.status === 'stopping'
                      ? <Button variant="tertiary" disabled>Stopping…</Button>
                      : deployment.desired_state !== 'stopped' && STOPPABLE_DEPLOYMENT_STATUSES.has(deployment.status)
                      ? (supportsAdditionalNodes(deployment)
                        ? <SplitButton
                            label="Stop"
                            disabled={busy === deployment.id || Boolean(deployment.launch_phase && PRE_CONTAINER_LAUNCH_PHASES.has(deployment.launch_phase))}
                            onMainAction={() => void act(deployment, 'stop')}
                            toggleAriaLabel={`More actions for ${deployment.alias}`}
                            items={[{ key: 'additional', label: 'Launch on additional nodes…', onSelect: () => openAdditionalPicker(deployment) }]}
                          />
                        : <Button variant="tertiary" disabled={busy === deployment.id || Boolean(deployment.launch_phase && PRE_CONTAINER_LAUNCH_PHASES.has(deployment.launch_phase))} onClick={() => void act(deployment, 'stop')}>Stop</Button>)
                      : <Button variant="tertiary" disabled={busy === deployment.id} onClick={() => {
                        if (isDiscoveredExternal(deployment) && !canPromoteDiscovered(deployment)) {
                          void act(deployment, 'start')
                        } else {
                          openStartPicker(deployment)
                        }
                      }}>{deployment.status === 'saved' ? 'Launch' : canPromoteDiscovered(deployment) ? 'Make managed' : 'Start'}</Button>)}
                    {deployment.managed && deployment.status === 'saved' && (
                      <Button variant="tertiary" disabled={busy === deployment.id} aria-label={`Edit ${deployment.alias}`} title="Edit deployment" onClick={() => openEditor(deployment)}><Settings2 size={16} /></Button>
                    )}
                    {(deployment.managed || deployment.logs_available) && deployment.status !== 'saved' && <Button variant="tertiary" disabled={busy === deployment.id} aria-label={`Logs for ${deployment.alias}`} title="Logs" onClick={() => openLogs(deployment)}><ScrollText size={16} /></Button>}
                    {deployment.id.startsWith('container:') && (
                      <Button variant="tertiary" disabled={busy === deployment.id} aria-label={`Save ${deployment.alias} as recipe`} title="Save as recipe" onClick={() => void importContainerRecipe(deployment)}><FolderPlus size={16} /></Button>
                    )}
                    {!deployment.id.startsWith('container:') && (
                      <Button variant="tertiary" disabled={busy === deployment.id} aria-label={`Clone ${deployment.alias}`} title="Clone deployment" onClick={() => void cloneDeployment(deployment)}><Copy size={16} /></Button>
                    )}
                    <Button variant="tertiary" disabled={busy === deployment.id} aria-label={`Rename ${deployment.alias}`} onClick={() => setRenaming({ id: deployment.id, value: deployment.alias })}><Pencil size={16} /></Button>
                    <Button variant="tertiary" disabled={busy === deployment.id} aria-label={`Remove ${deployment.alias}`} onClick={() => void confirm({
                      title: `Remove ${deployment.alias}?`,
                      message: 'SparkDeck will remove this deployment and tear down its managed runtime.',
                      confirmLabel: 'Remove deployment',
                      danger: true,
                    }).then((accepted) => {
                      if (accepted) return act(deployment, 'remove')
                      return undefined
                    })}><Trash2 size={17} /></Button>
                  </div>
                </div>
              ))}
            </div>
          </Panel>
        </section>
      )}
      {recipes.data && recipes.data.length > 0 && <section className="saved-configurations" aria-labelledby="saved-configurations-title">
        <div className="section-heading"><div><h2 id="saved-configurations-title">Recipes</h2><p>Saved launch configurations — choose nodes and launch one as a deployment.</p></div></div>
        {recipeGroups.map(([company, items]) => {
          const expanded = expandedGroups.includes(company)
          return <div className="saved-configuration-group" key={company}>
            <h3 className="saved-configuration-group-title">
              <button type="button" className="group-toggle" aria-expanded={expanded} onClick={() => toggleGroup(company)}>
                {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                {company} <span className="saved-configuration-group-count">{items.length}</span>
              </button>
            </h3>
            {expanded && <div className="saved-configuration-grid">
              {items.map((recipe) => {
                const targets = (recipe.node_ids?.length ? recipe.node_ids : ['local']).map((id) => nodes.data?.find((node) => node.id === id))
                const targetNames = targets.map((node, index) => node?.name ?? recipe.node_ids?.[index] ?? 'This device')
                const unavailable = targets.some((node) => !node || !isNodeSelectable(node))
                const disabled = recipe.supported === false
                const isPinned = pinned.includes(recipe.id)
                const editor = argsEditors[recipe.id]
                const isVllm = (recipe.engine || 'vllm') !== 'sglang'
                const deleting = busy === `recipe-delete:${recipe.id}`
                return <Panel className="saved-configuration-card" key={recipe.id}>
                  <div className="saved-configuration-heading">
                    <button
                      type="button"
                      className={`icon-button pin-button${isPinned ? ' pinned' : ''}`}
                      aria-label={`${isPinned ? 'Unpin' : 'Pin'} ${recipe.name || recipe.model}`}
                      aria-pressed={isPinned}
                      onClick={() => togglePin(recipe.id)}
                    ><Bookmark size={17} fill={isPinned ? 'currentColor' : 'none'} /></button>
                    <div><h3>{recipe.name || recipe.model}</h3><p>{recipe.model}</p></div>
                  </div>
                  <dl>
                    <div><dt>Runtime</dt><dd><RuntimeMark runtime={recipe.engine || 'vllm'} /></dd></div>
                    <div><dt>Layout</dt><dd>{recipe.tensor_parallel_size > 1 ? `TP${recipe.tensor_parallel_size} · ` : ''}{recipe.deployment_mode || 'single'} · {recipe.required_node_count} {recipe.required_node_count === 1 ? 'node' : 'nodes'}</dd></div>
                    <div><dt>Targets</dt><dd>{targetNames.join(', ')}</dd></div>
                    <div><dt>Arguments</dt><dd>{recipe.extra_args_count ?? 0} saved</dd></div>
                  </dl>
                  {(recipe.error || unavailable) && <p className="saved-configuration-warning">{recipe.error || 'One or more saved nodes are missing, offline, or not ready.'}</p>}
                  <div className="saved-configuration-actions">
                    <Button variant="primary" disabled={disabled || deleting || busy === `recipe:${recipe.id}`} onClick={() => openRecipeDeployment(recipe)}><Play size={15} /> Choose nodes &amp; deploy</Button>
                    <Button variant="tertiary" disabled={deleting} aria-expanded={editor?.open ?? false} onClick={() => void toggleArgsEditor(recipe)}><Settings2 size={15} /> Arguments</Button>
                    <Button variant="danger" disabled={deleting} aria-label={`Delete recipe ${recipe.name || recipe.model}`} onClick={() => void deleteRecipe(recipe)}><Trash2 size={15} /> {deleting ? 'Deleting…' : 'Delete'}</Button>
                  </div>
                  {editor?.open && <div className="args-editor">
                    {editor.loading && <p className="field-note">Loading arguments…</p>}
                    {editor.error && <p className="form-error" role="alert">{editor.error}</p>}
                    {!editor.loading && Object.keys(editor.form).length > 0 && <>
                      <div className="field-grid">
                        <label className="field"><span>Context window</span><input type="number" min="1" value={editor.form.context_window} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, context_window: event.target.value } })} /></label>
                        <label className="field"><span>Max concurrency</span><input type="number" min="1" value={editor.form.max_concurrency} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, max_concurrency: event.target.value } })} /></label>
                        <label className="field"><span>KV cache dtype</span><KvCacheDtypeSelect runtime={recipe.engine || 'vllm'} value={editor.form.kv_cache_dtype} onChange={(value) => setArgsEditor(recipe.id, { form: { ...editor.form, kv_cache_dtype: value } })} /></label>
                        <label className="field"><span>Thinking</span><select value={editor.form.thinking_mode} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, thinking_mode: event.target.value } })}><option value="default">Default</option><option value="enabled">Enabled</option><option value="disabled">Disabled</option></select></label>
                        {isVllm && <>
                          <label className="field"><span>Speculative method</span><select value={editor.form.speculative_method} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, speculative_method: event.target.value } })}><option value="">Auto / unset</option>{editor.form.speculative_method && !SPECULATIVE_METHODS.includes(editor.form.speculative_method) && <option value={editor.form.speculative_method}>{editor.form.speculative_method}</option>}{SPECULATIVE_METHODS.map((method) => <option key={method} value={method}>{method}</option>)}</select></label>
                          <label className="field"><span>Draft sample method</span><select value={editor.form.draft_sample_method} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, draft_sample_method: event.target.value } })}><option value="">Default</option>{editor.form.draft_sample_method && !DRAFT_SAMPLE_METHODS.includes(editor.form.draft_sample_method) && <option value={editor.form.draft_sample_method}>{editor.form.draft_sample_method}</option>}{DRAFT_SAMPLE_METHODS.map((method) => <option key={method} value={method}>{method}</option>)}</select></label>
                          <label className="field"><span>Speculative tokens</span><input type="number" min="1" value={editor.form.dspark_num_speculative_tokens} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, dspark_num_speculative_tokens: event.target.value } })} /></label>
                          <label className="field"><span>Cudagraph capture size</span><input type="number" min="1" value={editor.form.max_cudagraph_capture_size} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, max_cudagraph_capture_size: event.target.value } })} /></label>
                          <label className="field"><span>Batched tokens</span><input type="number" min="1" value={editor.form.max_num_batched_tokens} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, max_num_batched_tokens: event.target.value } })} /></label>
                          <label className="field"><span>GPU memory util</span><input type="number" step="0.05" min="0.1" max="0.98" value={editor.form.gpu_memory_utilization} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, gpu_memory_utilization: event.target.value } })} /></label>
                          <label className="field"><span>Reserve GB</span><input type="number" min="1" value={editor.form.gpu_memory_gb} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, gpu_memory_gb: event.target.value } })} /></label>
                        </>}
                        {!isVllm && <>
                          <label className="field"><span>TP size</span><input type="number" min="1" value={editor.form.sg_tp_size} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, sg_tp_size: event.target.value } })} /></label>
                          <label className="field"><span>Mem fraction (static)</span><input type="number" step="0.01" min="0.01" max="1" value={editor.form.sg_mem_fraction} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, sg_mem_fraction: event.target.value } })} /></label>
                        </>}
                      </div>
                      <label className="field"><span>Other flags</span><textarea rows={3} value={editor.form.remaining_flags} spellCheck={false} onChange={(event) => setArgsEditor(recipe.id, { form: { ...editor.form, remaining_flags: event.target.value } })} /></label>
                      <p className="field-note">Blank fields remove the flag. Structured fields above override matching flags in &quot;Other flags&quot;.</p>
                      <div className="args-editor-actions">
                        <Button variant="primary" disabled={editor.saving} onClick={() => void saveArgs(recipe)}>{editor.saving ? 'Saving…' : 'Save settings'}</Button>
                        {editor.saved && <span className="inline-success" role="status">Saved.</span>}
                      </div>
                    </>}
                  </div>}
                </Panel>
              })}
            </div>}
          </div>
        })}
      </section>}

      {recipeDeployment && (() => {
        const { recipe, nodeIds } = recipeDeployment
        const localPath = isLocalModelPath(recipe.model)
        const preflightTargets = new Map((transferPreflight.data?.targets ?? []).map((target) => [target.node_id, target]))
        const weighted = localPath
          ? nodesWithWeights(recipe)
          : new Set((transferPreflight.data?.targets ?? [])
            .filter((target) => target.has_required_weights)
            .map((target) => target.node_id))
        const missingNodes = (nodes.data ?? []).filter((node) => !weighted.has(node.id))
        const allowedIds = (nodes.data ?? []).filter((node) => {
          if (weighted.has(node.id)) return true
          const option = preflightTargets.get(node.id)
          return Boolean(option?.eligible || option?.download_eligible || option?.transfer_after_download_eligible)
        }).map((node) => node.id)
        const unavailableReasons = Object.fromEntries(missingNodes.map((node) => [
          node.id,
          localPath
            ? 'Local paths are available only on the controller'
            : preflightTargets.get(node.id)?.active_job_status
              ? `Model preparation ${preflightTargets.get(node.id)?.active_job_status}`
              : preflightTargets.get(node.id)?.reason
                ?? preflightTargets.get(node.id)?.download_reason
                ?? preflightTargets.get(node.id)?.transfer_after_download_reason
                ?? 'Model weights not cached',
        ]))
        const exactCount = nodeIds.length === recipe.required_node_count
        const allEligible = nodeIds.every((id) => allowedIds.includes(id) && nodes.data?.some((node) => node.id === id && isNodeSelectable(node)))
        const weightsReady = nodeIds.every((id) => weighted.has(id))
        const activeSelected = nodeIds.some((id) => recipeTransferJobs[recipeJobKey(recipe.id, recipe.model, id)])
        const ready = !nodes.loading && !nodes.error && exactCount && allEligible && weightsReady
        const canPrepare = !localPath && Boolean(transferPreflight.data?.enabled) && !transferPreflight.loading && exactCount && allEligible && !weightsReady && !activeSelected
        const recipeBusy = busy === `recipe:${recipe.id}` || busy === `recipe-prepare:${recipe.id}`
        return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && !recipeBusy && setRecipeDeployment(undefined)}>
          <section className="modal" role="dialog" aria-modal="true" aria-labelledby="deploy-saved-configuration-title">
            <div className="modal-heading"><div><p className="eyebrow">Recipe</p><h2 id="deploy-saved-configuration-title">Deploy {recipe.name || recipe.model}</h2></div><button className="icon-button" disabled={recipeBusy} onClick={() => setRecipeDeployment(undefined)} aria-label="Close dialog">×</button></div>
            <p className="modal-description">{recipe.tensor_parallel_size > 1 ? `TP${recipe.tensor_parallel_size} requires exactly ${recipe.required_node_count} nodes.` : `Select exactly ${recipe.required_node_count} ${recipe.required_node_count === 1 ? 'node' : 'nodes'}.`} Eligible nodes without weights can be prepared before deployment.</p>
            {recipeError && <p className="form-error" role="alert">{recipeError}</p>}
            {recipeTransferNotice && <p className="inline-success" role="status">{recipeTransferNotice}</p>}
            <NodeSelector
              nodes={nodes.data ?? []}
              selectedIds={nodeIds}
              onChange={(next) => {
                setRecipeDeployment({ recipe, nodeIds: next.length <= recipe.required_node_count ? next : nodeIds })
                if (recipeSeedNodeId && !next.includes(recipeSeedNodeId)) setRecipeSeedNodeId(undefined)
              }}
              loading={nodes.loading}
              error={nodes.error}
              onRetry={nodes.reload}
              multiple={recipe.required_node_count > 1}
              disabled={recipeBusy}
              allowedIds={allowedIds}
              unavailableReasons={unavailableReasons}
              localLabel={localLabel}
              primaryId={nodeIds[0]}
              legend={layoutLegend(recipe.deployment_mode, recipe.required_node_count)}
              help={localPath ? 'Local model paths can run only on the controller.' : `Choose the intended deployment nodes. ${layoutHelp(recipe.deployment_mode)} SparkDeck can seed ${recipe.model} from Hugging Face and fan it out through Virtual NAS.`}
            />
            {!localPath && missingNodes.length > 0 && <div className="recipe-transfer-options" aria-label="Virtual NAS transfer options">
              <div><strong>Prepare missing weights</strong><small>Select the deployment set above, then confirm one preparation workflow.</small></div>
              {transferPreflight.loading && <p role="status">Checking source weights, Hugging Face size, and cache-volume capacity…</p>}
              {transferPreflight.error && <div className="recipe-transfer-error" role="alert"><span>{transferPreflight.error}</span><Button type="button" variant="tertiary" onClick={transferPreflight.reload}>Retry</Button></div>}
              {transferPreflight.data && !transferPreflight.data.enabled && <p>Virtual NAS is disabled. Enable it on the Storage page before transferring weights.</p>}
              {transferPreflight.data?.enabled && transferPreflight.data.source && <p>A complete copy is available on {transferPreflight.data.source.node_name}; cache-empty nodes will receive it through Virtual NAS, while incomplete caches resume from Hugging Face.</p>}
              {transferPreflight.data?.enabled && !transferPreflight.data.source && transferPreflight.data.download && <p>No cluster node has the requested revision. Incomplete selected caches will resume from Hugging Face; a cache-empty selected node will seed any Virtual NAS fan-out.</p>}
              {transferPreflight.data?.enabled
                && !(transferPreflight.data.sources ?? []).some((source) => nodeIds.includes(source.node_id))
                && nodeIds.length > 1 && (
                <label className="field">
                  <span>Hub download seed (optional)</span>
                  <select
                    value={recipeSeedNodeId ?? ''}
                    disabled={recipeBusy}
                    onChange={(event) => setRecipeSeedNodeId(event.target.value || undefined)}
                  >
                    <option value="">Automatic</option>
                    {nodeIds.map((id) => {
                      const node = nodes.data?.find((item) => item.id === id)
                      return <option key={id} value={id}>{id === localNodeId ? localLabel : node?.name ?? id}</option>
                    })}
                  </select>
                  <small>The selected node downloads from Hugging Face first; the other nodes receive copies over Virtual NAS instead of downloading separately.</small>
                </label>
              )}
              {transferPreflight.data?.enabled && !transferPreflight.data.source && transferPreflight.data.download_error && <p>{transferPreflight.data.download_error}</p>}
              {transferPreflight.data?.enabled && <div className="recipe-transfer-targets">
                {missingNodes.filter((node) => nodeIds.includes(node.id)).map((node) => {
                  const option = preflightTargets.get(node.id)
                  const selectable = isNodeSelectable(node)
                  const trackedJob = recipeTransferJobs[recipeJobKey(recipe.id, recipe.model, node.id)]
                  const eligible = Boolean(allowedIds.includes(node.id) && selectable && !trackedJob)
                  const requiredBytes = recipePreparationRequiredBytes(
                    option, Boolean(transferPreflight.data?.source),
                  )
                  const capacity = option?.free_bytes != null && requiredBytes
                    ? `${formatBytes(option.free_bytes)} free · up to ${formatBytes(requiredBytes)} required`
                    : undefined
                  const reason = !selectable
                    ? node.online === false ? 'Node is offline' : node.docker_ready === false ? 'Docker unavailable' : 'Node unavailable for deployment'
                    : trackedJob ? `Model preparation ${option?.active_job_status ?? 'queued'}` : eligible ? undefined : unavailableReasons[node.id]
                  return <div className={`recipe-transfer-target${eligible ? '' : ' unavailable'}`} key={node.id}>
                    <span><strong>{node.name}</strong><small>{reason ?? capacity ?? 'Capacity verified at confirmation'}{reason && capacity ? ` · ${capacity}` : ''}</small></span>
                  </div>
                })}
              </div>}
            </div>}
            {!exactCount && <p className="field-note" role="status">Select exactly {recipe.required_node_count} {recipe.required_node_count === 1 ? 'node' : 'nodes'} to continue.</p>}
            <div className="modal-actions"><Button type="button" disabled={recipeBusy} onClick={() => setRecipeDeployment(undefined)}>Cancel</Button><Button variant="primary" disabled={(!ready && !canPrepare) || recipeBusy} onClick={() => void (ready ? deployRecipe() : prepareRecipeWeights())}>{ready ? <Play size={15} /> : <UploadCloud size={15} />} {recipeBusy ? (busy === `recipe:${recipe.id}` ? 'Deploying…' : 'Queueing…') : ready ? `Deploy on ${recipe.required_node_count} ${recipe.required_node_count === 1 ? 'node' : 'nodes'}` : 'Prepare selected nodes'}</Button></div>
          </section>
        </div>
      })()}

      {startSelection && (() => {
        const { deployment, nodeIds } = startSelection
        const required = deploymentRequiredNodes(deployment)
        const savedLaunch = deployment.status === 'saved'
        const converting = canPromoteDiscovered(deployment)
        const controllerArtifact = isControllerArtifact(deployment)
        const weighted = deploymentWeightedNodes(deployment)
        const plan = savedLaunch && !controllerArtifact ? startPreflight.data : undefined
        const planTargets = new Map((plan?.targets ?? []).map((target) => [target.node_id, target]))
        // Nodes without weights stay launchable whenever any of their own
        // preparation paths (transfer, Hugging Face download, or transfer
        // after a seed download) is feasible; the plan computes each node
        // independently, so one unrelated node out of cache space never
        // blocks a viable subset.
        const prepEligible = new Set((plan?.targets ?? [])
          .filter((target) => !weighted.has(target.node_id)
            && (target.eligible || target.download_eligible || target.transfer_after_download_eligible))
          .map((target) => target.node_id))
        const allowedIds = (nodes.data ?? [])
          .filter((node) => !deployment.managed || weighted.has(node.id) || prepEligible.has(node.id))
          .map((node) => node.id)
        const unavailableReasons = Object.fromEntries((nodes.data ?? [])
          .filter((node) => !allowedIds.includes(node.id)).map((node) => {
            const target = planTargets.get(node.id)
            return [node.id, controllerArtifact
              ? 'Local model artifacts are available only on the controller'
              : target?.active_job_status
                ? `Model preparation ${target.active_job_status}`
                : target?.reason
                  ?? target?.download_reason
                  ?? target?.transfer_after_download_reason
                  ?? 'Model weights not cached and the node cannot receive them']
          }))
        const weightWarnings = savedLaunch ? Object.fromEntries((nodes.data ?? [])
          .filter((node) => allowedIds.includes(node.id)
            && !(planTargets.get(node.id)?.has_required_weights ?? weighted.has(node.id)))
          .map((node) => [node.id, 'Weights need to be transferred before launch'])) : undefined
        const sharded = deployment.deployment_mode === 'sharded'
        const exactCount = nodeIds.length === required
        const allEligible = nodeIds.every((id) => allowedIds.includes(id) && nodes.data?.some((node) => node.id === id && isNodeSelectable(node)))
        const planReady = !savedLaunch || controllerArtifact || (!startPreflight.loading && !startPreflight.error)
        const ready = !nodes.loading && !nodes.error && planReady && exactCount && allEligible
        const needsPrep = savedLaunch && !controllerArtifact && nodeIds.some((id) => !weighted.has(id))
        const transferTargets = nodeIds
          .filter((id) => !weighted.has(id))
          .map((id) => nodes.data?.find((node) => node.id === id)?.name ?? id)
        const transferNotice = needsPrep && plan?.action === 'transfer' && plan.source && transferTargets.length
          ? `Weights will be transferred from ${plan.source.node_name} to ${transferTargets.join(', ')} via Virtual NAS before launch.`
          : undefined
        const startBusy = busy === deployment.id
        return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && !startBusy && setStartSelection(undefined)}>
          <section className="modal" role="dialog" aria-modal="true" aria-labelledby="start-deployment-title">
            <div className="modal-heading"><div><p className="eyebrow">{converting ? 'Convert deployment' : savedLaunch ? 'Launch deployment' : 'Start deployment'}</p><h2 id="start-deployment-title">{converting ? 'Make managed' : savedLaunch ? 'Launch' : 'Start'} {deployment.alias}</h2></div><button className="icon-button" disabled={startBusy} onClick={() => setStartSelection(undefined)} aria-label="Close dialog">×</button></div>
            <p className="modal-description">{sharded ? `TP${deployment.settings.tensor_parallel_size ?? required} requires exactly ${required} nodes.` : `Select ${required === 1 ? 'the node' : `exactly ${required} nodes`} to run ${deployment.model_id} on.`} {controllerArtifact ? 'This local artifact can run only on the controller.' : !deployment.managed ? 'SparkDeck will promote this discovered runtime into a managed deployment across the selected nodes.' : savedLaunch ? 'Nodes without the weights receive them automatically via Virtual NAS; nodes without enough free cache space are unavailable.' : 'Nodes without the complete model weights are disabled.'}</p>
            {startError && <p className="form-error" role="alert">{startError}</p>}
            {startNotice && <p className="inline-success" role="status">{startNotice}</p>}
            {transferNotice && <p className="field-note" role="status">{transferNotice}</p>}
            {!controllerArtifact && modelCache.error && <ErrorState message={`Model weights: ${modelCache.error}`} onRetry={modelCache.reload} />}
            {savedLaunch && !controllerArtifact && startPreflight.error && <ErrorState message={`Preparation plan: ${startPreflight.error}`} onRetry={startPreflight.reload} />}
            <NodeSelector
              nodes={nodes.data ?? []}
              selectedIds={nodeIds}
              onChange={(next) => setStartSelection({ deployment, nodeIds: next.length <= required ? next : nodeIds })}
              loading={nodes.loading || (!controllerArtifact && (modelCache.loading || (savedLaunch && startPreflight.loading)))}
              error={nodes.error}
              onRetry={() => { nodes.reload(); modelCache.reload(); if (savedLaunch) startPreflight.reload() }}
              multiple={required > 1}
              disabled={startBusy}
              allowedIds={allowedIds}
              unavailableReasons={unavailableReasons}
              warnings={weightWarnings}
              localLabel={localLabel}
              primaryId={nodeIds[0]}
              legend={layoutLegend(deployment.deployment_mode, required)}
              help={controllerArtifact ? 'Local model artifacts can run only on the controller.' : !deployment.managed ? `Choose the nodes SparkDeck should manage for this imported runtime. ${layoutHelp(deployment.deployment_mode)}` : savedLaunch ? `Choose where to launch. SparkDeck tracks which nodes hold ${deployment.model_id} and moves the weights to the rest via Virtual NAS.` : `Only nodes with ${deployment.model_id} already cached can be selected. ${layoutHelp(deployment.deployment_mode)}`}
            />
            {needsPrep && nodeIds.length > 1 && <label className="field"><span>Hub download seed (optional)</span>
              <select
                value={launchSeed && nodeIds.includes(launchSeed) ? launchSeed : ''}
                onChange={(event) => setLaunchSeed(event.target.value || undefined)}
              >
                <option value="">Automatic</option>
                {nodeIds.map((id) => {
                  const node = nodes.data?.find((item) => item.id === id)
                  return <option key={id} value={id}>{id === 'local' ? localLabel : node?.name ?? id}</option>
                })}
              </select>
              <small>One selected node downloads the GGUF from Hugging Face and the rest receive copies over the cluster network. Pick a node to control where that download runs.</small>
            </label>}
            {allowedIds.length < required && <p className="field-note">Only {allowedIds.length} of {required} required {required === 1 ? 'node is' : 'nodes are'} launchable. Free up model-cache space or copy the weights in Storage first.</p>}
            {!exactCount && <p className="field-note" role="status">Select exactly {required} {required === 1 ? 'node' : 'nodes'} to continue.</p>}
            <div className="modal-actions"><Button type="button" disabled={startBusy} onClick={() => setStartSelection(undefined)}>Cancel</Button><Button variant="primary" disabled={!ready || startBusy} onClick={() => void confirmStart()}><Play size={15} /> {startBusy ? (startNotice ? 'Preparing…' : converting ? 'Converting…' : 'Starting…') : converting ? `Make managed on ${required} ${required === 1 ? 'node' : 'nodes'}` : needsPrep ? `Transfer & launch on ${required} ${required === 1 ? 'node' : 'nodes'}` : `Launch on ${required} ${required === 1 ? 'node' : 'nodes'}`}</Button></div>
          </section>
        </div>
      })()}

      {additionalLaunch && (() => {
        const { deployment, currentIds, additionalIds } = additionalLaunch
        const weighted = deploymentWeightedNodes(deployment)
        // Gate on the cache predicate only: cached nodes that are offline or
        // Docker-unready stay in allowedIds so the selector reports their
        // real status ("Offline", "Docker unavailable") instead of a
        // misleading "weights not cached".
        const cachedIds = (nodes.data ?? []).filter((node) => !currentIds.includes(node.id) && weighted.has(node.id)).map((node) => node.id)
        const allowedIds = [...currentIds, ...cachedIds]
        const unavailableReasons = Object.fromEntries((nodes.data ?? []).filter((node) => !allowedIds.includes(node.id)).map((node) => [node.id, 'Model weights not cached']))
        const additionalBusy = busy === deployment.id
        const ready = additionalIds.length > 0 && !nodes.loading && !nodes.error
        return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && !additionalBusy && setAdditionalLaunch(undefined)}>
          <section className="modal" role="dialog" aria-modal="true" aria-labelledby="additional-nodes-title">
            <div className="modal-heading"><div><p className="eyebrow">Launch on additional nodes</p><h2 id="additional-nodes-title">Add nodes to {deployment.alias}</h2></div><button className="icon-button" disabled={additionalBusy} onClick={() => setAdditionalLaunch(undefined)} aria-label="Close dialog">×</button></div>
            <p className="modal-description">Currently running on {selectedNodeLabel(nodes.data ?? [], currentIds, localLabel)}. Choose the nodes that should also run {deployment.model_id}; the running nodes stay selected. SparkDeck relaunches the deployment, so existing replicas restart briefly.</p>
            {additionalError && <p className="form-error" role="alert">{additionalError}</p>}
            {modelCache.error && <ErrorState message={`Model weights: ${modelCache.error}`} onRetry={modelCache.reload} />}
            <NodeSelector
              nodes={nodes.data ?? []}
              selectedIds={[...currentIds, ...additionalIds]}
              onChange={(next) => setAdditionalLaunch({ deployment, currentIds, additionalIds: next.filter((id) => !currentIds.includes(id)) })}
              loading={nodes.loading || modelCache.loading}
              error={nodes.error}
              onRetry={() => { nodes.reload(); modelCache.reload() }}
              multiple
              disabled={additionalBusy}
              requiredIds={currentIds}
              allowedIds={allowedIds}
              unavailableReasons={unavailableReasons}
              localLabel={localLabel}
              primaryId={currentIds[0]}
              legend="Additional nodes · parallel instances"
              help={`Additional nodes run their own complete copy of ${deployment.model_id}. Only nodes with the model already cached can join, and the running nodes above cannot be removed here.`}
            />
            <div className="modal-actions"><Button type="button" disabled={additionalBusy} onClick={() => setAdditionalLaunch(undefined)}>Cancel</Button><Button variant="primary" disabled={!ready || additionalBusy} onClick={() => void confirmAdditionalLaunch()}><Play size={15} /> {additionalBusy ? 'Launching…' : `Launch on ${additionalIds.length} ${additionalIds.length === 1 ? 'node' : 'nodes'}`}</Button></div>
          </section>
        </div>
      })()}

      {creating && (
        <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) { setCreating(false); setEditingDeployment(undefined) } }}>
          <section className="modal" role="dialog" aria-modal="true" aria-labelledby="create-deployment-title">
            <div className="modal-heading"><div><p className="eyebrow">{editingDeployment ? 'Edit deployment' : 'New deployment'}</p><h2 id="create-deployment-title">{editingDeployment ? `Edit ${editingDeployment.alias}` : 'Create deployment'}</h2></div><button className="icon-button" onClick={() => { setCreating(false); setEditingDeployment(undefined) }} aria-label="Close dialog">×</button></div>
            <p className="modal-description">{editingDeployment ? 'Update the saved settings and node preferences. The runtime and model stay fixed; nothing launches until you choose Launch.' : 'Save a runtime, model, and node preferences as a deployment bookmark. Launch it from the deployments list whenever you are ready.'}</p>
            <form onSubmit={(event) => void create(event)}>
              {formError && <p className="form-error" role="alert">{formError}</p>}
              <div className="field-grid">
                <label className="field"><span>Display name</span><input autoFocus required value={form.alias} onChange={(event) => setForm({ ...form, alias: event.target.value })} /></label>
                <label className="field"><span>Runtime</span><select value={form.runtime} disabled={Boolean(editingDeployment)} onChange={(event) => updateRuntime(event.target.value as RuntimeKind)}><option value="vllm">vLLM</option><option value="sglang">SGLang</option><option value="llama.cpp">Llama server</option></select></label>
              </div>
              <label className="field"><span>Model repository or GGUF artifact</span><input required readOnly={Boolean(editingDeployment)} value={form.model_id} onChange={(event) => setForm({ ...form, model_id: event.target.value })} placeholder="org/model-name" /></label>
              {form.managed && form.runtime === 'vllm' && <label className="field"><span>vLLM image</span><input required value={form.settings.image ?? ''} onChange={(event) => setForm({ ...form, settings: { ...form.settings, image: event.target.value } })} placeholder="nvcr.io/nvidia/vllm:26.03.post1-py3" /><small>The container image pulled on every selected node. Change it to pin a different vLLM build or private registry tag.</small></label>}
              {!editingDeployment && cachedModels.length > 0 && <label className="field"><span>Or pick a model already on the cluster</span>
                <select
                  value={cachedModels.some((entry) => entry.modelId === form.model_id) ? form.model_id : ''}
                  onChange={(event) => {
                    const modelId = event.target.value
                    if (!modelId) return
                    setForm((current) => ({
                      ...current,
                      model_id: modelId,
                      alias: current.alias || modelId.split('/').at(-1) || modelId,
                    }))
                  }}
                >
                  <option value="">Select from cached models…</option>
                  {cachedModels.map((entry) => (
                    <option key={entry.modelId} value={entry.modelId}>
                      {entry.modelId} · {entry.nodeCount} {entry.nodeCount === 1 ? 'node' : 'nodes'} · {formatBytes(entry.sizeBytes)}
                    </option>
                  ))}
                </select>
              </label>}
              {form.runtime === 'llama.cpp' && createQuantizations.length > 0 ? <label className="field"><span>Quantization (optional)</span>
                <select
                  value={form.settings.quantization ?? ''}
                  onChange={(event) => updateCreateQuantization(event.target.value)}
                >
                  <option value="">Full precision (no quantization)</option>
                  {createQuantizationsByAvailability.map((variant) => (
                    <option key={variant.name} value={variant.name}>
                      {quantizationOptionLabel(variant, quantizationFilesDownloaded(variant, createCachedFileSets))}
                    </option>
                  ))}
                </select>
                <small>Quantizations published in the {form.model_id} repository, with their download size; ✓ Downloaded means the files are already in the cluster cache.</small>
              </label> : <label className="field"><span>Quantization (optional)</span><input value={form.settings.quantization ?? ''} onChange={(event) => setForm({ ...form, settings: { ...form.settings, quantization: event.target.value || undefined } })} placeholder="NVFP4, AWQ, Q4_K_M…" /></label>}
              {form.runtime === 'llama.cpp' && createArtifactOptions.length > 0 && !isLocalArtifact(form.settings.artifact) && !artifactManualEntry ? <label className="field"><span>GGUF artifact</span>
                <select
                  required
                  value={form.settings.artifact ?? ''}
                  onChange={(event) => {
                    if (event.target.value === MANUAL_ARTIFACT_OPTION) {
                      // Entering local-path mode starts from an empty field:
                      // a previously linked repository artifact must not
                      // stay prefilled and silently keep being used.
                      setArtifactManualEntry(true)
                      updateManualArtifact('')
                      return
                    }
                    updateCreateArtifact(event.target.value)
                  }}
                >
                  <option value="">Choose from the repository files…</option>
                  {createArtifactOptions.map((option) => (
                    <option key={option.key} value={option.filename}>
                      {artifactOptionLabel(option, artifactFilesDownloaded(option.files, createCachedFileSets))}
                    </option>
                  ))}
                  <option value={MANUAL_ARTIFACT_OPTION}>Use a local file path on this device…</option>
                </select>
                <small>The repository publishes one GGUF file per quantization; the chosen file is what nodes download from {form.model_id}. An absolute local path runs on this device only.</small>
              </label> : form.runtime === 'llama.cpp' && <>
                <label className="field"><span>GGUF artifact</span><input required value={form.settings.artifact ?? ''} onChange={(event) => updateManualArtifact(event.target.value)} placeholder="model-Q4_K_M.gguf" /><small>Repo-relative (e.g. subdir/model-Q4_K_M.gguf) to run on any node; an absolute local path runs on this device only.</small></label>
                {artifactManualEntry && createArtifactOptions.length > 0 && <Button type="button" variant="tertiary" onClick={() => { setArtifactManualEntry(false); updateManualArtifact('') }}>Choose from the repository files…</Button>}
              </>}
              {createModelCacheInfo && (createModelCacheInfo.cached.length > 0 || createModelCacheInfo.missing.length > 0) && <p className="field-note">
                {createModelCacheInfo.cached.length
                  ? `Weights cached on ${createModelCacheInfo.cached.join(', ')}.`
                  : 'Weights are not cached yet; they download onto the selected nodes at launch.'}
                {createModelCacheInfo.missing.length > 0 && createModelCacheInfo.cached.length > 0 && ` Virtual NAS transfers them to ${createModelCacheInfo.missing.join(', ')} at launch.`}
              </p>}
              {!editingDeployment && <label className="check-field"><input type="checkbox" checked={!form.managed} onChange={(event) => setForm({ ...form, managed: !event.target.checked })} /><span><strong>Connect an existing endpoint</strong><small>SparkDeck will not manage its process or container.</small></span></label>}
              {!form.managed && <label className="field"><span>Endpoint URL</span><input type="url" required value={form.endpoint_url} onChange={(event) => setForm({ ...form, endpoint_url: event.target.value })} placeholder="http://127.0.0.1:8001" /></label>}
              {!form.managed && <label className="field"><span>API key (optional)</span><input type="password" autoComplete="off" value={form.api_key ?? ''} onChange={(event) => setForm({ ...form, api_key: event.target.value })} /><small>Stored in your operating system credential store, never in SparkDeck's database.</small></label>}
              {form.managed && <NodeSelector
                nodes={nodes.data ?? []}
                selectedIds={form.node_ids ?? []}
                onChange={updateNodeSelection}
                loading={nodes.loading}
                error={nodes.error}
                onRetry={nodes.reload}
                multiple={form.runtime !== 'llama.cpp' || !isLocalArtifact(form.settings.artifact)}
                disabled={busy === 'create' || busy === 'edit'}
                localLabel={localLabel}
                primaryId={(form.node_ids?.length ?? 0) > 1 ? form.node_ids?.[0] : undefined}
                legend={layoutLegend(form.deployment_mode, form.node_ids?.length ?? 0)}
                help={`${layoutHelp(form.deployment_mode)} Saved with the deployment and preselected at launch; you can change the selection every time you launch.`}
              />}
              {form.managed && form.runtime === 'llama.cpp' && <p className="field-note">{isLocalArtifact(form.settings.artifact) ? 'Llama server runs on the local node for local GGUF artifacts.' : 'Llama server replicas run on each selected node; missing GGUF weights are fetched via Virtual NAS at launch.'}</p>}
              {form.managed && form.runtime !== 'llama.cpp' && (form.node_ids?.length ?? 0) > 1 && <label className="field"><span>Deployment layout</span><select value={form.deployment_mode === 'sharded' ? 'sharded' : 'replicated'} onChange={(event) => updateDeploymentMode(event.target.value as 'replicated' | 'sharded')}><option value="replicated">Parallel instances (replicated — a full model copy per node)</option><option value="sharded" disabled={!shardedAvailable}>Tensor parallelism (sharded — split one model across the nodes)</option></select><small>{form.deployment_mode === 'sharded' ? 'The first selected node is the coordinator, and tensor parallel size follows the selected node count.' : 'Each selected node runs a complete model replica.'}</small></label>}
              <div className="field-grid">
                <label className="field"><span>Context length</span><input type="number" min="256" value={form.settings.context_length} onChange={(event) => {
                  contextLengthTouched.current = true
                  setForm({ ...form, settings: { ...form.settings, context_length: Number(event.target.value) } })
                }} /></label>
                {form.runtime === 'llama.cpp' ? (
                  <label className="field"><span>Parallel slots</span><input type="number" min="1" value={form.settings.parallel_slots} onChange={(event) => setForm({ ...form, settings: { ...form.settings, parallel_slots: Number(event.target.value) } })} /></label>
                ) : (
                  <label className="field"><span>Tensor parallel size</span><input type="number" min="1" readOnly={form.deployment_mode === 'sharded'} value={form.settings.tensor_parallel_size} onChange={(event) => setForm({ ...form, settings: { ...form.settings, tensor_parallel_size: Number(event.target.value) } })} />{form.deployment_mode === 'sharded' && <small>Derived from the {form.node_ids?.length ?? 0} selected nodes.</small>}</label>
                )}
              </div>
              {form.managed && <>
                <Button type="button" variant="tertiary" aria-expanded={launchArgsOpen} onClick={() => setLaunchArgsOpen((open) => !open)}><Settings2 size={15} /> Launch arguments</Button>
                {launchArgsOpen && <div className="args-editor">
                  {form.runtime !== 'llama.cpp' && <label className="field"><span>GPU memory util</span><input type="number" step="0.05" min="0.1" max="0.98" placeholder="default" value={gpuMemoryUtil} onChange={(event) => setGpuMemoryUtil(event.target.value)} /></label>}
                  {form.runtime === 'vllm' && <label className="field"><span>Runtime environment variables</span><textarea rows={8} spellCheck={false} placeholder="HF_HUB_OFFLINE=1&#10;VLLM_CACHE_ROOT=/cache/clusterops-runtime/vllm" value={runtimeEnvironment} onChange={(event) => setRuntimeEnvironment(event.target.value)} /><small>One NAME=value per line. Stored as plain text and applied to every vLLM rank; do not enter secrets.</small></label>}
                  <label className="field"><span>Extra flags</span><textarea rows={3} spellCheck={false} placeholder="--kv-cache-dtype fp8 --max-num-seqs 32 --enable-prefix-caching" value={extraFlags} onChange={(event) => setExtraFlags(event.target.value)} /></label>
                  <p className="field-note">Passed to the runtime as-is. Context length and tensor parallel size above take precedence over duplicate flags here.</p>
                </div>}
              </>}
              <div className="modal-actions"><Button type="button" onClick={() => { setCreating(false); setEditingDeployment(undefined) }}>Cancel</Button><Button type="submit" variant="primary" disabled={busy === 'create' || busy === 'edit' || (form.managed && !selectionReady)}>{busy === 'create' || busy === 'edit' ? 'Saving…' : <><Server size={16} /> {editingDeployment ? 'Save changes' : 'Save deployment'}</>}</Button></div>
            </form>
          </section>
        </div>
      )}

      {logViewer && (
        <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && closeLogs()}>
          <section className="modal modal-wide" role="dialog" aria-modal="true" aria-labelledby="deployment-logs-title">
            <div className="modal-heading"><div><p className="eyebrow">Deployment logs</p><h2 id="deployment-logs-title">{logViewer.alias}</h2></div><button className="icon-button" onClick={closeLogs} aria-label="Close dialog">×</button></div>
            {logError && <p className="form-error" role="alert">{logError}</p>}
            {logMembers.length > 1 && (
              <div className="deployment-log-tabs" role="tablist" aria-label="Deployment log nodes">
                {logMembers.map((member, index) => {
                  const selected = member.node_id === selectedLogMember?.node_id
                  const detail = member.rank === 0 ? 'Primary' : member.rank === undefined ? 'Node' : `Rank ${member.rank}`
                  return <button
                    type="button"
                    role="tab"
                    id={`deployment-log-tab-${index}`}
                    aria-controls="deployment-log-panel"
                    aria-selected={selected}
                    tabIndex={selected ? 0 : -1}
                    key={`${member.node_id}-${member.rank ?? index}`}
                    onClick={() => setSelectedLogNodeId(member.node_id)}
                    onKeyDown={(event) => selectLogTabFromKeyboard(event, index)}
                  ><span>{logNodeLabel(member.node_id, member.node_name)}</span><small>{detail}</small></button>
                })}
              </div>
            )}
            <div
              id="deployment-log-panel"
              className="log-view deployment-log-view"
              ref={logPanelRef}
              role={logMembers.length > 1 ? 'tabpanel' : undefined}
              aria-labelledby={logMembers.length > 1 ? `deployment-log-tab-${Math.max(0, logMembers.indexOf(selectedLogMember!))}` : undefined}
              aria-label={logMembers.length > 1 ? undefined : `Logs for ${logViewer.alias}`}
              tabIndex={0}
            >
              {!logData && logLoading
                ? <span className="deployment-log-status">Loading logs…</span>
                : <pre>{selectedLogMember?.logs || logData?.logs || 'No log output.'}</pre>}
            </div>
            <div className="modal-actions">
              <Button type="button" variant={logTailing ? 'primary' : 'tertiary'} aria-pressed={logTailing} onClick={() => setLogTailing((current) => !current)}><ArrowDownToLine size={15} /> {logTailing ? 'Tailing' : 'Tail'}</Button>
              <Button type="button" disabled={logLoading} onClick={() => void loadLogs(logViewer.id)}><ScrollText size={15} /> {logLoading ? 'Refreshing…' : 'Refresh'}</Button>
              <Button type="button" onClick={closeLogs}>Close</Button>
            </div>
          </section>
        </div>
      )}
      {confirmationDialog}
    </div>
  )
}
