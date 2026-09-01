import { useEffect, useState, type FormEvent } from 'react'
import { ArrowLeft, Play, Save } from 'lucide-react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api/client'
import type { DeploymentDetail, DeploymentLaunchControls, DeploymentUpdateInput } from '../api/types'
import { isNodeSelectable, NodeSelector } from '../components/NodeSelector'
import { Button, ErrorState, LoadingState, PageHeader, Panel, RuntimeMark, Status } from '../components/ui'
import { useResource } from '../hooks/useResource'
import { formatEnvironment, parseEnvironment } from '../utils/environment'

const quoteArg = (arg: string) => (arg === '' || /[^A-Za-z0-9_./:=+-]/.test(arg) ? `'${arg.replace(/'/g, `'\\''`)}'` : arg)

function splitFlags(input: string): string[] {
  const values: string[] = []
  let value = ''
  let quote: string | undefined
  let inWord = false
  for (let index = 0; index < input.length; index += 1) {
    const character = input[index]
    if (quote) {
      if (character === quote) quote = undefined
      else if (quote === '"' && character === '\\') {
        const next = input[index + 1]
        if (next === '\n') index += 1
        else if (next === '$' || next === '`' || next === '"' || next === '\\') {
          value += next
          index += 1
          inWord = true
        } else {
          value += character
          inWord = true
        }
      }
      else value += character
      continue
    }
    if (character === '\\') {
      const next = input[index + 1]
      if (next === undefined) value += character
      else { value += next; index += 1 }
      inWord = true
      continue
    }
    if (character === '"' || character === "'") { quote = character; inWord = true; continue }
    if (/\s/.test(character)) {
      if (inWord) { values.push(value); value = ''; inWord = false }
      continue
    }
    value += character
    inWord = true
  }
  if (quote) throw new Error('Flags contain an unmatched quote')
  if (inWord) values.push(value)
  return values
}

const flagValue = (args: string[], names: string[]) => {
  for (let index = 0; index < args.length; index += 1) {
    if (names.includes(args[index])) return args[index + 1]
    const name = names.find((candidate) => args[index].startsWith(`${candidate}=`))
    if (name) return args[index].slice(name.length + 1)
  }
  return undefined
}

type Editor = Record<keyof DeploymentLaunchControls | 'gpu_memory_utilization' | 'gpu_memory_gb' | 'sg_tp_size' | 'sg_mem_fraction' | 'environment' | 'extra_args', string>

const editorFrom = (detail: DeploymentDetail): Editor => ({
  context_window: detail.launch_controls.context_window?.toString() ?? '',
  max_concurrency: detail.launch_controls.max_concurrency?.toString() ?? '',
  tensor_parallel_size: detail.launch_controls.tensor_parallel_size?.toString()
    ?? flagValue(detail.extra_args, ['--tensor-parallel-size', '-tp']) ?? '',
  pipeline_parallel_size: detail.launch_controls.pipeline_parallel_size?.toString()
    ?? flagValue(detail.extra_args, ['--pipeline-parallel-size', '-pp']) ?? '',
  kv_cache_dtype: detail.launch_controls.kv_cache_dtype ?? '',
  thinking_mode: detail.launch_controls.thinking_mode ?? 'default',
  speculative_method: detail.launch_controls.speculative_method ?? '',
  draft_sample_method: detail.launch_controls.draft_sample_method ?? '',
  dspark_num_speculative_tokens: detail.launch_controls.dspark_num_speculative_tokens?.toString() ?? '',
  max_cudagraph_capture_size: detail.launch_controls.max_cudagraph_capture_size?.toString() ?? '',
  max_num_batched_tokens: detail.launch_controls.max_num_batched_tokens?.toString() ?? '',
  gpu_memory_utilization: detail.gpu_memory_utilization?.toString() ?? '',
  gpu_memory_gb: detail.gpu_memory_gb?.toString() ?? '',
  sg_tp_size: detail.sg_tp_size?.toString() ?? '',
  sg_mem_fraction: detail.sg_mem_fraction?.toString() ?? '',
  environment: formatEnvironment(detail.environment ?? detail.settings.environment),
  extra_args: detail.extra_args.map(quoteArg).join(' '),
})

const editorFingerprint = (editor: Editor) => JSON.stringify(editor)

const optionalNumber = (value: string) => value.trim() ? Number(value) : null

const SPECULATIVE_METHODS = ['dspark', 'dflash', 'draft_model', 'eagle3', 'mtp', 'ngram', 'ngram_gpu', 'suffix']
const DRAFT_SAMPLE_METHODS = ['greedy', 'probabilistic']
const TRANSITIONAL_DEPLOYMENT_STATUSES = new Set(['launching', 'starting', 'stopping'])

function updateInput(editor: Editor): DeploymentUpdateInput {
  return {
    extra_args: splitFlags(editor.extra_args),
    environment: parseEnvironment(editor.environment),
    launch_controls: {
      context_window: optionalNumber(editor.context_window),
      max_concurrency: optionalNumber(editor.max_concurrency),
      tensor_parallel_size: optionalNumber(editor.tensor_parallel_size),
      pipeline_parallel_size: optionalNumber(editor.pipeline_parallel_size),
      kv_cache_dtype: editor.kv_cache_dtype.trim() || null,
      thinking_mode: editor.thinking_mode || 'default',
      speculative_method: editor.speculative_method || null,
      draft_sample_method: editor.draft_sample_method || null,
      dspark_num_speculative_tokens: optionalNumber(editor.dspark_num_speculative_tokens),
      max_cudagraph_capture_size: optionalNumber(editor.max_cudagraph_capture_size),
      max_num_batched_tokens: optionalNumber(editor.max_num_batched_tokens),
    },
    gpu_memory_utilization: optionalNumber(editor.gpu_memory_utilization),
    gpu_memory_gb: optionalNumber(editor.gpu_memory_gb),
    sg_tp_size: optionalNumber(editor.sg_tp_size),
    sg_mem_fraction: optionalNumber(editor.sg_mem_fraction),
  }
}

export function DeploymentPage() {
  const { deploymentId = '' } = useParams()
  const navigate = useNavigate()
  const resource = useResource((signal) => api.deployments.get(deploymentId, signal), [deploymentId])
  const nodes = useResource((signal) => api.nodes.list(signal))
  const [editor, setEditor] = useState<Editor>()
  const [savedEditorFingerprint, setSavedEditorFingerprint] = useState<string>()
  const [busy, setBusy] = useState<'save' | 'run' | 'stop'>()
  const [error, setError] = useState<string>()
  const [notice, setNotice] = useState<string>()
  const [runSelection, setRunSelection] = useState<string[]>()
  const [finalFlags, setFinalFlags] = useState('')
  const [previewError, setPreviewError] = useState<string>()

  useEffect(() => {
    if (resource.data) {
      const savedEditor = editorFrom(resource.data)
      setEditor(savedEditor)
      setSavedEditorFingerprint(editorFingerprint(savedEditor))
    }
  }, [resource.data])

  // Transitional statuses resolve without user input; keep the detail view
  // live so a finished stop/start is reflected without a manual reload.
  useEffect(() => {
    if (resource.loading || !resource.data || !TRANSITIONAL_DEPLOYMENT_STATUSES.has(resource.data.status)) return
    const timer = window.setTimeout(resource.reload, 2000)
    return () => window.clearTimeout(timer)
  }, [resource.data, resource.loading, resource.reload])

  useEffect(() => {
    if (!editor || !resource.data || resource.data.runtime === 'llama.cpp') {
      setFinalFlags('')
      setPreviewError(undefined)
      return
    }
    const deployment = resource.data
    const runtime = deployment.runtime
    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      try {
        const input = updateInput(editor)
        void api.deployments.previewFlags(
          runtime, input, deployment, controller.signal,
        )
          .then((preview) => {
            setFinalFlags(preview.command_flags)
            setPreviewError(undefined)
          })
          .catch((reason) => {
            if (controller.signal.aborted) return
            setFinalFlags('')
            setPreviewError(reason instanceof Error ? reason.message : 'Could not preview runtime flags')
          })
      } catch (reason) {
        setFinalFlags('')
        setPreviewError(reason instanceof Error ? reason.message : 'Could not preview runtime flags')
      }
    }, 200)
    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [editor, resource.data])

  const set = (key: keyof Editor, value: string) => setEditor((current) => current ? { ...current, [key]: value } : current)

  const persist = async () => {
    if (!editor) throw new Error('Deployment settings are not loaded')
    const updated = await api.deployments.update(deploymentId, updateInput(editor))
    const savedEditor = editorFrom(updated)
    setEditor(savedEditor)
    setSavedEditorFingerprint(editorFingerprint(savedEditor))
    return updated
  }

  const save = async (event: FormEvent) => {
    event.preventDefault()
    setBusy('save'); setError(undefined); setNotice(undefined)
    try {
      await persist()
      setNotice(deploymentId.startsWith('container:')
        ? 'Deployment settings saved and applied.'
        : 'Deployment settings saved. They will be applied on the next run.')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not save deployment settings')
    } finally { setBusy(undefined) }
  }

  const requiredRunNodes = () => {
    if (!editor) return 1
    const mode = resource.data?.deployment_mode
    if (mode === 'single') return 1
    if (mode === 'replicated') {
      const persisted = Number(resource.data?.required_node_count)
      return Number.isInteger(persisted) && persisted > 1
        ? persisted
        : Math.max(2, resource.data?.node_ids?.length ?? 0)
    }
    const tp = Number(resource.data?.runtime === 'sglang'
      ? editor.sg_tp_size
      : resource.data?.runtime === 'vllm' ? editor.tensor_parallel_size : '1')
    const tensor = Number.isInteger(tp) && tp > 0 ? tp : 1
    if (resource.data?.runtime !== 'vllm') return tensor
    const pp = Number(editor.pipeline_parallel_size)
    const pipeline = Number.isInteger(pp) && pp > 0 ? pp : 1
    const world = tensor * pipeline
    const savedCount = resource.data?.node_ids?.length ?? 0
    return world > 1 && savedCount > 1 && world % savedCount === 0
      ? savedCount
      : world
  }

  const openRun = (form: HTMLFormElement | null) => {
    if (resource.data?.editable && (!form || !form.reportValidity())) return
    const required = requiredRunNodes()
    const selectable = (nodes.data ?? []).filter(isNodeSelectable).map((node) => node.id)
    const preferred = (resource.data?.node_ids ?? []).filter((id) => selectable.includes(id))
    setError(undefined); setNotice(undefined)
    setRunSelection([...new Set([...preferred, ...selectable])].slice(0, required))
  }

  const run = async () => {
    if (!runSelection) return
    setBusy('run'); setError(undefined); setNotice(undefined)
    try {
      if (resource.data?.editable) await persist()
      await api.deployments.action(deploymentId, 'start', runSelection)
      navigate('/models')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not run deployment')
      setBusy(undefined)
    }
  }

  const stop = async () => {
    setBusy('stop'); setError(undefined); setNotice(undefined)
    try {
      await api.deployments.action(deploymentId, 'stop')
      navigate('/models')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not stop deployment')
      setBusy(undefined)
    }
  }

  if (resource.loading && !resource.data) return <div className="page"><LoadingState label="Loading deployment" /></div>
  if (resource.error && !resource.data) return <div className="page"><ErrorState message={resource.error} onRetry={resource.reload} /></div>
  const detail = resource.data
  if (!detail || !editor) return null
  const disabled = !detail.editable || Boolean(busy)
  const hasUnsavedChanges = editorFingerprint(editor) !== savedEditorFingerprint
  const active = ['launching', 'starting', 'stopping', 'running', 'ready'].includes(detail.status)
  const lifecycleDisabled = Boolean(busy) || detail.status === 'stopping' || (!detail.editable && !detail.controllable)

  return <div className="page">
    <PageHeader
      eyebrow="Deployment"
      title={detail.alias}
      description={`${detail.model_id} on ${detail.node_ids?.length ?? 1} node${detail.node_ids?.length === 1 ? '' : 's'}`}
      actions={<Link className="button button-secondary" to="/models"><ArrowLeft size={15} /> Models</Link>}
    />
    <Panel className="settings-section deployment-detail-section">
      <div className="settings-heading"><span><RuntimeMark runtime={detail.runtime} /></span><div><h2>Runtime object</h2><p>Saved launch settings and flags for this deployment.</p></div></div>
      <form className="settings-fields" onSubmit={(event) => void save(event)}>
        <div className="credential-state wide-field"><RuntimeMark runtime={detail.runtime} /><div><strong>{detail.model_id}</strong><span className="muted">{detail.deployment_mode ?? 'single'} · desired {detail.desired_state}</span></div><Status status={detail.status} /></div>
        {!detail.editable && <p className="form-error wide-field" role="status">{detail.edit_reason || 'Stop this deployment before editing its launch settings.'}</p>}
        {error && <p className="form-error wide-field" role="alert">{error}</p>}
        {notice && <p className="muted wide-field" role="status">{notice}</p>}
        <label className="field"><span>Context window</span><input disabled={disabled} type="number" min="1" value={editor.context_window} onChange={(event) => set('context_window', event.target.value)} /></label>
        <label className="field"><span>Max concurrency</span><input disabled={disabled} type="number" min="1" value={editor.max_concurrency} onChange={(event) => set('max_concurrency', event.target.value)} /></label>
        <label className="field"><span>KV cache dtype</span><input disabled={disabled} value={editor.kv_cache_dtype} onChange={(event) => set('kv_cache_dtype', event.target.value)} placeholder="auto" /></label>
        <label className="field"><span>Thinking mode</span><select disabled={disabled} value={editor.thinking_mode} onChange={(event) => set('thinking_mode', event.target.value)}><option value="default">Default</option><option value="enabled">Enabled</option><option value="disabled">Disabled</option></select></label>
        {detail.runtime === 'vllm' && <>
          <label className="field"><span>Speculative method</span><select disabled={disabled} value={editor.speculative_method} onChange={(event) => set('speculative_method', event.target.value)}><option value="">Auto / unset</option>{editor.speculative_method && !SPECULATIVE_METHODS.includes(editor.speculative_method) && <option value={editor.speculative_method}>{editor.speculative_method}</option>}{SPECULATIVE_METHODS.map((method) => <option key={method} value={method}>{method}</option>)}</select></label>
          <label className="field"><span>Draft sample method</span><select disabled={disabled} value={editor.draft_sample_method} onChange={(event) => set('draft_sample_method', event.target.value)}><option value="">Default</option>{editor.draft_sample_method && !DRAFT_SAMPLE_METHODS.includes(editor.draft_sample_method) && <option value={editor.draft_sample_method}>{editor.draft_sample_method}</option>}{DRAFT_SAMPLE_METHODS.map((method) => <option key={method} value={method}>{method}</option>)}</select></label>
        </>}
        <label className="field"><span>Speculative tokens</span><input disabled={disabled} type="number" min="1" value={editor.dspark_num_speculative_tokens} onChange={(event) => set('dspark_num_speculative_tokens', event.target.value)} /></label>
        <label className="field"><span>CUDA graph capture size</span><input disabled={disabled} type="number" min="1" value={editor.max_cudagraph_capture_size} onChange={(event) => set('max_cudagraph_capture_size', event.target.value)} /></label>
        <label className="field"><span>Max batched tokens</span><input disabled={disabled} type="number" min="1" value={editor.max_num_batched_tokens} onChange={(event) => set('max_num_batched_tokens', event.target.value)} /></label>
        {detail.runtime === 'vllm' && <>
          <label className="field"><span>Tensor parallel size</span><input disabled={disabled} type="number" min="1" value={editor.tensor_parallel_size} onChange={(event) => set('tensor_parallel_size', event.target.value)} /></label>
          <label className="field"><span>Pipeline parallel size</span><input disabled={disabled} type="number" min="1" value={editor.pipeline_parallel_size} onChange={(event) => set('pipeline_parallel_size', event.target.value)} /></label>
          <label className="field"><span>GPU memory utilization</span><input disabled={disabled} type="number" min="0.01" max="1" step="0.01" value={editor.gpu_memory_utilization} onChange={(event) => set('gpu_memory_utilization', event.target.value)} /></label>
          {!detail.id.startsWith('container:') && <label className="field"><span>GPU memory reserve (GB)</span><input disabled={disabled} type="number" min="0" step="0.1" value={editor.gpu_memory_gb} onChange={(event) => set('gpu_memory_gb', event.target.value)} /></label>}
          <label className="field wide-field"><span>Runtime environment variables</span><textarea disabled={disabled} rows={8} spellCheck={false} placeholder="VLLM_CACHE_ROOT=/cache/clusterops-runtime/vllm" value={editor.environment} onChange={(event) => set('environment', event.target.value)} /><small>One NAME=value per line. Stored as plain text and applied to every vLLM rank; do not enter secrets.</small></label>
        </>}
        {detail.runtime === 'sglang' && <>
          <label className="field"><span>TP size</span><input disabled={disabled} type="number" min="1" value={editor.sg_tp_size} onChange={(event) => set('sg_tp_size', event.target.value)} /></label>
          <label className="field"><span>Mem fraction (static)</span><input disabled={disabled} type="number" min="0.01" max="1" step="0.01" value={editor.sg_mem_fraction} onChange={(event) => set('sg_mem_fraction', event.target.value)} /></label>
        </>}
        <label className="field wide-field"><span>Runtime flags</span><textarea disabled={disabled} rows={6} spellCheck={false} value={editor.extra_args} onChange={(event) => set('extra_args', event.target.value)} /><small>Shell quoting is preserved when the flags are saved.</small></label>
        {detail.runtime !== 'llama.cpp' && <label className="field wide-field"><span>Final runtime flags (preview only)</span><textarea readOnly rows={6} spellCheck={false} value={finalFlags} /><small>Backend-normalized flags after dropdown values and environment references are resolved. This field is not submitted.</small>{previewError && <small className="form-error" role="alert">{previewError}</small>}</label>}
        <div className="settings-save wide-field">
          <Button type="submit" disabled={disabled || !hasUnsavedChanges}><Save size={15} /> {busy === 'save' ? 'Saving…' : 'Save'}</Button>
          {active && detail.controllable
            ? <Button type="button" variant="primary" disabled={lifecycleDisabled} onClick={() => void stop()}>{busy === 'stop' || detail.status === 'stopping' ? 'Stopping…' : 'Stop'}</Button>
            : <Button type="button" variant="primary" disabled={lifecycleDisabled} onClick={(event) => openRun(event.currentTarget.form)}><Play size={15} /> Run</Button>}
        </div>
      </form>
    </Panel>
    {runSelection && (() => {
      const required = requiredRunNodes()
      const tensor = Number(detail.runtime === 'sglang' ? editor.sg_tp_size : editor.tensor_parallel_size) || 1
      const layoutDescription = detail.deployment_mode === 'sharded'
        ? `TP${tensor} is distributed across exactly ${required} ${required === 1 ? 'node' : 'nodes'}.`
        : detail.deployment_mode === 'replicated'
          ? `This replicated layout runs on exactly ${required} nodes.`
          : `This single-node layout runs TP${tensor} on one physical node.`
      const exactCount = runSelection.length === required
      const allSelectable = runSelection.every((id) => nodes.data?.some((node) => node.id === id && isNodeSelectable(node)))
      const ready = !nodes.loading && !nodes.error && exactCount && allSelectable
      return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && !busy && setRunSelection(undefined)}>
        <section className="modal" role="dialog" aria-modal="true" aria-labelledby="run-deployment-title">
          <div className="modal-heading"><div><p className="eyebrow">Start deployment</p><h2 id="run-deployment-title">Start {detail.alias}</h2></div><button className="icon-button" disabled={Boolean(busy)} onClick={() => setRunSelection(undefined)} aria-label="Close dialog">×</button></div>
          <p className="modal-description">{layoutDescription} Select where SparkDeck should start the deployment.</p>
          {error && <p className="form-error" role="alert">{error}</p>}
          <NodeSelector
            nodes={nodes.data ?? []}
            selectedIds={runSelection}
            onChange={(next) => setRunSelection(next.length <= required ? next : runSelection)}
            loading={nodes.loading}
            error={nodes.error}
            onRetry={nodes.reload}
            multiple={required > 1}
            disabled={Boolean(busy)}
            primaryId={runSelection[0]}
            legend="Target nodes"
            help={`Choose exactly ${required} launch ${required === 1 ? 'node' : 'nodes'}. The first selected node coordinates the deployment.`}
          />
          {!exactCount && <p className="field-note" role="status">Select exactly {required} {required === 1 ? 'node' : 'nodes'} to continue.</p>}
          <div className="modal-actions"><Button type="button" disabled={Boolean(busy)} onClick={() => setRunSelection(undefined)}>Cancel</Button><Button variant="primary" disabled={!ready || Boolean(busy)} onClick={() => void run()}><Play size={15} /> {busy === 'run' ? 'Starting…' : `Start on ${required} ${required === 1 ? 'node' : 'nodes'}`}</Button></div>
        </section>
      </div>
    })()}
  </div>
}
