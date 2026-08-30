import { useEffect, useState, type FormEvent } from 'react'
import { ArrowLeft, Play, Save } from 'lucide-react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api/client'
import type { DeploymentDetail, DeploymentLaunchControls, DeploymentUpdateInput } from '../api/types'
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

const optionalNumber = (value: string) => value.trim() ? Number(value) : null

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
  const [editor, setEditor] = useState<Editor>()
  const [busy, setBusy] = useState<'save' | 'run' | 'stop'>()
  const [error, setError] = useState<string>()
  const [notice, setNotice] = useState<string>()

  useEffect(() => {
    if (resource.data) setEditor(editorFrom(resource.data))
  }, [resource.data])

  const set = (key: keyof Editor, value: string) => setEditor((current) => current ? { ...current, [key]: value } : current)

  const persist = async () => {
    if (!editor) throw new Error('Deployment settings are not loaded')
    const updated = await api.deployments.update(deploymentId, updateInput(editor))
    setEditor(editorFrom(updated))
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

  const run = async (form: HTMLFormElement | null) => {
    if (resource.data?.editable && (!form || !form.reportValidity())) return
    setBusy('run'); setError(undefined); setNotice(undefined)
    try {
      if (resource.data?.editable) await persist()
      await api.deployments.action(deploymentId, 'start')
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
  const active = ['launching', 'starting', 'running', 'ready'].includes(detail.status)
  const lifecycleDisabled = Boolean(busy) || (!detail.editable && !detail.controllable)

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
        <div className="settings-save wide-field">
          <Button type="submit" disabled={disabled}><Save size={15} /> {busy === 'save' ? 'Saving…' : 'Save'}</Button>
          {active && detail.controllable
            ? <Button type="button" variant="primary" disabled={lifecycleDisabled} onClick={() => void stop()}>{busy === 'stop' ? 'Stopping…' : 'Stop'}</Button>
            : <Button type="button" variant="primary" disabled={lifecycleDisabled} onClick={(event) => void run(event.currentTarget.form)}><Play size={15} /> {busy === 'run' ? 'Starting…' : 'Run'}</Button>}
        </div>
      </form>
    </Panel>
  </div>
}
