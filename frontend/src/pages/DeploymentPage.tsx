import { useEffect, useState, type FormEvent } from 'react'
import { ArrowLeft, Play, Save } from 'lucide-react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api/client'
import type { DeploymentDetail, DeploymentLaunchControls, DeploymentUpdateInput } from '../api/types'
import { Button, ErrorState, LoadingState, PageHeader, Panel, RuntimeMark, Status } from '../components/ui'
import { useResource } from '../hooks/useResource'

const quoteArg = (arg: string) => (arg === '' || /[^A-Za-z0-9_./:=+-]/.test(arg) ? `'${arg.replace(/'/g, `'\\''`)}'` : arg)

function splitFlags(input: string): string[] {
  const values: string[] = []
  let value = ''
  let quote: string | undefined
  let escaped = false
  let inWord = false
  for (const character of input) {
    if (escaped) { value += character; escaped = false; inWord = true; continue }
    if (quote) {
      if (character === quote) quote = undefined
      else if (quote === '"' && character === '\\') escaped = true
      else value += character
      continue
    }
    if (character === '\\') { escaped = true; inWord = true; continue }
    if (character === '"' || character === "'") { quote = character; inWord = true; continue }
    if (/\s/.test(character)) {
      if (inWord) { values.push(value); value = ''; inWord = false }
      continue
    }
    value += character
    inWord = true
  }
  if (quote) throw new Error('Flags contain an unmatched quote')
  if (escaped) { value += '\\'; inWord = true }
  if (inWord) values.push(value)
  return values
}

type Editor = Record<keyof DeploymentLaunchControls | 'gpu_memory_utilization' | 'gpu_memory_gb' | 'extra_args', string>

const editorFrom = (detail: DeploymentDetail): Editor => ({
  context_window: detail.launch_controls.context_window?.toString() ?? '',
  max_concurrency: detail.launch_controls.max_concurrency?.toString() ?? '',
  kv_cache_dtype: detail.launch_controls.kv_cache_dtype ?? '',
  thinking_mode: detail.launch_controls.thinking_mode ?? 'default',
  dspark_num_speculative_tokens: detail.launch_controls.dspark_num_speculative_tokens?.toString() ?? '',
  max_cudagraph_capture_size: detail.launch_controls.max_cudagraph_capture_size?.toString() ?? '',
  max_num_batched_tokens: detail.launch_controls.max_num_batched_tokens?.toString() ?? '',
  gpu_memory_utilization: detail.gpu_memory_utilization?.toString() ?? '',
  gpu_memory_gb: detail.gpu_memory_gb?.toString() ?? '',
  extra_args: detail.extra_args.map(quoteArg).join(' '),
})

const optionalNumber = (value: string) => value.trim() ? Number(value) : null

function updateInput(editor: Editor): DeploymentUpdateInput {
  return {
    extra_args: splitFlags(editor.extra_args),
    launch_controls: {
      context_window: optionalNumber(editor.context_window),
      max_concurrency: optionalNumber(editor.max_concurrency),
      kv_cache_dtype: editor.kv_cache_dtype.trim() || null,
      thinking_mode: editor.thinking_mode || 'default',
      dspark_num_speculative_tokens: optionalNumber(editor.dspark_num_speculative_tokens),
      max_cudagraph_capture_size: optionalNumber(editor.max_cudagraph_capture_size),
      max_num_batched_tokens: optionalNumber(editor.max_num_batched_tokens),
    },
    gpu_memory_utilization: optionalNumber(editor.gpu_memory_utilization),
    gpu_memory_gb: optionalNumber(editor.gpu_memory_gb),
  }
}

export function DeploymentPage() {
  const { deploymentId = '' } = useParams()
  const navigate = useNavigate()
  const resource = useResource((signal) => api.deployments.get(deploymentId, signal), [deploymentId])
  const [editor, setEditor] = useState<Editor>()
  const [busy, setBusy] = useState<'save' | 'run'>()
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
      setNotice('Deployment settings saved. They will be applied on the next run.')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not save deployment settings')
    } finally { setBusy(undefined) }
  }

  const run = async () => {
    setBusy('run'); setError(undefined); setNotice(undefined)
    try {
      await persist()
      await api.deployments.action(deploymentId, 'start')
      navigate('/models')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not run deployment')
      setBusy(undefined)
    }
  }

  if (resource.loading && !resource.data) return <LoadingState label="Loading deployment" />
  if (resource.error && !resource.data) return <ErrorState message={resource.error} onRetry={resource.reload} />
  const detail = resource.data
  if (!detail || !editor) return null
  const disabled = !detail.editable || Boolean(busy)

  return <>
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
        <label className="field"><span>GPU memory utilization</span><input disabled={disabled} type="number" min="0.01" max="1" step="0.01" value={editor.gpu_memory_utilization} onChange={(event) => set('gpu_memory_utilization', event.target.value)} /></label>
        <label className="field"><span>GPU memory reserve (GB)</span><input disabled={disabled} type="number" min="0" step="0.1" value={editor.gpu_memory_gb} onChange={(event) => set('gpu_memory_gb', event.target.value)} /></label>
        <label className="field wide-field"><span>Runtime flags</span><textarea disabled={disabled} rows={6} spellCheck={false} value={editor.extra_args} onChange={(event) => set('extra_args', event.target.value)} /><small>Shell quoting is preserved when the flags are saved.</small></label>
        <div className="settings-save wide-field"><Button type="submit" disabled={disabled}><Save size={15} /> {busy === 'save' ? 'Saving…' : 'Save'}</Button><Button type="button" variant="primary" disabled={disabled} onClick={() => void run()}><Play size={15} /> {busy === 'run' ? 'Starting…' : 'Run'}</Button></div>
      </form>
    </Panel>
  </>
}
