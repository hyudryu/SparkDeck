import { useEffect, useRef, useState, type FormEvent } from 'react'
import type { ContainerImage, NodeInventoryItem, PatchBuild } from '../api/types'
import { api } from '../api/client'
import { Button, Panel } from './ui'
import { isNodeSelectable, NodeSelector } from './NodeSelector'

const FILE_LIMIT = 1024 * 1024
const TOTAL_LIMIT = 4 * FILE_LIMIT
type DraftFile = { id: number; target: string; content: string; error?: string }
const active = (build: PatchBuild) => build.status === 'queued' || build.status === 'building'

export function PatchedImages({ images, nodes, localLabel, onBuilt }: {
  images: ContainerImage[]; nodes: NodeInventoryItem[]; localLabel: string; onBuilt: () => void
}) {
  const [open, setOpen] = useState(false)
  const [base, setBase] = useState('')
  const [image, setImage] = useState('')
  const [nodeIds, setNodeIds] = useState<string[]>(['local'])
  const [files, setFiles] = useState<DraftFile[]>([{ id: 1, target: '', content: '' }])
  const nextId = useRef(2)
  const [busy, setBusy] = useState(false)
  const [reading, setReading] = useState(0)
  const [error, setError] = useState<string>()
  const [historyError, setHistoryError] = useState<string>()
  const [builds, setBuilds] = useState<PatchBuild[]>([])
  const [retry, setRetry] = useState(0)
  const known = useRef(new Map<string, string>())

  useEffect(() => {
    if (!nodes.length) return
    setNodeIds((current) => {
      const available = current.filter((id) => nodes.some((node) => node.id === id && isNodeSelectable(node)))
      const fallback = nodes.find((node) => node.local && isNodeSelectable(node)) ?? nodes.find(isNodeSelectable)
      return available.length ? available : fallback ? [fallback.id] : []
    })
  }, [nodes])

  useEffect(() => {
    let stopped = false
    let timer: ReturnType<typeof setTimeout> | undefined
    const controller = new AbortController()
    const refresh = async () => {
      try {
        const result = await api.images.patchBuilds(controller.signal)
        if (stopped) return
        const items = result.items ?? []
        if (items.some((item) => known.current.has(item.id) && known.current.get(item.id) !== item.status && !active(item))) onBuilt()
        known.current = new Map(items.map((item) => [item.id, item.status]))
        setBuilds(items)
        setHistoryError(undefined)
        if (items.some(active)) timer = setTimeout(() => void refresh(), 3000)
      } catch (reason) {
        if (!stopped) {
          setHistoryError(reason instanceof Error ? reason.message : 'Could not load patch builds')
          // Preserve active jobs and retry transient polling failures.
          if ([...known.current.values()].some((status) => status === 'queued' || status === 'building')) timer = setTimeout(() => void refresh(), 3000)
        }
      }
    }
    void refresh()
    return () => { stopped = true; controller.abort(); clearTimeout(timer) }
  }, [retry, onBuilt])

  const changeFile = (id: number, change: Partial<DraftFile>) => setFiles((current) => current.map((file) => file.id === id ? { ...file, ...change } : file))
  const upload = async (id: number, file?: File) => {
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.py') || file.size > FILE_LIMIT) {
      changeFile(id, { error: 'Choose a Python (.py) file no larger than 1 MiB.' })
      return
    }
    setReading((count) => count + 1)
    try {
      const bytes = await file.arrayBuffer()
      const content = new TextDecoder('utf-8', { fatal: true }).decode(bytes)
      if (content.includes('\0')) throw new Error('Binary file')
      changeFile(id, { content, error: undefined })
    } catch {
      changeFile(id, { error: 'Could not read this file as UTF-8 Python text. Your previous contents are unchanged.' })
    } finally { setReading((count) => count - 1) }
  }
  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setError(undefined)
    const sizes = files.map((file) => new TextEncoder().encode(file.content).length)
    const targets = files.map((file) => file.target.trim())
    if (files.some((file, index) => file.error || sizes[index] > FILE_LIMIT || file.content.includes('\0'))) {
      setError('Each patch needs UTF-8 Python contents, up to 1 MiB per file. Correct any upload errors before building.'); return
    }
    if (sizes.reduce((sum, size) => sum + size, 0) > TOTAL_LIMIT) { setError('Patch contents must total no more than 4 MiB.'); return }
    if (targets.some((target) => !target.startsWith('/') || target.endsWith('/') || target.split('/').some((part) => part === '..' || part === '.') || !target.endsWith('.py')) || new Set(targets).size !== targets.length) {
      setError('Use a unique absolute container path ending in .py for each file, without . or .. segments.'); return
    }
    if (!base.trim() || !image.trim() || image.trim() === base.trim()) { setError('Enter a base image and a different tag for the patched image.'); return }
    if (!nodeIds.length || nodeIds.some((id) => !nodes.some((node) => node.id === id && isNodeSelectable(node)))) { setError('Select at least one available build node.'); return }
    setBusy(true)
    try {
      const build = await api.images.createPatchBuild({ base_image: base.trim(), image: image.trim(), node_ids: nodeIds, files: files.map((file) => ({ target: file.target.trim(), content: file.content })) })
      known.current.set(build.id, build.status)
      setBuilds((current) => [build, ...current.filter((item) => item.id !== build.id)])
      setRetry((value) => value + 1)
      if (!active(build)) onBuilt()
      setOpen(false)
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not create patched image') }
    finally { setBusy(false) }
  }
  const tags = [...new Set(images.flatMap((item) => item.tags?.length ? item.tags : item.repository ? [`${item.repository}${item.tag ? `:${item.tag}` : ''}`] : []))].filter((tag) => !tag.includes('<none>'))

  return <section className="patched-images" aria-label="Patched images">
    <div className="section-heading"><div><h2>Patched images</h2><p>Build a new image with Python files added or replaced.</p></div><Button onClick={() => setOpen((value) => !value)}>{open ? 'Close patch editor' : 'Create patched image'}</Button></div>
    {open && <Panel className="patch-editor"><form onSubmit={(event) => void submit(event)}>
      <p>Only use patches you trust. Python code executes when the runtime loads it. The base image stays unchanged.</p>
      <div className="patch-fields">
        <label className="field"><span>Available base image</span><select value={tags.includes(base) ? base : ''} onChange={(event) => setBase(event.target.value)} disabled={busy}><option value="">Enter a registry reference below</option>{tags.map((tag) => <option key={tag} value={tag}>{tag}</option>)}</select></label>
        <label className="field"><span>Base image reference</span><input value={base} onChange={(event) => setBase(event.target.value)} disabled={busy} placeholder="registry.example/runtime:version" required /></label>
        <label className="field"><span>Patched image tag</span><input value={image} onChange={(event) => setImage(event.target.value)} disabled={busy} placeholder="sparkdeck/qwen-dflash:patch-v1" required /></label>
      </div>
      <NodeSelector nodes={nodes} selectedIds={nodeIds} onChange={setNodeIds} localLabel={localLabel} disabled={busy} legend="Build nodes" help="Build on every node that will run this image. Missing base images are pulled automatically." />
      <p>Upload a UTF-8 .py file or paste Python code. Up to 16 files, 1 MiB each and 4 MiB total.</p>
      {files.map((file, index) => <fieldset className="patch-file" key={file.id} disabled={busy || reading > 0}><legend>Patch file {index + 1}</legend>
        <label className="field"><span>Upload Python file {index + 1}</span><input type="file" accept=".py" onChange={(event) => { void upload(file.id, event.target.files?.[0]); event.target.value = '' }} /></label>
        <label className="field"><span>Container destination {index + 1}</span><input value={file.target} onChange={(event) => changeFile(file.id, { target: event.target.value })} placeholder="/usr/local/lib/python3.12/site-packages/package/module.py" required /></label>
        <label className="field"><span>Python contents {index + 1}</span><textarea className="mono" rows={8} value={file.content} onChange={(event) => changeFile(file.id, { content: event.target.value, error: undefined })} spellCheck={false} aria-describedby={`patch-empty-help-${file.id}`} /></label><small id={`patch-empty-help-${file.id}`}>May be empty, for example for a package’s __init__.py file.</small>
        {file.error && <p role="alert" className="inline-error">{file.error}</p>}
        {files.length > 1 && <Button type="button" onClick={() => setFiles((current) => current.filter((item) => item.id !== file.id))}>Remove patch file {index + 1}</Button>}
      </fieldset>)}
      <div className="patch-actions"><Button type="button" disabled={busy || reading > 0 || files.length >= 16} onClick={() => setFiles((current) => [...current, { id: nextId.current++, target: '', content: '' }])}>Add patch file</Button><Button type="submit" variant="primary" disabled={busy || reading > 0 || !nodes.length}>{busy ? 'Queueing build…' : reading ? 'Reading file…' : 'Build patched image'}</Button></div>
      {error && <p role="alert" className="inline-error">{error}</p>}
    </form></Panel>}
    {historyError && <div className="inline-error" role="alert">{historyError} <Button onClick={() => setRetry((value) => value + 1)}>Retry build history</Button></div>}
    {builds.length > 0 && <div className="patch-history"><h3>Build history</h3>{builds.map((build) => <Panel className="patch-build" key={build.id}>
      <h3>{build.image}</h3><p>From {build.base_image} · {new Date(build.created_at).toLocaleString()}</p>
      <p role="status">{build.status === 'succeeded' ? 'Succeeded — patch file hashes verified on all selected nodes.' : build.status === 'failed' ? 'Build failed. Check each node below.' : build.status === 'building' ? 'Building…' : 'Queued…'}</p>
      {build.error && <p className="inline-error">{build.error}</p>}
      {build.persistence_warning && <p role="alert" className="inline-error">{build.persistence_warning}. Keep this page open to see the current build result.</p>}
      {build.status === 'succeeded' && <p>In Models, create or edit a deployment, enter <code>{build.image}</code> as the container image, and select these build nodes.</p>}
      <details><summary>Patch files and SHA-256 hashes</summary>{build.files.map((file) => <p key={file.target}><code>{file.target}</code><br /><code>{file.sha256}</code></p>)}</details>
      {build.nodes.map((node) => <details key={node.node_id} open={node.status === 'failed'}><summary>{node.node_name} — {node.status}</summary>{node.error && <p className="inline-error">{node.error}</p>}{node.image_id && <p>Image: <code>{node.image_id}</code></p>}<pre aria-label={`Build logs for ${node.node_name}`}>{node.logs.join('\n') || 'Waiting for build output…'}</pre></details>)}
    </Panel>)}</div>}
  </section>
}
