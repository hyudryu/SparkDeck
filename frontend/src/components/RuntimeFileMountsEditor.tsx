import { Plus, Trash2 } from 'lucide-react'
import type { RuntimeFileMount } from '../api/types'
import { Button } from './ui'

export function RuntimeFileMountsEditor({ mounts, onChange, disabled = false }: {
  mounts: RuntimeFileMount[]
  onChange: (mounts: RuntimeFileMount[]) => void
  disabled?: boolean
}) {
  return <fieldset className="field runtime-file-mounts wide-field" disabled={disabled}>
    <legend>Runtime file mounts</legend>
    <small>Each host file must already exist at the same path on every selected node. Files are mounted read-only inside every vLLM container on the next run. Up to 16 files.</small>
    {mounts.map((mount, index) => <div className="field-grid" key={index}>
      <label className="field"><span>Host file path {index + 1}</span><input required placeholder="/home/user/patches/model.py" value={mount.source} onChange={(event) => onChange(mounts.map((item, position) => position === index ? { ...item, source: event.target.value } : item))} /></label>
      <label className="field"><span>Container file path {index + 1}</span><input required placeholder="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/model.py" value={mount.target} onChange={(event) => onChange(mounts.map((item, position) => position === index ? { ...item, target: event.target.value } : item))} /></label>
      <Button type="button" variant="tertiary" aria-label={`Remove runtime file mount ${index + 1}`} onClick={() => onChange(mounts.filter((_, position) => position !== index))}><Trash2 size={15} /> Remove file</Button>
    </div>)}
    <Button type="button" variant="tertiary" disabled={disabled || mounts.length >= 16} onClick={() => onChange([...mounts, { source: '', target: '' }])}><Plus size={15} /> Add runtime file</Button>
  </fieldset>
}
