import { HardDrive, RotateCw } from 'lucide-react'
import type { NodeInventoryItem } from '../api/types'
import { Button } from './ui'

function nodeLabel(node: NodeInventoryItem | undefined, id: string, localLabel?: string) {
  if (id === 'local' && localLabel) return localLabel
  return node?.name ?? (id === 'local' ? 'This device' : id)
}

export function selectedNodeLabel(nodes: NodeInventoryItem[], selectedIds: string[], localLabel?: string) {
  const names = selectedIds.map((id) => nodeLabel(nodes.find((node) => node.id === id), id, localLabel))
  if (names.length === 0) return 'No target selected'
  return names.join(', ')
}

export function isNodeSelectable(node: NodeInventoryItem) {
  return node.selectable !== false && node.online !== false && node.docker_ready !== false
}

export function NodeSelector({
  nodes,
  selectedIds,
  onChange,
  loading = false,
  error,
  onRetry,
  multiple = true,
  disabled = false,
  requiredIds = [],
  allowedIds,
  unavailableReasons,
  localLabel,
  primaryId,
  legend = 'Target nodes',
  help = 'This device is the default. Select more nodes to pull or deploy across the cluster.',
}: {
  nodes: NodeInventoryItem[]
  selectedIds: string[]
  onChange: (ids: string[]) => void
  loading?: boolean
  error?: string
  onRetry?: () => void
  multiple?: boolean
  disabled?: boolean
  requiredIds?: string[]
  allowedIds?: string[]
  unavailableReasons?: Record<string, string>
  localLabel?: string
  primaryId?: string
  legend?: string
  help?: string
}) {
  const toggle = (nodeId: string) => {
    if (!multiple) {
      onChange([nodeId])
      return
    }
    onChange(selectedIds.includes(nodeId)
      ? selectedIds.filter((id) => id !== nodeId || requiredIds.includes(id))
      : [...selectedIds, nodeId])
  }

  return (
    <fieldset className="node-selector" disabled={disabled}>
      <legend>{legend}</legend>
      <p className="node-selector-help">{help}</p>
      {loading && <p className="node-selector-state" role="status">Loading available nodes…</p>}
      {error && <div className="node-selector-state node-selector-error" role="alert"><span>Couldn’t load nodes.</span>{onRetry && <Button type="button" variant="tertiary" onClick={onRetry}><RotateCw size={14} /> Retry</Button>}</div>}
      {!loading && !error && (
        <div className="node-options">
          {nodes.length === 0 && <p className="node-selector-state">No nodes are available.</p>}
          {nodes.map((node) => {
            const allowed = !allowedIds || allowedIds.includes(node.id)
            const ready = isNodeSelectable(node) && allowed
            const status = !allowed ? unavailableReasons?.[node.id] ?? 'Not available for this runtime' : node.online === false ? 'Offline' : node.docker_ready === false ? node.status_message ?? 'Docker unavailable' : node.selectable === false ? node.status_message ?? 'Unavailable' : 'Ready'
            const required = requiredIds.includes(node.id)
            const displayName = nodeLabel(node, node.id, localLabel)
            const primary = primaryId === node.id
            return (
              <label className={`node-option${selectedIds.includes(node.id) ? ' selected' : ''}${!ready ? ' unavailable' : ''}`} key={node.id}>
                <input
                  type={multiple ? 'checkbox' : 'radio'}
                  name={multiple ? undefined : 'target-node'}
                  checked={selectedIds.includes(node.id)}
                  disabled={disabled || !ready || (required && selectedIds.includes(node.id))}
                  onChange={() => toggle(node.id)}
                />
                <HardDrive size={17} aria-hidden="true" />
                <span><strong>{displayName}</strong><small>{displayName !== node.name ? `${node.name} · ` : node.local ? 'This device · ' : `${node.id} · `}{status}{required ? ' · Required' : ''}{primary ? ' · Primary' : ''}</small></span>
              </label>
            )
          })}
        </div>
      )}
      <p className="node-selection-summary" aria-live="polite"><strong>{selectedIds.length === 1 ? 'Target' : 'Targets'}:</strong> {selectedNodeLabel(nodes, selectedIds, localLabel)}{primaryId && selectedIds.length > 1 ? ` · Primary: ${selectedNodeLabel(nodes, [primaryId], localLabel)}` : ''}</p>
    </fieldset>
  )
}
