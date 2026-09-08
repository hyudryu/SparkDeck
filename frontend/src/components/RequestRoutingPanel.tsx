import { useMemo, useState, type FormEvent } from 'react'
import { ArrowRight, Trash2 } from 'lucide-react'
import { api } from '../api/client'
import type { Deployment, InferenceRoutingRule } from '../api/types'
import { useResource } from '../hooks/useResource'
import { Button, LoadingState, Panel } from './ui'

interface InferenceRouteTarget {
  key: string
  deploymentId: string
  instanceId: number | null
  nodeIds: string[]
  label: string
  models: string[]
  available: boolean
}

const targetKey = (deploymentId: string, instanceId: number | null, nodeIds: string[]) =>
  JSON.stringify([deploymentId, instanceId, nodeIds])
const ROUTABLE_STATUSES = new Set(['running', 'ready'])

function servedModels(deployment: Deployment) {
  const candidates = deployment.served_models?.length
    ? [...deployment.served_models, deployment.alias]
    : [deployment.served_model, deployment.alias, deployment.model_id]
  return [...new Set(
    candidates
      .map((model) => model?.trim())
      .filter((model): model is string => Boolean(model)),
  )]
}

function targetAvailable(status: string, desiredState?: string) {
  return desiredState !== 'stopped' && ROUTABLE_STATUSES.has(status)
}

export function inferenceRouteTargets(deployments: Deployment[]): InferenceRouteTarget[] {
  return deployments.flatMap<InferenceRouteTarget>((deployment) => {
    if (!deployment.managed || !['vllm', 'sglang'].includes(deployment.runtime)) return []
    const models = servedModels(deployment)
    if (deployment.instances?.length) return deployment.instances.map((group) => {
      const nodeIds = group.node_ids ?? []
      return {
        key: targetKey(deployment.id, group.instance_id, nodeIds),
        deploymentId: deployment.id,
        instanceId: group.instance_id,
        nodeIds,
        label: `${deployment.alias} - Group ${group.instance_id + 1} - ${group.node_names.join(' + ')}`,
        models,
        available: deployment.desired_state !== 'stopped'
          && nodeIds.length > 0
          && targetAvailable(group.status, group.desired_state),
      }
    })

    const selectedNodes = deployment.selected_nodes ?? []
    if (deployment.deployment_mode === 'replicated' || deployment.replicas !== undefined) {
      const replicas = deployment.replicas?.length
        ? [...deployment.replicas].sort((left, right) => left.rank - right.rank)
        : selectedNodes.map((node, rank) => ({
            node_id: node.id, node_name: node.name || node.id, rank,
            status: 'unknown', desired_state: 'running' as const,
            online: false, available: false,
          }))
      return replicas.map((replica) => {
        const nodeIds = [replica.node_id]
        return {
          key: targetKey(deployment.id, null, nodeIds),
          deploymentId: deployment.id,
          instanceId: null,
          nodeIds,
          label: `${deployment.alias} - ${replica.node_name || replica.node_id}`,
          models,
          available: deployment.desired_state !== 'stopped' && replica.available === true,
        }
      })
    }

    const nodeIds = deployment.node_ids ?? selectedNodes.map((node) => node.id)
    const names = selectedNodes.map((node) => node.name || node.id)
    return [{
      key: targetKey(deployment.id, null, nodeIds),
      deploymentId: deployment.id,
      instanceId: null,
      nodeIds,
      label: `${deployment.alias}${names.length ? ` - ${names.join(' + ')}` : ''}`,
      models,
      available: nodeIds.length > 0 && targetAvailable(deployment.status, deployment.desired_state),
    }]
  })
}

const ruleKey = (rule: Pick<InferenceRoutingRule, 'source_ip' | 'requested_model'>) => `${rule.source_ip}\0${rule.requested_model}`
const sameNodes = (left: string[], right: string[]) => {
  return left.length === right.length && left.every((node, index) => node === right[index])
}

export function RequestRoutingPanel({ refreshGeneration = 0 }: { refreshGeneration?: number }) {
  const [refresh, setRefresh] = useState(0)
  const deployments = useResource(
    (signal) => api.deployments.list(signal), [refreshGeneration, refresh],
  )
  const rules = useResource(
    (signal) => api.inferenceRouting.list(signal), [refreshGeneration, refresh],
  )
  const [enabled, setEnabled] = useState(true)
  const [sourceIp, setSourceIp] = useState('')
  const [requestedModel, setRequestedModel] = useState('')
  const [selectedTarget, setSelectedTarget] = useState('')
  const [busy, setBusy] = useState<string>()
  const [error, setError] = useState<string>()
  const targets = useMemo(() => inferenceRouteTargets(deployments.data ?? []), [deployments.data])
  const modelNames = useMemo(() => [...new Set(targets.flatMap((target) => target.models))].sort(), [targets])
  const modelTargets = targets.filter((target) => target.models.includes(requestedModel))
  const selected = targets.find((target) => target.key === selectedTarget)

  const save = async (event: FormEvent) => {
    event.preventDefault()
    if (!sourceIp.trim() || !requestedModel || !selected?.available) return
    setBusy('save'); setError(undefined)
    try {
      await api.inferenceRouting.save({
        source_ip: sourceIp.trim(), requested_model: requestedModel, enabled,
        deployment_id: selected.deploymentId, instance_id: selected.instanceId,
        node_ids: selected.nodeIds,
      })
      setEnabled(true); setSourceIp(''); setRequestedModel(''); setSelectedTarget('')
      rules.reload()
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not save the request routing rule') } finally { setBusy(undefined) }
  }

  const toggle = async (rule: InferenceRoutingRule, next: boolean) => {
    setBusy(ruleKey(rule)); setError(undefined)
    try {
      await api.inferenceRouting.save({ ...rule, enabled: next })
      rules.reload()
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not update the request routing rule') } finally { setBusy(undefined) }
  }

  const remove = async (rule: InferenceRoutingRule) => {
    setBusy(ruleKey(rule)); setError(undefined)
    try {
      await api.inferenceRouting.remove(rule.source_ip, rule.requested_model)
      rules.reload()
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not remove the request routing rule') } finally { setBusy(undefined) }
  }

  return <Panel className="usage-routing-panel request-routing-panel" aria-labelledby="request-routing-title"><div className="usage-panel-heading"><div><h2 id="request-routing-title">IP routing rules</h2><p>Send an exact client IP and requested model to one deployment group.</p><p>Enabled rules stay on the selected group; unavailable targets return an error. Disabled rules use normal routing.</p></div><Button type="button" onClick={() => setRefresh((value) => value + 1)}>Refresh IP routing rules</Button></div>
    {(deployments.error || rules.error) && <div className="request-routing-load-error"><span>{deployments.error || rules.error}</span><Button type="button" variant="tertiary" onClick={() => { deployments.reload(); rules.reload() }}>Retry request routing</Button></div>}
    <form className="request-routing-form" aria-label="Add request routing rule" onSubmit={(event) => void save(event)}>
      <label className="field request-routing-toggle"><span>Enabled</span><span className="request-toggle-control"><span>Off</span><input type="checkbox" role="switch" aria-label="Enabled" checked={enabled} onChange={(event) => setEnabled(event.target.checked)} /><span className="request-toggle-track" aria-hidden="true"><span /></span><span>On</span></span></label>
      <label className="field"><span>Source IP</span><input required maxLength={45} value={sourceIp} onChange={(event) => { setSourceIp(event.target.value); setError(undefined) }} placeholder="100.100.20.40" /></label>
      <label className="field"><span>Requested model</span><select value={requestedModel} onChange={(event) => { setRequestedModel(event.target.value); setSelectedTarget(''); setError(undefined) }}><option value="">Select a model</option>{modelNames.map((model) => <option key={model} value={model}>{model}</option>)}</select></label>
      <label className="field"><span>Target deployment group</span><select value={selectedTarget} disabled={!requestedModel} onChange={(event) => { setSelectedTarget(event.target.value); setError(undefined) }}><option value="">Select a deployment group</option>{modelTargets.map((target) => <option key={target.key} value={target.key} disabled={!target.available}>{target.label}{target.available ? '' : ' - Unavailable'}</option>)}</select></label>
      <Button type="submit" variant="primary" disabled={!sourceIp.trim() || !requestedModel || !selected?.available || busy === 'save'}>{busy === 'save' ? 'Saving...' : 'Add request rule'}</Button>
    </form>
    {error && <p className="inline-error usage-routing-error" role="alert">{error}</p>}
    {rules.loading && !rules.data ? <LoadingState label="Loading request routing rules" /> : rules.data?.length ? <ul className="request-routing-list" aria-label="Current request routing rules">{rules.data.map((rule) => {
      const target = targets.find((item) => item.deploymentId === rule.deployment_id
        && item.instanceId === rule.instance_id
        && item.models.includes(rule.requested_model)
        && sameNodes(item.nodeIds, rule.node_ids))
      const fallback = `${rule.deployment_id}${rule.instance_id === null ? '' : ` - Group ${rule.instance_id + 1}`}`
      const unavailable = !target || !target.available
      const key = ruleKey(rule)
      return <li key={key}><label className="request-rule-toggle"><input type="checkbox" role="switch" aria-label={`Enable request routing for ${rule.source_ip} and ${rule.requested_model}`} checked={rule.enabled} disabled={busy === key} onChange={(event) => void toggle(rule, event.target.checked)} /><span className="request-toggle-track" aria-hidden="true"><span /></span><span>{rule.enabled ? 'Enabled' : 'Disabled'}</span></label><div><strong>{rule.source_ip}</strong><code>{rule.requested_model}</code></div><ArrowRight size={14} aria-hidden="true" /><div><strong>{target?.label ?? fallback}</strong>{unavailable && <small>Unavailable</small>}</div><Button type="button" variant="tertiary" aria-label={`Remove request routing for ${rule.source_ip} and ${rule.requested_model}`} disabled={busy === key} onClick={() => void remove(rule)}><Trash2 size={14} /> Remove</Button></li>
    })}</ul> : <p className="usage-routing-empty">No request routing rules yet.</p>}
  </Panel>
}
