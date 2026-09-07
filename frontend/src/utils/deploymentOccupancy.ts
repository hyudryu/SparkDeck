import type { Deployment, DeploymentInstance } from '../api/types'

const ACTIVE = new Set(['running', 'ready', 'starting', 'launching', 'recovering', 'stopping', 'degraded', 'error', 'unknown'])

function requestedNodeIds(deployment: Deployment): string[] {
  if (deployment.node_ids?.length) return deployment.node_ids
  if (deployment.selected_nodes?.length) return deployment.selected_nodes.map((node) => node.id)
  return deployment.settings.node_ids ?? []
}

export function groupNodeIds(deployment: Deployment, group: DeploymentInstance): string[] {
  if (group.node_ids?.length) return group.node_ids
  const count = deployment.instance_node_count ?? deployment.settings.tensor_parallel_size ?? group.node_names.length
  return requestedNodeIds(deployment).slice(group.instance_id * count, (group.instance_id + 1) * count)
}

export function occupiedNodeReasons(deployments: Deployment[], exceptId?: string): Record<string, string> {
  const reasons: Record<string, string> = {}
  for (const deployment of deployments) {
    if (deployment.id === exceptId || deployment.status === 'saved') continue
    // Aggregate health can be degraded solely because unrelated peers are
    // unreachable. Prefer the controller's per-member runtime reservations.
    if (deployment.occupied_node_ids !== undefined) {
      for (const id of deployment.occupied_node_ids) reasons[id] = `Already used by deployment ${deployment.alias}`
      continue
    }
    const groupOccupied = (group: DeploymentInstance) => ACTIVE.has(group.status)
      || (deployment.desired_state !== 'stopped' && group.desired_state === 'running')
    if (!ACTIVE.has(deployment.status) && !(deployment.desired_state === 'running' && deployment.instances?.some(groupOccupied))) continue
    const requested = requestedNodeIds(deployment)
    const ids = deployment.instances?.length
      ? deployment.instances.filter(groupOccupied).flatMap((group) => groupNodeIds(deployment, group))
      : requested.length ? requested
        : deployment.managed || deployment.id.startsWith('container:') ? ['local'] : []
    for (const id of ids) reasons[id] = `Already used by deployment ${deployment.alias}`
  }
  return reasons
}
