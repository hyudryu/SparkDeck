import type { Deployment, DeploymentInstance } from '../api/types'

const ACTIVE = new Set(['running', 'ready', 'starting', 'launching', 'recovering', 'stopping', 'degraded', 'error', 'unknown'])

export function groupNodeIds(deployment: Deployment, group: DeploymentInstance): string[] {
  if (group.node_ids?.length) return group.node_ids
  const count = deployment.instance_node_count ?? deployment.settings.tensor_parallel_size ?? group.node_names.length
  return (deployment.node_ids ?? deployment.selected_nodes?.map((node) => node.id) ?? []).slice(group.instance_id * count, (group.instance_id + 1) * count)
}

export function occupiedNodeReasons(deployments: Deployment[], exceptId?: string): Record<string, string> {
  const reasons: Record<string, string> = {}
  for (const deployment of deployments) {
    if (deployment.id === exceptId || deployment.status === 'saved') continue
    if (!ACTIVE.has(deployment.status) && !(deployment.desired_state === 'running' && deployment.instances?.some((group) => ACTIVE.has(group.status)))) continue
    const ids = deployment.instances?.length
      ? deployment.instances.filter((group) => ACTIVE.has(group.status)).flatMap((group) => groupNodeIds(deployment, group))
      : deployment.node_ids?.length ? deployment.node_ids
        : deployment.selected_nodes?.length ? deployment.selected_nodes.map((node) => node.id)
          : deployment.managed ? ['local'] : []
    for (const id of ids) reasons[id] = `Already used by deployment ${deployment.alias}`
  }
  return reasons
}
