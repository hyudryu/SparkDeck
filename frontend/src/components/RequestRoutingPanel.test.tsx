import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import type { Deployment, InferenceRoutingRule } from '../api/types'
import { RequestRoutingPanel, inferenceRouteTargets } from './RequestRoutingPanel'

const groupedDeployment: Deployment = {
  id: 'deployment-a', alias: 'PRODUCTION DeepSeek', model_id: 'deepseek/repo',
  served_models: ['shared-model'], runtime: 'vllm', status: 'running', managed: true,
  settings: {}, deployment_mode: 'grouped_sharded', desired_state: 'running',
  instances: [
    { instance_id: 0, status: 'running', desired_state: 'running', node_ids: ['node-1', 'node-2'], node_names: ['Node 1', 'Node 2'] },
    { instance_id: 1, status: 'running', desired_state: 'running', node_ids: ['node-3', 'node-4'], node_names: ['Node 3', 'Node 4'] },
  ],
}

beforeEach(() => {
  vi.spyOn(api.deployments, 'list').mockResolvedValue([groupedDeployment])
  vi.spyOn(api.inferenceRouting, 'list').mockResolvedValue([])
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('RequestRoutingPanel', () => {
  it('distinguishes duplicate served-model groups and saves the selected node fingerprint', async () => {
    const user = userEvent.setup()
    const save = vi.spyOn(api.inferenceRouting, 'save').mockResolvedValue({} as InferenceRoutingRule)
    render(<RequestRoutingPanel />)

    await user.selectOptions(await screen.findByLabelText('Requested model'), 'shared-model')
    const targetSelect = screen.getByLabelText('Target deployment group')
    expect(within(targetSelect).getByRole('option', { name: 'PRODUCTION DeepSeek - Group 1 - Node 1 + Node 2' })).toBeInTheDocument()
    expect(within(targetSelect).getByRole('option', { name: 'PRODUCTION DeepSeek - Group 2 - Node 3 + Node 4' })).toBeInTheDocument()
    await user.selectOptions(targetSelect, within(targetSelect).getByRole('option', { name: /Group 2/ }))
    await user.type(screen.getByLabelText('Source IP'), '100.100.20.40')
    expect(screen.getByRole('switch', { name: 'Enabled' })).toBeChecked()
    await user.click(screen.getByRole('button', { name: 'Add request rule' }))

    await waitFor(() => expect(save).toHaveBeenCalledWith({
      source_ip: '100.100.20.40', requested_model: 'shared-model', enabled: true,
      deployment_id: 'deployment-a', instance_id: 1, node_ids: ['node-3', 'node-4'],
    }))
  })

  it('offers one target per node for replicated deployments', () => {
    const targets = inferenceRouteTargets([{
      ...groupedDeployment, id: 'replicas', alias: 'Replica pool', deployment_mode: 'replicated', instances: undefined,
      selected_nodes: [
        { id: 'node-1', name: 'Node 1' },
        { id: 'node-2', name: 'Node 2' },
      ],
    }])

    expect(targets.map(({ label, nodeIds }) => ({ label, nodeIds }))).toEqual([
      { label: 'Replica pool - Node 1', nodeIds: ['node-1'] },
      { label: 'Replica pool - Node 2', nodeIds: ['node-2'] },
    ])
    expect(targets[0].key).not.toBe(targets[1].key)
  })

  it.each(['replicated', undefined] as const)('uses per-replica health when a degraded deployment has mode %s', (deployment_mode) => {
    const targets = inferenceRouteTargets([{
      ...groupedDeployment, status: 'degraded', deployment_mode, instances: undefined,
      replicas: [
        { node_id: 'node-1', node_name: 'Node 1', rank: 0, status: 'running', desired_state: 'running', online: true, available: true },
        { node_id: 'node-2', node_name: 'Node 2', rank: 1, status: 'stopped', desired_state: 'stopped', online: false, available: false },
      ],
    }])

    expect(targets.map(({ nodeIds, available }) => ({ nodeIds, available }))).toEqual([
      { nodeIds: ['node-1'], available: true },
      { nodeIds: ['node-2'], available: false },
    ])
  })

  it('rejects a degraded indivisible sharded deployment', () => {
    const [target] = inferenceRouteTargets([{
      ...groupedDeployment,
      status: 'degraded',
      deployment_mode: 'sharded',
      instances: undefined,
      node_ids: ['node-1', 'node-2'],
      selected_nodes: [
        { id: 'node-1', name: 'Node 1' },
        { id: 'node-2', name: 'Node 2' },
      ],
    }])

    expect(target.available).toBe(false)
  })

  it('offers the deployment alias and recognizes an API-saved alias rule', async () => {
    vi.mocked(api.inferenceRouting.list).mockResolvedValue([{
      source_ip: '10.0.0.10', requested_model: 'PRODUCTION DeepSeek', enabled: true,
      deployment_id: 'deployment-a', instance_id: 0, node_ids: ['node-1', 'node-2'],
    }])
    render(<RequestRoutingPanel />)

    const modelSelect = await screen.findByLabelText('Requested model')
    expect(within(modelSelect).getByRole('option', { name: 'shared-model' })).toBeInTheDocument()
    expect(within(modelSelect).getByRole('option', { name: 'PRODUCTION DeepSeek' })).toBeInTheDocument()
    expect(await screen.findByText('10.0.0.10')).toBeInTheDocument()
    expect(screen.queryByText('Unavailable')).not.toBeInTheDocument()
  })

  it('keeps a stale rule visible and allows disabling and removing it', async () => {
    const user = userEvent.setup()
    let rules: InferenceRoutingRule[] = [{
      source_ip: '10.0.0.7', requested_model: 'shared-model', enabled: true,
      deployment_id: 'deployment-a', instance_id: 0, node_ids: ['old-node-1', 'old-node-2'],
    }]
    vi.mocked(api.inferenceRouting.list).mockImplementation(async () => rules)
    const save = vi.spyOn(api.inferenceRouting, 'save').mockImplementation(async (rule) => {
      rules = [rule]
      return rule
    })
    const remove = vi.spyOn(api.inferenceRouting, 'remove').mockImplementation(async () => {
      rules = []
      return undefined
    })
    render(<RequestRoutingPanel />)

    expect(await screen.findByText('Unavailable')).toBeInTheDocument()
    await user.click(screen.getByRole('switch', { name: 'Enable request routing for 10.0.0.7 and shared-model' }))
    await waitFor(() => expect(save).toHaveBeenCalledWith({ ...rules[0], enabled: false }))
    await user.click(await screen.findByRole('button', { name: 'Remove request routing for 10.0.0.7 and shared-model' }))
    expect(remove).toHaveBeenCalledWith('10.0.0.7', 'shared-model')
  })

  it('treats a changed rank order as an unavailable target', async () => {
    vi.mocked(api.inferenceRouting.list).mockResolvedValue([{
      source_ip: '10.0.0.9', requested_model: 'shared-model', enabled: true,
      deployment_id: 'deployment-a', instance_id: 0, node_ids: ['node-2', 'node-1'],
    }])
    render(<RequestRoutingPanel />)

    expect(await screen.findByText('Unavailable')).toBeInTheDocument()
  })

  it('shows backend validation inline while retaining the editable form', async () => {
    const user = userEvent.setup()
    vi.spyOn(api.inferenceRouting, 'save').mockRejectedValue(new Error('Source IP must be a valid IPv4 or IPv6 address'))
    render(<RequestRoutingPanel />)

    await user.selectOptions(await screen.findByLabelText('Requested model'), 'shared-model')
    await user.selectOptions(screen.getByLabelText('Target deployment group'), screen.getByRole('option', { name: /Group 1/ }))
    await user.type(screen.getByLabelText('Source IP'), 'not-an-ip')
    await user.click(screen.getByRole('button', { name: 'Add request rule' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Source IP must be a valid IPv4 or IPv6 address')
    expect(screen.getByLabelText('Source IP')).toHaveValue('not-an-ip')
    expect(screen.getByLabelText('Requested model')).toHaveValue('shared-model')
  })

  it('keeps available rule data visible when deployment loading fails', async () => {
    vi.mocked(api.deployments.list).mockRejectedValue(new Error('Could not load deployments'))
    vi.mocked(api.inferenceRouting.list).mockResolvedValue([{
      source_ip: '10.0.0.8', requested_model: 'shared-model', enabled: false,
      deployment_id: 'missing', instance_id: null, node_ids: ['node-x'],
    }])
    render(<RequestRoutingPanel />)

    expect(await screen.findByText('Could not load deployments')).toBeInTheDocument()
    expect(screen.getByText('10.0.0.8')).toBeInTheDocument()
    expect(screen.getByText('Unavailable')).toBeInTheDocument()
    expect(screen.getByLabelText('Add request routing rule')).toBeInTheDocument()
  })
})
