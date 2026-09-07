import { describe, expect, it } from 'vitest'
import type { Deployment } from '../api/types'
import { occupiedNodeReasons } from './deploymentOccupancy'

const split = {
  id: 'split', alias: 'Production', managed: true, status: 'running', settings: { tensor_parallel_size: 2 }, node_ids: ['n1', 'n2', 'n3', 'n4'],
  instances: [
    { instance_id: 0, status: 'starting', desired_state: 'running', node_names: ['One', 'Two'], node_ids: ['n1', 'n2'] },
    { instance_id: 1, status: 'stopped', desired_state: 'stopped', node_names: ['Three', 'Four'], node_ids: ['n3', 'n4'] },
  ],
} as Deployment

describe('deployment node occupancy', () => {
  it('uses observed reservations rather than assigning degraded health to every node', () => {
    const offlinePeers = { ...split, status: 'degraded' as const, desired_state: 'stopped' as const, occupied_node_ids: ['n3', 'n4'] }
    expect(Object.keys(occupiedNodeReasons([offlinePeers]))).toEqual(['n3', 'n4'])
    expect(occupiedNodeReasons([{ ...offlinePeers, occupied_node_ids: [] }])).toEqual({})
    expect(occupiedNodeReasons([offlinePeers], split.id)).toEqual({})
  })
  it('preserves actual runtime reservations despite a stopped aggregate status', () => {
    expect(Object.keys(occupiedNodeReasons([{ ...split, status: 'stopped', desired_state: 'stopped', occupied_node_ids: ['n1'] }]))).toEqual(['n1'])
  })
  it('reserves persisted remote topology while a saved launch awaits Manager linkage', () => {
    for (const node_ids of [undefined, []]) {
      const launching = { ...split, status: 'starting' as const, instances: undefined, node_ids, settings: { node_ids: ['remote-1', 'remote-2'] } }
      expect(Object.keys(occupiedNodeReasons([launching]))).toEqual(['remote-1', 'remote-2'])
      expect(Object.keys(occupiedNodeReasons([{ ...launching, node_ids: ['current-node'] }]))).toEqual(['current-node'])
    }
  })
  it('reserves only active groups and keeps saved inactive peers free', () => {
    expect(Object.keys(occupiedNodeReasons([split]))).toEqual(['n1', 'n2'])
    expect(occupiedNodeReasons([{ ...split, status: 'saved' }])).toEqual({})
    expect(occupiedNodeReasons([split], 'split')).toEqual({})
  })
  it('holds stopping and failed group nodes even with stopped intent', () => {
    for (const status of ['stopping', 'error']) {
      expect(Object.keys(occupiedNodeReasons([{ ...split, instances: [{ ...split.instances![0], status, desired_state: 'stopped' }] }]))).toEqual(['n1', 'n2'])
    }
  })
  it('reserves stopped groups awaiting recovery but frees explicitly stopped groups', () => {
    for (const status of ['degraded', 'stopped'] as const) {
      const recovering = { ...split, status, desired_state: 'running' as const, instances: [
        { ...split.instances![0], status: 'stopped' }, split.instances![1],
      ] }
      expect(Object.keys(occupiedNodeReasons([recovering]))).toEqual(['n1', 'n2'])
      expect(occupiedNodeReasons([{ ...recovering, desired_state: 'stopped' }])).toEqual({})
    }
  })
  it('reserves local discovered containers without reserving remote external endpoints', () => {
    const external = { ...split, managed: false, instances: undefined, node_ids: undefined }
    expect(Object.keys(occupiedNodeReasons([{ ...external, id: 'container:vllm' }]))).toEqual(['local'])
    expect(occupiedNodeReasons([{ ...external, id: 'container:vllm', status: 'stopped' }])).toEqual({})
    expect(occupiedNodeReasons([{ ...external, id: 'remote-endpoint' }])).toEqual({})
  })
  it('maps legacy group topology and defaults only managed standalone to local', () => {
    expect(Object.keys(occupiedNodeReasons([{ ...split, instances: split.instances!.map((group) => ({ ...group, node_ids: undefined })) }]))).toEqual(['n1', 'n2'])
    expect(Object.keys(occupiedNodeReasons([{ ...split, instances: undefined, node_ids: undefined }]))).toEqual(['local'])
    expect(occupiedNodeReasons([{ ...split, managed: false, instances: undefined, node_ids: undefined }])).toEqual({})
  })
})
