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
  it('maps legacy group topology and defaults only managed standalone to local', () => {
    expect(Object.keys(occupiedNodeReasons([{ ...split, instances: split.instances!.map((group) => ({ ...group, node_ids: undefined })) }]))).toEqual(['n1', 'n2'])
    expect(Object.keys(occupiedNodeReasons([{ ...split, instances: undefined, node_ids: undefined }]))).toEqual(['local'])
    expect(occupiedNodeReasons([{ ...split, managed: false, instances: undefined, node_ids: undefined }])).toEqual({})
  })
})
