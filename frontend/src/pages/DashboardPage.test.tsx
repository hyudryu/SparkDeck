import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { Deployment, NodeInventoryItem, SystemStats } from '../api/types'
import { nodeResourceSnapshot, DashboardPage, inferenceSessionSnapshot, sharedModelGroups } from './DashboardPage'

function json(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
}

class MockWebSocket {
  static instances: MockWebSocket[] = []
  static autoOpen = true
  readonly url: string
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  onclose: (() => void) | null = null
  private closed = false

  constructor(url: string) {
    this.url = url
    MockWebSocket.instances.push(this)
    if (MockWebSocket.autoOpen) {
      queueMicrotask(() => { if (!this.closed) this.onopen?.() })
    }
  }

  close() {
    if (this.closed) return
    this.closed = true
    this.onclose?.()
  }

  emit(snapshot: unknown) {
    this.onmessage?.({ data: JSON.stringify(snapshot) })
  }
}

function stubDashboardFetch(stats: Record<string, unknown>) {
  return vi.fn<typeof fetch>().mockImplementation(async (input) => {
    const path = String(input)
    if (path.includes('/api/stats')) return json(stats)
    if (path.includes('/api/inference-queue')) return json({})
    if (path.includes('/api/v1/deployments')) return json({ items: [] })
    if (path.includes('/api/v1/community/sync')) return json({ consent: false, outbox: {} })
    if (path.includes('/api/v1/onboarding')) return json({
      role: 'controller',
      node: { id: 'local', name: 'This node', port: 7878, access_urls: [] },
      controller_reachable: true,
    })
    return json({ items: [] })
  })
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers() })

describe('DashboardPage', () => {
  it('labels an indexless GPU discovery error without inventing a card identity', async () => {
    const stats = { cpu_pct: 20, gpus: [{ error: 'nvidia-smi unavailable' }], active_requests: {} } satisfies SystemStats
    vi.stubGlobal('fetch', stubDashboardFetch(stats))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    const discovery = await screen.findByRole('region', { name: 'Local node GPU telemetry' })
    expect(within(discovery).getByText('GPU telemetry', { exact: true })).toBeInTheDocument()
    expect(within(discovery).getByText('GPU telemetry unavailable: nvidia-smi unavailable')).toBeInTheDocument()
    expect(within(discovery).queryByRole('progressbar')).not.toBeInTheDocument()
    expect(screen.queryByText(/GPU undefined/)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(/GPU undefined/)).not.toBeInTheDocument()
  })

  it('shows each group per-stage prompt/output/thinking rates without merging separate thinking into output', async () => {
    vi.stubGlobal('fetch', stubDashboardFetch({ active_request_groups: {
      first: { group_id: 'first', instance_id: 0, model: 'shared', node_names: ['Node 1', 'Node 2'], connections: 2, pp_tok_s: 1200, output_tok_s: 40, thinking_tok_s: 10 },
      second: { group_id: 'second', instance_id: 1, model: 'shared', node_names: ['Node 3', 'Node 4'], connections: 1, pp_tok_s: null, output_tok_s: 70, thinking_tok_s: 0 },
    } }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    const first = within((await screen.findByText('Group 1 · Node 1 + Node 2')).closest('.session-row') as HTMLElement)
    expect(first.getByText('Prompt processing')).toBeInTheDocument()
    expect(first.getByText('1200.0 tok/s')).toBeInTheDocument()
    expect(first.getByText('40.0 tok/s')).toBeInTheDocument()
    expect(first.getByText('Thinking')).toBeInTheDocument()
    expect(first.getByText('10.0 tok/s')).toBeInTheDocument()
    expect(first.queryByText('50.0 tok/s')).not.toBeInTheDocument()
    const second = within(screen.getByText('Group 2 · Node 3 + Node 4').closest('.session-row') as HTMLElement)
    expect(second.getByText('Unavailable')).toBeInTheDocument()
    expect(second.getAllByText('Measuring…')).toHaveLength(1)
    expect(second.getByText('70.0 tok/s')).toBeInTheDocument()
    expect(second.getByText('Thinking')).toBeInTheDocument()
  })

  it('reports how many sessions are outputting, thinking, and prompt processing', async () => {
    vi.stubGlobal('fetch', stubDashboardFetch({ active_request_groups: {
      first: {
        group_id: 'first', instance_id: 0, model: 'shared', node_names: ['Node 1'],
        connections: 4, pp_tok_s: 1200, output_tok_s: 40, thinking_tok_s: 10,
        output_sessions: 2, thinking_sessions: 1, prefill_sessions: 1, prefill_seconds: 7.4,
      },
      second: {
        group_id: 'second', instance_id: 1, model: 'shared', node_names: ['Node 2'],
        connections: 1, pp_tok_s: null, output_tok_s: 70, thinking_tok_s: 0,
        output_sessions: 1, thinking_sessions: 0, prefill_sessions: 0,
      },
    } }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    // Panel total across both groups: 3 outputting, 1 thinking, 1 prompt processing.
    expect(await screen.findByText('5 active · 3 outputting · 1 thinking · 1 prompt processing · 0 queued')).toBeInTheDocument()
    const firstRow = screen.getByText('4 active · 0 queued').closest('.session-row')
    expect(firstRow).not.toBeNull()
    expect(within(firstRow as HTMLElement).getByText('2 outputting · 1 thinking · 1 prompt processing')).toBeInTheDocument()
    const secondRow = screen.getByText('1 active · 0 queued').closest('.session-row')
    expect(within(secondRow as HTMLElement).getByText('1 outputting')).toBeInTheDocument()
    expect(within(secondRow as HTMLElement).queryByText(/thinking/)).not.toBeInTheDocument()
  })

  it('shows prompt processing as a held prefill instead of an unmeasurable rate', async () => {
    vi.stubGlobal('fetch', stubDashboardFetch({ active_request_groups: {
      first: {
        group_id: 'first', instance_id: 0, model: 'shared', node_names: ['Node 1'],
        connections: 2, pp_tok_s: 900, output_tok_s: 40, thinking_tok_s: 0,
        output_sessions: 1, thinking_sessions: 0, prefill_sessions: 1, prefill_seconds: 12.6,
      },
    } }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    const rates = within((await screen.findByText('Group 1 · Node 1')).closest('.session-row') as HTMLElement)
    // The prefill has no first token yet, so no tok/s is invented for it.
    expect(rates.getByText('Prefilling 13s')).toBeInTheDocument()
    expect(rates.queryByText('900.0 tok/s')).not.toBeInTheDocument()
    expect(rates.getByText('40.0 tok/s')).toBeInTheDocument()
  })

  it('falls back to the measured prompt rate once the prefill completes', async () => {
    vi.stubGlobal('fetch', stubDashboardFetch({ active_request_groups: {
      first: {
        group_id: 'first', instance_id: 0, model: 'shared', node_names: ['Node 1'],
        connections: 1, pp_tok_s: 1200, output_tok_s: 40, thinking_tok_s: 0,
        output_sessions: 1, thinking_sessions: 0, prefill_sessions: 0, prefill_seconds: null,
      },
    } }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    const rates = within((await screen.findByText('Group 1 · Node 1')).closest('.session-row') as HTMLElement)
    expect(rates.getByText('1200.0 tok/s')).toBeInTheDocument()
    expect(rates.queryByText(/Prefilling/)).not.toBeInTheDocument()
  })

  it('shows runtime prompt throughput during prefill and retains it while thinking', async () => {
    MockWebSocket.instances = []
    MockWebSocket.autoOpen = true
    vi.stubGlobal('WebSocket', MockWebSocket)
    const request = {
      group_id: 'first', instance_id: 0, model: 'shared', node_names: ['Node 1'],
      connections: 1, pp_tok_s: null as number | null, output_tok_s: 0, thinking_tok_s: 0,
      prefill_sessions: 1, prefill_seconds: 3, thinking_sessions: 0,
    }
    vi.stubGlobal('fetch', stubDashboardFetch({ active_request_groups: { first: request } }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    const row = (await screen.findByText('Group 1 · Node 1')).closest('.session-row') as HTMLElement
    expect(within(row).getByText('Prefilling 3s')).toBeInTheDocument()
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    await act(async () => MockWebSocket.instances[0].emit({ type: 'snapshot', stats: { active_request_groups: {
      first: { ...request, pp_tok_s: 800, pp_rate_source: 'runtime_ttft', pp_sample_seconds: 1 },
    } } }))
    expect(within(row).getByText('800.0 tok/s (est.)')).toBeInTheDocument()
    expect(within(row).queryByText(/Prefilling/)).not.toBeInTheDocument()
    await act(async () => MockWebSocket.instances[0].emit({ type: 'snapshot', stats: { active_request_groups: {
      first: { ...request, pp_tok_s: 800, pp_rate_source: 'runtime_ttft', pp_sample_seconds: 1,
        prefill_sessions: 0, prefill_seconds: null, thinking_sessions: 1, thinking_tok_s: 20 },
    } } }))
    expect(within(row).getByText('800.0 tok/s (est.)')).toBeInTheDocument()
    expect(within(row).getByText('20.0 tok/s')).toBeInTheDocument()
    expect(within(row).queryByText('Unavailable')).not.toBeInTheDocument()
  })

  it('ends prompt measurement when generation has begun without usable telemetry', async () => {
    vi.stubGlobal('fetch', stubDashboardFetch({ active_request_groups: {
      first: { group_id: 'first', instance_id: 0, model: 'shared', node_names: ['Node 1'],
        connections: 1, pp_tok_s: null, thinking_tok_s: 30, prefill_sessions: 0, thinking_sessions: 1 },
    } }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    const row = (await screen.findByText('Group 1 · Node 1')).closest('.session-row') as HTMLElement
    const promptStage = within(row).getByText('Prompt processing').parentElement as HTMLElement
    expect(promptStage).toHaveTextContent('Unavailable')
    expect(promptStage).not.toHaveTextContent('Measuring')
  })

  it('omits state counts for entries from an older telemetry payload', async () => {
    vi.stubGlobal('fetch', stubDashboardFetch({ active_request_groups: {
      first: { group_id: 'first', instance_id: 0, model: 'shared', node_names: ['Node 1'], connections: 1, pp_tok_s: 500, output_tok_s: 20, thinking_tok_s: 0 },
    } }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    expect(await screen.findByRole('heading', { name: 'Current inference' })).toBeInTheDocument()
    const rates = within(screen.getByText('Group 1 · Node 1').closest('.session-row') as HTMLElement)
    expect(rates.getByText('500.0 tok/s')).toBeInTheDocument()
    // No session-state payload means no state line is claimed at all.
    expect(document.querySelector('.session-states')).toBeNull()
  })

  it('keeps a booting group yellow independently of the ready peer group', async () => {
    const fallback = stubDashboardFetch({})
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      if (String(input).includes('/api/v1/deployments')) return json({ items: [{
        id: 'dep', alias: 'Split model', model: { repository: 'org/model' }, runtime: 'vllm', kind: 'managed', status: 'running', settings: {},
        instances: [
          { instance_id: 0, node_names: ['Node 3', 'Node 4'], status: 'running' },
          { instance_id: 1, node_names: ['Node 1', 'Node 2'], status: 'starting' },
        ],
      }] })
      return fallback(input, init)
    }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    const ready = await screen.findByText('Status: running')
    const starting = screen.getByText('Status: starting')
    expect(ready.previousElementSibling).toHaveClass('status-running')
    expect(starting.previousElementSibling).toHaveClass('status-starting')
    expect(screen.getByText('Group 2 · Node 1 + Node 2')).toBeInTheDocument()
  })

  it('shows a still-running group with stopped intent despite a deployment error', async () => {
    const fallback = stubDashboardFetch({})
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      if (String(input).includes('/api/v1/deployments')) return json({ items: [{
        id: 'dep', alias: 'Failed stop', model: { repository: 'org/model' }, runtime: 'vllm', kind: 'managed',
        status: 'error', desired_state: 'stopped', settings: {}, instances: [
          { instance_id: 0, node_names: ['Node 3', 'Node 4'], status: 'running', desired_state: 'stopped', has_live_containers: true },
          { instance_id: 1, node_names: ['Node 1', 'Node 2'], status: 'stopped', desired_state: 'stopped' },
        ],
      }, {
        id: 'stopped', alias: 'Stopped model', model: { repository: 'org/other' }, runtime: 'vllm', kind: 'managed',
        status: 'stopped', settings: {},
      }] })
      return fallback(input, init)
    }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    expect(await screen.findByText('Failed stop')).toBeInTheDocument()
    expect(screen.getByText('Group 1 · Node 3 + Node 4')).toBeInTheDocument()
    expect(screen.getByText('Stop pending')).toBeInTheDocument()
    expect(screen.getByText('Deployment status: error')).toBeInTheDocument()
    expect(screen.getByText('1 of 2 deployments active')).toBeInTheDocument()
    expect(screen.queryByText('Group 2 · Node 1 + Node 2')).not.toBeInTheDocument()
    expect(screen.queryByText('Stopped model')).not.toBeInTheDocument()
  })

  it('does not keep reporting a finished stop as a running model', async () => {
    const fallback = stubDashboardFetch({})
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      if (String(input).includes('/api/v1/deployments')) return json({ items: [{
        // A stop the controller could not verify latches at degraded, but its
        // inventory already confirms every rank is idle.
        id: 'dep', alias: 'Stopped long ago', model: { repository: 'org/model' }, runtime: 'vllm', kind: 'managed',
        status: 'degraded', desired_state: 'stopped', settings: {},
        node_ids: ['gx10-node-1', 'gx10-node-2'], occupied_node_ids: [],
      }, {
        id: 'live', alias: 'Still serving', model: { repository: 'org/other' }, runtime: 'vllm', kind: 'managed',
        status: 'degraded', desired_state: 'stopped', settings: {},
        node_ids: ['gx10-node-3'], occupied_node_ids: ['gx10-node-3'],
      }] })
      return fallback(input, init)
    }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    expect(await screen.findByText('Still serving')).toBeInTheDocument()
    expect(screen.getByText('Stop pending')).toBeInTheDocument()
    expect(screen.getByText('gx10-node-3')).toBeInTheDocument()
    expect(screen.queryByText('Stopped long ago')).not.toBeInTheDocument()
    expect(screen.queryByText('gx10-node-1 + gx10-node-2')).not.toBeInTheDocument()
    expect(screen.getByText('1 of 2 deployments active')).toBeInTheDocument()
  })

  it('keeps a stop pending while no inventory has judged its ranks', async () => {
    const fallback = stubDashboardFetch({})
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      if (String(input).includes('/api/v1/deployments')) return json({ items: [{
        id: 'dep', alias: 'Unjudged stop', model: { repository: 'org/model' }, runtime: 'vllm', kind: 'managed',
        status: 'degraded', desired_state: 'stopped', settings: {}, node_ids: ['gx10-node-1'],
      }] })
      return fallback(input, init)
    }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    expect(await screen.findByText('Unjudged stop')).toBeInTheDocument()
    expect(screen.getByText('Stop pending')).toBeInTheDocument()
  })

  it('hides stopped groups and keeps the ones still holding containers', async () => {
    const fallback = stubDashboardFetch({})
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      if (String(input).includes('/api/v1/deployments')) return json({ items: [{
        id: 'dep', alias: 'Split stop', model: { repository: 'org/model' }, runtime: 'vllm', kind: 'managed',
        status: 'degraded', desired_state: 'stopped', settings: {}, instances: [
          { instance_id: 0, node_names: ['Node 1', 'Node 2'], status: 'degraded', desired_state: 'stopped', has_live_containers: false },
          { instance_id: 1, node_names: ['Node 3', 'Node 4'], status: 'running', desired_state: 'stopped', has_live_containers: true },
        ],
      }] })
      return fallback(input, init)
    }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    expect(await screen.findByText('Group 2 · Node 3 + Node 4')).toBeInTheDocument()
    expect(screen.getByText('Stop pending')).toBeInTheDocument()
    expect(screen.queryByText('Group 1 · Node 1 + Node 2')).not.toBeInTheDocument()
    expect(screen.getByText('1 of 1 deployments active')).toBeInTheDocument()
  })

  it.each([false, undefined])('hides rolled-back groups without live inventory confirmation (%s)', async (hasLiveContainers) => {
    const fallback = stubDashboardFetch({})
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      if (String(input).includes('/api/v1/deployments')) return json({ items: [{
        id: 'dep', alias: 'Failed launch', model: { repository: 'org/model' }, runtime: 'vllm', kind: 'managed',
        status: 'error', desired_state: 'running', settings: {}, instances: [
          { instance_id: 0, node_names: ['Node 3', 'Node 4'], status: 'starting', desired_state: 'running', has_live_containers: hasLiveContainers },
          { instance_id: 1, node_names: ['Node 1', 'Node 2'], status: 'running', desired_state: 'running', has_live_containers: hasLiveContainers },
        ],
      }] })
      return fallback(input, init)
    }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    expect(await screen.findByText('0 of 1 deployments active')).toBeInTheDocument()
    expect(screen.queryByText('Failed launch')).not.toBeInTheDocument()
    expect(screen.queryByText(/Group 1/)).not.toBeInTheDocument()
    expect(screen.queryByText(/Group 2/)).not.toBeInTheDocument()
  })

  it('shows running models and inference separately for each engine group', async () => {
    const groups = [
      { instance_id: 0, node_names: ['Node 1', 'Node 2'], status: 'running', desired_state: 'running' },
      { instance_id: 1, node_names: ['Node 3', 'Node 4'], status: 'running', desired_state: 'running' },
      { instance_id: 2, node_names: ['Node 5', 'Node 6'], status: 'stopped', desired_state: 'stopped' },
    ]
    const stats = {
      active_requests: { 'shared-model': { connections: 2 } },
      active_request_groups: Object.fromEntries(groups.slice(0, 2).map((group) => [String(group.instance_id), {
        ...group, group_id: String(group.instance_id), deployment_id: 'dep', model: 'shared-model', connections: 1,
      }])),
    }
    const fallback = stubDashboardFetch(stats)
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      if (String(input).includes('/api/v1/deployments')) return json({ items: [{
        id: 'dep', alias: 'Split model', model: { repository: 'org/model' }, runtime: 'vllm', kind: 'managed',
        status: 'degraded', deployment_mode: 'grouped_sharded', settings: {}, instances: groups,
      }] })
      return fallback(input, init)
    }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    expect(await screen.findAllByText('Group 1 · Node 1 + Node 2')).toHaveLength(2)
    expect(screen.getAllByText('Group 2 · Node 3 + Node 4')).toHaveLength(2)
    expect(screen.getAllByText('1 active · 0 queued')).toHaveLength(2)
    expect(screen.queryByText(/Node 5 \+ Node 6/)).not.toBeInTheDocument()
    expect(screen.getByText('2 active · 0 queued')).toBeInTheDocument()
  })

  it('merges admission by group without combining shared model sessions', () => {
    const group = { group_id: 'dep:instance:0', deployment_id: 'dep', instance_id: 0, node_names: ['Node 1', 'Node 2'], model: 'shared' }
    const snapshot = inferenceSessionSnapshot({ active_request_groups: {
      [group.group_id]: { ...group, connections: 1, queued: 2 },
    } }, {
      first: { ...group, running: 1, queued: 0 },
      second: { ...group, group_id: 'dep:instance:1', instance_id: 1, node_names: ['Node 3', 'Node 4'], running: 1, queued: 1 },
    })
    expect(snapshot.map(({ request }) => [request.connections, request.queued])).toEqual([[1, 0], [1, 1]])
    expect(snapshot[1].groupLabel).toBe('Group 2 · Node 3 + Node 4')
    expect(inferenceSessionSnapshot({ active_requests: { legacy: { connections: 2 } } })[0].model).toBe('legacy')
  })

  it('keeps admission-only groups separate when stats are unavailable and preserves legacy targets', () => {
    const group = { deployment_id: 'dep', instance_id: 0, node_names: ['Node 1', 'Node 2'], model: 'shared', running: 1, queued: 0 }
    const snapshot = inferenceSessionSnapshot(undefined, {
      first: { ...group, group_id: 'dep:instance:0' },
      second: { ...group, deployment_id: 'other', group_id: 'other:instance:0', node_names: ['Node 3', 'Node 4'] },
      legacy: { model: 'legacy', running: 1, queued: 2 },
    })
    expect(snapshot.map(({ key }) => key)).toEqual(['dep:instance:0', 'other:instance:0', 'legacy:legacy'])
    expect(snapshot.map(({ request }) => request.connections)).toEqual([1, 1, 1])
    expect(snapshot[2].request.queued).toBe(2)
  })

  it('shows the empty state when a degraded deployment has no active groups', async () => {
    const fallback = stubDashboardFetch({ active_requests: {}, active_request_groups: {} })
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      if (String(input).includes('/api/v1/deployments')) return json({ items: [{
        id: 'dep', alias: 'Split model', model: { repository: 'org/model' }, runtime: 'vllm', kind: 'managed',
        status: 'degraded', settings: {}, instances: [
          { instance_id: 0, node_names: ['Node 1', 'Node 2'], status: 'error', desired_state: 'running' },
          { instance_id: 1, node_names: ['Node 3', 'Node 4'], status: 'stopped', desired_state: 'stopped' },
        ],
      }] })
      return fallback(input, init)
    }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    expect(await screen.findByText('No models running')).toBeInTheDocument()
  })

  it('renders individual node and GPU usage while excluding hidden nodes', async () => {
    const localStats = {
      cpu_pct: 20, cpu_logical_count: 4, cpu_temp_c: 54, mem: { used: 64 * 1024 ** 3, total: 128 * 1024 ** 3, pct: 50 },
      gpus: [{ index: 0, name: 'NVIDIA GB10', util: 35, temp: 62, mem_used_mib: null, mem_total_mib: null }],
      active_requests: {}, ts: 1_777_000_000,
    }
    const remoteStats = {
      cpu_pct: 60, cpu_logical_count: 12, cpu_temp_c: 58, mem: { used: 32 * 1024 ** 3, total: 64 * 1024 ** 3, pct: 50 },
      gpus: [
        { index: 0, name: 'NVIDIA RTX', util: 55, temp: 65, mem_used_mib: 8 * 1024, mem_total_mib: 16 * 1024 },
        { index: 1, name: 'NVIDIA RTX', util: 90, temp: 67, mem_used_mib: 4 * 1024, mem_total_mib: 16 * 1024 },
        { index: 2, name: 'NVIDIA RTX', util: null },
        { index: 3, name: 'NVIDIA RTX', util: 99, error: 'GPU probe failed' },
      ],
      active_requests: {}, ts: 1_777_000_000,
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/stats')) return json(localStats)
      if (path.includes('/api/inference-queue')) return json({})
      if (path.includes('/api/v1/deployments')) return json({ items: [] })
      if (path.includes('/api/v1/community/sync')) return json({ consent: false, outbox: {} })
      return json({ items: [
        { id: 'local', name: 'Spark Four', local: true, online: true, stats: localStats },
        { id: 'node-2', name: 'Spark Two', online: true, stats: remoteStats },
        { id: 'node-3', name: 'Spark Three', online: false },
        { id: 'windows-pc', name: 'Inference PC', online: true, hidden_from_dashboard: true, stats: { ...localStats, cpu_pct: 100 } },
      ] })
    }))

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    const localCard = await screen.findByLabelText('Resource usage for Spark Four')
    const remoteCard = screen.getByLabelText('Resource usage for Spark Two')
    expect(within(localCard).getByRole('progressbar', { name: 'Spark Four CPU load' })).toHaveAttribute('aria-valuenow', '20')
    expect(within(remoteCard).getByRole('progressbar', { name: 'Spark Two CPU load' })).toHaveAttribute('aria-valuenow', '60')
    expect(within(localCard).getByText('64.0 GB used / 128.0 GB')).toBeInTheDocument()
    expect(within(remoteCard).getByText('32.0 GB used / 64.0 GB')).toBeInTheDocument()
    expect(within(localCard).getByRole('progressbar', { name: 'Spark Four GPU 0 utilization' })).toHaveAttribute('aria-valuenow', '35')
    expect(within(remoteCard).getByRole('progressbar', { name: 'Spark Two GPU 0 utilization' })).toHaveAttribute('aria-valuenow', '55')
    expect(within(remoteCard).getByRole('progressbar', { name: 'Spark Two GPU 1 utilization' })).toHaveAttribute('aria-valuenow', '90')
    const unmeasured = within(remoteCard).getByRole('progressbar', { name: 'Spark Two GPU 2 utilization' })
    expect(unmeasured).not.toHaveAttribute('aria-valuenow')
    expect(unmeasured).toHaveAttribute('aria-valuetext', 'Unavailable')
    expect(within(remoteCard).getByText('8.0 GB used / 16.0 GB')).toBeInTheDocument()
    expect(within(remoteCard).getByText('4.0 GB used / 16.0 GB')).toBeInTheDocument()
    expect(within(remoteCard).getByText('GPU telemetry unavailable: GPU probe failed')).toBeInTheDocument()
    expect(within(remoteCard).queryByRole('progressbar', { name: 'Spark Two GPU 3 utilization' })).not.toBeInTheDocument()
    expect(within(screen.getByLabelText('Resource usage for Spark Three')).queryByRole('progressbar')).not.toBeInTheDocument()
    expect(screen.queryByText(/Pooled/)).not.toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Cluster nodes' })).toBeInTheDocument()
    expect(screen.getByText('Spark Four')).toBeInTheDocument()
    expect(screen.getByText('Spark Two')).toBeInTheDocument()
    expect(screen.getByText('Spark Three')).toBeInTheDocument()
    expect(screen.getByText(/2 of 3 visible nodes online.*1 hidden/)).toBeInTheDocument()
    expect(screen.queryByText('Inference PC')).not.toBeInTheDocument()
    expect(document.querySelector('.community-strip')).toHaveAttribute(
      'title', 'Sign in under Settings → Community Features to see community data.')
    expect(screen.getByRole('link', { name: 'Open community settings' })).toHaveAttribute('href', '/settings')
  })

  it('keeps local telemetry visible when cluster inventory fails', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/stats')) return json({
        cpu_pct: 20, cpu_temp_c: 54,
        mem: { used: 64 * 1024 ** 3, total: 128 * 1024 ** 3, pct: 50 },
        gpus: [], active_requests: {}, ts: 1_777_000_000,
      })
      if (path.includes('/api/inference-queue')) return json({})
      if (path.includes('/api/v1/deployments')) return json({ items: [] })
      if (path.includes('/api/v1/community/sync')) return json({ consent: false, outbox: {} })
      return new Response(JSON.stringify({ detail: 'node probe failed' }), {
        status: 503,
        headers: { 'Content-Type': 'application/json' },
      })
    }))

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    expect(await screen.findByText('CPU load')).toBeInTheDocument()
    expect(screen.getByText('20.0%')).toBeInTheDocument()
    expect(screen.getByText('64.0 GB used / 128.0 GB')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Cluster nodes' })).toBeInTheDocument()
    expect(screen.getByText(/0 of 0 visible nodes online/)).toBeInTheDocument()
    expect(screen.getByText('Cluster inventory unavailable')).toBeInTheDocument()
    expect(screen.queryByText('No nodes shown on the dashboard')).not.toBeInTheDocument()
    expect(screen.queryByText('node probe failed')).not.toBeInTheDocument()
  })

  it('includes starting deployments in the running models card', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/stats')) return json({ cpu_pct: 20, mem: {}, gpus: [], active_requests: {} })
      if (path.includes('/api/inference-queue')) return json({})
      if (path.includes('/api/v1/deployments')) return json({ items: [
        { id: 'running', alias: 'Ready model', model: { repository: 'org/ready' }, settings: {}, kind: 'managed', runtime: 'vllm', status: 'running' },
        { id: 'starting', alias: 'Loading model', model: { repository: 'org/loading' }, settings: {}, kind: 'managed', runtime: 'vllm', status: 'starting' },
        { id: 'stopped', alias: 'Stopped model', model: { repository: 'org/stopped' }, settings: {}, kind: 'managed', runtime: 'vllm', status: 'stopped' },
      ] })
      if (path.includes('/api/v1/community/sync')) return json({ consent: false, outbox: {} })
      return json({ items: [] })
    }))

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    expect(await screen.findByText('2 of 3 deployments active')).toBeInTheDocument()
    expect(screen.getByText('Ready model')).toBeInTheDocument()
    expect(screen.getByText('Loading model')).toBeInTheDocument()
    expect(screen.queryByText('Stopped model')).not.toBeInTheDocument()
    expect(screen.getByText('Loading model').closest('.dashboard-list-row')?.querySelector('.status-dot'))
      .toHaveClass('status-starting')
  })

  it('shows an explicit empty state when every cluster node is hidden', async () => {
    const stats = { cpu_pct: 25, mem: { used: 8, total: 16 }, gpus: [], active_requests: {} }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/stats')) return json(stats)
      if (path.includes('/api/inference-queue')) return json({})
      if (path.includes('/api/v1/deployments')) return json({ items: [] })
      if (path.includes('/api/v1/community/sync')) return json({ consent: false, outbox: {} })
      return json({ items: [{ id: 'local', name: 'Hidden node', local: true, online: true, hidden_from_dashboard: true, stats }] })
    }))

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    expect(await screen.findByText('No nodes shown on the dashboard')).toBeInTheDocument()
    expect(screen.getByText(/0 of 0 visible nodes online.*1 hidden/)).toBeInTheDocument()
    expect(screen.queryByText('Hidden node')).not.toBeInTheDocument()
  })

  it('renders core telemetry without waiting for secondary dashboard requests', async () => {
    const pending = new Promise<Response>(() => undefined)
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/stats')) return json({
        cpu_pct: 42, cpu_logical_count: 8,
        mem: { used: 8 * 1024 ** 3, total: 16 * 1024 ** 3 },
        gpus: [], active_requests: {}, ts: 1_777_000_000,
      })
      if (
        path.includes('/api/inference-queue')
        || path.includes('/api/v1/deployments')
        || path.includes('/api/v1/community/sync')
        || path.includes('/api/v1/nodes')
      ) return pending
      return json({ enabled: false })
    }))

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    expect(await screen.findByText('CPU load')).toBeInTheDocument()
    expect(screen.getByText('42.0%')).toBeInTheDocument()
    expect(screen.queryByText('Loading system overview')).not.toBeInTheDocument()
    expect(screen.getByText('Loading cluster nodes')).toBeInTheDocument()
    expect(screen.getAllByText('Loading deployments')).toHaveLength(2)
    expect(screen.getByText('No active inference')).toBeInTheDocument()
    expect(screen.getAllByText(/queue loading/)).toHaveLength(2)
    expect(screen.queryByText('Idle')).not.toBeInTheDocument()
  })

  it('does not abort a slow initial load when the refresh interval elapses', async () => {
    vi.useFakeTimers()
    let finishNodeInventory: ((response: Response) => void) | undefined
    let nodeInventorySignal: AbortSignal | undefined
    let nodeInventoryCalls = 0
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.includes('/api/stats')) return json({ cpu_pct: 25, mem: {}, gpus: [], active_requests: {} })
      if (path.includes('/api/inference-queue')) return json({})
      if (path.includes('/api/v1/deployments')) return json({ items: [] })
      if (path.includes('/api/v1/community/sync')) return json({ consent: false, outbox: {} })
      if (path.includes('/api/v1/model-routing-policies')) return json({ items: [] })
      if (path.includes('/api/v1/onboarding')) return json({
        role: 'controller',
        node: { id: 'local', name: 'This node', port: 7878, access_urls: [] },
        controller_reachable: true,
      })
      nodeInventoryCalls += 1
      nodeInventorySignal ??= init?.signal ?? undefined
      return new Promise<Response>((resolve) => { finishNodeInventory = resolve })
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    await act(async () => { await vi.advanceTimersByTimeAsync(10_000) })
    expect(nodeInventoryCalls).toBe(1)
    expect(nodeInventorySignal?.aborted).toBe(false)

    await act(async () => { finishNodeInventory?.(json({ items: [] })) })
    expect(screen.getByRole('region', { name: 'Inference overview' })).toBeInTheDocument()
  })

  it('surfaces a stuck core telemetry request after the short dashboard timeout', async () => {
    vi.useFakeTimers()
    let statsSignal: AbortSignal | undefined
    let statsCalls = 0
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.includes('/api/stats')) {
        statsCalls += 1
        statsSignal = init?.signal ?? undefined
        return new Promise<Response>((_resolve, reject) => {
          statsSignal?.addEventListener('abort', () => {
            reject(new DOMException('The operation was aborted.', 'AbortError'))
          }, { once: true })
        })
      }
      if (path.includes('/api/inference-queue')) return json({ target: { model: 'live-model', running: 1, queued: 0 } })
      if (path.includes('/api/v1/deployments')) return json({ items: [] })
      if (path.includes('/api/v1/community/sync')) return json({ consent: false, outbox: {} })
      return json({ items: [] })
    }))

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    await act(async () => { await vi.advanceTimersByTimeAsync(9_999) })
    expect(statsCalls).toBe(1)
    expect(statsSignal?.aborted).toBe(false)

    await act(async () => { await vi.advanceTimersByTimeAsync(1) })
    expect(screen.getByText(/The request timed out\. Check the node connection and retry\./)).toBeInTheDocument()
    expect(screen.getByRole('region', { name: 'Inference overview' })).toBeInTheDocument()
    expect(screen.getByText('live-model')).toBeInTheDocument()
    expect(screen.getAllByText(/^1 active/)).toHaveLength(2)
    expect(screen.getByText('Processing')).toBeInTheDocument()
    expect(screen.queryByText('Active session status unavailable')).not.toBeInTheDocument()
    expect(screen.queryByText("Couldn't load this view")).not.toBeInTheDocument()

    await act(async () => { await vi.advanceTimersByTimeAsync(10_000) })
    expect(statsCalls).toBe(2)
  })

  it('merges per-model admission sessions with rates from local telemetry', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/stats')) return json({
        cpu_pct: 25,
        mem: {},
        gpus: [],
        active_requests: { 'live-model': { connections: 1, queued: 0, output_tok_s: 12 } },
      })
      if (path.includes('/api/inference-queue')) return json({
        'target-a': { model: 'live-model', running: 1, queued: 1 },
        'target-b': { model: 'live-model', running: 1, queued: 0 },
      })
      if (path.includes('/api/v1/deployments')) return json({ items: [] })
      if (path.includes('/api/v1/community/sync')) return json({ consent: false, outbox: {} })
      return json({ items: [] })
    }))

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    expect(await screen.findByText('live-model')).toBeInTheDocument()
    expect(screen.getAllByText('live-model')).toHaveLength(1)
    expect(screen.getAllByText('2 active · 1 queued')).toHaveLength(2)
    expect(screen.getByText('12.0 tok/s')).toBeInTheDocument()
  })

  it('uses fresh admission queue depth instead of stale stats queue depth', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/stats')) return json({
        cpu_pct: 25,
        mem: {},
        gpus: [],
        active_requests: { 'live-model': { connections: 1, queued: 5, output_tok_s: 12 } },
      })
      if (path.includes('/api/inference-queue')) return json({
        target: { model: 'live-model', running: 1, queued: 0 },
      })
      if (path.includes('/api/v1/deployments')) return json({ items: [] })
      if (path.includes('/api/v1/community/sync')) return json({ consent: false, outbox: {} })
      return json({ items: [] })
    }))

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    const model = await screen.findByText('live-model')
    expect(model.closest('.session-row')).toHaveTextContent('1 active · 0 queued')
    expect(screen.queryByText(/5 queued/)).not.toBeInTheDocument()
  })

  it('keeps healthy retained admission sessions visible during a normal refresh', async () => {
    let admissionCalls = 0
    let finishAdmission: ((response: Response) => void) | undefined
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/stats')) return json({ cpu_pct: 25, mem: {}, gpus: [], active_requests: {} })
      if (path.includes('/api/inference-queue')) {
        admissionCalls += 1
        if (admissionCalls === 1) return json({ target: { model: 'healthy-model', running: 1, queued: 0 } })
        return new Promise<Response>((resolve) => { finishAdmission = resolve })
      }
      if (path.includes('/api/v1/deployments')) return json({ items: [] })
      if (path.includes('/api/v1/community/sync')) return json({ consent: false, outbox: {} })
      return json({ items: [] })
    }))

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    expect(await screen.findByText('healthy-model')).toBeInTheDocument()

    act(() => { screen.getByRole('button', { name: 'Refresh' }).click() })

    expect(admissionCalls).toBe(2)
    expect(screen.getByText('healthy-model')).toBeInTheDocument()
    await act(async () => { finishAdmission?.(json({ target: { model: 'healthy-model', running: 1, queued: 0 } })) })
  })

  it('renders queued-only admission sessions as waiting', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/stats')) return new Response(JSON.stringify({ detail: 'telemetry unavailable' }), {
        status: 503,
        headers: { 'Content-Type': 'application/json' },
      })
      if (path.includes('/api/inference-queue')) return json({ target: { model: 'queued-model', running: 0, queued: 2 } })
      if (path.includes('/api/v1/deployments')) return json({ items: [] })
      if (path.includes('/api/v1/community/sync')) return json({ consent: false, outbox: {} })
      return json({ items: [] })
    }))

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    const model = await screen.findByText('queued-model')
    const row = model.closest('.session-row')!
    expect(row.querySelector('.status-dot')).toHaveClass('status-waiting')
    expect(row).toHaveTextContent('0 active · 2 queued')
    expect(row).toHaveTextContent('Waiting')
    expect(row).not.toHaveTextContent('Measuring')
    expect(screen.getByText('Waiting', { selector: '.status' })).toBeInTheDocument()
  })

  it('keeps inference unavailable when admission is empty and stats fail', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/stats')) return new Response(JSON.stringify({ detail: 'telemetry unavailable' }), {
        status: 503,
        headers: { 'Content-Type': 'application/json' },
      })
      if (path.includes('/api/inference-queue')) return json({})
      if (path.includes('/api/v1/deployments')) return json({ items: [] })
      if (path.includes('/api/v1/community/sync')) return json({ consent: false, outbox: {} })
      return json({ items: [] })
    }))

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    expect(await screen.findByText('Active session status unavailable')).toBeInTheDocument()
    expect(screen.getByText('Unavailable', { selector: '.status' })).toBeInTheDocument()
    expect(screen.queryByText('No active inference')).not.toBeInTheDocument()
    expect(screen.queryByText('Idle')).not.toBeInTheDocument()
  })

  it('does not let retained errored admission override fresh stats', async () => {
    MockWebSocket.instances = []
    MockWebSocket.autoOpen = true
    vi.stubGlobal('WebSocket', MockWebSocket)
    let admissionCalls = 0
    let finishAdmission: ((response: Response) => void) | undefined
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/stats')) return json({ cpu_pct: 25, mem: {}, gpus: [], active_requests: {} })
      if (path.includes('/api/inference-queue')) {
        admissionCalls += 1
        if (admissionCalls === 1) return json({ target: { model: 'finished-model', running: 1, queued: 0 } })
        if (admissionCalls === 2) return new Response(JSON.stringify({ detail: 'queue unavailable' }), {
          status: 503,
          headers: { 'Content-Type': 'application/json' },
        })
        return new Promise<Response>((resolve) => { finishAdmission = resolve })
      }
      if (path.includes('/api/v1/deployments')) return json({ items: [] })
      if (path.includes('/api/v1/community/sync')) return json({ consent: false, outbox: {} })
      if (path.includes('/api/v1/onboarding')) return json({ role: 'controller' })
      return json({ items: [] })
    }))

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    expect(await screen.findByText('finished-model')).toBeInTheDocument()

    await act(async () => {
      MockWebSocket.instances[0].emit({
        type: 'snapshot',
        stats: { cpu_pct: 25, mem: {}, gpus: [], active_requests: {} },
        admission: null,
      })
      await Promise.resolve()
      screen.getByRole('button', { name: 'Refresh' }).click()
    })

    expect(await screen.findByText(/Queue refresh paused: queue unavailable/)).toBeInTheDocument()
    expect(screen.queryByText('finished-model')).not.toBeInTheDocument()
    expect(screen.queryByText('Processing')).not.toBeInTheDocument()
    expect(screen.getByText('Unavailable', { selector: '.status' })).toBeInTheDocument()

    act(() => { screen.getByRole('button', { name: 'Refresh' }).click() })
    expect(admissionCalls).toBe(3)
    expect(screen.queryByText('finished-model')).not.toBeInTheDocument()

    await act(async () => {
      finishAdmission?.(json({ target: { model: 'replacement-model', running: 0, queued: 1 } }))
    })
    expect(await screen.findByText('replacement-model')).toBeInTheDocument()
    expect(screen.queryByText('finished-model')).not.toBeInTheDocument()
  })

  it('marks retained section data stale when an independent refresh fails', async () => {
    vi.useFakeTimers()
    const attempts = new Map<string, number>()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      const tracked = [
        '/api/stats', '/api/inference-queue', '/api/v1/deployments',
        '/api/v1/community/sync', '/api/v1/nodes',
      ].find((item) => path.includes(item))
      if (!tracked) return json({ enabled: false })
      const attempt = (attempts.get(tracked) ?? 0) + 1
      attempts.set(tracked, attempt)
      if (attempt > 1) return new Response(JSON.stringify({ detail: 'refresh failed' }), {
        status: 503,
        headers: { 'Content-Type': 'application/json' },
      })
      if (tracked === '/api/stats') return json({ cpu_pct: 25, mem: {}, gpus: [], active_requests: {} })
      if (tracked === '/api/inference-queue') return json({ model: { running: 0, queued: 2 } })
      if (tracked === '/api/v1/deployments') return json({ items: [] })
      if (tracked === '/api/v1/community/sync') return json({ consent: false, outbox: {} })
      return json({ items: [] })
    }))

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    await act(async () => { await Promise.resolve() })
    expect(screen.getByText('Sharing off')).toBeInTheDocument()

    await act(async () => { await vi.advanceTimersByTimeAsync(10_000) })

    expect(screen.getByText(/Local telemetry refresh paused: refresh failed/)).toBeInTheDocument()
    expect(screen.getByText(/Cluster inventory refresh paused: refresh failed/)).toBeInTheDocument()
    expect(screen.getByText(/Deployment refresh paused: refresh failed/)).toBeInTheDocument()
    expect(screen.getByText(/Queue refresh paused: refresh failed/)).toBeInTheDocument()
    expect(screen.getAllByText(/2 queued · refresh paused/)).toHaveLength(2)
    expect(screen.getByText(/0 pending · 0 synced · refresh paused/)).toBeInTheDocument()
  })

  it('keeps live session details visible when initial queue loading fails', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/stats')) return json({
        cpu_pct: 25, mem: {}, gpus: [],
        active_requests: { 'live-model': { connections: 1, output_tok_s: 12 } },
      })
      if (path.includes('/api/inference-queue')) return new Response(
        JSON.stringify({ detail: 'queue unavailable' }),
        { status: 503, headers: { 'Content-Type': 'application/json' } },
      )
      if (path.includes('/api/v1/deployments')) return json({ items: [] })
      if (path.includes('/api/v1/community/sync')) return json({ consent: false, outbox: {} })
      return json({ items: [] })
    }))

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    expect(await screen.findByText('live-model')).toBeInTheDocument()
    expect(screen.getByText('Processing')).toBeInTheDocument()
    expect(screen.getByText(/Queue status unavailable: queue unavailable/)).toBeInTheDocument()
    expect(screen.queryByText('Active session status unavailable')).not.toBeInTheDocument()
  })

  it('lists active inference counts by caller IP', async () => {
    vi.stubGlobal('fetch', stubDashboardFetch({
      cpu_pct: 25, mem: {}, gpus: [],
      active_requests: {
        'live-model': {
          connections: 3,
          output_tok_s: 12,
          caller_ips: { '192.0.2.20': 1, '192.0.2.10': 2 },
        },
      },
    }))

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    expect(await screen.findByText('live-model')).toBeInTheDocument()
    expect(screen.getByText('2 from 192.0.2.10 · 1 from 192.0.2.20')).toBeInTheDocument()
  })

  it('shows live prompt processing, output, and thinking token rates per stage', async () => {
    vi.stubGlobal('fetch', stubDashboardFetch({
      cpu_pct: 25, mem: {}, gpus: [],
      active_requests: {
        'live-model': {
          connections: 1,
          pp_tok_s: 2400,
          output_tok_s: 12.5,
          thinking_tok_s: 4,
        },
      },
    }))

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    expect(await screen.findByText('live-model')).toBeInTheDocument()
    expect(screen.getByText('Prompt processing')).toBeInTheDocument()
    expect(screen.getByText('2400.0 tok/s')).toBeInTheDocument()
    expect(screen.getByText('Output')).toBeInTheDocument()
    expect(screen.getByText('12.5 tok/s')).toBeInTheDocument()
    expect(screen.getByText('Thinking')).toBeInTheDocument()
    expect(screen.getByText('4.0 tok/s')).toBeInTheDocument()
  })

  it('shows measuring placeholders when prompt processing speed is not yet available', async () => {
    vi.stubGlobal('fetch', stubDashboardFetch({
      cpu_pct: 25, mem: {}, gpus: [],
      active_requests: {
        'live-model': { connections: 1, pp_tok_s: null, output_tok_s: 0, thinking_tok_s: 0 },
      },
    }))

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    expect(await screen.findByText('live-model')).toBeInTheDocument()
    expect(screen.getByText('Unavailable')).toBeInTheDocument()
    expect(screen.getAllByText('Measuring…')).toHaveLength(2)
  })

  it('prefers fresh local stats over retained local node telemetry', () => {
    const snapshot = nodeResourceSnapshot([{
      id: 'local', name: 'This node', local: true, online: true,
      stats: {
        cpu_pct: 10, cpu_logical_count: 8,
        mem: { used: 2, total: 10 },
        gpus: [{ index: 0, util: 10 }],
      },
    }], {
      cpu_pct: 80, cpu_logical_count: 8,
      mem: { used: 8, total: 10 },
      gpus: [{ index: 0, util: 70 }],
    })

    expect(snapshot[0].stats?.cpu_pct).toBe(80)
    expect(snapshot[0].stats?.mem?.used).toBe(8)
    expect(snapshot[0].stats?.gpus?.[0].util).toBe(70)
    expect(snapshot[0].source).toBe('stats')
  })

  it('retains the telemetry source when local inventory is newer than the local stats feed', () => {
    const snapshot = nodeResourceSnapshot([{
      id: 'local', name: 'This node', local: true, online: true,
      stats: { cpu_pct: 90, ts: 200 },
    }], { cpu_pct: 10, ts: 100 })
    expect(snapshot[0].stats?.cpu_pct).toBe(90)
    expect(snapshot[0].source).toBe('nodes')
  })

  it('marks stale telemetry and leaves missing CPU, GPU, and RAM readings unmeasured', async () => {
    const freshTimestamp = Date.now() / 1000
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/stats')) return json({ cpu_pct: 5, ts: freshTimestamp })
      if (path.includes('/api/inference-queue')) return json({})
      if (path.includes('/api/v1/deployments')) return json({ items: [] })
      if (path.includes('/api/v1/community/sync')) return json({ consent: false, outbox: {} })
      return json({ items: [
        { id: 'missing', name: 'Missing readings', online: true, stats: { mem: { total: 16 * 1024 ** 3 }, gpus: [{ index: 0, name: 'NVIDIA RTX', mem_total_mib: 24 * 1024 }], ts: freshTimestamp } },
        { id: 'stale', name: 'Stale worker', online: true, stats: { cpu_pct: 80, ts: freshTimestamp - 60 } },
      ] })
    }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    const missing = await screen.findByLabelText('Resource usage for Missing readings')
    expect(within(missing).getAllByRole('progressbar')).toHaveLength(4)
    for (const bar of within(missing).getAllByRole('progressbar')) {
      expect(bar).not.toHaveAttribute('aria-valuenow')
      expect(bar).toHaveAttribute('aria-valuetext', 'Unavailable')
    }
    expect(within(missing).queryByText('0.0%')).not.toBeInTheDocument()
    expect(within(missing).getByText('Usage unavailable / 16.0 GB')).toBeInTheDocument()
    const stale = screen.getByLabelText('Resource usage for Stale worker')
    expect(within(stale).getByText(/Last reported values - telemetry stale/)).toBeInTheDocument()
    expect(within(stale).getByText('80.0%')).toBeInTheDocument()
  })

  it('applies pushed stream snapshots without extra fetches', async () => {
    MockWebSocket.instances = []
    MockWebSocket.autoOpen = true
    const fetchMock = stubDashboardFetch({ cpu_pct: 20, mem: {}, gpus: [], active_requests: {}, ts: 1_777_000_000 })
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('WebSocket', MockWebSocket)

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    expect(await screen.findByText('20.0%')).toBeInTheDocument()
    const baselineCalls = fetchMock.mock.calls.length
    const socket = MockWebSocket.instances[0]
    expect(socket.url).toBe('ws://localhost:3000/api/ws/dashboard')

    act(() => socket.emit({
      type: 'snapshot',
      stats: { cpu_pct: 90, mem: {}, gpus: [], active_requests: {}, ts: 1_777_000_000 },
      admission: { chat: { running: 1, queued: 3 } },
      deployments: { items: [] },
      community_sync: { consent: false, outbox: {} },
      nodes: { items: [] },
    }))

    expect(screen.getByText('90.0%')).toBeInTheDocument()
    expect(screen.getAllByText(/3 queued/).length).toBeGreaterThan(0)
    expect(screen.getByText(/· live/)).toBeInTheDocument()
    expect(fetchMock.mock.calls.length).toBe(baselineCalls)
  })

  it('uses REST polling without opening a rejected stream on joined workers', async () => {
    MockWebSocket.instances = []
    const fetchMock = stubDashboardFetch({ cpu_pct: 20, mem: {}, gpus: [], active_requests: {} })
    const baseImplementation = fetchMock.getMockImplementation()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      if (String(input).includes('/api/v1/onboarding')) {
        return json({
          role: 'worker',
          node: { id: 'worker-1', name: 'Worker 1', port: 7878, access_urls: [] },
          controller_reachable: true,
        })
      }
      return baseImplementation!(input, init)
    }))
    vi.stubGlobal('WebSocket', MockWebSocket)

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    expect(await screen.findByText('20.0%')).toBeInTheDocument()
    expect(MockWebSocket.instances).toHaveLength(0)
  })

  it('rechecks the worker role before reconnecting after role discovery recovers', async () => {
    vi.useFakeTimers()
    MockWebSocket.instances = []
    MockWebSocket.autoOpen = true
    let onboardingCalls = 0
    const fetchMock = stubDashboardFetch({ cpu_pct: 20, mem: {}, gpus: [], active_requests: {} })
    const baseImplementation = fetchMock.getMockImplementation()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      if (String(input).includes('/api/v1/onboarding')) {
        onboardingCalls += 1
        if (onboardingCalls === 1) throw new TypeError('status unavailable')
        return json({
          role: 'worker',
          node: { id: 'worker-1', name: 'Worker 1', port: 7878, access_urls: [] },
          controller_reachable: true,
        })
      }
      return baseImplementation!(input, init)
    }))
    vi.stubGlobal('WebSocket', MockWebSocket)

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    expect(onboardingCalls).toBe(1)
    expect(MockWebSocket.instances).toHaveLength(1)

    act(() => { MockWebSocket.instances[0].close() })
    await act(async () => { await vi.advanceTimersByTimeAsync(5_000) })

    expect(onboardingCalls).toBe(2)
    expect(MockWebSocket.instances).toHaveLength(1)
  })

  it('resumes the 10s polling fallback when the stream closes', async () => {
    vi.useFakeTimers()
    MockWebSocket.instances = []
    MockWebSocket.autoOpen = true
    let statsCalls = 0
    const fetchMock = stubDashboardFetch({ cpu_pct: 25, mem: {}, gpus: [], active_requests: {} })
    const baseImplementation = fetchMock.getMockImplementation()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      if (String(input).includes('/api/stats')) statsCalls += 1
      return baseImplementation!(input, init)
    }))
    vi.stubGlobal('WebSocket', MockWebSocket)

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    await act(async () => { await Promise.resolve() })
    expect(statsCalls).toBe(1)

    // While the socket is live the 10s polling stays paused.
    await act(async () => { await vi.advanceTimersByTimeAsync(20_000) })
    expect(statsCalls).toBe(1)

    // The stream drops and reconnects never open: polling resumes.
    MockWebSocket.autoOpen = false
    act(() => { MockWebSocket.instances[0].close() })
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000) })
    expect(statsCalls).toBe(2)
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000) })
    expect(statsCalls).toBe(3)
  })

  it('keeps polling a source whose stream value is null', async () => {
    vi.useFakeTimers()
    MockWebSocket.instances = []
    MockWebSocket.autoOpen = true
    let statsCalls = 0
    let nodesCalls = 0
    const fetchMock = stubDashboardFetch({ cpu_pct: 25, mem: {}, gpus: [], active_requests: {} })
    const baseImplementation = fetchMock.getMockImplementation()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.includes('/api/stats')) statsCalls += 1
      if (path.includes('/api/v1/nodes')) nodesCalls += 1
      return baseImplementation!(input, init)
    }))
    vi.stubGlobal('WebSocket', MockWebSocket)

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    await act(async () => { await Promise.resolve() })
    expect(statsCalls).toBe(1)
    expect(nodesCalls).toBe(1)

    // stats failed server-side: only its REST polling stays active.
    act(() => MockWebSocket.instances[0].emit({
      type: 'snapshot',
      stats: null,
      admission: {},
      deployments: { items: [] },
      community_sync: { consent: false, outbox: {} },
      nodes: { items: [] },
    }))
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000) })
    expect(statsCalls).toBe(2)
    expect(nodesCalls).toBe(1)

    // A later healthy stats value clears the failure and pauses polling again.
    act(() => MockWebSocket.instances[0].emit({
      type: 'snapshot',
      stats: { cpu_pct: 30, mem: {}, gpus: [], active_requests: {} },
      admission: {},
      deployments: { items: [] },
      community_sync: { consent: false, outbox: {} },
      nodes: { items: [] },
    }))
    await act(async () => { await vi.advanceTimersByTimeAsync(20_000) })
    expect(statsCalls).toBe(2)
  })

  it('clears a resource error when stream data arrives', async () => {
    MockWebSocket.instances = []
    MockWebSocket.autoOpen = true
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/stats')) {
        return new Response(JSON.stringify({ detail: 'telemetry down' }), {
          status: 503,
          headers: { 'Content-Type': 'application/json' },
        })
      }
      if (path.includes('/api/inference-queue')) return json({})
      if (path.includes('/api/v1/deployments')) return json({ items: [] })
      if (path.includes('/api/v1/community/sync')) return json({ consent: false, outbox: {} })
      return json({ items: [] })
    }))
    vi.stubGlobal('WebSocket', MockWebSocket)

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    expect(await screen.findByText(/telemetry down/)).toBeInTheDocument()

    act(() => MockWebSocket.instances[0].emit({
      type: 'snapshot',
      stats: { cpu_pct: 90, mem: {}, gpus: [], active_requests: {}, ts: 1_777_000_000 },
      admission: {},
      deployments: { items: [] },
      community_sync: { consent: false, outbox: {} },
      nodes: { items: [] },
    }))

    expect(screen.queryByText('telemetry down')).not.toBeInTheDocument()
    expect(screen.getByText('90.0%')).toBeInTheDocument()
  })

  it('keeps CPU measurements separate when logical CPU counts are incomplete', () => {
    const snapshot = nodeResourceSnapshot([
      { id: 'older-node', online: true, stats: { cpu_pct: 0 } },
      { id: 'newer-node', online: true, stats: { cpu_pct: 100, cpu_logical_count: 64 } },
    ] as NodeInventoryItem[])

    expect(snapshot.map((item) => item.stats?.cpu_pct)).toEqual([0, 100])
  })
})

describe('sharedModelGroups', () => {
  const deployment = (id: string, modelId: string, servedModels?: string[]) => ({
    id, alias: id, model_id: modelId, served_models: servedModels,
    runtime: 'vllm', status: 'running', managed: true, settings: {},
  }) as unknown as Deployment

  it('groups deployments that publish a shared request id', () => {
    const groups = sharedModelGroups([
      deployment('dgx', 'org/model'),
      deployment('ws1', 'org/model'),
      deployment('other', 'org/different'),
    ])
    expect(groups).toHaveLength(1)
    expect(groups[0].model).toBe('org/model')
    expect(groups[0].deployments.map((item) => item.id)).toEqual(['dgx', 'ws1'])
  })

  it('groups deployments that share only a served name', () => {
    const groups = sharedModelGroups([
      deployment('a', 'org/one', ['shared']),
      deployment('b', 'org/two', ['shared']),
    ])
    expect(groups).toHaveLength(1)
    expect(groups[0].model).toBe('shared')
    expect(groups[0].deployments.map((item) => item.id)).toEqual(['a', 'b'])
  })

  it('builds one group per shared request id instead of merging transitively', () => {
    const groups = sharedModelGroups([
      deployment('a', 'x'),
      deployment('b', 'ignored', ['x', 'y']),
      deployment('c', 'y'),
    ])
    expect(groups.map((group) => [group.model, group.deployments.map((d) => d.id)]))
      .toEqual([['x', ['a', 'b']], ['y', ['b', 'c']]])
  })

  it('ignores the backing model id when explicit served names exist', () => {
    const groups = sharedModelGroups([
      deployment('a', 'org/model', ['served-a']),
      deployment('b', 'org/model'),
    ])
    expect(groups).toHaveLength(0)
  })
})

describe('Running models shared-instance routing', () => {
  const wire = (id: string, alias: string, nodeName: string) => ({
    id, alias, runtime: 'vllm', kind: 'managed', status: 'running',
    model: { repository: 'org/shared' }, served_models: ['shared-name'],
    selected_nodes: [{ id: nodeName, name: nodeName }],
  })

  it('groups shared instances and saves the priority policy', async () => {
    const base = stubDashboardFetch({ active_requests: {} })
    const puts: unknown[] = []
    let savedPolicy: { model: string; members: { deployment_id: string; max_concurrency: number | null }[] } | undefined
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.includes('/api/v1/deployments')) {
        return json({ items: [wire('dgx', 'dgx-model', 'DGX Spark'), wire('ws1', 'ws1-model', 'RTX Pro 6000')] })
      }
      if (path.includes('/api/v1/model-routing-policies')) {
        if (init?.method === 'PUT') {
          const body = JSON.parse(String(init.body))
          puts.push(body)
          savedPolicy = body
          return json(body)
        }
        if (!savedPolicy) return json({ items: [] })
        return json({ items: [{
          model: savedPolicy.model,
          members: savedPolicy.members.map((member) => ({
            ...member,
            alias: member.deployment_id === 'ws1' ? 'ws1-model' : 'dgx-model',
            status: 'running',
            node_names: [],
            live: true,
            inflight: 0,
          })),
        }] })
      }
      return base(input as RequestInfo, init)
    }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    await screen.findByText('2 instances share this request id')
    const heading = screen.getByText('shared-name', { selector: '.dashboard-cluster-heading strong' })
    expect(heading.closest('.dashboard-cluster')).not.toBeNull()

    fireEvent.change(screen.getByLabelText('Priority for ws1-model'), { target: { value: '0' } })
    fireEvent.change(screen.getByLabelText('Max concurrent requests for ws1-model'), { target: { value: '3' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save routing' }))

    await waitFor(() => expect(puts).toHaveLength(1))
    expect(puts[0]).toEqual({
      model: 'shared-name',
      members: [
        { deployment_id: 'ws1', max_concurrency: 3 },
        { deployment_id: 'dgx', max_concurrency: null },
      ],
    })
    await waitFor(() => expect(screen.getByText('Routing saved')).toBeInTheDocument())
  })

  it('keeps configured membership separate from the live inventory', async () => {
    const base = stubDashboardFetch({ active_requests: {} })
    const puts: unknown[] = []
    const policy = {
      model: 'shared-name',
      members: [
        { deployment_id: 'ws1', max_concurrency: 3, alias: 'ws1-model', status: 'running', node_names: [], live: true, inflight: 0 },
        { deployment_id: 'old', max_concurrency: null, alias: 'old-model', status: 'stopped', node_names: [], live: false, inflight: 0 },
      ],
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.includes('/api/v1/deployments')) {
        return json({ items: [
          wire('dgx', 'dgx-model', 'DGX Spark'),
          wire('ws1', 'ws1-model', 'RTX Pro 6000'),
          wire('extra', 'extra-model', 'Extra node'),
        ] })
      }
      if (path.includes('/api/v1/model-routing-policies')) {
        if (init?.method === 'PUT') {
          puts.push(JSON.parse(String(init.body)))
          return json({ model: 'shared-name', members: [] })
        }
        return json({ items: [policy] })
      }
      return base(input as RequestInfo, init)
    }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    await screen.findByText('3 instances share this request id')
    // The configured members render (offline one included); live deployments
    // outside the policy are only offered as explicit Add candidates.
    expect(screen.getByText('old-model')).toBeInTheDocument()
    expect(screen.getByText('Offline')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Save routing' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Add dgx-model to routing' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Add extra-model to routing' })).toBeEnabled()

    fireEvent.click(screen.getByRole('button', { name: 'Add dgx-model to routing' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save routing' }))
    await waitFor(() => expect(puts).toHaveLength(1))
    expect(puts[0]).toEqual({
      model: 'shared-name',
      members: [
        { deployment_id: 'ws1', max_concurrency: 3 },
        { deployment_id: 'old', max_concurrency: null },
        { deployment_id: 'dgx', max_concurrency: null },
      ],
    })
  })

  it('renders a plain row for a deployment with no shared request id', async () => {
    const base = stubDashboardFetch({ active_requests: {} })
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.includes('/api/v1/deployments')) {
        return json({ items: [wire('solo', 'solo-model', 'Local node')] })
      }
      return base(input as RequestInfo, init)
    }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)
    await screen.findByText('solo-model')
    expect(screen.queryByText('1 instances share this request id')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Save routing' })).not.toBeInTheDocument()
  })

  it('keeps a saved policy editable when only one member is live', async () => {
    const base = stubDashboardFetch({ active_requests: {} })
    const policy = {
      model: 'shared-name',
      members: [
        { deployment_id: 'ws1', max_concurrency: 3, alias: 'ws1-model', status: 'running', node_names: [], live: true, inflight: 0 },
        { deployment_id: 'old', max_concurrency: null, alias: 'old-model', status: 'stopped', node_names: [], live: false, inflight: 0 },
      ],
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.includes('/api/v1/deployments')) {
        return json({ items: [wire('ws1', 'ws1-model', 'RTX Pro 6000')] })
      }
      if (path.includes('/api/v1/model-routing-policies')) {
        return json({ items: [policy] })
      }
      return base(input as RequestInfo, init)
    }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    // A policy survives an outage of the other members: the cluster still
    // renders from the saved policy, sized by configured membership.
    await screen.findByText('shared-name', { selector: '.dashboard-cluster-heading strong' })
    expect(screen.getByText('2 instances share this request id')).toBeInTheDocument()
    expect(screen.getByText('old-model')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Clear' })).toBeEnabled()
  })

  it('renders a saved cap outside the preset choices as selected', async () => {
    const base = stubDashboardFetch({ active_requests: {} })
    const policy = {
      model: 'shared-name',
      members: [
        { deployment_id: 'ws1', max_concurrency: 100, alias: 'ws1-model', status: 'running', node_names: [], live: true, inflight: 0 },
        { deployment_id: 'dgx', max_concurrency: null, alias: 'dgx-model', status: 'running', node_names: [], live: true, inflight: 0 },
      ],
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.includes('/api/v1/deployments')) {
        return json({ items: [wire('dgx', 'dgx-model', 'DGX Spark'), wire('ws1', 'ws1-model', 'RTX Pro 6000')] })
      }
      if (path.includes('/api/v1/model-routing-policies')) {
        return json({ items: [policy] })
      }
      return base(input as RequestInfo, init)
    }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    await screen.findByText('2 instances share this request id')
    const limit = screen.getByLabelText<HTMLSelectElement>('Max concurrent requests for ws1-model')
    await waitFor(() => expect(limit).toHaveValue('100'))
    expect(screen.getByRole('option', { name: '100' })).toBeInTheDocument()
  })

  it('locks the editor while routing policies are loading', async () => {
    const base = stubDashboardFetch({ active_requests: {} })
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.includes('/api/v1/deployments')) {
        return json({ items: [wire('dgx', 'dgx-model', 'DGX Spark'), wire('ws1', 'ws1-model', 'RTX Pro 6000')] })
      }
      if (path.includes('/api/v1/model-routing-policies')) {
        // Never resolves: the saved policy stays unknown.
        return new Promise<Response>(() => {})
      }
      return base(input as RequestInfo, init)
    }))
    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    await screen.findByText('2 instances share this request id')
    expect(screen.getByText('Loading routing policies…')).toBeInTheDocument()
    expect(screen.getByLabelText('Priority for ws1-model')).toBeDisabled()
    expect(screen.getByLabelText('Max concurrent requests for ws1-model')).toBeDisabled()
    // Even a dispatched edit cannot arm Save while the policy load is pending.
    fireEvent.change(screen.getByLabelText('Max concurrent requests for ws1-model'), { target: { value: '3' } })
    expect(screen.getByRole('button', { name: 'Save routing' })).toBeDisabled()
  })
})
