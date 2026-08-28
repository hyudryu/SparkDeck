import { act, cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { NodeInventoryItem } from '../api/types'
import { clusterResourceSnapshot, DashboardPage } from './DashboardPage'

function json(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers() })

describe('DashboardPage', () => {
  it('renders pooled CPU, GPU, and RAM while excluding hidden nodes', async () => {
    const localStats = {
      cpu_pct: 20, cpu_logical_count: 4, cpu_temp_c: 54, mem: { used: 64 * 1024 ** 3, total: 128 * 1024 ** 3, pct: 50 },
      gpus: [{ index: 0, name: 'NVIDIA GB10', util: 35, temp: 62, mem_used_mib: null, mem_total_mib: null }],
      active_requests: {}, ts: 1_777_000_000,
    }
    const remoteStats = {
      cpu_pct: 60, cpu_logical_count: 12, cpu_temp_c: 58, mem: { used: 32 * 1024 ** 3, total: 64 * 1024 ** 3, pct: 50 },
      gpus: [
        { index: 0, name: 'NVIDIA RTX', util: 55, temp: 65, mem_used_mib: 8 * 1024, mem_total_mib: 16 * 1024 },
        { index: 1, name: 'NVIDIA RTX', util: null, temp: 67, mem_used_mib: 4 * 1024, mem_total_mib: 16 * 1024 },
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

    expect(await screen.findByText('Pooled CPU')).toBeInTheDocument()
    expect(screen.getByText('50.0%')).toBeInTheDocument()
    expect(screen.getByText('16 logical processors · 2 measured nodes')).toBeInTheDocument()
    expect(screen.getByText('Pooled GPU')).toBeInTheDocument()
    expect(screen.getByText('45.0%')).toBeInTheDocument()
    expect(screen.getByText('3 GPUs across 2 nodes · 2 measured')).toBeInTheDocument()
    expect(screen.getByText('Pooled RAM')).toBeInTheDocument()
    expect(screen.getByText('96.0 GB')).toBeInTheDocument()
    expect(screen.getByText('of 192.0 GB across 2 nodes')).toBeInTheDocument()
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

    expect(await screen.findByText('Pooled CPU')).toBeInTheDocument()
    expect(screen.getByText('20.0%')).toBeInTheDocument()
    expect(screen.getByText('64.0 GB')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Cluster nodes' })).toBeInTheDocument()
    expect(screen.getByText(/0 of 0 visible nodes online/)).toBeInTheDocument()
    expect(screen.getByText('Cluster inventory unavailable')).toBeInTheDocument()
    expect(screen.queryByText('No nodes shown on the dashboard')).not.toBeInTheDocument()
    expect(screen.queryByText('node probe failed')).not.toBeInTheDocument()
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

    expect(await screen.findByText('Pooled CPU')).toBeInTheDocument()
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
    expect(screen.getByText('Pooled CPU')).toBeInTheDocument()
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
      if (path.includes('/api/inference-queue')) return json({})
      if (path.includes('/api/v1/deployments')) return json({ items: [] })
      if (path.includes('/api/v1/community/sync')) return json({ consent: false, outbox: {} })
      return json({ items: [] })
    }))

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    await act(async () => { await vi.advanceTimersByTimeAsync(9_999) })
    expect(statsCalls).toBe(1)
    expect(statsSignal?.aborted).toBe(false)

    await act(async () => { await vi.advanceTimersByTimeAsync(1) })
    expect(screen.getByText('The request timed out. Check the node connection and retry.')).toBeInTheDocument()
    expect(screen.queryByText('Loading system overview')).not.toBeInTheDocument()
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

  it('prefers fresh local stats over retained local node telemetry', () => {
    const snapshot = clusterResourceSnapshot([{
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

    expect(snapshot.cpuPct).toBe(80)
    expect(snapshot.ramUsed).toBe(8)
    expect(snapshot.gpuPct).toBe(70)
  })

  it('uses equal node weighting when logical CPU counts are incomplete', () => {
    const snapshot = clusterResourceSnapshot([
      { id: 'older-node', online: true, stats: { cpu_pct: 0 } },
      { id: 'newer-node', online: true, stats: { cpu_pct: 100, cpu_logical_count: 64 } },
    ] as NodeInventoryItem[])

    expect(snapshot.cpuPct).toBe(50)
    expect(snapshot.logicalProcessors).toBeUndefined()
  })
})
