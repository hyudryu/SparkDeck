import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { DashboardPage } from './DashboardPage'

function json(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

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
        { index: 1, name: 'NVIDIA RTX', util: 75, temp: 67, mem_used_mib: 4 * 1024, mem_total_mib: 16 * 1024 },
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
    expect(screen.getByText('55.0%')).toBeInTheDocument()
    expect(screen.getByText('3 GPUs across 2 nodes')).toBeInTheDocument()
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
})
