import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { DashboardPage } from './DashboardPage'

function json(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

describe('DashboardPage', () => {
  it('uses DGX unified memory and renders telemetry for every cluster node', async () => {
    const localStats = {
      cpu_pct: 20, cpu_temp_c: 54, mem: { used: 64 * 1024 ** 3, total: 128 * 1024 ** 3, pct: 50 },
      gpus: [{ index: 0, name: 'NVIDIA GB10', util: 35, temp: 62, mem_used_mib: null, mem_total_mib: null }],
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
        { id: 'node-2', name: 'Spark Two', online: true, stats: { ...localStats, cpu_temp_c: 58 } },
        { id: 'node-3', name: 'Spark Three', online: false },
      ] })
    }))

    render(<MemoryRouter><DashboardPage /></MemoryRouter>)

    expect((await screen.findAllByText('Unified memory')).length).toBeGreaterThanOrEqual(3)
    expect(screen.getByText('64.0 GB')).toBeInTheDocument()
    expect(screen.getByText(/of 128\.0 GB · shared CPU\/GPU pool/)).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Cluster nodes' })).toBeInTheDocument()
    expect(screen.getByText('Spark Four')).toBeInTheDocument()
    expect(screen.getByText('Spark Two')).toBeInTheDocument()
    expect(screen.getByText('Spark Three')).toBeInTheDocument()
    expect(screen.getByText(/2 of 3 nodes online/)).toBeInTheDocument()
    expect(document.querySelector('.community-strip')).toHaveAttribute(
      'title', 'Sign in and enable telemetry in Settings → Community Features to see community data.')
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

    expect(await screen.findByText('54.0°C')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Cluster nodes' })).toBeInTheDocument()
    expect(screen.getByText(/0 of 0 nodes online/)).toBeInTheDocument()
    expect(screen.queryByText('node probe failed')).not.toBeInTheDocument()
  })
})
