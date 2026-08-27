import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { StorageState } from '../api/types'
import { StoragePage } from './StoragePage'

const enabledStorage: StorageState = {
  enabled: true,
  nodes: [
    {
      id: 'node-a', name: 'Studio Spark', online: true, total_size: 2_000_000_000,
      models: [{ model_id: 'org/model', size_bytes: 1_000_000_000, revision: 'main', file_count: 4 }],
    },
    { id: 'node-b', name: 'Backup Spark', online: true, total_size: 3_000_000_000, models: [] },
    {
      id: 'node-c', name: 'Archive Spark', online: true, total_size: 3_000_000_000,
      models: [{ model_id: 'org/model', size_bytes: 1_000_000_000 }],
    },
    {
      id: 'node-d', name: 'Cold Spark', online: false, total_size: 3_000_000_000,
      models: [{ model_id: 'org/offline-model', size_bytes: 500_000_000 }],
    },
  ],
  jobs: [{
    id: 'job-1', model_id: 'org/other', source_node_id: 'node-a', source_node_name: 'Studio Spark',
    target_node_id: 'node-b', target_node_name: 'Backup Spark', status: 'running', bytes_total: 1000,
    bytes_transferred: 400, progress: 0.4, created_at: '2026-08-26T12:00:00Z',
  }],
  instructions: ['Keep both nodes online until the transfer completes.'],
}

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('StoragePage', () => {
  it('enables Virtual NAS from its explanatory disabled state', async () => {
    let enabled = false
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.endsWith('/api/v1/storage/settings') && init?.method === 'PUT') {
        enabled = true
        return json({ ...enabledStorage, enabled })
      }
      return json({ ...enabledStorage, enabled })
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<StoragePage />)

    await user.click(await screen.findByRole('button', { name: 'Enable Virtual NAS' }))

    expect(await screen.findByRole('heading', { name: 'Node storage' })).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/storage/settings', expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({ enabled: true }),
    }))
  })

  it('queues a transfer from the keyboard and touch-friendly form', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.endsWith('/api/v1/storage/transfers') && init?.method === 'POST') return json({ id: 'new-job' }, 201)
      return json(enabledStorage)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<StoragePage />)

    await user.click(await screen.findByRole('checkbox', { name: /Backup Spark/ }))
    expect(screen.getAllByText('954 MB used').length).toBeGreaterThan(0)
    expect(screen.queryByText(/cataloged/i)).not.toBeInTheDocument()
    expect(screen.getByRole('checkbox', { name: /Archive Spark/ })).toBeDisabled()
    expect(screen.getByText('Model already available')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Queue transfer' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/v1/storage/transfers', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ model_id: 'org/model', source_node_id: 'node-a', target_node_ids: ['node-b'] }),
    })))
    expect(await screen.findByRole('status')).toHaveTextContent('Queued org/model for transfer to Backup Spark.')
  })

  it('shows partial caches with a warning and excludes them from transfers', async () => {
    const storage: StorageState = {
      ...enabledStorage,
      nodes: enabledStorage.nodes.map((node) => node.id === 'node-a'
        ? {
            ...node,
            models: [
              ...node.models,
              { model_id: 'org/partial-model', size_bytes: 400_000_000, partial: true },
            ],
          }
        : node),
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(json(storage)))
    render(<StoragePage />)

    const partial = await screen.findByLabelText('Partial cache org/partial-model on Studio Spark')
    expect(partial).toHaveTextContent('Partial')
    expect(partial).toHaveAttribute('draggable', 'false')
    expect(screen.getByLabelText('Partial cache')).toBeInTheDocument()
    expect(screen.queryByRole('option', { name: 'org/partial-model' })).not.toBeInTheDocument()
  })

  it('supports drag transfer, per-node deletion, and queue cancellation', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (init?.method === 'DELETE') return new Response(null, { status: 204 })
      if (path.endsWith('/api/v1/storage/transfers') && init?.method === 'POST') return json({ id: 'new-job' }, 201)
      return json(enabledStorage)
    })
    vi.stubGlobal('fetch', fetchMock)
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    const user = userEvent.setup()
    render(<StoragePage />)

    await user.click(await screen.findByRole('button', { name: 'Delete org/model from Studio Spark' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/storage/nodes/node-a/models/org%2Fmodel',
      expect.objectContaining({ method: 'DELETE' }),
    ))
    expect(screen.getByRole('button', { name: 'Delete org/offline-model from Cold Spark' })).toBeDisabled()

    const dataTransfer = {
      effectAllowed: 'none', dropEffect: 'none', setData: vi.fn(), getData: vi.fn(() => ''),
    } as unknown as DataTransfer
    const weight = screen.getByLabelText('Transfer org/model from Studio Spark')
    const target = screen.getByRole('region', { name: 'Storage on Backup Spark' })
    fireEvent.dragStart(weight, { dataTransfer })
    fireEvent.dragEnter(target, { dataTransfer })
    expect(target).toHaveClass('drop-active')
    expect(screen.getByText('Drop to copy org/model')).toBeInTheDocument()
    fireEvent.drop(target, { dataTransfer })

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/v1/storage/transfers', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ model_id: 'org/model', source_node_id: 'node-a', target_node_ids: ['node-b'] }),
    })))

    await user.click(screen.getByRole('button', { name: 'Cancel' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/storage/transfers/job-1',
      expect.objectContaining({ method: 'DELETE' }),
    ))
    expect(screen.getByRole('progressbar', { name: 'Transfer org/other progress' })).toHaveValue(40)
  })
})
