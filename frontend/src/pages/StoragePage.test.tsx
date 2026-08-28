import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
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

  it('reports node capacity as model usage plus real free space, not raw disk size', async () => {
    const gib = 1024 ** 3
    const storage: StorageState = {
      ...enabledStorage,
      nodes: [
        {
          id: 'node-e', name: 'Tight Spark', online: true, total_size: 900 * gib, free_size: 100 * gib,
          models: [{ model_id: 'org/big', size_bytes: 400 * gib, revision: 'main', file_count: 4 }],
        },
        {
          id: 'node-f', name: 'Full Spark', online: true, total_size: 900 * gib, free_size: 0,
          models: [{ model_id: 'org/full', size_bytes: 100 * gib }],
        },
        {
          id: 'node-d', name: 'Cold Spark', online: false, total_size: 3_000_000_000, free_size: 50 * gib,
          models: [{ model_id: 'org/offline-model', size_bytes: 500_000_000 }],
        },
      ],
    }
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async () => json(storage))
    vi.stubGlobal('fetch', fetchMock)
    render(<StoragePage />)

    expect(await screen.findByText('400 GB used')).toBeInTheDocument()
    expect(screen.getByText('100 GB free')).toBeInTheDocument()
    expect(screen.getByText('500 GB total')).toBeInTheDocument()
    expect(screen.queryByText('900 GB total')).not.toBeInTheDocument()
    const track = screen.getByLabelText('Tight Spark used model storage')
    expect(track.firstElementChild).toHaveStyle({ width: '80%' })
    // A zero free reading is a genuinely full disk, not missing telemetry.
    expect(screen.getByText('0 B free')).toBeInTheDocument()
    expect(screen.getByText('100 GB total')).toBeInTheDocument()
    const fullTrack = screen.getByLabelText('Full Spark used model storage')
    expect(fullTrack.firstElementChild).toHaveStyle({ width: '100%' })
    // Offline nodes keep the raw disk total because their model inventory
    // cannot be validated.
    expect(screen.getByText('2.8 GB total')).toBeInTheDocument()
    expect(screen.queryByText('50 GB free')).not.toBeInTheDocument()
  })

  it('sorts Virtual NAS models by size in descending order', async () => {
    const storage: StorageState = {
      ...enabledStorage,
      nodes: enabledStorage.nodes.map((node) => node.id === 'node-a'
        ? {
            ...node,
            models: [
              { model_id: 'org/alpha', size_bytes: 300 },
              { model_id: 'org/zeta', size_bytes: 100 },
              { model_id: 'org/model', size_bytes: 200 },
            ],
          }
        : node),
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(json(storage)))
    render(<StoragePage />)

    const nodePanel = await screen.findByRole('region', { name: 'Storage on Studio Spark' })
    expect([...nodePanel.querySelectorAll('.storage-weight-list strong')].map((item) => item.textContent)).toEqual([
      'org/alpha', 'org/model', 'org/zeta',
    ])

    const modelSelect = screen.getByRole('combobox', { name: 'Model weights' })
    await waitFor(() => expect(
      within(modelSelect).getAllByRole('option').map((option) => option.textContent),
    ).toEqual(['Select a model', 'org/alpha', 'org/model', 'org/zeta']))

    const inventory = screen.getByRole('table', { name: 'Model storage inventory' })
    expect([...inventory.querySelectorAll('.table-row:not(.table-header) [data-label="Model"] strong')].map((item) => item.textContent)).toEqual([
      'org/offline-model', 'org/alpha', 'org/model', 'org/zeta',
    ])
    expect(storage.nodes[0].models.map((model) => model.model_id)).toEqual([
      'org/alpha', 'org/zeta', 'org/model',
    ])
  })

  it('keeps in-progress downloads in strict descending size order', async () => {
    const storage: StorageState = {
      ...enabledStorage,
      nodes: enabledStorage.nodes.map((node) => node.id === 'node-a'
        ? {
            ...node,
            models: [
              { model_id: 'org/alpha', size_bytes: 100, partial: true },
              { model_id: 'org/zeta', size_bytes: 300 },
              { model_id: 'org/model', size_bytes: 200 },
              { model_id: 'org/aardvark', size_bytes: 50, partial: true },
              { model_id: 'org/beta', size_bytes: 150 },
            ],
          }
        : node),
      jobs: [
        {
          id: 'download-alpha', kind: 'download', model_id: 'org/alpha',
          source_node_id: 'huggingface', source_node_name: 'Hugging Face',
          target_node_id: 'node-a', target_node_name: 'Studio Spark', status: 'running',
          bytes_total: 100, bytes_transferred: 50, created_at: '2026-08-27T12:00:00Z',
        },
        {
          id: 'transfer-aardvark', kind: 'transfer', model_id: 'org/aardvark',
          source_node_id: 'node-b', source_node_name: 'Backup Spark',
          target_node_id: 'node-a', target_node_name: 'Studio Spark', status: 'running',
          bytes_total: 50, bytes_transferred: 25, created_at: '2026-08-27T12:01:00Z',
        },
        {
          id: 'download-beta', kind: 'download', model_id: 'org/beta',
          source_node_id: 'huggingface', source_node_name: 'Hugging Face',
          target_node_id: 'node-a', target_node_name: 'Studio Spark', status: 'completed',
          bytes_total: 150, bytes_transferred: 150, created_at: '2026-08-27T11:00:00Z',
        },
      ],
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(json(storage)))
    render(<StoragePage />)

    const nodePanel = await screen.findByRole('region', { name: 'Storage on Studio Spark' })
    expect([...nodePanel.querySelectorAll('.storage-weight-list strong')].map((item) => item.textContent)).toEqual([
      'org/zeta', 'org/model', 'org/beta', 'org/alpha', 'org/aardvark',
    ])

    const modelSelect = screen.getByRole('combobox', { name: 'Model weights' })
    await waitFor(() => expect(
      within(modelSelect).getAllByRole('option').map((option) => option.textContent),
    ).toEqual(['Select a model', 'org/zeta', 'org/model', 'org/beta']))

    const inventory = screen.getByRole('table', { name: 'Model storage inventory' })
    expect([...inventory.querySelectorAll('.table-row:not(.table-header) [data-label="Model"] strong')].map((item) => item.textContent)).toEqual([
      'org/offline-model', 'org/zeta', 'org/model', 'org/beta', 'org/alpha', 'org/aardvark',
    ])
    expect(storage.nodes[0].models.map((model) => model.model_id)).toEqual([
      'org/alpha', 'org/zeta', 'org/model', 'org/aardvark', 'org/beta',
    ])
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
              {
                model_id: 'org/mixed-model', size_bytes: 600_000_000,
                partial: false, has_partial_download: true,
                partial_size_bytes: 50_000_000, revision: 'complete-a',
              },
            ],
          }
        : node),
    }
    let finished = false
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.endsWith('/api/v1/storage/nodes/node-a/models/org%2Fpartial-model/download') && init?.method === 'POST') {
        finished = true
        return json({ job_ids: ['download-1'], jobs: [] }, 202)
      }
      return json(finished ? {
        ...storage,
        nodes: storage.nodes.map((node) => node.id === 'node-a' ? {
          ...node,
          models: node.models.map((model) => model.model_id === 'org/partial-model'
            ? { ...model, partial: false, revision: 'release-1' }
            : model),
        } : node),
      } : storage)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<StoragePage />)

    const partial = await screen.findByLabelText('Partial cache org/partial-model on Studio Spark')
    expect(partial).toHaveTextContent('Partial')
    expect(partial).toHaveAttribute('draggable', 'false')
    const mixed = screen.getByLabelText('Transfer org/mixed-model from Studio Spark')
    expect(mixed).not.toHaveTextContent('Incomplete download')
    expect(mixed).toHaveAttribute('draggable', 'true')
    expect(screen.queryByRole('button', { name: 'Finish download of org/mixed-model on Studio Spark' })).not.toBeInTheDocument()
    expect(await screen.findByRole('option', { name: 'org/mixed-model' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Finish download of org/partial-model on Studio Spark' }))
    let dialog = await screen.findByRole('dialog', { name: 'Finish downloading org/partial-model?' })
    await user.click(within(dialog).getByRole('button', { name: 'Cancel' }))
    expect(fetchMock.mock.calls.some(([, init]) => init?.method === 'POST')).toBe(false)

    await user.click(screen.getByRole('button', { name: 'Finish download of org/partial-model on Studio Spark from inventory' }))
    dialog = await screen.findByRole('dialog', { name: 'Finish downloading org/partial-model?' })
    await user.click(within(dialog).getByRole('button', { name: 'Finish download' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/storage/nodes/node-a/models/org%2Fpartial-model/download',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({}) }),
    ))
    expect(await screen.findByRole('status')).toHaveTextContent('Queued org/partial-model to finish downloading on Studio Spark.')
    await waitFor(() => expect(screen.queryByRole('button', { name: /Finish download of org\/partial-model/ })).not.toBeInTheDocument())
    expect(screen.getByRole('option', { name: 'org/partial-model' })).toBeInTheDocument()
  })

  it('supports drag transfer, per-node deletion, and queue cancellation', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (init?.method === 'DELETE') return new Response(null, { status: 204 })
      if (path.endsWith('/api/v1/storage/transfers') && init?.method === 'POST') return json({ id: 'new-job' }, 201)
      return json(enabledStorage)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<StoragePage />)

    await user.click(await screen.findByRole('button', { name: 'Delete org/model from Studio Spark' }))
    const dialog = await screen.findByRole('dialog', { name: 'Delete org/model?' })
    await user.click(within(dialog).getByRole('button', { name: 'Delete weights' }))
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

    await user.click(screen.getAllByRole('button', { name: 'Cancel org/other transfer' })[0])
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/storage/transfers/job-1',
      expect.objectContaining({ method: 'DELETE' }),
    ))
    expect(screen.getByRole('progressbar', { name: 'Transfer org/other progress' })).toHaveAttribute('aria-valuenow', '40')
  })

  it('shows active downloads and transfers on their target NAS cards', async () => {
    const gib = 1024 ** 3
    const storage: StorageState = {
      ...enabledStorage,
      nodes: enabledStorage.nodes.map((node) => node.id === 'node-b'
        ? { ...node, models: [{ model_id: 'org/download', size_bytes: 97.5 * gib, partial: true }] }
        : node),
      jobs: [
        {
          id: 'download-1', model_id: 'org/download', source_node_id: 'huggingface', source_node_name: 'Hugging Face',
          target_node_id: 'node-b', target_node_name: 'Backup Spark', status: 'running', kind: 'download',
          bytes_total: 100 * gib, bytes_transferred: 97.5 * gib, progress: 0.975, created_at: '2026-08-26T12:00:00Z',
        },
        {
          id: 'transfer-1', model_id: 'org/copy', source_node_id: 'node-a', source_node_name: 'Studio Spark',
          target_node_id: 'node-b', target_node_name: 'Backup Spark', status: 'running', kind: 'transfer',
          bytes_total: 1000, bytes_transferred: 400, progress: 0.4, created_at: '2026-08-26T12:00:00Z',
        },
      ],
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(json(storage)))
    render(<StoragePage />)

    const nodePanel = await screen.findByRole('region', { name: 'Storage on Backup Spark' })
    const download = within(nodePanel).getByLabelText('Downloading from Hugging Face org/download on Backup Spark')
    const downloadProgress = within(download).getByRole('progressbar', { name: 'Downloading from Hugging Face org/download progress' })
    expect(downloadProgress).toHaveAttribute('aria-valuenow', '97.5')
    expect(downloadProgress.firstElementChild).toHaveStyle({ transform: 'scaleX(0.975)' })
    expect(within(download).getByText(/^97\.5%/)).toBeInTheDocument()
    expect(within(nodePanel).getAllByText('org/download')).toHaveLength(1)
    expect(within(nodePanel).queryByText('No model weights reported')).not.toBeInTheDocument()
    expect(within(nodePanel).getByRole('progressbar', { name: 'Transferring from Studio Spark org/copy progress' })).toHaveAttribute('aria-valuenow', '40')
    expect(within(nodePanel).getByRole('button', { name: 'Cancel org/copy transfer' })).toBeInTheDocument()
    expect(within(nodePanel).queryByRole('button', { name: 'Cancel org/download download' })).not.toBeInTheDocument()
  })
})
