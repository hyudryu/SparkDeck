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
          models: [
            { model_id: 'org/big', size_bytes: 400 * gib, revision: 'main', file_count: 4 },
            {
              model_id: 'org/comfyui', size_bytes: 200 * gib,
              externally_managed: true,
            },
          ],
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
    expect(screen.queryByText('200 GB in ComfyUI')).not.toBeInTheDocument()
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

  it('shows the quantizations cached on each node', async () => {
    const storage: StorageState = {
      ...enabledStorage,
      nodes: enabledStorage.nodes.map((node) => node.id === 'node-a'
        ? {
            ...node,
            models: [{
              model_id: 'unsloth/Qwen3.6-35B-A3B-MTP-GGUF',
              size_bytes: 68_000_000_000,
              quantizations: ['Q6_K_XL', 'Q8_0'],
            }],
          }
        : node),
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(json(storage)))
    render(<StoragePage />)

    const nodePanel = await screen.findByRole('region', { name: 'Storage on Studio Spark' })
    expect(within(nodePanel).getByText(/Q6_K_XL, Q8_0/)).toBeInTheDocument()
  })

  it('pins newest in-progress tasks above models sorted by descending size', async () => {
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
      'org/aardvark', 'org/alpha', 'org/zeta', 'org/model', 'org/beta',
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

  it('stops polling optimistic jobs after Virtual NAS is disabled', async () => {
    let enabled = true
    const storage = { ...enabledStorage, jobs: [] }
    const queuedJob = {
      id: 'optimistic-job', kind: 'transfer' as const, model_id: 'org/model',
      source_node_id: 'node-a', source_node_name: 'Studio Spark',
      target_node_id: 'node-b', target_node_name: 'Backup Spark', status: 'queued',
      bytes_total: 1_000_000_000, bytes_transferred: 0, progress: 0,
      created_at: '2026-08-29T12:00:00Z',
    }
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.endsWith('/api/v1/storage/transfers') && init?.method === 'POST') {
        return json({ job_ids: [queuedJob.id], jobs: [queuedJob] }, 202)
      }
      if (path.endsWith('/api/v1/storage/settings') && init?.method === 'PUT') enabled = false
      return json(enabled ? storage : { enabled: false, nodes: [], jobs: [], instructions: [] })
    })
    const setTimeoutSpy = vi.spyOn(window, 'setTimeout')
    const clearTimeoutSpy = vi.spyOn(window, 'clearTimeout')
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<StoragePage />)

    await user.click(await screen.findByRole('checkbox', { name: /Backup Spark/ }))
    await user.click(screen.getByRole('button', { name: 'Queue transfer' }))
    await waitFor(() => expect(setTimeoutSpy).toHaveBeenCalled())
    await user.click(screen.getByRole('button', { name: 'Disable Virtual NAS' }))

    expect(await screen.findByRole('heading', { name: 'Virtual NAS is off' })).toBeInTheDocument()
    await waitFor(() => expect(clearTimeoutSpy).toHaveBeenCalled())
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
              {
                model_id: 'Lightricks/LTX-2.5', size_bytes: 68_000_000_000,
                partial: false, revision: 'ComfyUI', source: 'ComfyUI',
                externally_managed: true,
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
    const external = screen.getByLabelText('Installed weights Lightricks/LTX-2.5 on Studio Spark')
    expect(external).toHaveTextContent('Lightricks/LTX-2.5')
    expect(external).not.toHaveTextContent('Externally managed')
    expect(external).toHaveAttribute('draggable', 'true')
    expect(screen.getByRole('button', { name: 'Delete Lightricks/LTX-2.5 from Studio Spark' })).toBeInTheDocument()
    expect(await screen.findByRole('option', { name: 'Lightricks/LTX-2.5' })).toBeInTheDocument()
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
      if (path.endsWith('/api/v1/storage/transfers') && init?.method === 'POST') return json({
        job_ids: ['new-job'],
        jobs: [{
          id: 'new-job', kind: 'transfer', model_id: 'org/model',
          source_node_id: 'node-a', source_node_name: 'Studio Spark',
          target_node_id: 'node-b', target_node_name: 'Backup Spark', status: 'queued',
          bytes_total: 1_000_000_000, bytes_transferred: 0, progress: 0,
          created_at: '2026-08-29T12:00:00Z',
        }],
      }, 202)
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
    const queuedCopy = await within(target).findByLabelText('Transfer queued org/model on Backup Spark')
    expect(queuedCopy).toHaveClass('storage-active-weight')
    expect(queuedCopy).toHaveStyle({ '--storage-active-progress': '0%' })
    expect(within(target).getAllByRole('listitem')[0]).toBe(queuedCopy)

    await user.click(screen.getAllByRole('button', { name: 'Cancel org/other transfer' })[0])
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/storage/transfers/job-1',
      expect.objectContaining({ method: 'DELETE' }),
    ))
    expect(screen.getByRole('progressbar', { name: 'Transfer org/other progress' })).toHaveAttribute('aria-valuenow', '40')
  })

  it('warns that deleting externally managed weights removes them from ComfyUI', async () => {
    const storage: StorageState = {
      ...enabledStorage,
      nodes: enabledStorage.nodes.map((node) => node.id === 'node-a'
        ? {
            ...node,
            models: [
              ...node.models,
              {
                model_id: 'Comfy-Org/MiniMax-Music-3', size_bytes: 21_000_000_000,
                partial: false, revision: 'ComfyUI', source: 'ComfyUI',
                externally_managed: true,
              },
            ],
          }
        : node),
    }
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (_input, init) => {
      if (init?.method === 'DELETE') return new Response(null, { status: 204 })
      return json(storage)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<StoragePage />)

    const external = await screen.findByLabelText('Installed weights Comfy-Org/MiniMax-Music-3 on Studio Spark')
    expect(external).toHaveAttribute('draggable', 'true')

    await user.click(screen.getByRole('button', { name: 'Delete Comfy-Org/MiniMax-Music-3 from Studio Spark' }))
    const dialog = await screen.findByRole('dialog', { name: 'Delete Comfy-Org/MiniMax-Music-3?' })
    expect(dialog).toHaveTextContent("The files will be deleted from ComfyUI's model folders")
    await user.click(within(dialog).getByRole('button', { name: 'Delete weights' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/storage/nodes/node-a/models/Comfy-Org%2FMiniMax-Music-3',
      expect.objectContaining({ method: 'DELETE' }),
    ))

    // Managed weights keep the original confirmation copy.
    await user.click(screen.getByRole('button', { name: 'Delete org/model from Studio Spark' }))
    const managedDialog = await screen.findByRole('dialog', { name: 'Delete org/model?' })
    expect(managedDialog).not.toHaveTextContent('ComfyUI')
    await user.click(within(managedDialog).getByRole('button', { name: 'Cancel' }))
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
          bytes_per_second: 2_500_000_000,
        },
        {
          id: 'transfer-1', model_id: 'org/copy', source_node_id: 'node-a', source_node_name: 'Studio Spark',
          target_node_id: 'node-b', target_node_name: 'Backup Spark', status: 'running', kind: 'transfer',
          bytes_total: 1000, bytes_transferred: 400, progress: 0.4, created_at: '2026-08-26T12:00:00Z',
          bytes_per_second: 1_250_000_000,
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
    expect(download).toHaveStyle({ '--storage-active-progress': '97.5%' })
    expect(within(download).getByText(/2\.50 GB\/s/)).toBeInTheDocument()
    expect(download).not.toHaveTextContent('avg')
    expect(within(nodePanel).getAllByText('org/download')).toHaveLength(1)
    expect(within(nodePanel).queryByText('No model weights reported')).not.toBeInTheDocument()
    const copy = within(nodePanel).getByLabelText('Transferring from Studio Spark org/copy on Backup Spark')
    expect(within(copy).getByRole('progressbar', { name: 'Transferring from Studio Spark org/copy progress' })).toHaveAttribute('aria-valuenow', '40')
    expect(within(copy).getByText(/1\.25 GB\/s/)).toBeInTheDocument()
    expect(copy).not.toHaveTextContent('avg')
    expect(within(nodePanel).getByRole('button', { name: 'Cancel org/copy transfer' })).toBeInTheDocument()
    expect(within(nodePanel).queryByRole('button', { name: 'Cancel org/download download' })).not.toBeInTheDocument()
  })

  it('keeps completed bytes visible while the receiver syncs and registers the cache', async () => {
    const storage: StorageState = {
      ...enabledStorage,
      jobs: [{
        id: 'finalizing-transfer', model_id: 'org/finalizing', source_node_id: 'node-a', source_node_name: 'Studio Spark',
        target_node_id: 'node-b', target_node_name: 'Backup Spark', status: 'running', kind: 'transfer',
        bytes_total: 1000, bytes_transferred: 1000, progress: 1, phase: 'syncing', created_at: '2026-08-29T12:00:00Z',
      }],
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(json(storage)))
    render(<StoragePage />)

    const nodePanel = await screen.findByRole('region', { name: 'Storage on Backup Spark' })
    const finalizing = within(nodePanel).getByLabelText('Finalizing on Backup Spark org/finalizing on Backup Spark')
    expect(within(finalizing).getByText('Syncing model files to disk')).toBeInTheDocument()
    const phaseBar = within(finalizing).getByRole('progressbar', { name: 'Syncing model files to disk progress' })
    expect(phaseBar).toHaveAttribute('aria-valuetext', 'Syncing model files to disk')
    expect(within(finalizing).getByText(/^100%/)).toBeInTheDocument()
  })

  it('does not call a reported copy phase finalization just because tar bytes exceed model bytes', async () => {
    const storage: StorageState = {
      ...enabledStorage,
      jobs: [{
        id: 'receiving-transfer', model_id: 'org/receiving', source_node_id: 'node-a', source_node_name: 'Studio Spark',
        target_node_id: 'node-b', target_node_name: 'Backup Spark', status: 'running', kind: 'transfer',
        bytes_total: 1000, bytes_transferred: 1024, progress: 1, phase: 'receiving', created_at: '2026-08-29T12:00:00Z',
      }],
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(json(storage)))
    render(<StoragePage />)

    const nodePanel = await screen.findByRole('region', { name: 'Storage on Backup Spark' })
    expect(within(nodePanel).getByLabelText('Transferring from Studio Spark org/receiving on Backup Spark')).toBeInTheDocument()
    expect(within(nodePanel).queryByText('Finalizing model cache')).not.toBeInTheDocument()
  })

  it('filters models across every node from one global search field', async () => {
    const storage: StorageState = {
      ...enabledStorage,
      nodes: [
        {
          id: 'node-a', name: 'Studio Spark', online: true, total_size: 3_000_000_000,
          models: [
            { model_id: 'Qwen/Qwen3.8-Flash-Next-FP8', size_bytes: 2_000_000_000 },
            { model_id: 'IBM/granite', size_bytes: 1_000_000_000 },
          ],
        },
        {
          id: 'node-b', name: 'Backup Spark', online: true, total_size: 3_000_000_000,
          models: [
            { model_id: 'community/qwen-small', size_bytes: 800_000_000 },
            { model_id: 'team/unrelated', size_bytes: 700_000_000 },
          ],
        },
      ],
      jobs: [],
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(json(storage)))
    const user = userEvent.setup()
    render(<StoragePage />)

    await user.type(
      await screen.findByRole('searchbox', { name: 'Search models across all storage nodes' }),
      'QWEN',
    )

    const studio = screen.getByRole('region', { name: 'Storage on Studio Spark' })
    const backup = screen.getByRole('region', { name: 'Storage on Backup Spark' })
    expect(within(studio).getByText('Qwen/Qwen3.8-Flash-Next-FP8')).toBeInTheDocument()
    expect(within(studio).queryByText('IBM/granite')).not.toBeInTheDocument()
    expect(within(backup).getByText('community/qwen-small')).toBeInTheDocument()
    expect(within(backup).queryByText('team/unrelated')).not.toBeInTheDocument()

    const inventory = screen.getByRole('table', { name: 'Model storage inventory' })
    expect(within(inventory).getByText('Qwen/Qwen3.8-Flash-Next-FP8')).toBeInTheDocument()
    expect(within(inventory).getByText('community/qwen-small')).toBeInTheDocument()
    expect(within(inventory).queryByText('IBM/granite')).not.toBeInTheDocument()

    await user.clear(screen.getByRole('searchbox', { name: 'Search models across all storage nodes' }))
    await user.type(screen.getByRole('searchbox', { name: 'Search models across all storage nodes' }), 'ibm')
    expect(within(studio).getByText('IBM/granite')).toBeInTheDocument()
  })

  it('uses MB/s for sub-gigabyte transfer rates and truncates failed-job errors with a tooltip', async () => {
    const error = 'gx10-node-3 agent error: HTTP 409: download finished without a complete requested revision'
    const storage: StorageState = {
      ...enabledStorage,
      jobs: [{
        id: 'failed-transfer', model_id: 'org/failed', source_node_id: 'node-a', source_node_name: 'Studio Spark',
        target_node_id: 'node-b', target_node_name: 'Backup Spark', status: 'failed', kind: 'transfer',
        bytes_total: 13_000_000_000, bytes_transferred: 354_000_000, bytes_per_second: 40_000_000,
        created_at: '2026-08-29T12:00:00Z', error,
      }],
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(json(storage)))
    render(<StoragePage />)

    expect(await screen.findByText(/40\.0 MB\/s/)).toBeInTheDocument()
    expect(screen.queryByText(/MB\/s avg/)).not.toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveAttribute('title', error)
  })

  it('hides dashboard-hidden nodes from cards, the transfer form, and availability', async () => {
    const storage: StorageState = {
      ...enabledStorage,
      nodes: [
        {
          id: 'node-a', name: 'Studio Spark', online: true, total_size: 2_000_000_000,
          models: [{ model_id: 'org/model', size_bytes: 1_000_000_000, revision: 'main', file_count: 4 }],
        },
        {
          id: 'node-h', name: 'Secret Spark', online: true, total_size: 8_000_000_000,
          hidden_from_dashboard: true,
          models: [{ model_id: 'org/hidden-model', size_bytes: 5_000_000_000 }],
        },
      ],
      jobs: [],
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(json(storage)))
    render(<StoragePage />)

    expect(await screen.findByRole('region', { name: 'Storage on Studio Spark' })).toBeInTheDocument()
    expect(screen.queryByRole('region', { name: 'Storage on Secret Spark' })).not.toBeInTheDocument()

    const sourceSelect = screen.getByRole('combobox', { name: 'Source node' })
    expect(within(sourceSelect).queryByRole('option', { name: /Secret Spark/ })).not.toBeInTheDocument()
    expect(screen.queryByRole('checkbox', { name: /Secret Spark/ })).not.toBeInTheDocument()
    expect(await screen.findByText('No other nodes are available.')).toBeInTheDocument()

    // Weights stored only on hidden nodes never enter the inventory, and the
    // availability matrix lists visible nodes only.
    expect(screen.queryByText('org/hidden-model')).not.toBeInTheDocument()
    const inventory = screen.getByRole('table', { name: 'Model storage inventory' })
    expect(within(inventory).getByText('org/model')).toBeInTheDocument()
    expect(within(inventory).queryByText(/Secret Spark/)).not.toBeInTheDocument()
  })

  it('shows only the five newest transfer tasks above model inventory', async () => {
    const createdOrder = [5, 1, 7, 2, 6, 3, 4]
    const storage: StorageState = {
      ...enabledStorage,
      jobs: createdOrder.map((number) => ({
        id: `job-${number}`, model_id: `org/job-${number}`,
        source_node_id: 'node-a', source_node_name: 'Studio Spark',
        target_node_id: 'node-b', target_node_name: 'Backup Spark', status: 'completed',
        bytes_total: 1_000_000_000, bytes_transferred: 1_000_000_000,
        created_at: `2026-08-${20 + number}T12:00:00Z`,
      })),
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(json(storage)))
    render(<StoragePage />)

    const queue = await screen.findByRole('table', { name: 'Model transfer queue' })
    expect([...queue.querySelectorAll('.table-row:not(.table-header) [data-label="Model"] strong')].map((item) => item.textContent)).toEqual([
      'org/job-7', 'org/job-6', 'org/job-5', 'org/job-4', 'org/job-3',
    ])
    expect(createdOrder).toEqual([5, 1, 7, 2, 6, 3, 4])
    const queueHeading = screen.getByRole('heading', { name: 'Transfer queue' })
    const inventoryHeading = screen.getByRole('heading', { name: 'Model inventory' })
    expect(queueHeading.compareDocumentPosition(inventoryHeading) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('renders expected totals for partial caches and titles on truncated names', async () => {
    const gib = 1024 ** 3
    const storage: StorageState = {
      ...enabledStorage,
      nodes: [
        {
          id: 'node-a', name: 'Studio Spark', online: true, total_size: 50 * gib,
          models: [
            {
              model_id: 'org/partial-model', size_bytes: 12 * gib,
              partial: true, expected_size_bytes: 40 * gib, revision: 'main',
            },
            { model_id: 'org/plain-partial', size_bytes: 1 * gib, partial: true },
          ],
        },
      ],
      jobs: [],
    }
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(json(storage)))
    render(<StoragePage />)

    const partial = await screen.findByLabelText('Partial cache org/partial-model on Studio Spark')
    expect(partial).toHaveTextContent('12 GB of 40 GB')
    const plain = screen.getByLabelText('Partial cache org/plain-partial on Studio Spark')
    expect(plain).toHaveTextContent('1.0 GB')
    expect(plain).not.toHaveTextContent(' of ')

    expect(within(partial).getByText('org/partial-model')).toHaveAttribute('title', 'org/partial-model')
    expect(screen.getByRole('heading', { name: 'Studio Spark' })).toHaveAttribute('title', 'Studio Spark')

    const inventory = screen.getByRole('table', { name: 'Model storage inventory' })
    const row = within(inventory).getByText('org/partial-model').closest('.table-row') as HTMLElement
    expect(within(row).getByText('12 GB of 40 GB')).toBeInTheDocument()
    expect(within(row).getByText('org/partial-model')).toHaveAttribute('title', 'org/partial-model')
  })
})
