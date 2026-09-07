import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import { LogsPage } from './LogsPage'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('LogsPage', () => {
  it('shows lifecycle events and all error severities while hiding routine output', async () => {
    vi.spyOn(api.logs, 'list').mockResolvedValue([
      { event: 'launched', level: 'info', message: 'Model launched' },
      { event: 'stopped', level: 'info', message: 'Model shut down' },
      { event: 'crashed', level: 'error', message: 'Model crashed' },
      { level: 'ERROR', message: 'Download failed' },
      { level: 'critical', message: 'Disk unavailable' },
      { level: 'fatal', message: 'Worker terminated' },
      { level: 'info', message: 'GET /api/state 200 OK' },
      { level: 'debug', message: 'Health check tick' },
      { level: 'warning', message: 'Retrying request' },
      { level: 'info', message: 'The error counter is zero' },
    ])
    render(<LogsPage />)
    expect(await screen.findByText('Model launched')).toBeInTheDocument()
    for (const message of ['Model shut down', 'Model crashed', 'Download failed', 'Disk unavailable', 'Worker terminated']) {
      expect(screen.getByText(message)).toBeInTheDocument()
    }
    for (const message of ['GET /api/state 200 OK', 'Health check tick', 'Retrying request', 'The error counter is zero']) {
      expect(screen.queryByText(message)).not.toBeInTheDocument()
    }

    fireEvent.change(screen.getByRole('combobox', { name: 'Event type' }), { target: { value: 'error' } })
    expect(screen.queryByText('Model launched')).not.toBeInTheDocument()
    expect(screen.getByText('Disk unavailable')).toBeInTheDocument()
    expect(screen.getByText('Model crashed')).toBeInTheDocument()

    fireEvent.change(screen.getByRole('combobox', { name: 'Event type' }), { target: { value: 'lifecycle' } })
    expect(screen.getByText('Model launched')).toBeInTheDocument()
    expect(screen.queryByText('Download failed')).not.toBeInTheDocument()
    fireEvent.change(screen.getByRole('textbox', { name: 'Filter logs' }), { target: { value: 'shut down' } })
    expect(screen.getByText('Model shut down')).toBeInTheDocument()
    expect(screen.queryByText('Model launched')).not.toBeInTheDocument()
  })

  it('exports only the displayed events', async () => {
    vi.spyOn(api.logs, 'list').mockResolvedValue([
      { event: 'launched', level: 'info', message: 'Model launched' },
      { level: 'info', message: 'Routine request' },
    ])
    const blobs: Blob[] = []
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, writable: true, value: vi.fn((blob: Blob) => { blobs.push(blob); return 'blob:logs' }) })
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, writable: true, value: vi.fn() })
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
    render(<LogsPage />)
    await screen.findByText('Model launched')
    fireEvent.click(screen.getByRole('button', { name: 'Export' }))
    const reader = new FileReader()
    reader.readAsText(blobs[0])
    await waitFor(() => expect(reader.result).toBe('info  Model launched'))
  })
})
