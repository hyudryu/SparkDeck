import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import { LogsPage } from './LogsPage'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.useRealTimers()
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

  it('lets slow requests settle before scheduling another poll', async () => {
    vi.useFakeTimers()
    let finish!: (value: Awaited<ReturnType<typeof api.logs.list>>) => void
    let signal!: AbortSignal
    const list = vi.spyOn(api.logs, 'list').mockImplementation((requestSignal) => {
      signal = requestSignal!
      return new Promise((resolve) => { finish = resolve })
    })
    render(<LogsPage />)
    await act(() => vi.advanceTimersByTimeAsync(15000))
    expect(list).toHaveBeenCalledTimes(1)
    expect(signal.aborted).toBe(false)
    await act(async () => finish([{ event: 'launched', message: 'Slow response' }]))
    expect(screen.getByText('Slow response')).toBeInTheDocument()
    await act(() => vi.advanceTimersByTimeAsync(5000))
    expect(list).toHaveBeenCalledTimes(2)
    await act(() => vi.advanceTimersByTimeAsync(15000))
    expect(list).toHaveBeenCalledTimes(2)
    expect(signal.aborted).toBe(false)
  })

  it('exports only the displayed events', async () => {
    vi.spyOn(api.logs, 'list').mockResolvedValue([
      { event: 'launched', level: 'info', message: 'Model launched' },
      { level: 'error', message: 'POST /api/deployments 400', details: { status: 400, reason: 'Bad Request', response: { detail: 'Not enough memory' } } },
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
    await waitFor(() => expect(typeof reader.result).toBe('string'))
    expect(JSON.parse(reader.result as string)).toEqual([
      { event: 'launched', level: 'info', message: 'Model launched' },
      { level: 'error', message: 'POST /api/deployments 400', details: { status: 400, reason: 'Bad Request', response: { detail: 'Not enough memory' } } },
    ])
    expect(blobs[0].type).toBe('application/json')
  })

  it('displays and searches structured response details', async () => {
    const details = { status: 400, reason: 'Bad Request', response: { detail: 'Missing model weights', node: 'node-3' } }
    vi.spyOn(api.logs, 'list').mockResolvedValue([
      { level: 'error', message: 'Deployment request failed', details },
      { event: 'launched', message: 'Another model launched' },
    ])
    render(<LogsPage />)
    const summary = await screen.findByText('Error details (JSON)')
    fireEvent.click(summary)
    expect(summary.parentElement).toHaveAttribute('open')
    expect(summary.parentElement?.querySelector('pre')?.textContent).toBe(JSON.stringify(details, null, 2))
    fireEvent.change(screen.getByRole('textbox', { name: 'Filter logs' }), { target: { value: 'node-3' } })
    expect(screen.getByText('Deployment request failed')).toBeInTheDocument()
    expect(screen.queryByText('Another model launched')).not.toBeInTheDocument()
  })
})
