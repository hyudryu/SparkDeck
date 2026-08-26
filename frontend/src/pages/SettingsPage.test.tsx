import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { SettingsPage } from './SettingsPage'
import { THEME_STORAGE_KEY } from '../theme'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  localStorage.clear()
  delete document.documentElement.dataset.theme
})

describe('theme persistence', () => {
  it('restores the saved theme and only persists a new selection after save', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.includes('system-update')) return new Response(JSON.stringify({ can_update: false, blockers: ['No published GitHub release is available'], nodes: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      return new Response(JSON.stringify(init?.method === 'PUT' ? {
        theme: 'light', default_runtime: 'vllm', default_context_length: 8192, community_api_url: '',
      } : {
        theme: 'dark',
        default_runtime: 'vllm',
        default_context_length: 8192,
        community_api_url: '',
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()

    render(<MemoryRouter><SettingsPage /></MemoryRouter>)

    const appearance = await screen.findByRole('combobox', { name: 'Appearance' })
    const save = screen.getByRole('button', { name: 'Save settings' })
    await waitFor(() => expect(appearance).toHaveValue('dark'))
    expect(save).toBeDisabled()
    expect(document.documentElement.dataset.theme).toBe('dark')
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark')

    await user.selectOptions(appearance, 'light')
    expect(save).toBeEnabled()
    expect(document.documentElement.dataset.theme).toBe('light')
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark')

    await user.click(save)
    await screen.findByText('Saved')
    expect(save).toBeDisabled()
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('light')
    expect(document.documentElement.dataset.theme).toBe('light')
    expect(fetchMock).toHaveBeenLastCalledWith('/api/v1/settings', expect.objectContaining({
      method: 'PUT',
      body: expect.stringContaining('"theme":"light"'),
    }))
  })

  it('enables save only while an editable value differs from the loaded settings', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => new Response(JSON.stringify(
      String(input).includes('system-update')
        ? { can_update: false, blockers: [], nodes: [] }
        : { theme: 'system', default_runtime: 'vllm', default_context_length: 8192, community_api_url: '' },
    ), { status: 200, headers: { 'Content-Type': 'application/json' } })))
    const user = userEvent.setup()

    render(<MemoryRouter><SettingsPage /></MemoryRouter>)

    const save = await screen.findByRole('button', { name: 'Save settings' })
    const contextLength = screen.getByRole('spinbutton', { name: /Default context length/ })
    expect(save).toBeDisabled()

    await user.clear(contextLength)
    await user.type(contextLength, '16384')
    await waitFor(() => expect(save).toBeEnabled())

    await user.clear(contextLength)
    await user.type(contextLength, '8192')
    expect(save).toBeDisabled()
  })

  it('keeps save enabled when the update fails', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      if (String(input).includes('system-update')) return new Response(JSON.stringify({ can_update: false, blockers: [], nodes: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (init?.method === 'PUT') return new Response(JSON.stringify({ detail: 'save failed' }), {
        status: 500, headers: { 'Content-Type': 'application/json' },
      })
      return new Response(JSON.stringify({
        theme: 'system',
        default_runtime: 'vllm',
        default_context_length: 8192,
        community_api_url: '',
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()

    render(<MemoryRouter><SettingsPage /></MemoryRouter>)

    const runtime = await screen.findByRole('combobox', { name: 'Default runtime' })
    const save = screen.getByRole('button', { name: 'Save settings' })
    await user.selectOptions(runtime, 'sglang')
    await user.click(save)

    expect(await screen.findByRole('alert')).toHaveTextContent('save failed')
    expect(save).toBeEnabled()
  })

  it('confirms and starts one cluster-wide release update', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.includes('system-update')) return new Response(JSON.stringify(init?.method === 'POST' ? {
        id: 'job-1', active: true, phase: 'preflight', target_tag: 'v1.0.0', target_revision: 'b'.repeat(40), nodes: [],
      } : {
        repository: 'hyudryu/SparkDeck', current_revision: 'a'.repeat(40),
        latest_release: { tag: 'v1.0.0', revision: 'b'.repeat(40), name: 'v1.0.0' },
        can_update: true, blockers: [], nodes: [{ id: 'local', name: 'Controller', local: true, online: true, blockers: [] }],
      }), { status: init?.method === 'POST' ? 202 : 200, headers: { 'Content-Type': 'application/json' } })
      return new Response(JSON.stringify({ theme: 'system', default_runtime: 'vllm', default_context_length: 8192 }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    vi.stubGlobal('fetch', fetchMock)
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    const user = userEvent.setup()
    render(<MemoryRouter><SettingsPage /></MemoryRouter>)

    await user.click(await screen.findByRole('button', { name: 'Update all nodes' }))

    expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining('controller restarts last'))
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/system-update', expect.objectContaining({
      method: 'POST', body: JSON.stringify({ confirm: 'update-entire-cluster' }),
    }))
  })
})
