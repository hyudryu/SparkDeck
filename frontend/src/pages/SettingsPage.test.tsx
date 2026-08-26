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
    const fetchMock = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(new Response(JSON.stringify({
        theme: 'dark',
        default_runtime: 'vllm',
        default_context_length: 8192,
        community_api_url: '',
      }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        theme: 'light',
        default_runtime: 'vllm',
        default_context_length: 8192,
        community_api_url: '',
      }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
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
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({
      theme: 'system',
      default_runtime: 'vllm',
      default_context_length: 8192,
      community_api_url: '',
    }), { status: 200, headers: { 'Content-Type': 'application/json' } })))
    const user = userEvent.setup()

    render(<MemoryRouter><SettingsPage /></MemoryRouter>)

    const save = await screen.findByRole('button', { name: 'Save settings' })
    const contextLength = screen.getByRole('spinbutton', { name: /Default context length/ })
    expect(save).toBeDisabled()

    await user.clear(contextLength)
    await user.type(contextLength, '16384')
    expect(save).toBeEnabled()

    await user.clear(contextLength)
    await user.type(contextLength, '8192')
    expect(save).toBeDisabled()
  })

  it('keeps save enabled when the update fails', async () => {
    const fetchMock = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(new Response(JSON.stringify({
        theme: 'system',
        default_runtime: 'vllm',
        default_context_length: 8192,
        community_api_url: '',
      }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'save failed' }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }))
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
})
