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
    await waitFor(() => expect(appearance).toHaveValue('dark'))
    expect(document.documentElement.dataset.theme).toBe('dark')
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark')

    await user.selectOptions(appearance, 'light')
    expect(document.documentElement.dataset.theme).toBe('light')
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark')

    await user.click(screen.getByRole('button', { name: 'Save settings' }))
    await screen.findByText('Saved')
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('light')
    expect(document.documentElement.dataset.theme).toBe('light')
    expect(fetchMock).toHaveBeenLastCalledWith('/api/v1/settings', expect.objectContaining({
      method: 'PUT',
      body: expect.stringContaining('"theme":"light"'),
    }))
  })
})
