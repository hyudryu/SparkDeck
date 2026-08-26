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
        hf_token: '',
        hf_token_configured: false,
        community_api_url: '',
      }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        theme: 'light',
        hf_token: '',
        hf_token_configured: false,
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

  it('keeps a configured Hugging Face key masked and saves a replacement write-only', async () => {
    const fetchMock = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(new Response(JSON.stringify({
        theme: 'system', hf_token: '', hf_token_configured: true, community_api_url: '',
      }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        theme: 'system', hf_token: '', hf_token_configured: true, community_api_url: '',
      }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()

    render(<MemoryRouter><SettingsPage /></MemoryRouter>)

    const save = await screen.findByRole('button', { name: 'Save settings' })
    const credential = screen.getByLabelText('Hugging Face API key')
    expect(save).toBeDisabled()
    expect(credential).toHaveValue('')
    expect(screen.getByText('Configured')).toBeInTheDocument()
    expect(screen.queryByText('Default runtime')).not.toBeInTheDocument()
    expect(screen.queryByText('Default context length')).not.toBeInTheDocument()

    await user.type(credential, 'hf_replacement_secret')
    expect(save).toBeEnabled()
    expect(screen.queryByText('hf_replacement_secret')).not.toBeInTheDocument()
    await user.click(save)

    await screen.findByText('Saved')
    const request = fetchMock.mock.calls.at(-1)
    expect(request?.[0]).toBe('/api/v1/settings')
    expect(JSON.parse(String(request?.[1]?.body))).toEqual(expect.objectContaining({
      hf_token: 'hf_replacement_secret',
    }))
    expect(credential).toHaveValue('')
    expect(save).toBeDisabled()
  })

  it('keeps save enabled when the update fails', async () => {
    const fetchMock = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(new Response(JSON.stringify({
        theme: 'system',
        hf_token: '',
        hf_token_configured: false,
        community_api_url: '',
      }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'save failed' }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }))
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()

    render(<MemoryRouter><SettingsPage /></MemoryRouter>)

    const credential = await screen.findByLabelText('Hugging Face API key')
    const save = screen.getByRole('button', { name: 'Save settings' })
    await user.type(credential, 'hf_retry_secret')
    await user.click(save)

    expect(await screen.findByRole('alert')).toHaveTextContent('save failed')
    expect(credential).toHaveValue('hf_retry_secret')
    expect(save).toBeEnabled()
  })
})
