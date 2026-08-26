import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { SettingsPage } from './SettingsPage'
import { THEME_STORAGE_KEY } from '../theme'
import { SPARKDECK_VERSION } from '../buildInfo'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  localStorage.clear()
  delete document.documentElement.dataset.theme
})

describe('settings page', () => {
  it('shows the version embedded when the frontend was built', () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async () => new Response(JSON.stringify({
      theme: 'system', default_runtime: 'vllm', default_context_length: 8192, community_api_url: '',
    }), { status: 200, headers: { 'Content-Type': 'application/json' } })))

    render(<MemoryRouter><SettingsPage /></MemoryRouter>)

    expect(screen.getByLabelText('SparkDeck version')).toHaveTextContent(`Version ${SPARKDECK_VERSION}`)
  })

  it('restores the saved theme and only persists a new selection after save', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.includes('system-update')) return new Response(JSON.stringify({ can_update: false, blockers: ['No published GitHub release is available'], nodes: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      return new Response(JSON.stringify(init?.method === 'PUT' ? {
        theme: 'light', default_runtime: 'vllm', default_context_length: 8192, community_api_url: '',
      } : {
        theme: 'dark',
        hf_token: '',
        hf_token_configured: false,
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
    expect(screen.getByRole('link', { name: 'Open switch setup' })).toHaveAttribute('href', '/switch')

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
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input) => {
      if (String(input).includes('system-update')) return new Response(JSON.stringify({ can_update: false, blockers: [], nodes: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      return new Response(JSON.stringify({
        theme: 'system', hf_token: '', hf_token_configured: true, community_api_url: '',
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()

    render(<MemoryRouter><SettingsPage /></MemoryRouter>)

    const save = await screen.findByRole('button', { name: 'Save settings' })
    const credential = screen.getByLabelText('Hugging Face API key')
    expect(save).toBeDisabled()
    expect(credential).toHaveValue('')
    expect(await screen.findByText('Configured')).toBeInTheDocument()
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
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      if (String(input).includes('system-update')) return new Response(JSON.stringify({ can_update: false, blockers: [], nodes: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (init?.method === 'PUT') return new Response(JSON.stringify({ detail: 'save failed' }), {
        status: 500, headers: { 'Content-Type': 'application/json' },
      })
      return new Response(JSON.stringify({
        theme: 'system',
        hf_token: '',
        hf_token_configured: false,
        community_api_url: '',
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
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

  it('explicitly removes a saved Hugging Face key after confirmation', async () => {
    let configured = true
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      if (String(input).includes('system-update')) return new Response(JSON.stringify({ can_update: false, blockers: [], nodes: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (init?.method === 'DELETE') configured = false
      return new Response(JSON.stringify({
        theme: 'system', hf_token_configured: configured, community_api_url: '',
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    vi.stubGlobal('fetch', fetchMock)
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    const user = userEvent.setup()

    render(<MemoryRouter><SettingsPage /></MemoryRouter>)

    const appearance = await screen.findByRole('combobox', { name: 'Appearance' })
    await user.selectOptions(appearance, 'dark')
    await user.click(await screen.findByRole('button', { name: 'Remove saved key' }))

    expect(window.confirm).toHaveBeenCalledWith(expect.stringMatching(/entire cluster/i))
    await screen.findByText('Not configured')
    expect(appearance).toHaveValue('dark')
    expect(screen.getByRole('button', { name: 'Save settings' })).toBeEnabled()
    expect(screen.queryByRole('button', { name: 'Remove saved key' })).not.toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/settings/hf-token', expect.objectContaining({
      method: 'DELETE',
    }))
  })

  it('confirms and starts one cluster-wide release update', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.includes('system-update')) return new Response(JSON.stringify(init?.method === 'POST' ? {
        id: 'job-1', active: true, phase: 'preflight', target_tag: 'v0.9.0', target_revision: 'b'.repeat(40), nodes: [],
      } : {
        repository: 'hyudryu/SparkDeck', current_revision: 'a'.repeat(40),
        current_release_tag: 'v1.0.0',
        releases: [
          { tag: 'v1.1.0', name: 'Version 1.1' },
          { tag: 'v1.0.0', name: 'Version 1.0' },
          { tag: 'v0.9.0', name: 'Version 0.9' },
        ],
        latest_release: { tag: 'v1.1.0', name: 'Version 1.1' },
        can_update: true, blockers: [], nodes: [{ id: 'local', name: 'Controller', local: true, online: true, current_revision: 'a'.repeat(40), blockers: [] }],
      }), { status: init?.method === 'POST' ? 202 : 200, headers: { 'Content-Type': 'application/json' } })
      return new Response(JSON.stringify({ theme: 'system', default_runtime: 'vllm', default_context_length: 8192 }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    vi.stubGlobal('fetch', fetchMock)
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    const user = userEvent.setup()
    render(<MemoryRouter><SettingsPage /></MemoryRouter>)

    const release = await screen.findByRole('combobox', { name: 'Release version' })
    expect(release).toHaveValue('v1.1.0')
    expect(screen.getByRole('option', { name: /Version 1.0.*installed/ })).toBeInTheDocument()
    await user.selectOptions(release, 'v1.0.0')
    expect(screen.getByRole('button', { name: 'Installed on all nodes' })).toBeDisabled()
    await user.selectOptions(release, 'v0.9.0')
    await user.click(screen.getByRole('button', { name: 'Install on all nodes' }))

    expect(window.confirm).toHaveBeenCalledWith(expect.stringMatching(/v0\.9\.0.*may upgrade or downgrade.*controller restarts last/i))
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/system-update', expect.objectContaining({
      method: 'POST', body: JSON.stringify({ confirm: 'update-entire-cluster', tag: 'v0.9.0' }),
    }))
  })
})
