import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { useEffect } from 'react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { SettingsPage } from './SettingsPage'
import { AuthProvider, useAuth } from '../auth/AuthContext'
import { api } from '../api/client'
import { THEME_STORAGE_KEY } from '../theme'
import { SPARKDECK_VERSION } from '../buildInfo'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  localStorage.clear()
  localStorage.clear()
  delete document.documentElement.dataset.theme
})

function fakeIdToken(claims: Record<string, unknown>) {
  const encode = (value: Record<string, unknown>) => btoa(JSON.stringify(value))
    .replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '')
  return `${encode({ alg: 'RS256', typ: 'JWT' })}.${encode(claims)}.signature`
}

function ProtectedRequestOnSignIn() {
  const auth = useAuth()
  useEffect(() => {
    if (auth.status === 'signed-in') void api.benchmarks.aggregates()
  }, [auth.status])
  return <span>{auth.status}</span>
}

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

describe('community features sign-in', () => {
  function authResult(email: string) {
    return {
      AuthenticationResult: {
        IdToken: fakeIdToken({ email, sub: 'user-sub-1', exp: Math.floor(Date.now() / 1000) + 3600 }),
        AccessToken: 'access-token',
        RefreshToken: 'refresh-token',
      },
    }
  }

  function stubSettingsFetch(fetchMock: ReturnType<typeof vi.fn<typeof fetch>>, options: {
    idpResponse?: (body: Record<string, unknown>) => Response
    cluster?: { applied: string[]; conflicts: { node: string; email?: string }[]; errors: string[] }
    pairResponse?: () => Response
    unpairCluster?: { applied: string[]; conflicts: { node: string; email?: string }[]; errors: string[] }
  } = {}) {
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.endsWith('/api/v1/community/auth-config')) return new Response(JSON.stringify({
        idp_endpoint: 'https://cognito-idp.us-east-2.amazonaws.com/',
        client_id: '30ihrkeg4k1rn95d4mmkq00fvl',
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path.includes('cognito-idp')) {
        if (options.idpResponse) return options.idpResponse(JSON.parse(String(init?.body)))
        return new Response(JSON.stringify({}), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path.includes('system-update')) return new Response(JSON.stringify({ can_update: false, blockers: [], nodes: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path.endsWith('/api/v1/community/pair') && init?.method === 'POST') {
        if (options.pairResponse) return options.pairResponse()
        return new Response(JSON.stringify({
          pairing: { status: 'paired' },
          cluster: options.cluster ?? { applied: [], conflicts: [], errors: [] },
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path.endsWith('/api/v1/community/pair') && init?.method === 'DELETE') return new Response(JSON.stringify({
        pairing: { status: 'not_paired' },
        cluster: options.unpairCluster ?? { applied: [], conflicts: [], errors: [] },
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      return new Response(JSON.stringify({
        theme: 'system', hf_token: '', hf_token_configured: false, community_api_url: '',
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    vi.stubGlobal('fetch', fetchMock)
    return fetchMock
  }

  function cognitoCalls(fetchMock: ReturnType<typeof vi.fn<typeof fetch>>, target: string) {
    return fetchMock.mock.calls.filter(([input, init]) => (
      String(input).includes('cognito-idp')
      && String((init?.headers as Record<string, string>)?.['X-Amz-Target']).endsWith(target)
    ))
  }

  it('renders the email and password sign-in form when signed out', async () => {
    stubSettingsFetch(vi.fn<typeof fetch>())

    render(<MemoryRouter><SettingsPage /></MemoryRouter>)

    expect(await screen.findByRole('heading', { name: 'Community Features' })).toBeInTheDocument()
    expect(screen.getByText('Create an account or sign in to share anonymized benchmark telemetry and see community data.')).toBeInTheDocument()
    expect(screen.getByLabelText('Email')).toBeInTheDocument()
    expect(screen.getByLabelText('Password')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Sign in' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Create account' })).toBeEnabled()
    expect(screen.queryByRole('button', { name: 'Pair account' })).not.toBeInTheDocument()
  })

  it('signs in with email and password and pairs the device', async () => {
    const fetchMock = stubSettingsFetch(vi.fn<typeof fetch>(), {
      idpResponse: () => new Response(JSON.stringify(authResult('driver@example.com')), {
        status: 200, headers: { 'Content-Type': 'application/json' },
      }),
    })
    const user = userEvent.setup()

    render(<MemoryRouter><AuthProvider><SettingsPage /></AuthProvider></MemoryRouter>)

    await user.type(await screen.findByLabelText('Email'), 'driver@example.com')
    await user.type(screen.getByLabelText('Password'), 'Password1')
    await user.click(screen.getByRole('button', { name: 'Sign in' }))

    expect(await screen.findByText('driver@example.com')).toBeInTheDocument()
    expect(screen.getByText('Signed in')).toBeInTheDocument()
    expect(cognitoCalls(fetchMock, 'InitiateAuth')).toHaveLength(1)
    expect(JSON.parse(String(cognitoCalls(fetchMock, 'InitiateAuth')[0][1]?.body))).toMatchObject({
      AuthFlow: 'USER_PASSWORD_AUTH',
      AuthParameters: { USERNAME: 'driver@example.com', PASSWORD: 'Password1' },
    })
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/v1/community/pair', expect.objectContaining({
      method: 'POST',
      body: expect.stringContaining('"id_token"'),
    })))
    const pairCall = fetchMock.mock.calls.find(([input, init]) => (
      String(input).endsWith('/api/v1/community/pair') && init?.method === 'POST'
    ))
    expect(JSON.parse(String(pairCall?.[1]?.body))).toMatchObject({
      refresh_token: 'refresh-token',
    })
  })

  it('reports cluster conflicts and unreachable nodes after sign-in', async () => {
    stubSettingsFetch(vi.fn<typeof fetch>(), {
      idpResponse: () => new Response(JSON.stringify(authResult('driver@example.com')), {
        status: 200, headers: { 'Content-Type': 'application/json' },
      }),
      cluster: {
        applied: [],
        conflicts: [{ node: 'Spark Three', email: 'other@example.com' }],
        errors: ['Spark Four: Spark Four agent error: unreachable'],
      },
    })
    const user = userEvent.setup()

    render(<MemoryRouter><AuthProvider><SettingsPage /></AuthProvider></MemoryRouter>)

    await user.type(await screen.findByLabelText('Email'), 'driver@example.com')
    await user.type(screen.getByLabelText('Password'), 'Password1')
    await user.click(screen.getByRole('button', { name: 'Sign in' }))

    expect(await screen.findByText('Sign-in was not applied to: Spark Three (already signed in as other@example.com).')).toBeInTheDocument()
    expect(screen.getByText("Could not reach: Spark Four — they'll stay signed out until synced.")).toBeInTheDocument()
    expect(screen.queryByText(/Sign-in synced to/)).not.toBeInTheDocument()
  })

  it('confirms when sign-in synced to peer nodes', async () => {
    stubSettingsFetch(vi.fn<typeof fetch>(), {
      idpResponse: () => new Response(JSON.stringify(authResult('driver@example.com')), {
        status: 200, headers: { 'Content-Type': 'application/json' },
      }),
      cluster: { applied: ['Spark Two', 'Spark Three'], conflicts: [], errors: [] },
    })
    const user = userEvent.setup()

    render(<MemoryRouter><AuthProvider><SettingsPage /></AuthProvider></MemoryRouter>)

    await user.type(await screen.findByLabelText('Email'), 'driver@example.com')
    await user.type(screen.getByLabelText('Password'), 'Password1')
    await user.click(screen.getByRole('button', { name: 'Sign in' }))

    expect(await screen.findByText('Sign-in synced to 2 nodes.')).toBeInTheDocument()
  })

  it('signs back out and shows the error when backend pairing fails', async () => {
    stubSettingsFetch(vi.fn<typeof fetch>(), {
      idpResponse: () => new Response(JSON.stringify(authResult('driver@example.com')), {
        status: 200, headers: { 'Content-Type': 'application/json' },
      }),
      pairResponse: () => new Response(JSON.stringify({ detail: 'pairing backend exploded' }), {
        status: 500, headers: { 'Content-Type': 'application/json' },
      }),
    })
    const user = userEvent.setup()

    render(<MemoryRouter><AuthProvider><SettingsPage /></AuthProvider></MemoryRouter>)

    await user.type(await screen.findByLabelText('Email'), 'driver@example.com')
    await user.type(screen.getByLabelText('Password'), 'Password1')
    await user.click(screen.getByRole('button', { name: 'Sign in' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('pairing backend exploded')
    expect(screen.getByRole('button', { name: 'Sign in' })).toBeInTheDocument()
    expect(screen.queryByText('Signed in')).not.toBeInTheDocument()
    expect(localStorage.getItem('sparkdeck.cognito.id_token')).toBeNull()
  })

  it('signs back out with a conflict message when the node is paired with another account', async () => {
    stubSettingsFetch(vi.fn<typeof fetch>(), {
      idpResponse: () => new Response(JSON.stringify(authResult('driver@example.com')), {
        status: 200, headers: { 'Content-Type': 'application/json' },
      }),
      pairResponse: () => new Response(JSON.stringify({
        error: 'already_paired', existing: { email: 'other@example.com' },
      }), { status: 409, headers: { 'Content-Type': 'application/json' } }),
    })
    const user = userEvent.setup()

    render(<MemoryRouter><AuthProvider><SettingsPage /></AuthProvider></MemoryRouter>)

    await user.type(await screen.findByLabelText('Email'), 'driver@example.com')
    await user.type(screen.getByLabelText('Password'), 'Password1')
    await user.click(screen.getByRole('button', { name: 'Sign in' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('This node is already signed in as other@example.com')
    expect(screen.queryByText('Signed in')).not.toBeInTheDocument()
    expect(localStorage.getItem('sparkdeck.cognito.id_token')).toBeNull()
  })

  it('rejects a restored session when backend re-pairing fails', async () => {
    const fetchMock = stubSettingsFetch(vi.fn<typeof fetch>(), {
      pairResponse: () => new Response(JSON.stringify({ detail: 'pairing backend exploded' }), {
        status: 500, headers: { 'Content-Type': 'application/json' },
      }),
    })
    localStorage.setItem('sparkdeck.cognito.id_token', fakeIdToken({
      email: 'driver@example.com', sub: 'user-sub-1', exp: Math.floor(Date.now() / 1000) + 3600,
    }))

    render(<MemoryRouter><AuthProvider><SettingsPage /></AuthProvider></MemoryRouter>)

    expect(await screen.findByLabelText('Email')).toBeInTheDocument()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/v1/community/pair', expect.objectContaining({
      method: 'POST',
    })))
    expect(screen.queryByText('Signed in')).not.toBeInTheDocument()
    expect(screen.queryByText('driver@example.com')).not.toBeInTheDocument()
    expect(localStorage.getItem('sparkdeck.cognito.id_token')).toBeNull()
  })

  it('pressing Enter in credentials does not submit dirty application settings', async () => {
    const fetchMock = stubSettingsFetch(vi.fn<typeof fetch>(), {
      idpResponse: () => new Response(JSON.stringify(authResult('driver@example.com')), {
        status: 200, headers: { 'Content-Type': 'application/json' },
      }),
    })
    const user = userEvent.setup()
    render(<MemoryRouter><AuthProvider><SettingsPage /></AuthProvider></MemoryRouter>)

    await user.selectOptions(await screen.findByLabelText('Appearance'), 'dark')
    await user.type(screen.getByLabelText('Email'), 'driver@example.com')
    await user.type(screen.getByLabelText('Password'), 'Password1{Enter}')

    expect(await screen.findByText('Signed in')).toBeInTheDocument()
    expect(fetchMock.mock.calls.some(([input, init]) => (
      String(input).endsWith('/api/v1/settings') && init?.method === 'PUT'
    ))).toBe(false)
  })

  it('shows unpair propagation failures after signing out', async () => {
    stubSettingsFetch(vi.fn<typeof fetch>(), {
      unpairCluster: {
        applied: [],
        conflicts: [{ node: 'Spark Three', email: 'other@example.com' }],
        errors: ['Spark Four: Spark Four agent error: unreachable'],
      },
    })
    const user = userEvent.setup()
    localStorage.setItem('sparkdeck.cognito.id_token', fakeIdToken({
      email: 'driver@example.com', sub: 'user-sub-1', exp: Math.floor(Date.now() / 1000) + 3600,
    }))

    render(<MemoryRouter><AuthProvider><SettingsPage /></AuthProvider></MemoryRouter>)

    await user.click(await screen.findByRole('button', { name: 'Sign out' }))

    expect(await screen.findByLabelText('Email')).toBeInTheDocument()
    expect(screen.getByText('Sign-out was not applied to: Spark Three (signed in as other@example.com).')).toBeInTheDocument()
    expect(screen.getByText('Some nodes are still signed in: Spark Four — they could not be reached.')).toBeInTheDocument()
  })

  it('creates an account and asks for the emailed confirmation code', async () => {
    const fetchMock = stubSettingsFetch(vi.fn<typeof fetch>())
    const user = userEvent.setup()

    render(<MemoryRouter><SettingsPage /></MemoryRouter>)

    await user.click(await screen.findByRole('button', { name: 'Create account' }))
    await user.type(screen.getByLabelText('Email'), 'new@example.com')
    await user.type(screen.getByLabelText('Password'), 'Password1')
    await user.type(screen.getByLabelText('Confirm password'), 'Password1')
    await user.click(screen.getByRole('button', { name: 'Create account' }))

    expect(await screen.findByLabelText('Confirmation code')).toBeInTheDocument()
    expect(screen.getByText('We emailed a confirmation code to new@example.com.')).toBeInTheDocument()
    expect(cognitoCalls(fetchMock, 'SignUp')).toHaveLength(1)

    await user.type(screen.getByLabelText('Confirmation code'), '123456')
    await user.click(screen.getByRole('button', { name: 'Confirm' }))

    expect(await screen.findByText('Account confirmed — sign in with your password.')).toBeInTheDocument()
    expect(JSON.parse(String(cognitoCalls(fetchMock, 'ConfirmSignUp')[0][1]?.body))).toMatchObject({
      Username: 'new@example.com', ConfirmationCode: '123456',
    })
    expect(screen.getByRole('button', { name: 'Sign in' })).toBeInTheDocument()
  })

  it('rejects mismatched passwords before calling Cognito', async () => {
    const fetchMock = stubSettingsFetch(vi.fn<typeof fetch>())
    const user = userEvent.setup()

    render(<MemoryRouter><SettingsPage /></MemoryRouter>)

    await user.click(await screen.findByRole('button', { name: 'Create account' }))
    await user.type(screen.getByLabelText('Email'), 'new@example.com')
    await user.type(screen.getByLabelText('Password'), 'Password1')
    await user.type(screen.getByLabelText('Confirm password'), 'Password2')
    await user.click(screen.getByRole('button', { name: 'Create account' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Passwords do not match')
    expect(cognitoCalls(fetchMock, 'SignUp')).toHaveLength(0)
  })

  it('shows a friendly message when the password is wrong', async () => {
    stubSettingsFetch(vi.fn<typeof fetch>(), {
      idpResponse: () => new Response(JSON.stringify({ __type: 'NotAuthorizedException', message: 'Incorrect username or password.' }), {
        status: 400, headers: { 'Content-Type': 'application/x-amz-json-1.1' },
      }),
    })
    const user = userEvent.setup()

    render(<MemoryRouter><SettingsPage /></MemoryRouter>)

    await user.type(await screen.findByLabelText('Email'), 'driver@example.com')
    await user.type(screen.getByLabelText('Password'), 'wrong-pass')
    await user.click(screen.getByRole('button', { name: 'Sign in' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Incorrect email or password')
    expect(screen.getByRole('button', { name: 'Sign in' })).toBeInTheDocument()
  })

  it('jumps to the confirmation step when the account is unconfirmed', async () => {
    stubSettingsFetch(vi.fn<typeof fetch>(), {
      idpResponse: () => new Response(JSON.stringify({ __type: 'UserNotConfirmedException' }), {
        status: 400, headers: { 'Content-Type': 'application/x-amz-json-1.1' },
      }),
    })
    const user = userEvent.setup()

    render(<MemoryRouter><SettingsPage /></MemoryRouter>)

    await user.type(await screen.findByLabelText('Email'), 'pending@example.com')
    await user.type(screen.getByLabelText('Password'), 'Password1')
    await user.click(screen.getByRole('button', { name: 'Sign in' }))

    expect(await screen.findByLabelText('Confirmation code')).toBeInTheDocument()
    expect(screen.getByText(/not confirmed yet/)).toBeInTheDocument()
  })

  it('resets a forgotten password and returns to sign-in', async () => {
    const fetchMock = stubSettingsFetch(vi.fn<typeof fetch>())
    const user = userEvent.setup()

    render(<MemoryRouter><SettingsPage /></MemoryRouter>)

    await user.type(await screen.findByLabelText('Email'), 'driver@example.com')
    await user.click(screen.getByRole('button', { name: 'Forgot password?' }))

    expect(screen.getByLabelText('Email')).toHaveValue('driver@example.com')
    await user.click(screen.getByRole('button', { name: 'Send reset code' }))

    expect(await screen.findByText('If an account exists for driver@example.com, we emailed a reset code.')).toBeInTheDocument()
    expect(JSON.parse(String(cognitoCalls(fetchMock, 'ForgotPassword')[0][1]?.body))).toMatchObject({
      Username: 'driver@example.com',
    })

    await user.type(screen.getByLabelText('Confirmation code'), '654321')
    await user.type(screen.getByLabelText('New password'), 'NewPassword1')
    await user.type(screen.getByLabelText('Confirm new password'), 'NewPassword1')
    await user.click(screen.getByRole('button', { name: 'Reset password' }))

    expect(await screen.findByText('Password updated — sign in with your new password.')).toBeInTheDocument()
    expect(JSON.parse(String(cognitoCalls(fetchMock, 'ConfirmForgotPassword')[0][1]?.body))).toMatchObject({
      Username: 'driver@example.com', ConfirmationCode: '654321', Password: 'NewPassword1',
    })
    expect(screen.getByRole('button', { name: 'Sign in' })).toBeInTheDocument()
  })

  it('shows a friendly error when the reset code is wrong', async () => {
    stubSettingsFetch(vi.fn<typeof fetch>(), {
      idpResponse: (body) => ('ConfirmationCode' in body
        ? new Response(JSON.stringify({ __type: 'CodeMismatchException' }), {
            status: 400, headers: { 'Content-Type': 'application/x-amz-json-1.1' },
          })
        : new Response(JSON.stringify({}), { status: 200, headers: { 'Content-Type': 'application/json' } })),
    })
    const user = userEvent.setup()

    render(<MemoryRouter><SettingsPage /></MemoryRouter>)

    await user.type(await screen.findByLabelText('Email'), 'driver@example.com')
    await user.click(screen.getByRole('button', { name: 'Forgot password?' }))
    await user.click(screen.getByRole('button', { name: 'Send reset code' }))

    await user.type(await screen.findByLabelText('Confirmation code'), '000000')
    await user.type(screen.getByLabelText('New password'), 'NewPassword1')
    await user.type(screen.getByLabelText('Confirm new password'), 'NewPassword1')
    await user.click(screen.getByRole('button', { name: 'Reset password' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('That confirmation code is not correct')
    expect(screen.getByRole('button', { name: 'Reset password' })).toBeInTheDocument()
  })

  it('rejects mismatched new passwords before calling Cognito', async () => {
    const fetchMock = stubSettingsFetch(vi.fn<typeof fetch>())
    const user = userEvent.setup()

    render(<MemoryRouter><SettingsPage /></MemoryRouter>)

    await user.type(await screen.findByLabelText('Email'), 'driver@example.com')
    await user.click(screen.getByRole('button', { name: 'Forgot password?' }))
    await user.click(screen.getByRole('button', { name: 'Send reset code' }))

    await user.type(await screen.findByLabelText('Confirmation code'), '654321')
    await user.type(screen.getByLabelText('New password'), 'NewPassword1')
    await user.type(screen.getByLabelText('Confirm new password'), 'NewPassword2')
    await user.click(screen.getByRole('button', { name: 'Reset password' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Passwords do not match')
    expect(cognitoCalls(fetchMock, 'ConfirmForgotPassword')).toHaveLength(0)
  })

  it('shows the account email when signed in and unpairs on sign out', async () => {
    const fetchMock = stubSettingsFetch(vi.fn<typeof fetch>())
    const user = userEvent.setup()
    localStorage.setItem('sparkdeck.cognito.id_token', fakeIdToken({
      email: 'driver@example.com', sub: 'user-sub-1', exp: Math.floor(Date.now() / 1000) + 3600,
    }))

    render(<MemoryRouter><AuthProvider><SettingsPage /></AuthProvider></MemoryRouter>)

    expect(await screen.findByText('driver@example.com')).toBeInTheDocument()
    expect(screen.getByText('Signed in')).toBeInTheDocument()
    expect(screen.queryByLabelText('Email')).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Sign out' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/v1/community/pair', expect.objectContaining({
      method: 'DELETE',
    })))
    expect(await screen.findByLabelText('Email')).toBeInTheDocument()
    expect(localStorage.getItem('sparkdeck.cognito.id_token')).toBeNull()
  })

  it('silently refreshes an expired stored session on load', async () => {
    const fetchMock = stubSettingsFetch(vi.fn<typeof fetch>(), {
      idpResponse: () => new Response(JSON.stringify(authResult('driver@example.com')), {
        status: 200, headers: { 'Content-Type': 'application/json' },
      }),
    })
    localStorage.setItem('sparkdeck.cognito.id_token', fakeIdToken({
      email: 'driver@example.com', sub: 'user-sub-1', exp: 1,
    }))
    localStorage.setItem('sparkdeck.cognito.refresh_token', 'saved-refresh-token')

    render(<MemoryRouter><AuthProvider><SettingsPage /></AuthProvider></MemoryRouter>)

    expect(await screen.findByText('driver@example.com')).toBeInTheDocument()
    expect(screen.getByText('Signed in')).toBeInTheDocument()
    expect(cognitoCalls(fetchMock, 'InitiateAuth')).toHaveLength(1)
    expect(JSON.parse(String(cognitoCalls(fetchMock, 'InitiateAuth')[0][1]?.body))).toMatchObject({
      AuthFlow: 'REFRESH_TOKEN_AUTH',
      AuthParameters: { REFRESH_TOKEN: 'saved-refresh-token' },
    })
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/v1/community/pair', expect.objectContaining({
      method: 'POST',
      body: expect.stringContaining('"id_token"'),
    })))
  })

  it('installs the restored bearer before signed-in children request community data', async () => {
    const fetchMock = stubSettingsFetch(vi.fn<typeof fetch>())
    const token = fakeIdToken({
      email: 'driver@example.com', sub: 'user-sub-1',
      exp: Math.floor(Date.now() / 1000) + 3600,
    })
    localStorage.setItem('sparkdeck.cognito.id_token', token)

    render(<AuthProvider><ProtectedRequestOnSignIn /></AuthProvider>)

    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => (
      String(input).endsWith('/api/v1/community/aggregates')
    ))).toBe(true))
    const aggregate = fetchMock.mock.calls.find(([input]) => (
      String(input).endsWith('/api/v1/community/aggregates')
    ))
    expect(aggregate?.[1]?.headers).toEqual(expect.objectContaining({
      Authorization: `Bearer ${token}`,
    }))
  })

  it('refreshes a near-expiry bearer before unpairing a long-open session', async () => {
    const refreshed = fakeIdToken({
      email: 'driver@example.com', sub: 'user-sub-1',
      exp: Math.floor(Date.now() / 1000) + 3600,
    })
    const fetchMock = stubSettingsFetch(vi.fn<typeof fetch>(), {
      idpResponse: (body) => new Response(JSON.stringify(
        body.AuthFlow === 'REFRESH_TOKEN_AUTH'
          ? { AuthenticationResult: { IdToken: refreshed, AccessToken: 'new-access' } }
          : {},
      ), { status: 200, headers: { 'Content-Type': 'application/json' } }),
    })
    const user = userEvent.setup()
    localStorage.setItem('sparkdeck.cognito.id_token', fakeIdToken({
      email: 'driver@example.com', sub: 'user-sub-1',
      exp: Math.floor(Date.now() / 1000) + 30,
    }))
    localStorage.setItem('sparkdeck.cognito.refresh_token', 'saved-refresh-token')

    render(<MemoryRouter><AuthProvider><SettingsPage /></AuthProvider></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'Sign out' }))

    expect(cognitoCalls(fetchMock, 'InitiateAuth')).toHaveLength(1)
    const unpair = fetchMock.mock.calls.find(([input, init]) => (
      String(input).endsWith('/api/v1/community/pair') && init?.method === 'DELETE'
    ))
    expect(unpair?.[1]?.headers).toEqual(expect.objectContaining({
      Authorization: `Bearer ${refreshed}`,
    }))
    expect(await screen.findByLabelText('Email')).toBeInTheDocument()
  })
})
