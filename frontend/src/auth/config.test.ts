import { afterEach, describe, expect, it, vi } from 'vitest'
import { cognitoConfig } from './config'

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('Cognito runtime configuration', () => {
  it('loads the backend-provided endpoint and client ID', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({
      idp_endpoint: 'https://cognito-idp.eu-west-1.amazonaws.com/',
      client_id: 'fork-client-id',
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(cognitoConfig()).resolves.toEqual({
      idp_endpoint: 'https://cognito-idp.eu-west-1.amazonaws.com/',
      client_id: 'fork-client-id',
    })
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/community/auth-config', expect.any(Object),
    )
  })
})
