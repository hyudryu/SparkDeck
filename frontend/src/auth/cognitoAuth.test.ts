import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  CognitoAuthError,
  confirmForgotPassword,
  confirmSignUp,
  decodeIdToken,
  forgotPassword,
  refresh,
  resendCode,
  signIn,
  signOut,
  signUp,
  storedTokens,
  UserNotConfirmedError,
} from './cognitoAuth'

const IDP_ENDPOINT = 'https://cognito-idp.us-east-2.amazonaws.com/'
const CLIENT_ID = '30ihrkeg4k1rn95d4mmkq00fvl'

vi.mock('./config', () => ({
  cognitoConfig: async () => ({
    idp_endpoint: 'https://cognito-idp.us-east-2.amazonaws.com/',
    client_id: '30ihrkeg4k1rn95d4mmkq00fvl',
  }),
}))

function fakeIdToken(claims: Record<string, unknown>) {
  const encode = (value: Record<string, unknown>) => btoa(JSON.stringify(value))
    .replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '')
  return `${encode({ alg: 'RS256', typ: 'JWT' })}.${encode(claims)}.signature`
}

function authResult(body: unknown = {
  AuthenticationResult: {
    IdToken: fakeIdToken({ email: 'user@example.com', exp: 4_000_000_000 }),
    AccessToken: 'access-token',
    RefreshToken: 'refresh-token',
  },
}) {
  return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
}

function cognitoError(type: string, message = 'cognito says no', status = 400) {
  return new Response(JSON.stringify({ __type: type, message }), {
    status, headers: { 'Content-Type': 'application/x-amz-json-1.1' },
  })
}

function stubFetch(response: Response | ((input: unknown) => Response)) {
  const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input) => (
    typeof response === 'function' ? response(input) : response
  ))
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  localStorage.clear()
  localStorage.clear()
})

describe('cognito IDP calls', () => {
  it('signs up with the email as username and email attribute', async () => {
    const fetchMock = stubFetch(authResult({}))

    await signUp('person@example.com', 'Password1')

    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe(IDP_ENDPOINT)
    expect(init?.method).toBe('POST')
    expect(init?.headers).toEqual({
      'Content-Type': 'application/x-amz-json-1.1',
      'X-Amz-Target': 'AWSCognitoIdentityProviderService.SignUp',
    })
    expect(JSON.parse(String(init?.body))).toEqual({
      ClientId: CLIENT_ID,
      Username: 'person@example.com',
      Password: 'Password1',
      UserAttributes: [{ Name: 'email', Value: 'person@example.com' }],
    })
  })

  it('confirms sign-up and resends the confirmation code', async () => {
    const fetchMock = stubFetch(authResult({}))

    await confirmSignUp('person@example.com', '123456')
    await resendCode('person@example.com')

    expect(fetchMock.mock.calls[0][1]?.headers).toEqual(expect.objectContaining({
      'X-Amz-Target': 'AWSCognitoIdentityProviderService.ConfirmSignUp',
    }))
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({
      ClientId: CLIENT_ID, Username: 'person@example.com', ConfirmationCode: '123456',
    })
    expect(fetchMock.mock.calls[1][1]?.headers).toEqual(expect.objectContaining({
      'X-Amz-Target': 'AWSCognitoIdentityProviderService.ResendConfirmationCode',
    }))
  })

  it('signs in with USER_PASSWORD_AUTH and stores the tokens', async () => {
    const fetchMock = stubFetch(authResult())

    const tokens = await signIn('person@example.com', 'Password1')

    expect(fetchMock.mock.calls[0][1]?.headers).toEqual(expect.objectContaining({
      'X-Amz-Target': 'AWSCognitoIdentityProviderService.InitiateAuth',
    }))
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({
      ClientId: CLIENT_ID,
      AuthFlow: 'USER_PASSWORD_AUTH',
      AuthParameters: { USERNAME: 'person@example.com', PASSWORD: 'Password1' },
    })
    expect(decodeIdToken(tokens.idToken).email).toBe('user@example.com')
    expect(storedTokens()).toEqual({
      idToken: tokens.idToken,
      accessToken: 'access-token',
      refreshToken: 'refresh-token',
    })
  })

  it('refreshes an expired session through REFRESH_TOKEN_AUTH', async () => {
    localStorage.setItem('sparkdeck.cognito.id_token', fakeIdToken({ exp: 1 }))
    localStorage.setItem('sparkdeck.cognito.refresh_token', 'saved-refresh-token')
    const fetchMock = stubFetch(authResult())

    const tokens = await refresh()

    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({
      ClientId: CLIENT_ID,
      AuthFlow: 'REFRESH_TOKEN_AUTH',
      AuthParameters: { REFRESH_TOKEN: 'saved-refresh-token' },
    })
    expect(tokens?.idToken).toBeTruthy()
  })

  it('preserves stored tokens when refresh fails transiently', async () => {
    const expired = fakeIdToken({ exp: 1 })
    localStorage.setItem('sparkdeck.cognito.id_token', expired)
    localStorage.setItem('sparkdeck.cognito.refresh_token', 'saved-refresh-token')
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockRejectedValue(new TypeError('offline')))

    await expect(refresh()).rejects.toMatchObject({ code: 'NetworkError' })
    expect(localStorage.getItem('sparkdeck.cognito.id_token')).toBe(expired)
    expect(localStorage.getItem('sparkdeck.cognito.refresh_token')).toBe('saved-refresh-token')
  })

  it('clears stored tokens only when Cognito rejects the refresh token', async () => {
    localStorage.setItem('sparkdeck.cognito.id_token', fakeIdToken({ exp: 1 }))
    localStorage.setItem('sparkdeck.cognito.refresh_token', 'revoked-refresh-token')
    stubFetch(cognitoError('NotAuthorizedException'))

    await expect(refresh()).resolves.toBeNull()
    expect(storedTokens()).toBeNull()
  })

  it('preserves stored tokens when Cognito throttles refresh', async () => {
    const expired = fakeIdToken({ exp: 1 })
    localStorage.setItem('sparkdeck.cognito.id_token', expired)
    localStorage.setItem('sparkdeck.cognito.refresh_token', 'saved-refresh-token')
    stubFetch(cognitoError('TooManyRequestsException', 'slow down', 429))

    await expect(refresh()).rejects.toMatchObject({ code: 'TooManyRequestsException' })
    expect(localStorage.getItem('sparkdeck.cognito.id_token')).toBe(expired)
    expect(localStorage.getItem('sparkdeck.cognito.refresh_token')).toBe('saved-refresh-token')
  })

  it('persists tokens in local storage so sessions survive browser restarts', async () => {
    stubFetch(authResult())

    await signIn('person@example.com', 'Password1')

    expect(localStorage.getItem('sparkdeck.cognito.id_token')).toBeTruthy()
    expect(localStorage.getItem('sparkdeck.cognito.refresh_token')).toBe('refresh-token')
    expect(sessionStorage.getItem('sparkdeck.cognito.id_token')).toBeNull()
  })

  it('revokes the refresh token on sign-out', async () => {
    localStorage.setItem('sparkdeck.cognito.id_token', fakeIdToken({ exp: 4_000_000_000 }))
    localStorage.setItem('sparkdeck.cognito.refresh_token', 'saved-refresh-token')
    const fetchMock = stubFetch(authResult({}))

    await signOut()

    expect(fetchMock.mock.calls[0][1]?.headers).toEqual(expect.objectContaining({
      'X-Amz-Target': 'AWSCognitoIdentityProviderService.RevokeToken',
    }))
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({
      ClientId: CLIENT_ID, Token: 'saved-refresh-token',
    })
    expect(storedTokens()).toBeNull()
  })

  it('starts a password reset with only the email', async () => {
    const fetchMock = stubFetch(authResult({}))

    await forgotPassword('person@example.com')

    expect(fetchMock.mock.calls[0][1]?.headers).toEqual(expect.objectContaining({
      'X-Amz-Target': 'AWSCognitoIdentityProviderService.ForgotPassword',
    }))
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({
      ClientId: CLIENT_ID, Username: 'person@example.com',
    })
  })

  it('confirms a password reset with the code and new password', async () => {
    const fetchMock = stubFetch(authResult({}))

    await confirmForgotPassword('person@example.com', '654321', 'NewPassword1')

    expect(fetchMock.mock.calls[0][1]?.headers).toEqual(expect.objectContaining({
      'X-Amz-Target': 'AWSCognitoIdentityProviderService.ConfirmForgotPassword',
    }))
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({
      ClientId: CLIENT_ID,
      Username: 'person@example.com',
      ConfirmationCode: '654321',
      Password: 'NewPassword1',
    })
  })
})

describe('cognito error mapping', () => {
  it('maps common Cognito errors to friendly messages', async () => {
    stubFetch(cognitoError('NotAuthorizedException', 'Incorrect username or password.'))

    await expect(signIn('person@example.com', 'wrong')).rejects.toMatchObject({
      name: 'CognitoAuthError',
      message: 'Incorrect email or password',
      code: 'NotAuthorizedException',
    })
  })

  it('strips the namespace prefix from error types', async () => {
    stubFetch(cognitoError('com.amazonaws.cognito.identity.provider.model#UsernameExistsException'))

    await expect(signUp('person@example.com', 'Password1')).rejects.toMatchObject({
      message: 'An account with this email already exists — try signing in',
      code: 'UsernameExistsException',
    })
  })

  it('throws a typed error for unconfirmed accounts during sign-in', async () => {
    stubFetch(cognitoError('UserNotConfirmedException', 'User is not confirmed.'))

    await expect(signIn('person@example.com', 'Password1')).rejects.toBeInstanceOf(UserNotConfirmedError)
  })

  it('maps confirmation-code errors', async () => {
    stubFetch(cognitoError('ExpiredCodeException'))

    await expect(confirmSignUp('person@example.com', '000000')).rejects.toMatchObject({
      message: 'That confirmation code has expired — request a new one',
    })
  })

  it('maps reset-code and password-policy errors during password reset', async () => {
    stubFetch(cognitoError('CodeMismatchException'))
    await expect(confirmForgotPassword('person@example.com', '000000', 'NewPassword1')).rejects.toMatchObject({
      message: 'That confirmation code is not correct',
      code: 'CodeMismatchException',
    })

    stubFetch(cognitoError('InvalidPasswordException'))
    await expect(confirmForgotPassword('person@example.com', '123456', 'short')).rejects.toMatchObject({
      message: 'Password must be at least 8 characters with upper, lower case and a number',
    })
  })

  it('maps rate limiting to a friendly message', async () => {
    stubFetch(cognitoError('LimitExceededException'))
    await expect(forgotPassword('person@example.com')).rejects.toMatchObject({
      message: 'Too many attempts — wait a moment and try again',
      code: 'LimitExceededException',
    })

    stubFetch(cognitoError('TooManyRequestsException'))
    await expect(forgotPassword('person@example.com')).rejects.toMatchObject({
      message: 'Too many attempts — wait a moment and try again',
    })
  })

  it('wraps network failures in a CognitoAuthError', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockRejectedValue(new TypeError('fetch failed')))

    await expect(signIn('person@example.com', 'Password1')).rejects.toMatchObject({
      name: 'CognitoAuthError',
      code: 'NetworkError',
    })
    await expect(signIn('person@example.com', 'Password1')).rejects.toBeInstanceOf(CognitoAuthError)
  })
})
