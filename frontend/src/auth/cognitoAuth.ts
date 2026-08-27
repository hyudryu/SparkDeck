// Native Cognito email+password authentication: the frontend calls the Cognito
// IDP HTTPS API directly (no hosted UI, no redirect, no PKCE). Tokens live in
// localStorage so the session survives browser restarts; the refresh token is
// valid for years and expired ID tokens are refreshed silently on load.

import { cognitoConfig } from './config'

const ID_TOKEN_KEY = 'sparkdeck.cognito.id_token'
const ACCESS_TOKEN_KEY = 'sparkdeck.cognito.access_token'
const REFRESH_TOKEN_KEY = 'sparkdeck.cognito.refresh_token'

export interface TokenSet {
  idToken: string
  accessToken?: string
  refreshToken?: string
}

export interface IdTokenClaims {
  sub?: string
  email?: string
  exp?: number
}

// Sign-in rejected because the account exists but has not been confirmed yet;
// the UI uses this to switch to the confirmation-code step.
export class UserNotConfirmedError extends Error {
  constructor() {
    super('Account is not confirmed yet — enter the code from your email')
    this.name = 'UserNotConfirmedError'
  }
}

export class CognitoAuthError extends Error {
  constructor(
    message: string,
    readonly code: string,
  ) {
    super(message)
    this.name = 'CognitoAuthError'
  }
}

const ERROR_MESSAGES: Record<string, string> = {
  UsernameExistsException: 'An account with this email already exists — try signing in',
  CodeMismatchException: 'That confirmation code is not correct',
  ExpiredCodeException: 'That confirmation code has expired — request a new one',
  NotAuthorizedException: 'Incorrect email or password',
  InvalidPasswordException: 'Password must be at least 8 characters with upper, lower case and a number',
  UserNotFoundException: 'No account found for this email',
  LimitExceededException: 'Too many attempts — wait a moment and try again',
  TooManyRequestsException: 'Too many attempts — wait a moment and try again',
}

export function decodeIdToken(idToken: string): IdTokenClaims {
  const payload = idToken.split('.')[1]
  if (!payload) return {}
  try {
    const json = atob(payload.replaceAll('-', '+').replaceAll('_', '/'))
    return JSON.parse(json) as IdTokenClaims
  } catch {
    return {}
  }
}

export function storedTokens(): TokenSet | null {
  const idToken = localStorage.getItem(ID_TOKEN_KEY)
  if (!idToken) return null
  return {
    idToken,
    accessToken: localStorage.getItem(ACCESS_TOKEN_KEY) ?? undefined,
    refreshToken: localStorage.getItem(REFRESH_TOKEN_KEY) ?? undefined,
  }
}

function storeTokens(tokens: TokenSet) {
  localStorage.setItem(ID_TOKEN_KEY, tokens.idToken)
  if (tokens.accessToken) localStorage.setItem(ACCESS_TOKEN_KEY, tokens.accessToken)
  if (tokens.refreshToken) localStorage.setItem(REFRESH_TOKEN_KEY, tokens.refreshToken)
}

function clearTokens() {
  localStorage.removeItem(ID_TOKEN_KEY)
  localStorage.removeItem(ACCESS_TOKEN_KEY)
  localStorage.removeItem(REFRESH_TOKEN_KEY)
}

async function callIdp(operation: string, body: Record<string, unknown>): Promise<Record<string, unknown>> {
  let config
  try {
    config = await cognitoConfig()
  } catch {
    throw new CognitoAuthError('Could not load the community sign-in configuration', 'ConfigurationError')
  }
  let response: Response
  try {
    response = await fetch(config.idp_endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-amz-json-1.1',
        'X-Amz-Target': `AWSCognitoIdentityProviderService.${operation}`,
      },
      body: JSON.stringify({ ClientId: config.client_id, ...body }),
    })
  } catch {
    throw new CognitoAuthError('Could not reach the community sign-in service', 'NetworkError')
  }
  const data = (await response.json().catch(() => ({}))) as { __type?: string; message?: string }
  if (!response.ok) {
    const code = (data.__type ?? 'UnknownException').split('#').at(-1) ?? 'UnknownException'
    if (code === 'UserNotConfirmedException') throw new UserNotConfirmedError()
    throw new CognitoAuthError(ERROR_MESSAGES[code] ?? data.message ?? `Sign-in failed (${code})`, code)
  }
  return data
}

export async function signUp(email: string, password: string): Promise<void> {
  await callIdp('SignUp', {
    Username: email,
    Password: password,
    UserAttributes: [{ Name: 'email', Value: email }],
  })
}

export async function confirmSignUp(email: string, code: string): Promise<void> {
  await callIdp('ConfirmSignUp', { Username: email, ConfirmationCode: code })
}

export async function resendCode(email: string): Promise<void> {
  await callIdp('ResendConfirmationCode', { Username: email })
}

// The pool sets PreventUserExistenceErrors, so this succeeds even for unknown
// emails by design — callers should keep the wording existence-neutral.
export async function forgotPassword(email: string): Promise<void> {
  await callIdp('ForgotPassword', { Username: email })
}

export async function confirmForgotPassword(email: string, code: string, newPassword: string): Promise<void> {
  await callIdp('ConfirmForgotPassword', {
    Username: email,
    ConfirmationCode: code,
    Password: newPassword,
  })
}

function tokensFromAuthResult(data: Record<string, unknown>): TokenSet {
  const result = data.AuthenticationResult as
    | { IdToken?: string; AccessToken?: string; RefreshToken?: string }
    | undefined
  if (!result?.IdToken) throw new CognitoAuthError('Sign-in did not return tokens', 'MissingTokens')
  const tokens: TokenSet = {
    idToken: result.IdToken,
    accessToken: result.AccessToken,
    refreshToken: result.RefreshToken,
  }
  storeTokens(tokens)
  return tokens
}

export async function signIn(email: string, password: string): Promise<TokenSet> {
  const data = await callIdp('InitiateAuth', {
    AuthFlow: 'USER_PASSWORD_AUTH',
    AuthParameters: { USERNAME: email, PASSWORD: password },
  })
  return tokensFromAuthResult(data)
}

export async function refresh(): Promise<TokenSet | null> {
  const refreshToken = localStorage.getItem(REFRESH_TOKEN_KEY)
  if (!refreshToken) {
    clearTokens()
    return null
  }
  try {
    const data = await callIdp('InitiateAuth', {
      AuthFlow: 'REFRESH_TOKEN_AUTH',
      AuthParameters: { REFRESH_TOKEN: refreshToken },
    })
    return tokensFromAuthResult(data)
  } catch (reason) {
    if (
      reason instanceof CognitoAuthError
      && (reason.code === 'NotAuthorizedException' || reason.code === 'UserNotFoundException')
    ) {
      clearTokens()
      return null
    }
    throw reason
  }
}

export async function signOut(): Promise<void> {
  const refreshToken = localStorage.getItem(REFRESH_TOKEN_KEY)
  clearTokens()
  if (refreshToken) {
    try {
      await callIdp('RevokeToken', { Token: refreshToken })
    } catch {
      // Revocation is best-effort; local sign-out already happened.
    }
  }
}
