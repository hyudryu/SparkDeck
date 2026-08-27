import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { api, ApiError, setAuthTokenProvider } from '../api/client'
import type { CommunityClusterSync } from '../api/types'
import {
  confirmForgotPassword,
  confirmSignUp,
  decodeIdToken,
  forgotPassword,
  refresh,
  resendCode,
  signIn as cognitoSignIn,
  signOut as cognitoSignOut,
  signUp,
  storedTokens,
} from './cognitoAuth'

export interface AuthState {
  status: 'restoring' | 'signed-out' | 'signing-in' | 'signed-in' | 'reauth-required'
  email?: string
  idToken?: string
  /** Cluster fan-out result from the latest sign-in/out, if the backend reported one. */
  clusterSync?: CommunityClusterSync | null
  signIn: (email: string, password: string) => Promise<void>
  signUp: (email: string, password: string) => Promise<void>
  confirmSignUp: (email: string, code: string) => Promise<void>
  resendCode: (email: string) => Promise<void>
  forgotPassword: (email: string) => Promise<void>
  confirmForgotPassword: (email: string, code: string, newPassword: string) => Promise<void>
  signOut: () => Promise<void>
}

// A signed-out default keeps components usable without a provider (tests,
// isolated renders); the real session is supplied by AuthProvider.
const signedOutDefault: AuthState = {
  status: 'signed-out',
  signIn: async (email, password) => {
    const tokens = await cognitoSignIn(email, password)
    await api.community.pair(tokens.idToken, tokens.refreshToken).catch(() => undefined)
  },
  signUp,
  confirmSignUp,
  resendCode,
  forgotPassword,
  confirmForgotPassword,
  signOut: () => cognitoSignOut(),
}

const AuthContext = createContext<AuthState>(signedOutDefault)

export function useAuth(): AuthState {
  return useContext(AuthContext)
}

function isExpired(idToken: string): boolean {
  const exp = decodeIdToken(idToken).exp
  return !exp || exp * 1000 <= Date.now()
}

function pairingConflictMessage(error: ApiError): Error {
  const existing = (error.body as { existing?: { email?: string } } | undefined)?.existing?.email
  return new Error(`This node is already signed in as ${existing ?? 'another account'}`)
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthState['status']>('restoring')
  const [idToken, setIdToken] = useState<string>()
  const [clusterSync, setClusterSync] = useState<CommunityClusterSync | null>(null)
  const idTokenRef = useRef<string | undefined>(undefined)
  const refreshPromiseRef = useRef<Promise<string | undefined> | null>(null)
  const publishIdToken = useCallback((token: string | undefined) => {
    idTokenRef.current = token
    setIdToken(token)
  }, [])

  const validIdToken = useCallback(async () => {
    const current = idTokenRef.current
    if (!current) return undefined
    const expiresAt = (decodeIdToken(current).exp ?? 0) * 1000
    if (expiresAt - Date.now() > 60_000) return current
    if (!refreshPromiseRef.current) {
      refreshPromiseRef.current = (async () => {
        const tokens = await refresh()
        if (!tokens || isExpired(tokens.idToken)) {
          publishIdToken(undefined)
          setStatus('reauth-required')
          throw new Error('Your community session expired before this node could be signed out. Sign in again, then retry Sign out.')
        }
        publishIdToken(tokens.idToken)
        return tokens.idToken
      })().finally(() => {
        refreshPromiseRef.current = null
      })
    }
    return refreshPromiseRef.current
  }, [publishIdToken])

  useEffect(() => {
    let cancelled = false
    const restore = async () => {
      try {
        let tokens = storedTokens()
        if (tokens && isExpired(tokens.idToken)) tokens = await refresh()
        if (cancelled) return
        if (tokens && !isExpired(tokens.idToken)) {
          try {
            // A refreshed TokenSet omits the refresh token, so the backend
            // re-pair reads it from storage (the uploader needs it to mint
            // short-lived ID tokens for consented telemetry).
            const paired = await api.community.pair(
              tokens.idToken,
              storedTokens()?.refreshToken ?? tokens.refreshToken,
            )
            if (cancelled) return
            setClusterSync(paired.cluster ?? null)
            publishIdToken(tokens.idToken)
            setStatus('signed-in')
            return
          } catch {
            await cognitoSignOut()
          }
        }
      } catch {
        // Retryable refresh failures retain browser credentials for the next
        // restore attempt, but never create an unpaired signed-in UI session.
      }
      if (cancelled) return
      setClusterSync(null)
      publishIdToken(undefined)
      setStatus('signed-out')
    }
    void restore()
    return () => {
      cancelled = true
    }
  }, [publishIdToken])

  useEffect(() => {
    setAuthTokenProvider(validIdToken)
    return () => {
      setAuthTokenProvider(undefined)
    }
  }, [validIdToken])

  const value = useMemo<AuthState>(() => ({
    status,
    idToken,
    clusterSync,
    email: idToken ? decodeIdToken(idToken).email : undefined,
    signIn: async (email, password) => {
      if (status === 'restoring') {
        throw new Error('Wait for the saved community session to finish restoring, then try again.')
      }
      setStatus('signing-in')
      setClusterSync(null)
      try {
        const tokens = await cognitoSignIn(email, password)
        try {
          const paired = await api.community.pair(tokens.idToken, tokens.refreshToken)
          setClusterSync(paired.cluster ?? null)
        } catch (pairError) {
          // Pairing is what makes community features work; if it fails, do not
          // leave the UI looking signed in.
          await cognitoSignOut()
          throw pairError instanceof ApiError && pairError.status === 409
            ? pairingConflictMessage(pairError)
            : pairError
        }
        publishIdToken(tokens.idToken)
        setStatus('signed-in')
      } catch (reason) {
        setStatus('signed-out')
        throw reason
      }
    },
    signUp,
    confirmSignUp,
    resendCode,
    forgotPassword,
    confirmForgotPassword,
    signOut: async () => {
      setClusterSync(null)
      const unpaired = await api.community.unpair()
      await cognitoSignOut()
      setClusterSync(unpaired?.cluster ?? null)
      publishIdToken(undefined)
      setStatus('signed-out')
    },
  }), [status, idToken, clusterSync, publishIdToken])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
