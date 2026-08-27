import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
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
  status: 'signed-out' | 'signing-in' | 'signed-in'
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
    await api.community.pair(tokens.idToken).catch(() => undefined)
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
  const [status, setStatus] = useState<AuthState['status']>('signed-out')
  const [idToken, setIdToken] = useState<string>()
  const [clusterSync, setClusterSync] = useState<CommunityClusterSync | null>(null)

  useEffect(() => {
    let cancelled = false
    const restore = async () => {
      try {
        let tokens = storedTokens()
        if (tokens && isExpired(tokens.idToken)) tokens = await refresh()
        if (cancelled) return
        if (tokens && !isExpired(tokens.idToken)) {
          try {
            const paired = await api.community.pair(tokens.idToken)
            if (cancelled) return
            setClusterSync(paired.cluster ?? null)
            setIdToken(tokens.idToken)
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
      setIdToken(undefined)
      setStatus('signed-out')
    }
    void restore()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    setAuthTokenProvider(() => idToken)
    return () => setAuthTokenProvider(undefined)
  }, [idToken])

  const value = useMemo<AuthState>(() => ({
    status,
    idToken,
    clusterSync,
    email: idToken ? decodeIdToken(idToken).email : undefined,
    signIn: async (email, password) => {
      setStatus('signing-in')
      setClusterSync(null)
      try {
        const tokens = await cognitoSignIn(email, password)
        try {
          const paired = await api.community.pair(tokens.idToken)
          setClusterSync(paired.cluster ?? null)
        } catch (pairError) {
          // Pairing is what makes community features work; if it fails, do not
          // leave the UI looking signed in.
          await cognitoSignOut()
          throw pairError instanceof ApiError && pairError.status === 409
            ? pairingConflictMessage(pairError)
            : pairError
        }
        setIdToken(tokens.idToken)
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
      const unpaired = await api.community.unpair().catch(() => undefined)
      await cognitoSignOut()
      setClusterSync(unpaired?.cluster ?? null)
      setIdToken(undefined)
      setStatus('signed-out')
    },
  }), [status, idToken, clusterSync])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
