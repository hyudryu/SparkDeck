import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { api, ApiError } from '../api/client'
import type { CommunityClusterSync } from '../api/types'
import {
  confirmForgotPassword,
  confirmSignUp,
  decodeIdToken,
  forgetStoredTokens,
  forgotPassword,
  resendCode,
  signIn as cognitoSignIn,
  signOut as cognitoSignOut,
  signUp,
} from './cognitoAuth'

export const COMMUNITY_SESSION_RENEW_MS = 30 * 60 * 1000
export const COMMUNITY_SESSION_RETRY_MS = 2 * 1000
export const COMMUNITY_SESSION_RETRY_MAX_MS = 30 * 1000

export interface AuthState {
  status: 'restoring' | 'signed-out' | 'signing-in' | 'signed-in' | 'reauth-required'
  email?: string
  /** Cluster fan-out result from the latest sign-in/out, if the backend reported one. */
  clusterSync?: CommunityClusterSync | null
  signIn: (email: string, password: string) => Promise<void>
  signUp: (email: string, password: string) => Promise<void>
  confirmSignUp: (email: string, code: string) => Promise<void>
  resendCode: (email: string) => Promise<void>
  forgotPassword: (email: string) => Promise<void>
  confirmForgotPassword: (email: string, code: string, newPassword: string) => Promise<void>
  signOut: (password: string) => Promise<void>
}

// A signed-out default keeps components usable without a provider (tests,
// isolated renders); the real session is supplied by AuthProvider.
const signedOutDefault: AuthState = {
  status: 'signed-out',
  signIn: async (email, password) => {
    const tokens = await cognitoSignIn(email, password)
    await api.community.pair(tokens.idToken, tokens.refreshToken)
    forgetStoredTokens()
  },
  signUp,
  confirmSignUp,
  resendCode,
  forgotPassword,
  confirmForgotPassword,
  signOut: async () => {
    throw new Error('Community session is unavailable')
  },
}

const AuthContext = createContext<AuthState>(signedOutDefault)

export function useAuth(): AuthState {
  return useContext(AuthContext)
}

function pairingConflictMessage(error: ApiError): Error {
  const existing = (error.body as { existing?: { email?: string } } | undefined)?.existing?.email
  return new Error(`This node is already signed in as ${existing ?? 'another account'}`)
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthState['status']>('restoring')
  const [email, setEmail] = useState<string>()
  const [clusterSync, setClusterSync] = useState<CommunityClusterSync | null>(null)
  const sessionGeneration = useRef(0)

  useEffect(() => {
    let cancelled = false
    let retryTimer: number | undefined
    let retryDelay = COMMUNITY_SESSION_RETRY_MS
    const restore = async () => {
      try {
        // The node is authoritative. Browser-origin tokens from an older
        // session must never silently re-pair a cluster that was signed out.
        const session = await api.community.session()
        if (cancelled) return
        forgetStoredTokens()
        setClusterSync(null)
        setEmail(session.email)
        setStatus(session.status)
        retryDelay = COMMUNITY_SESSION_RETRY_MS
      } catch {
        if (cancelled) return
        // A failed request is not an authoritative sign-out. Keep the account
        // UI in its non-interactive restore state and retry so a transient
        // controller, forwarding, or Cognito failure cannot expose the sign-in
        // form for a node that is still paired.
        const delay = retryDelay
        retryDelay = Math.min(retryDelay * 2, COMMUNITY_SESSION_RETRY_MAX_MS)
        retryTimer = window.setTimeout(() => {
          void restore()
        }, delay)
      }
    }
    void restore()
    return () => {
      cancelled = true
      if (retryTimer !== undefined) window.clearTimeout(retryTimer)
    }
  }, [])

  useEffect(() => {
    if (status !== 'signed-in') return
    const generation = sessionGeneration.current
    const timer = window.setInterval(() => {
      void api.community.session().then((session) => {
        if (sessionGeneration.current !== generation) return
        forgetStoredTokens()
        setEmail(session.email)
        setStatus(session.status)
      }).catch(() => {
        // A transient renewal failure must not sign out a working page. The
        // next interval or a protected-request retry can renew the cookie.
      })
    }, COMMUNITY_SESSION_RENEW_MS)
    return () => window.clearInterval(timer)
  }, [status])

  const value = useMemo<AuthState>(() => ({
    status,
    email,
    clusterSync,
    signIn: async (accountEmail, password) => {
      if (status === 'restoring') {
        throw new Error('Wait for the saved community session to finish restoring, then try again.')
      }
      setStatus('signing-in')
      setClusterSync(null)
      try {
        const tokens = await cognitoSignIn(accountEmail, password)
        try {
          const paired = await api.community.pair(tokens.idToken, tokens.refreshToken)
          setClusterSync(paired.cluster ?? null)
          forgetStoredTokens()
        } catch (pairError) {
          // Pairing is what makes community features work; if it fails, do not
          // leave the UI looking signed in.
          await cognitoSignOut()
          throw pairError instanceof ApiError && pairError.status === 409
            ? pairingConflictMessage(pairError)
            : pairError
        }
        setEmail(decodeIdToken(tokens.idToken).email)
        setStatus('signed-in')
      } catch (reason) {
        setEmail(undefined)
        setStatus('signed-out')
        throw reason
      }
    },
    signUp,
    confirmSignUp,
    resendCode,
    forgotPassword,
    confirmForgotPassword,
    signOut: async (password) => {
      setClusterSync(null)
      if (!email) throw new Error('The paired account email is unavailable; reload and try again.')
      const tokens = await cognitoSignIn(email, password)
      try {
        const unpaired = await api.community.unpair(tokens.idToken)
        sessionGeneration.current += 1
        setClusterSync(unpaired?.cluster ?? null)
        setEmail(undefined)
        setStatus('signed-out')
      } finally {
        await cognitoSignOut()
      }
    },
  }), [status, email, clusterSync])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
