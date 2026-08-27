import { api } from '../api/client'
import { useAuth } from '../auth/AuthContext'
import { useResource } from './useResource'

export const COMMUNITY_ACCESS_HINT = 'Sign in and enable telemetry in Settings → Community Features to see community data.'

// Community features are enabled only for a signed-in user who has also opted
// in to telemetry sharing. Call reload() after consent or pairing changes so
// the gated UI does not go stale.
export function useCommunityAccess() {
  const { status } = useAuth()
  const sync = useResource((signal) => api.benchmarks.syncStatus(signal))
  const signedIn = status === 'signed-in'
  const sharingEnabled = Boolean(sync.data?.sharing_enabled)
  return {
    signedIn,
    sharingEnabled,
    loading: sync.loading,
    enabled: signedIn && sharingEnabled,
    reload: sync.reload,
  }
}
