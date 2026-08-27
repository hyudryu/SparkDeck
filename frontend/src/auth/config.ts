// The backend owns Cognito runtime configuration so its verifier, CSP, and the
// browser always target the same pool after environment overrides.
import { api } from '../api/client'
import type { CommunityAuthConfig } from '../api/types'

export async function cognitoConfig(): Promise<CommunityAuthConfig> {
  const config = await api.community.authConfig()
  let endpoint: URL
  try {
    endpoint = new URL(config.idp_endpoint)
  } catch {
    throw new Error('The community sign-in configuration is invalid')
  }
  if (endpoint.protocol !== 'https:' || !config.client_id.trim()) {
    throw new Error('The community sign-in configuration is invalid')
  }
  return {
    idp_endpoint: endpoint.toString(),
    client_id: config.client_id.trim(),
  }
}
