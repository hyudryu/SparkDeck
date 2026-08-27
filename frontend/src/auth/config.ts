// Cognito IDP configuration for community sign-in. The defaults point at the
// deployed sparkdeck-community-auth stack; localStorage overrides exist so
// forks can point a dev build at their own pool without rebuilding.

const DEFAULT_COGNITO_IDP_ENDPOINT = 'https://cognito-idp.us-east-2.amazonaws.com/'
const DEFAULT_COGNITO_CLIENT_ID = '30ihrkeg4k1rn95d4mmkq00fvl'

export function cognitoIdpEndpoint(): string {
  return localStorage.getItem('sparkdeck.cognito.idp_endpoint') ?? DEFAULT_COGNITO_IDP_ENDPOINT
}

export function cognitoClientId(): string {
  return localStorage.getItem('sparkdeck.cognito.client_id') ?? DEFAULT_COGNITO_CLIENT_ID
}
