const ENVIRONMENT_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/
const PROTECTED_NAMES = new Set(['HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'])
const SECRET_NAME = /(?:^|_)(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|PRIVATE_KEY|ACCESS_KEY)(?:$|_)/i

export function parseEnvironment(input: string): Record<string, string> {
  const environment: Record<string, string> = {}
  input.split(/\r?\n/).forEach((line, index) => {
    if (!line.trim()) return
    const separator = line.indexOf('=')
    if (separator < 1) throw new Error(`Environment line ${index + 1} must use NAME=value`)
    const name = line.slice(0, separator).trim()
    const value = line.slice(separator + 1)
    if (!ENVIRONMENT_NAME.test(name)) throw new Error(`Environment line ${index + 1} has an invalid variable name`)
    if (PROTECTED_NAMES.has(name)) throw new Error(`${name} is managed in Settings, not deployment environment variables`)
    if (SECRET_NAME.test(name)) throw new Error(`${name} looks secret and cannot be stored in deployment environment variables`)
    if (Object.prototype.hasOwnProperty.call(environment, name)) throw new Error(`Environment variable ${name} is duplicated`)
    environment[name] = value
  })
  return environment
}

export function formatEnvironment(environment?: Record<string, string>): string {
  return Object.entries(environment ?? {}).map(([name, value]) => `${name}=${value}`).join('\n')
}
