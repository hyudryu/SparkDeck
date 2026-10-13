export type IpAliases = Record<string, string>

/**
 * Resolve the friendly name configured for a caller address. Keys arrive in
 * canonical form from the backend, so an exact lookup is the common path;
 * the case-insensitive retry covers hand-written IPv6 keys that kept their
 * original casing before canonicalization existed.
 */
export function ipAliasFor(aliases: IpAliases | undefined, ip: string): string | undefined {
  if (!aliases) return undefined
  const key = ip.trim()
  if (!key) return undefined
  const direct = aliases[key]
  if (direct) return direct
  const lowered = key.toLowerCase()
  for (const [candidate, alias] of Object.entries(aliases)) {
    if (candidate.toLowerCase() === lowered) return alias
  }
  return undefined
}

/** Label for a live-session caller: the alias with the raw IP kept visible. */
export function callerLabel(aliases: IpAliases | undefined, ip: string): string {
  const alias = ipAliasFor(aliases, ip)
  return alias && alias !== ip ? `${alias} (${ip})` : ip
}

const IPV4_PATTERN = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/

/** Literal IPv4 or IPv6 text, for editor-side checks before a round-trip. */
export function isIpLiteral(value: string): boolean {
  const text = value.trim()
  if (!text) return false
  const ipv4 = IPV4_PATTERN.exec(text)
  if (ipv4) {
    return ipv4.slice(1).every((octet) => Number(octet) <= 255 && String(Number(octet)) === octet)
  }
  // A literal IPv6 may end with one embedded IPv4 tail (for example
  // '::ffff:10.0.0.5'); dots anywhere else in the body are typos. A loose
  // shape is acceptable here because the backend parser is the authority and
  // answers a bad one with a 400.
  if (!text.includes(':')) return false
  if (!/^[0-9a-fA-F:.]+$/.test(text)) return false
  const withoutIpv4Tail = text.replace(/(\d{1,3}\.){3}\d{1,3}$/, '')
  return !withoutIpv4Tail.includes('.') && text.split('::').length <= 2
}
