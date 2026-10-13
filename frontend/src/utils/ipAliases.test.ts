import { describe, expect, it } from 'vitest'
import { callerLabel, ipAliasFor, isIpLiteral } from './ipAliases'

describe('ipAliasFor', () => {
  it('resolves canonical keys, trims the lookup, and retries case-insensitively', () => {
    const aliases = { '100.114.180.111': 'Studio laptop', 'FE80::1': 'Router' }
    expect(ipAliasFor(aliases, '100.114.180.111')).toBe('Studio laptop')
    expect(ipAliasFor(aliases, ' fe80::1 ')).toBe('Router')
    expect(ipAliasFor(aliases, '100.114.180.99')).toBeUndefined()
    expect(ipAliasFor(undefined, '10.0.0.1')).toBeUndefined()
    expect(ipAliasFor(aliases, '   ')).toBeUndefined()
  })
})

describe('callerLabel', () => {
  it('keeps the raw address visible next to the alias', () => {
    expect(callerLabel({ '10.0.0.5': 'Front desk' }, '10.0.0.5')).toBe('Front desk (10.0.0.5)')
  })

  it('falls back to the bare address when no alias is configured', () => {
    expect(callerLabel({}, '10.0.0.5')).toBe('10.0.0.5')
    expect(callerLabel(undefined, '10.0.0.5')).toBe('10.0.0.5')
  })
})

describe('isIpLiteral', () => {
  it.each(['10.0.0.5', '100.114.180.111', '255.255.255.255', 'fe80::1', '::1', '::ffff:10.0.0.5'])(
    'accepts %s', (value) => {
      expect(isIpLiteral(value)).toBe(true)
    },
  )

  it.each(['', '   ', 'front-desk', '10.0.0', '10.0.0.256', '010.0.0.5', '10.0.0.5:8080', 'g::1', 'fe80::1::2'])(
    'rejects %s', (value) => {
      expect(isIpLiteral(value)).toBe(false)
    },
  )
})
