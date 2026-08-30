import { describe, expect, it } from 'vitest'
import { formatEnvironment, parseEnvironment } from './environment'

describe('runtime environment helpers', () => {
  it('round-trips literal values including empty and equals signs', () => {
    const environment = parseEnvironment('HF_HUB_OFFLINE=1\nEMPTY=\nJSON={"key":"a=b"}\n')
    expect(environment).toEqual({ HF_HUB_OFFLINE: '1', EMPTY: '', JSON: '{"key":"a=b"}' })
    expect(formatEnvironment(environment)).toBe('HF_HUB_OFFLINE=1\nEMPTY=\nJSON={"key":"a=b"}')
  })

  it('reports malformed and duplicate lines', () => {
    expect(() => parseEnvironment('NOT A NAME=value')).toThrow('line 1')
    expect(() => parseEnvironment('NO_EQUALS')).toThrow('line 1')
    expect(() => parseEnvironment('DUP=one\nDUP=two')).toThrow('duplicated')
  })

  it('keeps credentials in the secure settings flow', () => {
    expect(() => parseEnvironment('HF_TOKEN=secret')).toThrow('managed in Settings')
    expect(() => parseEnvironment('SERVICE_API_KEY=secret')).toThrow('looks secret')
  })
})
