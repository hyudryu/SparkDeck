import { describe, expect, it } from 'vitest'
import { formatBytes } from './format'

describe('formatBytes', () => {
  it('renders empty values as zero bytes', () => {
    expect(formatBytes(0)).toBe('0 B')
    expect(formatBytes(-1)).toBe('0 B')
    expect(formatBytes(null)).toBe('0 B')
    expect(formatBytes(undefined)).toBe('0 B')
    expect(formatBytes(Number.NaN)).toBe('0 B')
    expect(formatBytes(Number.POSITIVE_INFINITY)).toBe('0 B')
  })

  it('uses decimal units so sizes match the Hugging Face Hub', () => {
    // The DeepSeek-V4.1-Flash repository as listed by the Hub. A binary
    // divisor labelled "GB" reported this same cache as 475 GB, which made
    // transfer progress disagree with the Hub page it was started from.
    expect(formatBytes(510_313_353_565)).toBe('510 GB')
    expect(formatBytes(307_240_296_645)).toBe('307 GB')
    expect(formatBytes(203_073_056_920)).toBe('203 GB')
  })

  it('steps through decimal units at 1000-byte boundaries', () => {
    expect(formatBytes(999)).toBe('999 B')
    expect(formatBytes(1_000)).toBe('1.0 KB')
    expect(formatBytes(1_000_000)).toBe('1.0 MB')
    expect(formatBytes(1_000_000_000)).toBe('1.0 GB')
    expect(formatBytes(1_000_000_000_000)).toBe('1.0 TB')
  })

  it('keeps one decimal below ten units and rounds above', () => {
    expect(formatBytes(2_000_000_000)).toBe('2.0 GB')
    expect(formatBytes(9_400_000_000)).toBe('9.4 GB')
    expect(formatBytes(15_032_385_536)).toBe('15 GB')
  })

  it('stops at terabytes for values beyond the unit table', () => {
    expect(formatBytes(1_000_000_000_000_000)).toBe('1000 TB')
  })
})
