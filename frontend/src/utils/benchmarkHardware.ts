import type { BenchmarkHardwareIdentity } from '../api/types'

export const SPARK_HARDWARE = 'dgx-spark'
export const UNKNOWN_HARDWARE = 'unknown'

export function hardwareKey(item: BenchmarkHardwareIdentity) {
  return item.hardware_key || UNKNOWN_HARDWARE
}

export function hardwareLabel(item: BenchmarkHardwareIdentity) {
  return item.hardware_label || 'Unknown hardware'
}

export function matchesHardware(item: BenchmarkHardwareIdentity, selection: string) {
  return selection === SPARK_HARDWARE
    ? item.hardware?.hardware_class === SPARK_HARDWARE && hardwareKey(item) !== UNKNOWN_HARDWARE
    : hardwareKey(item) === selection
}

export function hardwareOptions(items: BenchmarkHardwareIdentity[]) {
  const options = new Map<string, string>()
  for (const item of items) options.set(hardwareKey(item), hardwareLabel(item))
  return [...options].sort((a, b) => a[1].localeCompare(b[1]))
}
