/**
 * Shared bounds for the History sampling interval.
 *
 * These mirror the backend's clamped range in `sparkdeck/live_metrics.py`: the
 * interval is the bucket span, so changing it changes the graph's time
 * resolution rather than the length of the trailing window.
 */
export const MIN_HISTORY_SAMPLE_SECONDS = 1
export const MAX_HISTORY_SAMPLE_SECONDS = 30
export const DEFAULT_HISTORY_SAMPLE_SECONDS = 5

export function clampHistorySampleSeconds(value: unknown): number {
  const seconds = Math.round(Number(value))
  if (!Number.isFinite(seconds)) return DEFAULT_HISTORY_SAMPLE_SECONDS
  return Math.min(MAX_HISTORY_SAMPLE_SECONDS, Math.max(MIN_HISTORY_SAMPLE_SECONDS, seconds))
}
