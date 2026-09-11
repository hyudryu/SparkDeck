/**
 * Render a byte count using decimal (SI) units.
 *
 * SparkDeck deliberately uses the same 1000-based convention as the Hugging
 * Face Hub, container registries, and disk vendors, so a model the Hub lists as
 * 510 GB reads as "510 GB" here as well. A binary divisor labelled "GB" would
 * render that same repository as 475 GB and make transfer progress disagree
 * with the Hub page it was started from.
 */
export function formatBytes(bytes?: number | null) {
  if (!Number.isFinite(bytes) || Number(bytes) <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const exponent = Math.min(Math.floor(Math.log(Number(bytes)) / Math.log(1000)), units.length - 1)
  const value = Number(bytes) / 1000 ** exponent
  return `${value >= 10 || exponent === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[exponent]}`
}
