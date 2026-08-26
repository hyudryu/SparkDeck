export function formatBytes(bytes?: number | null) {
  if (!Number.isFinite(bytes) || Number(bytes) <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const exponent = Math.min(Math.floor(Math.log(Number(bytes)) / Math.log(1024)), units.length - 1)
  const value = Number(bytes) / 1024 ** exponent
  return `${value >= 10 || exponent === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[exponent]}`
}
