function copyWithSelection(value: string) {
  const input = document.createElement('textarea')
  input.value = value
  input.setAttribute('readonly', '')
  input.setAttribute('aria-hidden', 'true')
  input.style.position = 'fixed'
  input.style.inset = '0 auto auto -9999px'
  input.style.opacity = '0'
  const activeElement = document.activeElement instanceof HTMLElement
    ? document.activeElement
    : undefined
  document.body.appendChild(input)
  let copied: boolean
  try {
    input.select()
    copied = document.execCommand('copy')
  } finally {
    input.remove()
    activeElement?.focus()
  }
  if (!copied) throw new Error('Browser rejected the copy command')
}

/**
 * Write a value to the clipboard from within a user gesture. Falls back to a
 * hidden textarea and execCommand when the modern API is unavailable or the
 * page is not on a secure origin (the cluster tailnet serves plain HTTP).
 */
export async function copyToClipboard(value: string) {
  if (window.isSecureContext && navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(value)
      return
    } catch {
      // Permission policies can block the modern API even on secure origins.
      // Continue within the click handler using the selection-based fallback.
    }
  }
  copyWithSelection(value)
}
