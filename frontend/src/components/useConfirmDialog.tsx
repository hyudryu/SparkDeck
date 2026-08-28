import { useCallback, useEffect, useId, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import { AlertTriangle, X } from 'lucide-react'
import { Button } from './ui'

type Confirmation = {
  title: string
  message: ReactNode
  confirmLabel?: string
  danger?: boolean
}

export function useConfirmDialog() {
  const titleId = useId()
  const [confirmation, setConfirmation] = useState<Confirmation>()
  const resolver = useRef<((accepted: boolean) => void) | undefined>(undefined)
  const cancelRef = useRef<HTMLButtonElement>(null)
  const returnFocusRef = useRef<HTMLElement | null>(null)

  const close = useCallback((accepted: boolean) => {
    resolver.current?.(accepted)
    resolver.current = undefined
    setConfirmation(undefined)
  }, [])

  const confirm = useCallback((next: Confirmation) => new Promise<boolean>((resolve) => {
    resolver.current?.(false)
    resolver.current = resolve
    returnFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null
    setConfirmation(next)
  }), [])

  useEffect(() => () => resolver.current?.(false), [])
  useEffect(() => {
    if (!confirmation) return
    cancelRef.current?.focus()
    const returnFocusTarget = returnFocusRef.current
    return () => returnFocusTarget?.focus()
  }, [confirmation])

  const keepFocusInside = (event: KeyboardEvent<HTMLElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault()
      close(false)
      return
    }
    if (event.key !== 'Tab') return
    const focusable = Array.from(event.currentTarget.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    ))
    if (!focusable.length) return
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault()
      first.focus()
    }
  }

  const confirmationDialog = confirmation ? (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && close(false)}>
      <section
        className="modal confirmation-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onKeyDown={keepFocusInside}
      >
        <div className="modal-heading">
          <div className="confirmation-title"><AlertTriangle size={18} aria-hidden="true" /><h2 id={titleId}>{confirmation.title}</h2></div>
          <button ref={cancelRef} className="icon-button" onClick={() => close(false)} aria-label="Close dialog"><X size={17} /></button>
        </div>
        <div className="modal-description confirmation-message">{confirmation.message}</div>
        <div className="modal-actions">
          <Button onClick={() => close(false)}>Cancel</Button>
          <Button variant={confirmation.danger ? 'danger' : 'primary'} onClick={() => close(true)}>{confirmation.confirmLabel ?? 'Continue'}</Button>
        </div>
      </section>
    </div>
  ) : null

  return { confirm, confirmationDialog }
}
