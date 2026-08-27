import { useEffect, useRef, type KeyboardEvent, type ReactNode, type RefObject } from 'react'
import { X } from 'lucide-react'

interface LegalDialogProps {
  eyebrow: string
  title: string
  titleId: string
  children: ReactNode
  onClose: () => void
  returnFocusRef: RefObject<HTMLButtonElement | null>
}

export function LegalDialog({ eyebrow, title, titleId, children, onClose, returnFocusRef }: LegalDialogProps) {
  const closeRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    const returnFocusTarget = returnFocusRef.current
    closeRef.current?.focus()
    const closeOnEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', closeOnEscape)
    return () => {
      window.removeEventListener('keydown', closeOnEscape)
      returnFocusTarget?.focus()
    }
  }, [onClose, returnFocusRef])

  const keepFocusInside = (event: KeyboardEvent<HTMLElement>) => {
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

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="modal legal-modal" role="dialog" aria-modal="true" aria-labelledby={titleId} onKeyDown={keepFocusInside}>
        <div className="modal-heading">
          <div><p className="eyebrow">{eyebrow}</p><h2 id={titleId}>{title}</h2></div>
          <button ref={closeRef} className="icon-button" type="button" onClick={onClose} aria-label={`Close ${title}`}><X size={17} /></button>
        </div>
        <div className="legal-content">{children}</div>
        <div className="modal-actions"><button className="button button-primary" type="button" onClick={onClose}>Done</button></div>
      </section>
    </div>
  )
}
