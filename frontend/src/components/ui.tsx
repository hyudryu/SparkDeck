import { useEffect, useRef, useState, type ButtonHTMLAttributes, type HTMLAttributes, type ReactNode } from 'react'
import { AlertCircle, ChevronDown, Inbox, LoaderCircle, RefreshCw } from 'lucide-react'
import { createPortal } from 'react-dom'

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string
  title: string
  description: string
  actions?: ReactNode
}) {
  return (
    <header className="page-header">
      <div>
        {eyebrow && <p className="eyebrow">{eyebrow}</p>}
        <h1>{title}</h1>
        <p className="page-description">{description}</p>
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  )
}

export function Button({
  variant = 'secondary',
  className = '',
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'secondary' | 'tertiary' | 'danger'
}) {
  return <button className={`button button-${variant} ${className}`.trim()} {...props} />
}

export function Status({ status, children }: { status: string; children?: ReactNode }) {
  const normalized = status.toLowerCase().replaceAll(' ', '-')
  return (
    <span className="status">
      <span className={`status-dot status-${normalized}`} aria-hidden="true" />
      {children ?? status}
    </span>
  )
}

export function RuntimeMark({ runtime }: { runtime: string }) {
  const labels: Record<string, string> = {
    vllm: 'vLLM',
    'llama.cpp': 'Llama server',
    sglang: 'SGLang',
  }
  return <span className="runtime-mark">{labels[runtime] ?? runtime}</span>
}

export function LoadingState({ label = 'Loading' }: { label?: string }) {
  return (
    <div className="state-panel" role="status">
      <LoaderCircle className="spin" size={20} aria-hidden="true" />
      <span>{label}</span>
    </div>
  )
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="state-panel state-error" role="alert">
      <AlertCircle size={20} aria-hidden="true" />
      <div>
        <strong>Couldn’t load this view</strong>
        <p>{message}</p>
      </div>
      {onRetry && (
        <Button onClick={onRetry}>
          <RefreshCw size={15} aria-hidden="true" /> Retry
        </Button>
      )}
    </div>
  )
}

export function EmptyState({
  title,
  description,
  action,
}: {
  title: string
  description: string
  action?: ReactNode
}) {
  return (
    <div className="empty-state">
      <Inbox size={22} aria-hidden="true" />
      <h2>{title}</h2>
      <p>{description}</p>
      {action}
    </div>
  )
}

export function Panel({ className = '', ...props }: HTMLAttributes<HTMLElement>) {
  return <section className={`panel ${className}`.trim()} {...props} />
}

// Hover/focus tooltip rendered through a portal with fixed positioning so
// table overflow clipping and stacking contexts cannot hide it.
export function Tooltip({ label, children }: { label: ReactNode; children: ReactNode }) {
  const [anchor, setAnchor] = useState<{ x: number; y: number; below: boolean }>()
  const show = (target: HTMLElement) => {
    const rect = target.getBoundingClientRect()
    // Near the top edge there is no room above the badge, so flip below when
    // that side has more space; horizontally clamp the centered tooltip
    // inside the viewport.
    const below = rect.top < 72 && window.innerHeight - rect.bottom > rect.top
    const halfWidth = Math.min(170, window.innerWidth * 0.45)
    setAnchor({
      x: Math.min(
        Math.max(rect.left + rect.width / 2, halfWidth + 8),
        window.innerWidth - halfWidth - 8,
      ),
      y: below ? rect.bottom : rect.top,
      below,
    })
  }
  return (
    <span
      tabIndex={0}
      onMouseEnter={(event) => show(event.currentTarget)}
      onMouseLeave={() => setAnchor(undefined)}
      onFocus={(event) => show(event.currentTarget)}
      onBlur={() => setAnchor(undefined)}
    >
      {children}
      {anchor && createPortal(
        <div
          className={`tooltip${anchor.below ? ' tooltip-below' : ''}`}
          role="tooltip"
          style={{ left: anchor.x, top: anchor.y }}
        >{label}</div>,
        document.body,
      )}
    </span>
  )
}

export interface SplitButtonItem {
  key: string
  label: ReactNode
  disabled?: boolean
  onSelect: () => void
}

// Primary action with a side caret that opens a small action menu, similar to
// GitHub's merge button. The menu renders through a portal anchored to the
// caret so enclosing panels with overflow: hidden cannot clip it.
export function SplitButton({
  label,
  onMainAction,
  items,
  disabled = false,
  mainAriaLabel,
  toggleAriaLabel,
}: {
  label: ReactNode
  onMainAction: () => void
  items: SplitButtonItem[]
  disabled?: boolean
  mainAriaLabel?: string
  toggleAriaLabel: string
}) {
  const toggleRef = useRef<HTMLButtonElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const [anchor, setAnchor] = useState<{ top?: number; bottom?: number; right: number }>()
  const closeMenu = () => {
    setAnchor(undefined)
    // The portal lives at the end of document.body, so returning focus to the
    // toggle keeps keyboard users anchored in the row they started from.
    toggleRef.current?.focus()
  }
  const toggleMenu = () => {
    if (anchor) {
      closeMenu()
      return
    }
    const rect = toggleRef.current?.getBoundingClientRect()
    if (!rect) return
    const right = window.innerWidth - rect.right
    // Rows near the bottom of the viewport would push a below-anchored menu
    // off-screen (fixed elements cannot be scrolled into view), so flip to
    // whichever side has room.
    const estimatedHeight = items.length * 40 + 12
    const spaceBelow = window.innerHeight - rect.bottom
    const spaceAbove = rect.top
    setAnchor(spaceBelow < estimatedHeight && spaceAbove > spaceBelow
      ? { bottom: window.innerHeight - rect.top + 4, right }
      : { top: rect.bottom + 4, right })
  }
  const select = (item: SplitButtonItem) => {
    closeMenu()
    item.onSelect()
  }

  // Move keyboard focus into the portal menu once it is committed.
  useEffect(() => {
    if (!anchor) return
    menuRef.current
      ?.querySelector<HTMLButtonElement>('button[role="menuitem"]:not(:disabled)')
      ?.focus()
  }, [anchor])
  return (
    <span className="split-button">
      <Button variant="tertiary" disabled={disabled} aria-label={mainAriaLabel} onClick={onMainAction}>{label}</Button>
      <button
        ref={toggleRef}
        type="button"
        className="button button-tertiary split-button-toggle"
        aria-label={toggleAriaLabel}
        aria-haspopup="menu"
        aria-expanded={anchor !== undefined}
        disabled={disabled}
        onClick={toggleMenu}
      ><ChevronDown size={14} /></button>
      {anchor && createPortal(
        <>
          <div className="menu-backdrop" onMouseDown={closeMenu} />
          <div
            ref={menuRef}
            className="menu"
            role="menu"
            aria-label={toggleAriaLabel}
            style={{
              right: anchor.right,
              ...(anchor.top !== undefined ? { top: anchor.top } : { bottom: anchor.bottom }),
            }}
            onKeyDown={(event) => event.key === 'Escape' && closeMenu()}
          >
            {items.map((item) => (
              <button
                key={item.key}
                type="button"
                role="menuitem"
                disabled={item.disabled}
                onClick={() => select(item)}
              >{item.label}</button>
            ))}
          </div>
        </>,
        document.body,
      )}
    </span>
  )
}

export function formatNumber(value?: number | null) {
  if (value == null) return '—'
  return Intl.NumberFormat('en', { notation: value >= 10_000 ? 'compact' : 'standard' }).format(value)
}

export function formatRate(value?: number | null) {
  return value == null ? '—' : `${value.toFixed(1)} tok/s`
}

export function formatDuration(value?: number | null) {
  if (value == null) return '—'
  return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`
}
