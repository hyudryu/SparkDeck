import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode } from 'react'
import { AlertCircle, Inbox, LoaderCircle, RefreshCw } from 'lucide-react'

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
    'llama.cpp': 'llama.cpp',
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

export function formatNumber(value?: number) {
  if (value === undefined) return '—'
  return Intl.NumberFormat('en', { notation: value >= 10_000 ? 'compact' : 'standard' }).format(value)
}

export function formatRate(value?: number) {
  return value === undefined ? '—' : `${value.toFixed(1)} tok/s`
}

export function formatDuration(value?: number) {
  if (value === undefined) return '—'
  return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`
}
