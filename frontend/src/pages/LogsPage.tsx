import { useEffect, useMemo, useState } from 'react'
import { Download, RefreshCw, Search } from 'lucide-react'
import { api } from '../api/client'
import { Button, EmptyState, ErrorState, LoadingState, PageHeader, Panel } from '../components/ui'
import { useResource } from '../hooks/useResource'

export function LogsPage() {
  const resource = useResource((signal) => api.logs.list(signal))
  const { reload } = resource
  useEffect(() => {
    const timer = window.setInterval(reload, 5000)
    return () => window.clearInterval(timer)
  }, [reload])
  const [query, setQuery] = useState('')
  const [level, setLevel] = useState('')
  const activity = useMemo(() => (resource.data ?? []).filter((entry) => {
    const isError = ['error', 'critical', 'fatal'].includes(entry.level?.toLowerCase() ?? '')
    const isLifecycle = ['launched', 'stopped', 'crashed'].includes(entry.event ?? '')
    return isError || isLifecycle
  }), [resource.data])
  const entries = useMemo(() => activity.filter((entry) => {
    const matchesQuery = !query || `${entry.source ?? ''} ${entry.message}`.toLowerCase().includes(query.toLowerCase())
    const isError = ['error', 'critical', 'fatal'].includes(entry.level?.toLowerCase() ?? '')
    const isLifecycle = ['launched', 'stopped', 'crashed'].includes(entry.event ?? '')
    const matchesLevel = !level || (level === 'error' ? isError : isLifecycle)
    return matchesQuery && matchesLevel
  }), [activity, query, level])

  const download = () => {
    const blob = new Blob([entries.map((entry) => `${entry.timestamp ?? ''} ${entry.level ?? ''} ${entry.source ?? ''} ${entry.message}`.trim()).join('\n')], { type: 'text/plain' })
    const href = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = href
    anchor.download = `sparkdeck-logs-${new Date().toISOString().slice(0, 10)}.txt`
    anchor.click()
    URL.revokeObjectURL(href)
  }

  return (
    <div className="page logs-page">
      <PageHeader eyebrow="Diagnostics" title="Logs" description="Deployment launches, shutdowns, crashes, and errors. Secrets are redacted before entries reach this view." actions={<><Button onClick={resource.reload}><RefreshCw size={15} /> Refresh</Button><Button onClick={download} disabled={!entries.length}><Download size={15} /> Export</Button></>} />
      <div className="log-filters">
        <label className="search-field"><span className="sr-only">Filter logs</span><Search size={17} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Filter log messages" /></label>
        <label className="select-field"><span className="sr-only">Event type</span><select value={level} onChange={(event) => setLevel(event.target.value)}><option value="">All events</option><option value="lifecycle">Deployment activity</option><option value="error">Errors</option></select></label>
      </div>
      {resource.loading && <LoadingState label="Loading logs" />}
      {resource.error && <ErrorState message={resource.error} onRetry={resource.reload} />}
      {!resource.loading && !resource.error && entries.length === 0 && <EmptyState title={activity.length ? 'No matching entries' : 'No log entries'} description={activity.length ? 'Change the filters to see more activity.' : 'Deployment launches, shutdowns, crashes, and errors will appear here.'} />}
      {entries.length > 0 && <Panel className="log-view" aria-label="Application logs" tabIndex={0}>{entries.map((entry, index) => <div className="log-line" key={`${entry.timestamp}-${index}`}><time>{entry.timestamp ?? '—'}</time><span className={`log-level log-${entry.level?.toLowerCase() ?? 'info'}`}>{entry.level ?? 'info'}</span><span className="log-source">{entry.source ?? 'sparkdeck'}</span><span>{entry.message}</span></div>)}</Panel>}
    </div>
  )
}
