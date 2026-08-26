import { useMemo, useState } from 'react'
import { Download, RefreshCw, Search } from 'lucide-react'
import { api } from '../api/client'
import { Button, EmptyState, ErrorState, LoadingState, PageHeader, Panel } from '../components/ui'
import { useResource } from '../hooks/useResource'

export function LogsPage() {
  const resource = useResource((signal) => api.logs.list(signal))
  const [query, setQuery] = useState('')
  const [level, setLevel] = useState('')
  const entries = useMemo(() => (resource.data ?? []).filter((entry) => {
    const matchesQuery = !query || `${entry.source ?? ''} ${entry.message}`.toLowerCase().includes(query.toLowerCase())
    const matchesLevel = !level || entry.level?.toLowerCase() === level
    return matchesQuery && matchesLevel
  }), [resource.data, query, level])

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
      <PageHeader eyebrow="Diagnostics" title="Logs" description="Inspect local SparkDeck and runtime activity. Secrets are redacted before entries reach this view." actions={<><Button onClick={resource.reload}><RefreshCw size={15} /> Refresh</Button><Button onClick={download} disabled={!entries.length}><Download size={15} /> Export</Button></>} />
      <div className="log-filters">
        <label className="search-field"><span className="sr-only">Filter logs</span><Search size={17} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Filter log messages" /></label>
        <label className="select-field"><span className="sr-only">Log level</span><select value={level} onChange={(event) => setLevel(event.target.value)}><option value="">All levels</option><option value="debug">Debug</option><option value="info">Info</option><option value="warning">Warning</option><option value="error">Error</option></select></label>
      </div>
      {resource.loading && <LoadingState label="Loading logs" />}
      {resource.error && <ErrorState message={resource.error} onRetry={resource.reload} />}
      {!resource.loading && !resource.error && entries.length === 0 && <EmptyState title={resource.data?.length ? 'No matching entries' : 'No log entries'} description={resource.data?.length ? 'Change the filters to see more activity.' : 'Runtime and application events will appear here.'} />}
      {entries.length > 0 && <Panel className="log-view" aria-label="Application logs" tabIndex={0}>{entries.map((entry, index) => <div className="log-line" key={`${entry.timestamp}-${index}`}><time>{entry.timestamp ?? '—'}</time><span className={`log-level log-${entry.level?.toLowerCase() ?? 'info'}`}>{entry.level ?? 'info'}</span><span className="log-source">{entry.source ?? 'sparkdeck'}</span><span>{entry.message}</span></div>)}</Panel>}
    </div>
  )
}
