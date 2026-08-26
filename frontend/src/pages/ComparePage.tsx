import { useMemo, useState, type FormEvent } from 'react'
import { ArrowRight, Gauge, GitCompareArrows } from 'lucide-react'
import { api } from '../api/client'
import type { ChatMessage, Deployment } from '../api/types'
import { Button, ErrorState, PageHeader, Panel, RuntimeMark, Status } from '../components/ui'
import { useResource } from '../hooks/useResource'

interface CompareResult {
  deployment: Deployment
  content?: string
  latency?: number
  error?: string
}

export function ComparePage() {
  const deployments = useResource((signal) => api.deployments.list(signal))
  const running = useMemo(() => deployments.data?.filter((item) => item.status === 'running') ?? [], [deployments.data])
  const [left, setLeft] = useState('')
  const [right, setRight] = useState('')
  const [prompt, setPrompt] = useState('')
  const [results, setResults] = useState<CompareResult[]>([])
  const [busy, setBusy] = useState(false)

  const run = async (event: FormEvent) => {
    event.preventDefault()
    const selected = [left || running[0]?.alias, right || running[1]?.alias].filter(Boolean)
    if (selected.length !== 2 || !prompt.trim()) return
    setBusy(true)
    const message: ChatMessage = { role: 'user', content: prompt.trim() }
    const next = await Promise.all(selected.map(async (alias): Promise<CompareResult> => {
      const deployment = running.find((item) => item.alias === alias)!
      const started = performance.now()
      try {
        const response = await api.chat(alias!, [message])
        return { deployment, content: response.choices[0]?.message.content ?? '', latency: performance.now() - started }
      } catch (reason) {
        return { deployment, error: reason instanceof Error ? reason.message : 'Request failed' }
      }
    }))
    setResults(next)
    setBusy(false)
  }

  return (
    <div className="page">
      <PageHeader eyebrow="Side-by-side evaluation" title="Compare" description="Send the same prompt to two local models and inspect quality and observed latency together." />
      {deployments.error && <ErrorState message={deployments.error} onRetry={deployments.reload} />}
      <form className="compare-setup" onSubmit={(event) => void run(event)}>
        <div className="compare-selectors">
          <label className="field"><span>Model A</span><select value={left || running[0]?.alias || ''} onChange={(event) => setLeft(event.target.value)}><option value="">Choose a running model</option>{running.map((item) => <option key={item.id} value={item.alias}>{item.alias}</option>)}</select></label>
          <span className="versus">vs</span>
          <label className="field"><span>Model B</span><select value={right || running[1]?.alias || ''} onChange={(event) => setRight(event.target.value)}><option value="">Choose a running model</option>{running.map((item) => <option key={item.id} value={item.alias}>{item.alias}</option>)}</select></label>
        </div>
        <label className="field"><span>Comparison prompt</span><textarea rows={4} value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Ask a question that will reveal meaningful differences…" /></label>
        <div className="compare-submit"><p><Gauge size={14} /> Both requests produce private local benchmark samples.</p><Button type="submit" variant="primary" disabled={busy || running.length < 2 || !prompt.trim()}>{busy ? 'Comparing…' : <><GitCompareArrows size={16} /> Compare models</>}</Button></div>
      </form>
      {results.length === 0 ? (
        <div className="comparison-placeholder"><ArrowRight size={18} /><p>Results will appear side by side on larger screens and stack vertically on mobile.</p></div>
      ) : (
        <div className="comparison-grid" aria-live="polite">
          {results.map((result) => (
            <Panel className="comparison-result" key={result.deployment.id}>
              <div className="comparison-heading"><div><h2>{result.deployment.alias}</h2><p>{result.deployment.model_id}</p></div><RuntimeMark runtime={result.deployment.runtime} /></div>
              <div className="comparison-meta"><Status status={result.error ? 'error' : 'completed'} />{result.latency && <span>{(result.latency / 1000).toFixed(2)} s observed</span>}</div>
              {result.error ? <p className="inline-error">{result.error}</p> : <div className="response-content">{result.content}</div>}
            </Panel>
          ))}
        </div>
      )}
    </div>
  )
}
