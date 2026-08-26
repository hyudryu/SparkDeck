import { useEffect, useState, type FormEvent } from 'react'
import { Check, Download, Search, Users } from 'lucide-react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { RuntimeKind } from '../api/types'
import { EmptyState, ErrorState, formatNumber, formatRate, LoadingState, PageHeader, Panel, RuntimeMark } from '../components/ui'
import { useResource } from '../hooks/useResource'

export function ExplorePage() {
  const [draft, setDraft] = useState('')
  const [query, setQuery] = useState('')
  const [runtime, setRuntime] = useState<RuntimeKind | ''>('')
  const { data, error, loading, reload } = useResource(
    (signal) => api.catalog.search(query, runtime || undefined, undefined, signal),
    [query, runtime],
  )

  useEffect(() => {
    const timeout = window.setTimeout(() => setQuery(draft.trim()), 350)
    return () => window.clearTimeout(timeout)
  }, [draft])

  const submit = (event: FormEvent) => {
    event.preventDefault()
    setQuery(draft.trim())
  }

  return (
    <div className="page">
      <PageHeader
        eyebrow="Model catalog"
        title="Find the right model for your hardware"
        description="Search open models, check runtime compatibility, and compare measured community performance before you deploy."
      />
      <form className="catalog-search" onSubmit={submit} role="search">
        <label className="search-field">
          <span className="sr-only">Search models</span>
          <Search size={18} aria-hidden="true" />
          <input value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Search models, tasks, or authors" />
        </label>
        <label className="select-field compact-select">
          <span className="sr-only">Runtime</span>
          <select value={runtime} onChange={(event) => setRuntime(event.target.value as RuntimeKind | '')}>
            <option value="">All runtimes</option>
            <option value="vllm">vLLM</option>
            <option value="llama.cpp">llama.cpp</option>
            <option value="sglang">SGLang</option>
          </select>
        </label>
        <button className="button button-primary" type="submit">Search</button>
      </form>

      <div className="result-heading" aria-live="polite">
        <div>
          <h2>{query ? `Results for “${query}”` : 'Recommended models'}</h2>
          {!loading && data && <span>{formatNumber(data.total)} models</span>}
        </div>
        <p>Community results never include prompt or response content.</p>
      </div>

      {loading && <LoadingState label="Searching the catalog" />}
      {error && <ErrorState message={error} onRetry={reload} />}
      {!loading && !error && data?.items.length === 0 && (
        <EmptyState title="No models found" description="Try a broader name or remove the runtime filter." />
      )}
      {data && data.items.length > 0 && (
        <div className="model-grid">
          {data.items.map((model) => (
            <Panel className="model-card" key={model.id}>
              <div className="model-card-topline">
                <p>{model.author ?? model.id.split('/')[0] ?? 'Community'}</p>
                {model.local_deployment_ids && model.local_deployment_ids.length > 0 && (
                  <span className="local-chip"><Check size={12} /> Local</span>
                )}
              </div>
              <h2>{model.name ?? model.id.split('/').at(-1) ?? model.id}</h2>
              <p className="model-id">{model.id}</p>
              <div className="runtime-row" aria-label="Compatible runtimes">
                {(model.runtime_compatibility ?? []).filter((item) => item.supported).map((item) => (
                  <RuntimeMark key={item.runtime} runtime={item.runtime} />
                ))}
                {(!model.runtime_compatibility || model.runtime_compatibility.length === 0) && <span className="muted">Compatibility unknown</span>}
              </div>
              <dl className="model-metrics">
                <div><dt><Download size={14} /> Downloads</dt><dd>{formatNumber(model.downloads)}</dd></div>
                <div><dt><Users size={14} /> Samples</dt><dd>{formatNumber(model.community?.sample_count)}</dd></div>
                <div><dt>Median speed</dt><dd>{formatRate(model.community?.median_tokens_per_second)}</dd></div>
              </dl>
              <div className="model-card-footer">
                {model.community?.community_proven ? (
                  <span className="proven"><Check size={14} /> Community proven</span>
                ) : (
                  <span className="muted">More community data needed</span>
                )}
                <div className="model-card-actions"><a className="text-link" href={`https://huggingface.co/${model.id}`} target="_blank" rel="noreferrer">Details</a><Link className="button button-primary" aria-label={`Deploy ${model.id}`} to={`/models?model=${encodeURIComponent(model.id)}`}>Deploy</Link></div>
              </div>
            </Panel>
          ))}
        </div>
      )}
    </div>
  )
}
