import { useMemo, useState, type FormEvent } from 'react'
import { ArrowUp, Bot, Gauge, Trash2, User } from 'lucide-react'
import { api } from '../api/client'
import type { ChatMessage } from '../api/types'
import { Button, ErrorState, PageHeader, RuntimeMark } from '../components/ui'
import { useResource } from '../hooks/useResource'

export function ChatPage() {
  const deployments = useResource((signal) => api.deployments.list(signal))
  const running = useMemo(() => deployments.data?.filter((item) => item.status === 'running') ?? [], [deployments.data])
  const [model, setModel] = useState('')
  const [draft, setDraft] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string>()
  const selectedModel = model || running[0]?.alias || ''

  const send = async (event: FormEvent) => {
    event.preventDefault()
    const content = draft.trim()
    if (!content || !selectedModel || sending) return
    const next: ChatMessage[] = [...messages, { role: 'user', content }]
    setMessages(next)
    setDraft('')
    setSending(true)
    setError(undefined)
    try {
      const response = await api.chat(selectedModel, next)
      const answer = response.choices[0]?.message
      if (answer) setMessages([...next, answer])
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'The model did not respond')
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="page chat-page">
      <PageHeader
        eyebrow="Inference workspace"
        title="Chat"
        description="Talk to any running model. Eligible performance measurements are captured locally without message content."
        actions={messages.length > 0 ? <Button onClick={() => setMessages([])}><Trash2 size={15} /> Clear</Button> : undefined}
      />
      {deployments.error && <ErrorState message={deployments.error} onRetry={deployments.reload} />}
      <div className="chat-toolbar">
        <label className="field inline-field"><span>Model</span><select value={selectedModel} onChange={(event) => setModel(event.target.value)} disabled={running.length === 0}><option value="">Select a running model</option>{running.map((item) => <option key={item.id} value={item.alias}>{item.alias} · {item.runtime}</option>)}</select></label>
        {selectedModel && running.find((item) => item.alias === selectedModel) && <RuntimeMark runtime={running.find((item) => item.alias === selectedModel)!.runtime} />}
        <span className="privacy-note"><Gauge size={14} /> Metrics on · content private</span>
      </div>
      <section className="conversation" aria-live="polite" aria-label="Conversation">
        {messages.length === 0 ? (
          <div className="chat-empty"><span className="chat-empty-icon"><Bot size={22} /></span><h2>Start a local conversation</h2><p>Choose a running model and send a message. SparkDeck records timing and token counts, never your content.</p></div>
        ) : messages.map((message, index) => (
          <article className={`message message-${message.role}`} key={`${message.role}-${index}`}>
            <div className="message-avatar" aria-hidden="true">{message.role === 'user' ? <User size={16} /> : <Bot size={16} />}</div>
            <div><p className="message-author">{message.role === 'user' ? 'You' : selectedModel}</p><p>{message.content}</p></div>
          </article>
        ))}
        {sending && <article className="message message-assistant"><div className="message-avatar"><Bot size={16} /></div><div><p className="message-author">{selectedModel}</p><p className="thinking">Generating response<span>…</span></p></div></article>}
        {error && <p className="inline-error" role="alert">{error}</p>}
      </section>
      <form className="composer" onSubmit={(event) => void send(event)}>
        <label><span className="sr-only">Message</span><textarea rows={2} value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit() } }} placeholder={running.length ? 'Message your model…' : 'Start a model to begin'} disabled={!running.length} /></label>
        <button type="submit" className="send-button" aria-label="Send message" disabled={!draft.trim() || !selectedModel || sending}><ArrowUp size={18} /></button>
      </form>
    </div>
  )
}
