import { lazy, Suspense, useEffect, useMemo, useRef, useState, type FormEvent } from 'react'
import { ArrowUp, Bot, Gauge, Square, Trash2 } from 'lucide-react'
import { api } from '../api/client'
import type { ChatMessage, ChatResponseMetrics } from '../api/types'
import { Button, ErrorState, PageHeader, RuntimeMark } from '../components/ui'
import { useResource } from '../hooks/useResource'

const MarkdownContent = lazy(() => import('../components/MarkdownContent').then((module) => ({
  default: module.MarkdownContent,
})))

interface ConversationMessage extends ChatMessage {
  id: string
  model?: string
  reasoning?: string
  metrics?: ChatResponseMetrics
  streaming?: boolean
  stopped?: boolean
}

function formatRate(value: number | undefined, streaming?: boolean) {
  return value === undefined ? (streaming ? 'Measuring…' : 'Unavailable') : `${value.toFixed(1)} tok/s`
}

function formatTtft(value: number | undefined, streaming?: boolean) {
  if (value === undefined) return streaming ? 'Waiting…' : 'Unavailable'
  return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`
}

function ResponseMetrics({ metrics, streaming }: { metrics?: ChatResponseMetrics; streaming?: boolean }) {
  if (!metrics && !streaming) return null
  return (
    <dl className="response-metrics" role="group" aria-label="Response performance">
      <div><dt>Prompt processing</dt><dd>{formatRate(metrics?.prompt_tokens_per_second, streaming)}</dd></div>
      <div><dt>TTFT</dt><dd>{formatTtft(metrics?.ttft_ms, streaming)}</dd></div>
      <div><dt>Output speed</dt><dd>{formatRate(metrics?.output_tokens_per_second, streaming)}</dd></div>
    </dl>
  )
}

export function ChatPage() {
  const deployments = useResource((signal) => api.deployments.list(signal))
  const running = useMemo(() => deployments.data?.filter((item) => item.status === 'running') ?? [], [deployments.data])
  const [model, setModel] = useState('')
  const [draft, setDraft] = useState('')
  const [messages, setMessages] = useState<ConversationMessage[]>([])
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string>()
  const abortRef = useRef<AbortController | null>(null)
  const messageIdRef = useRef(0)
  const conversationRef = useRef<HTMLElement>(null)
  const shouldAutoScrollRef = useRef(true)
  const selectedModel = model || running[0]?.alias || ''

  useEffect(() => () => abortRef.current?.abort(), [])
  useEffect(() => {
    const conversation = conversationRef.current
    if (conversation && shouldAutoScrollRef.current) conversation.scrollTop = conversation.scrollHeight
  }, [messages])

  const updateMessage = (id: string, update: (message: ConversationMessage) => ConversationMessage) => {
    setMessages((current) => current.map((message) => message.id === id ? update(message) : message))
  }

  const send = async (event: FormEvent) => {
    event.preventDefault()
    const content = draft.trim()
    if (!content || !selectedModel || sending) return

    const userMessage: ConversationMessage = {
      id: `message-${++messageIdRef.current}`,
      role: 'user',
      content,
    }
    const assistantId = `message-${++messageIdRef.current}`
    const assistantMessage: ConversationMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      model: selectedModel,
      streaming: true,
    }
    const history = [...messages, userMessage]
    const requestMessages: ChatMessage[] = history
      .filter((message) => message.role !== 'assistant' || message.content.trim())
      .map(({ role, content: messageContent }) => ({ role, content: messageContent }))
    const controller = new AbortController()
    abortRef.current = controller
    shouldAutoScrollRef.current = true
    setMessages([...history, assistantMessage])
    setDraft('')
    setSending(true)
    setError(undefined)

    try {
      const result = await api.chatStream(selectedModel, requestMessages, {
        signal: controller.signal,
        onUpdate: (update) => updateMessage(assistantId, (message) => ({
          ...message,
          content: message.content + (update.content ?? ''),
          reasoning: (message.reasoning ?? '') + (update.reasoning ?? ''),
          metrics: update.metrics ?? message.metrics,
        })),
      })
      updateMessage(assistantId, (message) => ({
        ...message,
        content: result.message.content || message.content,
        reasoning: result.reasoning || message.reasoning,
        metrics: result.metrics,
        streaming: false,
      }))
    } catch (reason) {
      const stopped = reason instanceof Error && reason.name === 'AbortError'
      setMessages((current) => current.flatMap((message) => {
        if (message.id !== assistantId) return [message]
        if (!stopped && !message.content && !message.reasoning) return []
        return [{ ...message, streaming: false, stopped }]
      }))
      if (!stopped) setError(reason instanceof Error ? reason.message : 'The model did not respond')
    } finally {
      if (abortRef.current === controller) abortRef.current = null
      setSending(false)
    }
  }

  const stop = () => abortRef.current?.abort()
  const clear = () => {
    abortRef.current?.abort()
    setMessages([])
    setError(undefined)
  }

  return (
    <div className="page chat-page">
      <PageHeader
        eyebrow="Inference workspace"
        title="Chat"
        description="Talk to any running model. Eligible performance measurements are captured locally without message content."
        actions={messages.length > 0 ? <Button onClick={clear}><Trash2 size={15} /> Clear</Button> : undefined}
      />
      {deployments.error && <ErrorState message={deployments.error} onRetry={deployments.reload} />}
      <div className="chat-toolbar">
        <label className="field inline-field"><span>Model</span><select value={selectedModel} onChange={(event) => setModel(event.target.value)} disabled={running.length === 0 || sending}><option value="">Select a running model</option>{running.map((item) => <option key={item.id} value={item.alias}>{item.alias} · {item.runtime}</option>)}</select></label>
        {selectedModel && running.find((item) => item.alias === selectedModel) && <RuntimeMark runtime={running.find((item) => item.alias === selectedModel)!.runtime} />}
        <span className="privacy-note"><Gauge size={14} /> Live metrics · content private</span>
      </div>
      <p className="sr-only" role="status" aria-live="polite">{sending ? 'Generating response' : messages.length ? 'Response complete' : ''}</p>
      <section ref={conversationRef} className="conversation" aria-label="Conversation" onScroll={(event) => {
        const target = event.currentTarget
        shouldAutoScrollRef.current = target.scrollHeight - target.scrollTop - target.clientHeight < 80
      }}>
        {messages.length === 0 ? (
          <div className="chat-empty"><span className="chat-empty-icon"><Bot size={22} /></span><h2>How can I help?</h2><p>Choose a running model and start a private, local conversation. Thinking and answer text stream as they are generated.</p></div>
        ) : messages.map((message) => message.role === 'user' ? (
          <article className="message message-user" key={message.id}>
            <div className="user-message-content">{message.content}</div>
          </article>
        ) : (
          <article className="message message-assistant" key={message.id}>
            <div className="message-avatar" aria-hidden="true"><Bot size={16} /></div>
            <div className="assistant-message-body">
              <p className="message-author">{message.model}</p>
              {message.reasoning && (
                <details className="reasoning-block" open={message.streaming && !message.content ? true : undefined}>
                  <summary>{message.streaming ? 'Thinking…' : 'Thinking'}</summary>
                  <div>{message.reasoning}</div>
                </details>
              )}
              {message.content ? <div className={`message-content${message.streaming ? ' is-streaming' : ''}`}><Suspense fallback={<div className="markdown-fallback">{message.content}</div>}><MarkdownContent content={message.content} /></Suspense></div> : (
                message.streaming && !message.reasoning ? <p className="thinking"><span className="thinking-dot" /> Thinking…</p> : null
              )}
              {message.stopped && <p className="generation-stopped">Response stopped</p>}
              <ResponseMetrics metrics={message.metrics} streaming={message.streaming} />
            </div>
          </article>
        ))}
        {error && <p className="inline-error chat-error" role="alert">{error}</p>}
      </section>
      <form className="composer" onSubmit={(event) => void send(event)}>
        <label><span className="sr-only">Message</span><textarea rows={2} value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit() } }} placeholder={running.length ? `Message ${selectedModel || 'your model'}…` : 'Start a model to begin'} disabled={!running.length} /></label>
        {sending ? (
          <button type="button" className="send-button stop-button" aria-label="Stop generating" onClick={stop}><Square size={15} fill="currentColor" /></button>
        ) : (
          <button type="submit" className="send-button" aria-label="Send message" disabled={!draft.trim() || !selectedModel}><ArrowUp size={18} /></button>
        )}
        <p className="composer-hint">Enter to send · Shift + Enter for a new line</p>
      </form>
    </div>
  )
}
