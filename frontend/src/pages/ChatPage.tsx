import { lazy, Suspense, useEffect, useMemo, useRef, useState, type ClipboardEvent, type FormEvent } from 'react'
import { ArrowUp, Bot, Gauge, ImagePlus, Square, Trash2, X } from 'lucide-react'
import { api } from '../api/client'
import type { ChatMessage, ChatResponseMetrics } from '../api/types'
import { Button, ErrorState, PageHeader, RuntimeMark } from '../components/ui'
import { useResource } from '../hooks/useResource'

const MarkdownContent = lazy(() => import('../components/MarkdownContent').then((module) => ({
  default: module.MarkdownContent,
})))

interface ChatImageAttachment {
  id: string
  name: string
  type: string
  size: number
  dataUrl: string
}

interface ConversationMessage {
  id: string
  role: ChatMessage['role']
  content: string
  images?: ChatImageAttachment[]
  model?: string
  reasoning?: string
  metrics?: ChatResponseMetrics
  streaming?: boolean
  stopped?: boolean
  failed?: boolean
}

const ACCEPTED_IMAGE_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp', 'image/gif'])
const ACCEPTED_IMAGE_INPUT = [...ACCEPTED_IMAGE_TYPES].join(',')
const MAX_IMAGES_PER_MESSAGE = 4
const MAX_IMAGE_BYTES = 10 * 1024 * 1024
const MAX_CONVERSATION_IMAGE_BYTES = 20 * 1024 * 1024

const formatImageSize = (bytes: number) => bytes < 1024 * 1024
  ? `${Math.max(1, Math.round(bytes / 1024))} KB`
  : `${(bytes / (1024 * 1024)).toFixed(1)} MB`

const hasImageSignature = async (file: File) => {
  const bytes = new Uint8Array(await file.slice(0, 12).arrayBuffer())
  if (file.type === 'image/png') return [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a].every((byte, index) => bytes[index] === byte)
  if (file.type === 'image/jpeg') return bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff
  if (file.type === 'image/gif') return new TextDecoder().decode(bytes.slice(0, 6)) === 'GIF87a' || new TextDecoder().decode(bytes.slice(0, 6)) === 'GIF89a'
  return new TextDecoder().decode(bytes.slice(0, 4)) === 'RIFF' && new TextDecoder().decode(bytes.slice(8, 12)) === 'WEBP'
}

const readImage = async (file: File, id: string): Promise<ChatImageAttachment> => {
  if (!await hasImageSignature(file)) throw new Error(`${file.name || 'Image'} does not contain valid ${file.type.replace('image/', '').toUpperCase()} image data.`)
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(new Error(`Could not read ${file.name}`))
    reader.onload = () => {
      if (typeof reader.result !== 'string' || !reader.result.startsWith(`data:${file.type};base64,`)) {
        reject(new Error(`Could not read ${file.name} as an image`))
        return
      }
      resolve({ id, name: file.name || 'Pasted image', type: file.type, size: file.size, dataUrl: reader.result })
    }
    reader.readAsDataURL(file)
  })
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
  const [images, setImages] = useState<ChatImageAttachment[]>([])
  const [imageError, setImageError] = useState<string>()
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string>()
  const abortRef = useRef<AbortController | null>(null)
  const messageIdRef = useRef(0)
  const imageIdRef = useRef(0)
  const imagesRef = useRef<ChatImageAttachment[]>([])
  const imageMutationEpochRef = useRef(0)
  const imageReadQueueRef = useRef(Promise.resolve())
  const imageInputRef = useRef<HTMLInputElement>(null)
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

  const replaceImages = (next: ChatImageAttachment[], cancelPendingReads = true) => {
    if (cancelPendingReads) imageMutationEpochRef.current += 1
    imagesRef.current = next
    setImages(next)
  }

  const addImages = (files: File[]) => {
    const invocationEpoch = imageMutationEpochRef.current
    imageReadQueueRef.current = imageReadQueueRef.current.then(async () => {
      if (invocationEpoch !== imageMutationEpochRef.current) return
      const additions: ChatImageAttachment[] = []
      const failures: string[] = []
      const historyBytes = messages.reduce(
        (total, message) => total + (message.images?.reduce((sum, image) => sum + image.size, 0) ?? 0),
        0,
      )
      for (const file of files) {
        const currentImages = [...imagesRef.current, ...additions]
        if (!ACCEPTED_IMAGE_TYPES.has(file.type)) {
          failures.push(`${file.name || 'Image'} is not a PNG, JPEG, WebP, or GIF image.`)
          continue
        }
        if (currentImages.length >= MAX_IMAGES_PER_MESSAGE) {
          failures.push(`You can attach up to ${MAX_IMAGES_PER_MESSAGE} images per message.`)
          break
        }
        if (file.size > MAX_IMAGE_BYTES) {
          failures.push(`${file.name || 'Image'} exceeds the 10 MB per-image limit.`)
          continue
        }
        if (historyBytes + currentImages.reduce((total, image) => total + image.size, 0) + file.size > MAX_CONVERSATION_IMAGE_BYTES) {
          failures.push('Images exceed the 20 MB conversation limit. Clear the chat to attach more.')
          continue
        }
        try {
          const image = await readImage(file, `image-${++imageIdRef.current}`)
          additions.push(image)
        } catch (reason) {
          failures.push(reason instanceof Error ? reason.message : `Could not read ${file.name || 'image'}.`)
        }
        if (invocationEpoch !== imageMutationEpochRef.current) return
      }
      replaceImages([...imagesRef.current, ...additions], false)
      setImageError(failures.length ? failures.join(' ') : undefined)
    })
  }

  const removeImage = (id: string) => {
    replaceImages(imagesRef.current.filter((image) => image.id !== id), false)
    setImageError(undefined)
  }

  const pasteImages = (event: ClipboardEvent<HTMLTextAreaElement>) => {
    const files = Array.from(event.clipboardData.items)
      .filter((item) => item.kind === 'file' && item.type.startsWith('image/'))
      .map((item) => item.getAsFile())
      .filter((file): file is File => file !== null)
    if (!files.length) return
    event.preventDefault()
    void addImages(files)
  }

  const send = async (event: FormEvent) => {
    event.preventDefault()
    const content = draft.trim()
    const attachedImages = [...imagesRef.current]
    if ((!content && !attachedImages.length) || !selectedModel || sending) return

    const userMessage: ConversationMessage = {
      id: `message-${++messageIdRef.current}`,
      role: 'user',
      content,
      images: attachedImages,
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
      .filter((message) => message.role !== 'assistant' || (message.content.trim() && !message.failed))
      .map(({ role, content: messageContent, images: messageImages }) => ({
        role,
        content: messageImages?.length ? [
          ...(messageContent ? [{ type: 'text' as const, text: messageContent }] : []),
          ...messageImages.map((image) => ({
            type: 'image_url' as const,
            image_url: { url: image.dataUrl },
          })),
        ] : messageContent,
      }))
    const controller = new AbortController()
    abortRef.current = controller
    shouldAutoScrollRef.current = true
    setMessages([...history, assistantMessage])
    setDraft('')
    replaceImages([])
    setImageError(undefined)
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
        return [{ ...message, streaming: false, stopped, failed: !stopped }]
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
    replaceImages([])
    setImageError(undefined)
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
            <div className="user-message-content">
              {message.images?.length ? <div className="user-message-images">{message.images.map((image) => <img key={image.id} src={image.dataUrl} alt={`Attached ${image.name}`} />)}</div> : null}
              {message.content && <div className="user-message-text">{message.content}</div>}
            </div>
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
              {message.failed && <p className="generation-failed">Response interrupted</p>}
              <ResponseMetrics metrics={message.metrics} streaming={message.streaming} />
            </div>
          </article>
        ))}
        {error && <p className="inline-error chat-error" role="alert">{error}</p>}
      </section>
      <form className="composer" onSubmit={(event) => void send(event)}>
        {images.length > 0 && <div className="composer-attachments" aria-label="Attached images">{images.map((image) => <div className="composer-attachment" key={image.id}>
          <img src={image.dataUrl} alt="" />
          <span><strong>{image.name}</strong><small>{formatImageSize(image.size)}</small></span>
          <button type="button" aria-label={`Remove ${image.name}`} onClick={() => removeImage(image.id)}><X size={14} /></button>
        </div>)}</div>}
        {imageError && <p className="composer-image-error" role="alert">{imageError}</p>}
        <input ref={imageInputRef} className="sr-only" type="file" accept={ACCEPTED_IMAGE_INPUT} multiple aria-label="Choose image files" onChange={(event) => {
          const files = Array.from(event.currentTarget.files ?? [])
          event.currentTarget.value = ''
          void addImages(files)
        }} disabled={!running.length} />
        <button type="button" className="attach-button" aria-label="Upload images" disabled={!running.length} onClick={() => imageInputRef.current?.click()}><ImagePlus size={18} /></button>
        <label><span className="sr-only">Message</span><textarea rows={2} value={draft} onChange={(event) => setDraft(event.target.value)} onPaste={pasteImages} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit() } }} placeholder={running.length ? `Message ${selectedModel || 'your model'}…` : 'Start a model to begin'} disabled={!running.length} /></label>
        {sending ? (
          <button type="button" className="send-button stop-button" aria-label="Stop generating" onClick={stop}><Square size={15} fill="currentColor" /></button>
        ) : (
          <button type="submit" className="send-button" aria-label="Send message" disabled={(!draft.trim() && !images.length) || !selectedModel}><ArrowUp size={18} /></button>
        )}
        <p className="composer-hint">Paste or upload images · Enter to send · Shift + Enter for a new line</p>
      </form>
    </div>
  )
}
