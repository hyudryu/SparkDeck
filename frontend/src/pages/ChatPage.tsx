import { lazy, Suspense, useEffect, useMemo, useRef, useState, type ClipboardEvent, type DragEvent as ReactDragEvent, type FormEvent } from 'react'
import { ArrowUp, Bot, Gauge, Paperclip, Square, Trash2, X } from 'lucide-react'
import { api } from '../api/client'
import type { ChatMessage, ChatResponseMetrics } from '../api/types'
import { Button, ErrorState, PageHeader, RuntimeMark } from '../components/ui'
import { useResource } from '../hooks/useResource'

const MarkdownContent = lazy(() => import('../components/MarkdownContent').then((module) => ({
  default: module.MarkdownContent,
})))

interface ChatMediaAttachment {
  id: string
  name: string
  type: string
  size: number
  kind: 'image' | 'video'
  dataUrl: string
}

interface ConversationMessage {
  id: string
  role: ChatMessage['role']
  content: string
  media?: ChatMediaAttachment[]
  model?: string
  reasoning?: string
  metrics?: ChatResponseMetrics
  streaming?: boolean
  stopped?: boolean
  failed?: boolean
}

const ACCEPTED_IMAGE_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp', 'image/gif'])
const ACCEPTED_VIDEO_TYPES = new Set(['video/mp4', 'video/webm', 'video/quicktime'])
const ACCEPTED_MEDIA_INPUT = [...ACCEPTED_IMAGE_TYPES, ...ACCEPTED_VIDEO_TYPES].join(',')
const IMAGE_TYPE_LABELS: Record<string, string> = { 'image/png': 'PNG', 'image/jpeg': 'JPEG', 'image/webp': 'WebP', 'image/gif': 'GIF' }
const VIDEO_TYPE_LABELS: Record<string, string> = { 'video/mp4': 'MP4', 'video/webm': 'WebM', 'video/quicktime': 'MOV' }
const MAX_IMAGES_PER_MESSAGE = 4
const MAX_VIDEOS_PER_MESSAGE = 2
const MAX_IMAGE_BYTES = 10 * 1024 * 1024
const MAX_VIDEO_BYTES = 16 * 1024 * 1024
const MAX_CONVERSATION_MEDIA_BYTES = 20 * 1024 * 1024

const formatMediaSize = (bytes: number) => bytes < 1024 * 1024
  ? `${Math.max(1, Math.round(bytes / 1024))} KB`
  : `${(bytes / (1024 * 1024)).toFixed(1)} MB`

const hasImageSignature = async (file: File) => {
  const bytes = new Uint8Array(await file.slice(0, 12).arrayBuffer())
  if (file.type === 'image/png') return [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a].every((byte, index) => bytes[index] === byte)
  if (file.type === 'image/jpeg') return bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff
  if (file.type === 'image/gif') return new TextDecoder().decode(bytes.slice(0, 6)) === 'GIF87a' || new TextDecoder().decode(bytes.slice(0, 6)) === 'GIF89a'
  return new TextDecoder().decode(bytes.slice(0, 4)) === 'RIFF' && new TextDecoder().decode(bytes.slice(8, 12)) === 'WEBP'
}

const hasVideoSignature = async (file: File) => {
  const bytes = new Uint8Array(await file.slice(0, 12).arrayBuffer())
  if (file.type === 'video/webm') return bytes[0] === 0x1a && bytes[1] === 0x45 && bytes[2] === 0xdf && bytes[3] === 0xa3
  // MP4 and QuickTime files both start with an ftyp box at offset 4.
  return new TextDecoder().decode(bytes.slice(4, 8)) === 'ftyp'
}

const mediaKind = (file: File) => ACCEPTED_VIDEO_TYPES.has(file.type)
  ? 'video' as const
  : ACCEPTED_IMAGE_TYPES.has(file.type) ? 'image' as const : null

const readMedia = async (file: File, id: string): Promise<ChatMediaAttachment> => {
  const kind = mediaKind(file)
  if (kind === 'image') {
    const label = IMAGE_TYPE_LABELS[file.type] ?? 'image'
    if (!await hasImageSignature(file)) throw new Error(`${file.name || 'Image'} does not contain valid ${label} image data.`)
  } else if (kind === 'video') {
    const label = VIDEO_TYPE_LABELS[file.type] ?? 'video'
    if (!await hasVideoSignature(file)) throw new Error(`${file.name || 'Video'} does not contain valid ${label} video data.`)
  } else {
    throw new Error(`${file.name || 'File'} is not a supported image or video (PNG, JPEG, WebP, GIF, MP4, WebM, MOV).`)
  }
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(new Error(`Could not read ${file.name}`))
    reader.onload = () => {
      if (typeof reader.result !== 'string' || !reader.result.startsWith(`data:${file.type};base64,`)) {
        reject(new Error(`Could not read ${file.name}`))
        return
      }
      resolve({
        id,
        name: file.name || (kind === 'video' ? 'Pasted video' : 'Pasted image'),
        type: file.type,
        size: file.size,
        kind,
        dataUrl: reader.result,
      })
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
  const promptMeasured = streaming || metrics?.prompt_tokens_per_second !== undefined
  return (
    <dl className="response-metrics" role="group" aria-label="Response performance">
      {promptMeasured && <div><dt>Prompt processing</dt><dd>{formatRate(metrics?.prompt_tokens_per_second, streaming)}</dd></div>}
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
  const [attachments, setAttachments] = useState<ChatMediaAttachment[]>([])
  const [attachmentError, setAttachmentError] = useState<string>()
  const [dragging, setDragging] = useState(false)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string>()
  const abortRef = useRef<AbortController | null>(null)
  const messageIdRef = useRef(0)
  const attachmentIdRef = useRef(0)
  const attachmentsRef = useRef<ChatMediaAttachment[]>([])
  const attachmentMutationEpochRef = useRef(0)
  const attachmentReadQueueRef = useRef(Promise.resolve())
  const dragDepthRef = useRef(0)
  const mediaInputRef = useRef<HTMLInputElement>(null)
  const conversationRef = useRef<HTMLElement>(null)
  const shouldAutoScrollRef = useRef(true)
  const selectedModel = model || running[0]?.alias || ''

  useEffect(() => () => abortRef.current?.abort(), [])
  useEffect(() => {
    const conversation = conversationRef.current
    if (conversation && shouldAutoScrollRef.current) conversation.scrollTop = conversation.scrollHeight
  }, [messages])
  // Keep file drops outside the composer from navigating the page away.
  useEffect(() => {
    const preventDropNavigation = (event: DragEvent) => {
      if (Array.from(event.dataTransfer?.types ?? []).includes('Files')) event.preventDefault()
    }
    document.addEventListener('dragover', preventDropNavigation)
    document.addEventListener('drop', preventDropNavigation)
    return () => {
      document.removeEventListener('dragover', preventDropNavigation)
      document.removeEventListener('drop', preventDropNavigation)
    }
  }, [])

  const updateMessage = (id: string, update: (message: ConversationMessage) => ConversationMessage) => {
    setMessages((current) => current.map((message) => message.id === id ? update(message) : message))
  }

  const replaceAttachments = (next: ChatMediaAttachment[], cancelPendingReads = true) => {
    if (cancelPendingReads) attachmentMutationEpochRef.current += 1
    attachmentsRef.current = next
    setAttachments(next)
  }

  const addMedia = (files: File[]) => {
    const invocationEpoch = attachmentMutationEpochRef.current
    attachmentReadQueueRef.current = attachmentReadQueueRef.current.then(async () => {
      if (invocationEpoch !== attachmentMutationEpochRef.current) return
      const additions: ChatMediaAttachment[] = []
      const failures: string[] = []
      const historyBytes = messages.reduce(
        (total, message) => total + (message.media?.reduce((sum, item) => sum + item.size, 0) ?? 0),
        0,
      )
      for (const file of files) {
        const currentMedia = [...attachmentsRef.current, ...additions]
        const kind = mediaKind(file)
        if (!kind) {
          failures.push(`${file.name || 'File'} is not a supported image or video (PNG, JPEG, WebP, GIF, MP4, WebM, MOV).`)
          continue
        }
        const maxCount = kind === 'video' ? MAX_VIDEOS_PER_MESSAGE : MAX_IMAGES_PER_MESSAGE
        if (currentMedia.filter((item) => item.kind === kind).length >= maxCount) {
          const message = kind === 'video'
            ? `You can attach up to ${MAX_VIDEOS_PER_MESSAGE} videos per message.`
            : `You can attach up to ${MAX_IMAGES_PER_MESSAGE} images per message.`
          if (!failures.includes(message)) failures.push(message)
          continue
        }
        const maxBytes = kind === 'video' ? MAX_VIDEO_BYTES : MAX_IMAGE_BYTES
        if (file.size > maxBytes) {
          failures.push(`${file.name || 'File'} exceeds the ${maxBytes / (1024 * 1024)} MB per-${kind} limit.`)
          continue
        }
        if (historyBytes + currentMedia.reduce((total, item) => total + item.size, 0) + file.size > MAX_CONVERSATION_MEDIA_BYTES) {
          failures.push('Media exceeds the 20 MB conversation limit. Clear the chat to attach more.')
          continue
        }
        try {
          additions.push(await readMedia(file, `${kind}-${++attachmentIdRef.current}`))
        } catch (reason) {
          failures.push(reason instanceof Error ? reason.message : `Could not read ${file.name || 'file'}.`)
        }
        if (invocationEpoch !== attachmentMutationEpochRef.current) return
      }
      replaceAttachments([...attachmentsRef.current, ...additions], false)
      setAttachmentError(failures.length ? failures.join(' ') : undefined)
    })
  }

  const removeAttachment = (id: string) => {
    replaceAttachments(attachmentsRef.current.filter((item) => item.id !== id), false)
    setAttachmentError(undefined)
  }

  const dragHasFiles = (event: ReactDragEvent) => Array.from(event.dataTransfer?.types ?? []).includes('Files')

  const dragEnter = (event: ReactDragEvent<HTMLFormElement>) => {
    if (!dragHasFiles(event)) return
    event.preventDefault()
    dragDepthRef.current += 1
    setDragging(true)
  }

  const dragOver = (event: ReactDragEvent<HTMLFormElement>) => {
    if (!dragHasFiles(event)) return
    event.preventDefault()
    event.dataTransfer.dropEffect = 'copy'
  }

  const dragLeave = (event: ReactDragEvent<HTMLFormElement>) => {
    if (!dragHasFiles(event)) return
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1)
    if (dragDepthRef.current === 0) setDragging(false)
  }

  const dropFiles = (event: ReactDragEvent<HTMLFormElement>) => {
    if (!dragHasFiles(event)) return
    event.preventDefault()
    dragDepthRef.current = 0
    setDragging(false)
    addMedia(Array.from(event.dataTransfer?.files ?? []))
  }

  const pasteImages = (event: ClipboardEvent<HTMLTextAreaElement>) => {
    const files = Array.from(event.clipboardData.items)
      .filter((item) => item.kind === 'file' && item.type.startsWith('image/'))
      .map((item) => item.getAsFile())
      .filter((file): file is File => file !== null)
    if (!files.length) return
    event.preventDefault()
    void addMedia(files)
  }

  const send = async (event: FormEvent) => {
    event.preventDefault()
    const content = draft.trim()
    const attachedMedia = [...attachmentsRef.current]
    if ((!content && !attachedMedia.length) || !selectedModel || sending) return

    const userMessage: ConversationMessage = {
      id: `message-${++messageIdRef.current}`,
      role: 'user',
      content,
      media: attachedMedia,
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
      .map(({ role, content: messageContent, media: messageMedia }) => ({
        role,
        content: messageMedia?.length ? [
          ...(messageContent ? [{ type: 'text' as const, text: messageContent }] : []),
          ...messageMedia.map((item) => item.kind === 'video'
            ? ({ type: 'video_url' as const, video_url: { url: item.dataUrl } })
            : ({ type: 'image_url' as const, image_url: { url: item.dataUrl } })),
        ] : messageContent,
      }))
    const controller = new AbortController()
    abortRef.current = controller
    shouldAutoScrollRef.current = true
    setMessages([...history, assistantMessage])
    setDraft('')
    replaceAttachments([])
    setAttachmentError(undefined)
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
    replaceAttachments([])
    setAttachmentError(undefined)
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
              {message.media?.length ? <div className="user-message-media">{message.media.map((item) => item.kind === 'video'
                ? <video key={item.id} src={item.dataUrl} controls preload="metadata" aria-label={`Attached ${item.name}`} />
                : <img key={item.id} src={item.dataUrl} alt={`Attached ${item.name}`} />)}</div> : null}
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
      <form
        className={`composer${dragging ? ' is-dragging' : ''}`}
        onSubmit={(event) => void send(event)}
        onDragEnter={dragEnter}
        onDragOver={dragOver}
        onDragLeave={dragLeave}
        onDrop={dropFiles}
      >
        {attachments.length > 0 && <div className="composer-attachments" aria-label="Attached media">{attachments.map((item) => <div className="composer-attachment" key={item.id}>
          {item.kind === 'video' ? <video src={item.dataUrl} muted preload="metadata" /> : <img src={item.dataUrl} alt="" />}
          <span><strong>{item.name}</strong><small>{formatMediaSize(item.size)}</small></span>
          <button type="button" aria-label={`Remove ${item.name}`} onClick={() => removeAttachment(item.id)}><X size={14} /></button>
        </div>)}</div>}
        {attachmentError && <p className="composer-attachment-error" role="alert">{attachmentError}</p>}
        {dragging && <div className="composer-drop-overlay" aria-hidden="true">Drop images or videos to attach</div>}
        <input ref={mediaInputRef} className="sr-only" type="file" accept={ACCEPTED_MEDIA_INPUT} multiple aria-label="Choose image or video files" onChange={(event) => {
          const files = Array.from(event.currentTarget.files ?? [])
          event.currentTarget.value = ''
          void addMedia(files)
        }} disabled={!running.length} />
        <button type="button" className="attach-button" aria-label="Upload images or videos" disabled={!running.length} onClick={() => mediaInputRef.current?.click()}><Paperclip size={18} /></button>
        <label><span className="sr-only">Message</span><textarea rows={2} value={draft} onChange={(event) => setDraft(event.target.value)} onPaste={pasteImages} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit() } }} placeholder={running.length ? `Message ${selectedModel || 'your model'}…` : 'Start a model to begin'} disabled={!running.length} /></label>
        {sending ? (
          <button type="button" className="send-button stop-button" aria-label="Stop generating" onClick={stop}><Square size={15} fill="currentColor" /></button>
        ) : (
          <button type="submit" className="send-button" aria-label="Send message" disabled={(!draft.trim() && !attachments.length) || !selectedModel}><ArrowUp size={18} /></button>
        )}
        <p className="composer-hint">Drag, paste, or upload images and videos · Enter to send · Shift + Enter for a new line</p>
      </form>
    </div>
  )
}
