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
  // Concurrent sends share a group id; runIndex (1-based) identifies which
  // parallel session an answer belongs to so follow-up turns replay only
  // their own branch of the history and seed the engine differently per
  // session (same prompt, different samples).
  groupId?: string
  runIndex?: number
  runTotal?: number
  startedAt?: number
  endedAt?: number
}

const CONCURRENCY_OPTIONS = [1, 2, 4, 8, 16]
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

// The thinking body scrolls in its own 240px box (styles.css), so the outer
// conversation's follow-the-bottom logic never reaches text that overflows
// it. This keeps the newest thinking line visible while reasoning streams,
// pausing the moment the reader scrolls up and resuming once they return.
const SCROLL_STICK_THRESHOLD_PX = 80

function stickToBottom(el: HTMLElement) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < SCROLL_STICK_THRESHOLD_PX
}

function ReasoningBlock({ streaming, reasoning }: { streaming: boolean; reasoning: string }) {
  const bodyRef = useRef<HTMLDivElement>(null)
  const stickToBottomRef = useRef(true)
  useEffect(() => {
    const body = bodyRef.current
    if (body && streaming && stickToBottomRef.current) body.scrollTop = body.scrollHeight
  }, [reasoning, streaming])
  return (
    <div ref={bodyRef} onScroll={(event) => {
      stickToBottomRef.current = stickToBottom(event.currentTarget)
    }}>{reasoning}</div>
  )
}

// Shown once every run of a concurrent send has finished: aggregate serving
// throughput over the shared wall-clock window (summing the per-run average
// rates would overstate it when runs finish at different times), the spread
// in time-to-first-token, and whether the engine produced byte-identical
// answers across the sessions. The window runs from request dispatch to
// stream close, so unlike a session's own Output speed it includes queueing
// and time-to-first-token; the label says so to keep the two apart.
function ConcurrencySummary({ runs }: { runs: ConversationMessage[] }) {
  if (runs.length < 2 || runs.some((run) => run.streaming)) return null
  const parts = [`${runs.length} concurrent runs`]
  const tokens = runs.map((run) => run.metrics?.completion_tokens)
  const starts = runs.map((run) => run.startedAt)
  const ends = runs.map((run) => run.endedAt)
  if (tokens.every((value): value is number => value !== undefined)
    && starts.every((value): value is number => value !== undefined)
    && ends.every((value): value is number => value !== undefined)) {
    const windowMs = Math.max(1, Math.max(...ends) - Math.min(...starts))
    const totalTokens = tokens.reduce((sum, count) => sum + count, 0)
    parts.push(`combined ${(totalTokens / (windowMs / 1000)).toFixed(1)} tok/s over ${(windowMs / 1000).toFixed(1)} s (TTFT included)`)
  }
  const ttfts = runs.map((run) => run.metrics?.ttft_ms).filter((value): value is number => value !== undefined)
  if (ttfts.length === runs.length) parts.push(`TTFT ${formatTtft(Math.min(...ttfts))} – ${formatTtft(Math.max(...ttfts))}`)
  const completed = runs.every((run) => !run.stopped && !run.failed && run.content.length > 0)
  parts.push(completed
    ? (runs.every((run) => run.content === runs[0]?.content) ? 'identical output' : 'outputs differ')
    : 'incomplete runs')
  return <p className="concurrency-summary">{parts.join(' · ')}</p>
}

function UserBubble({ message }: { message: ConversationMessage }) {
  return (
    <article className="message message-user">
      <div className="user-message-content">
        {message.media?.length ? <div className="user-message-media">{message.media.map((item) => item.kind === 'video'
          ? <video key={item.id} src={item.dataUrl} controls preload="metadata" aria-label={`Attached ${item.name}`} />
          : <img key={item.id} src={item.dataUrl} alt={`Attached ${item.name}`} />)}</div> : null}
        {message.content && <div className="user-message-text">{message.content}</div>}
      </div>
    </article>
  )
}

// runLabel is dropped inside a session pane, where the pane's own header
// already identifies which concurrent session an answer belongs to.
function AssistantMessage({ message, className, runLabel = true }: {
  message: ConversationMessage
  className: string
  runLabel?: boolean
}) {
  return (
    <article className={className}>
      <div className="message-avatar" aria-hidden="true"><Bot size={16} /></div>
      <div className="assistant-message-body">
        <p className="message-author">{message.model}{runLabel && message.runTotal ? ` · Run ${message.runIndex} of ${message.runTotal}` : ''}</p>
        {message.reasoning && (
          <details className="reasoning-block" open={message.streaming && !message.content ? true : undefined}>
            <summary>{message.streaming ? 'Thinking…' : 'Thinking'}</summary>
            <ReasoningBlock streaming={message.streaming === true} reasoning={message.reasoning} />
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
  )
}

// Conversation is rendered as user / assistant singles plus one entry per
// concurrent-send group. Once any group exists the whole box splits into
// that many independent session panes (see the pane rendering below), each
// replaying the shared turns and its own branch of each group. Runs of a
// group are always appended together, so contiguously shared group ids mean
// one entry.
type RenderEntry
  = { kind: 'single'; message: ConversationMessage }
  | { kind: 'runs'; groupId: string; runs: ConversationMessage[] }

const toRequestMessage = ({ role, content, media }: ConversationMessage): ChatMessage => ({
  role,
  content: media?.length ? [
    ...(content ? [{ type: 'text' as const, text: content }] : []),
    ...media.map((item) => item.kind === 'video'
      ? ({ type: 'video_url' as const, video_url: { url: item.dataUrl } })
      : ({ type: 'image_url' as const, image_url: { url: item.dataUrl } })),
  ] : content,
})

export function ChatPage() {
  const deployments = useResource((signal) => api.deployments.list(signal))
  const running = useMemo(() => deployments.data?.filter((item) => (
    item.status === 'running'
    // Laya is a non-autoregressive decision model: it takes `state` and typed
    // `questions` and returns calibrated answers, so a chat turn would fail
    // with a 400 every time. It is reachable on /v1 for callers that send a
    // decision payload, but it is not a chat target.
    && item.runtime !== 'laya'
  )) ?? [], [deployments.data])
  // The gateway routes /v1 requests by model name and picks an owning
  // deployment itself, so deployments sharing an alias are one chat target:
  // the dropdown lists each running model name once, not one option per
  // deployment.
  const modelNames = useMemo(() => Array.from(new Set(running.map((item) => item.alias))), [running])
  const [model, setModel] = useState('')
  const [draft, setDraft] = useState('')
  const [messages, setMessages] = useState<ConversationMessage[]>([])
  const [attachments, setAttachments] = useState<ChatMediaAttachment[]>([])
  const [attachmentError, setAttachmentError] = useState<string>()
  const [dragging, setDragging] = useState(false)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string>()
  const [concurrency, setConcurrency] = useState(1)
  const abortRef = useRef<AbortController[]>([])
  const messageIdRef = useRef(0)
  const attachmentIdRef = useRef(0)
  const attachmentsRef = useRef<ChatMediaAttachment[]>([])
  const attachmentMutationEpochRef = useRef(0)
  const attachmentReadQueueRef = useRef(Promise.resolve())
  const dragDepthRef = useRef(0)
  const mediaInputRef = useRef<HTMLInputElement>(null)
  const conversationRef = useRef<HTMLElement>(null)
  const shouldAutoScrollRef = useRef(true)
  const selectedModel = model || modelNames[0] || ''

  const renderEntries = useMemo(() => {
    const entries: RenderEntry[] = []
    for (const message of messages) {
      if (message.role === 'assistant' && message.groupId) {
        const last = entries[entries.length - 1]
        if (last && last.kind === 'runs' && last.groupId === message.groupId) {
          last.runs.push(message)
          continue
        }
        entries.push({ kind: 'runs', groupId: message.groupId, runs: [message] })
        continue
      }
      entries.push({ kind: 'single', message })
    }
    return entries
  }, [messages])
  // One pane per concurrent session: the conversation box splits into this
  // many full-height columns, each an independent chat with its own scroll.
  // The count is the widest group in the history so earlier branches keep
  // their column even after the concurrency selector is turned back down.
  // runTotal (not the live runs length) keeps failed runs that were dropped
  // before producing content from collapsing a session column.
  const paneCount = useMemo(() => renderEntries.reduce(
    (max, entry) => entry.kind === 'runs' ? Math.max(max, entry.runs[0]?.runTotal ?? entry.runs.length) : max,
    1,
  ), [renderEntries])

  useEffect(() => () => abortRef.current.forEach((controller) => controller.abort()), [])
  useEffect(() => {
    const conversation = conversationRef.current
    if (conversation && shouldAutoScrollRef.current) conversation.scrollTop = conversation.scrollHeight
  }, [messages])
  // Panes scroll on their own (the outer box stops scrolling), so each one
  // follows its own newest content until its reader scrolls up.
  const paneRefs = useRef<Array<HTMLDivElement | null>>([])
  const paneStickToBottom = useRef<boolean[]>([])
  useEffect(() => {
    paneRefs.current.forEach((pane, index) => {
      if (pane && paneStickToBottom.current[index] !== false) pane.scrollTop = pane.scrollHeight
    })
  }, [messages, paneCount])
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

    const runTotal = concurrency
    const groupId = runTotal > 1 ? `group-${++messageIdRef.current}` : undefined
    const userMessage: ConversationMessage = {
      id: `message-${++messageIdRef.current}`,
      role: 'user',
      content,
      media: attachedMedia,
    }
    const assistantMessages: ConversationMessage[] = Array.from({ length: runTotal }, (_, index) => ({
      id: `message-${++messageIdRef.current}`,
      role: 'assistant',
      content: '',
      model: selectedModel,
      streaming: true,
      groupId,
      runIndex: groupId ? index + 1 : undefined,
      runTotal: groupId ? runTotal : undefined,
      startedAt: Date.now(),
    }))
    // Each concurrent session replays only its own past answers, so the
    // sessions stay independent branches of the same conversation. A plain
    // (unrun) send replays the first usable answer of each earlier group —
    // every branch would otherwise duplicate into the next prompt.
    const representativeByGroup = new Map<string, string>()
    for (const entry of messages) {
      if (entry.role === 'assistant' && entry.groupId && !entry.failed && entry.content.trim()
        && !representativeByGroup.has(entry.groupId)) {
        representativeByGroup.set(entry.groupId, entry.id)
      }
    }
    const historyFor = (runIndex: number | undefined) => [...messages, userMessage]
      .filter((message) => message.role !== 'assistant'
        || (message.content.trim() && !message.failed
          && (message.groupId === undefined
            ? true
            : runIndex === undefined
              ? representativeByGroup.get(message.groupId) === message.id
              : message.runIndex === runIndex)))
      .map(toRequestMessage)

    const controllers = assistantMessages.map(() => new AbortController())
    abortRef.current = [...abortRef.current, ...controllers]
    shouldAutoScrollRef.current = true
    paneStickToBottom.current = []
    setMessages([...messages, userMessage, ...assistantMessages])
    setDraft('')
    replaceAttachments([])
    setAttachmentError(undefined)
    setSending(true)
    setError(undefined)

    const runAssistant = async (message: ConversationMessage, controller: AbortController) => {
      try {
        const result = await api.chatStream(selectedModel, historyFor(message.runIndex), {
          signal: controller.signal,
          // Parallel sessions get distinct seeds (their 1-based run number)
          // so the engine samples the shared prompt differently per session;
          // a single-session send leaves the seed to the engine default.
          seed: message.runIndex,
          onUpdate: (update) => updateMessage(message.id, (current) => ({
            ...current,
            content: current.content + (update.content ?? ''),
            reasoning: (current.reasoning ?? '') + (update.reasoning ?? ''),
            metrics: update.metrics ?? current.metrics,
          })),
        })
        updateMessage(message.id, (current) => ({
          ...current,
          content: result.message.content || current.content,
          reasoning: result.reasoning || current.reasoning,
          metrics: result.metrics,
          streaming: false,
          endedAt: Date.now(),
        }))
      } catch (reason) {
        const stopped = reason instanceof Error && reason.name === 'AbortError'
        setMessages((current) => current.flatMap((entry) => {
          if (entry.id !== message.id) return [entry]
          if (!stopped && !entry.content && !entry.reasoning) return []
          return [{ ...entry, streaming: false, stopped, failed: !stopped, endedAt: Date.now() }]
        }))
        if (!stopped) setError(reason instanceof Error ? reason.message : 'The model did not respond')
      } finally {
        abortRef.current = abortRef.current.filter((entry) => entry !== controller)
      }
    }

    try {
      await Promise.all(assistantMessages.map((message, index) => runAssistant(message, controllers[index]!)))
    } finally {
      setSending(false)
    }
  }

  const stop = () => abortRef.current.forEach((controller) => controller.abort())
  const clear = () => {
    abortRef.current.forEach((controller) => controller.abort())
    abortRef.current = []
    paneStickToBottom.current = []
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
        <label className="field inline-field"><span>Model</span><select value={selectedModel} onChange={(event) => setModel(event.target.value)} disabled={running.length === 0 || sending}><option value="">Select a running model</option>{modelNames.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
        {selectedModel && running.find((item) => item.alias === selectedModel) && <RuntimeMark runtime={running.find((item) => item.alias === selectedModel)!.runtime} />}
        <label className="field inline-field"><span>Concurrency</span><select value={String(concurrency)} onChange={(event) => setConcurrency(Number(event.target.value))} disabled={running.length === 0 || sending}>{CONCURRENCY_OPTIONS.map((count) => <option key={count} value={String(count)}>{count} {count === 1 ? 'session' : 'sessions'}</option>)}</select></label>
        <span className="privacy-note"><Gauge size={14} /> Live metrics · content private</span>
      </div>
      <p className="sr-only" role="status" aria-live="polite">{sending ? 'Generating response' : messages.length ? 'Response complete' : ''}</p>
      <section ref={conversationRef} className={`conversation${paneCount > 1 ? ' has-panes' : ''}`} aria-label="Conversation" onScroll={(event) => {
        shouldAutoScrollRef.current = stickToBottom(event.currentTarget)
      }}>
        {messages.length === 0 ? (
          <div className="chat-empty"><span className="chat-empty-icon"><Bot size={22} /></span><h2>How can I help?</h2><p>Choose a running model and start a private, local conversation. Thinking and answer text stream as they are generated.</p></div>
        ) : paneCount > 1 ? (
          <>
            <div className="chat-panes">
              {Array.from({ length: paneCount }, (_, paneIndex) => (
                <div
                  key={paneIndex}
                  className="chat-pane"
                  role="log"
                  aria-label={`Session ${paneIndex + 1}`}
                  ref={(element) => { paneRefs.current[paneIndex] = element }}
                  onScroll={(event) => { paneStickToBottom.current[paneIndex] = stickToBottom(event.currentTarget) }}
                >
                  <p className="chat-pane-label">Session {paneIndex + 1}</p>
                  {renderEntries.map((entry) => entry.kind === 'runs' ? (() => {
                    // Assign by the stable runIndex, never array position: a
                    // run that fails before producing content is removed from
                    // the group, and the survivors must stay with their own
                    // session so the panes keep showing the history that is
                    // actually replayed to the model.
                    const run = entry.runs.find((item) => item.runIndex === paneIndex + 1)
                    return run ? <AssistantMessage key={run.id} className="message message-assistant" message={run} runLabel={false} /> : null
                  })() : entry.message.role === 'user' ? (
                    <UserBubble key={entry.message.id} message={entry.message} />
                  ) : (
                    <AssistantMessage key={entry.message.id} className="message message-assistant" message={entry.message} />
                  ))}
                </div>
              ))}
            </div>
            <div className="conversation-footer">
              {renderEntries.flatMap((entry) => entry.kind === 'runs'
                ? [<ConcurrencySummary key={entry.groupId} runs={entry.runs} />]
                : [])}
            </div>
          </>
        ) : renderEntries.map((entry) => entry.kind === 'runs' ? (
          entry.runs.map((run) => <AssistantMessage key={run.id} className="message message-assistant" message={run} />)
        ) : entry.message.role === 'user' ? (
          <UserBubble key={entry.message.id} message={entry.message} />
        ) : (
          <AssistantMessage key={entry.message.id} className="message message-assistant" message={entry.message} />
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
