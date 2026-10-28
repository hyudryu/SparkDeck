import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ChatStreamOptions } from '../api/client'
import { api } from '../api/client'
import type { ChatStreamResult } from '../api/types'
import { ChatPage } from './ChatPage'

vi.mock('../api/client', () => ({
  api: {
    deployments: { list: vi.fn() },
    chatStream: vi.fn(),
  },
}))

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

const pngFile = (name: string) => new File(
  [new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])],
  name,
  { type: 'image/png' },
)

const mp4File = (name: string) => new File(
  [new Uint8Array([0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70, 0x6d, 0x70, 0x34, 0x32])],
  name,
  { type: 'video/mp4' },
)

const dropOnComposer = (container: HTMLElement, files: File[]) => {
  fireEvent.drop(container.querySelector('.composer')!, {
    dataTransfer: { types: ['Files'], files },
  })
}

describe('ChatPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.deployments.list).mockResolvedValue([{
      id: 'deployment-1', alias: 'reasoner', runtime: 'vllm', status: 'running',
    } as never])
  })

  it('renders thinking and answer deltas before the stream completes, then keeps metrics', async () => {
    let streamOptions: ChatStreamOptions | undefined
    let finishStream: ((result: ChatStreamResult) => void) | undefined
    vi.mocked(api.chatStream).mockImplementation((_model, _messages, options) => {
      streamOptions = options
      return new Promise((resolve) => {
        finishStream = resolve
        options?.onUpdate?.({
          reasoning: 'Inspecting the request',
          metrics: { ttft_ms: 240 },
        })
      })
    })
    const user = userEvent.setup()
    render(<ChatPage />)

    const composer = await screen.findByRole('textbox', { name: 'Message' })
    await user.type(composer, 'Explain streaming')
    await user.click(screen.getByRole('button', { name: 'Send message' }))

    expect(await screen.findByText('Inspecting the request')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Stop generating' })).toBeInTheDocument()
    expect(screen.getByText('240 ms')).toBeInTheDocument()

    act(() => streamOptions?.onUpdate?.({
      content: '## Live answer\n\nTokens now arrive live.',
      metrics: {
        prompt_tokens_per_second: 1200,
        ttft_ms: 240,
        output_tokens_per_second: 42.5,
      },
    }))
    expect(await screen.findByText('Tokens now arrive live.', {}, { timeout: 5000 })).toBeInTheDocument()
    expect(await screen.findByRole('heading', { name: 'Live answer' }, { timeout: 5000 })).toBeInTheDocument()
    expect(screen.getByText('1200.0 tok/s')).toBeInTheDocument()
    expect(screen.getByText('42.5 tok/s')).toBeInTheDocument()

    await act(async () => finishStream?.({
      message: { role: 'assistant', content: '## Live answer\n\nTokens now arrive live.' },
      reasoning: 'Inspecting the request',
      metrics: {
        prompt_tokens_per_second: 1200,
        ttft_ms: 240,
        output_tokens_per_second: 42.5,
      },
    }))

    await waitFor(() => expect(screen.queryByRole('button', { name: 'Stop generating' })).not.toBeInTheDocument())
    expect(screen.getAllByText('Tokens now arrive live.')).toHaveLength(1)
    expect(screen.getByRole('button', { name: 'Send message' })).toBeInTheDocument()
  })

  it('fans one prompt out into side-by-side session panes with a combined summary', async () => {
    const finishers: Array<(result: ChatStreamResult) => void> = []
    vi.mocked(api.chatStream).mockImplementation((_model, _messages, options) => {
      const index = finishers.length
      return new Promise<ChatStreamResult>((resolve) => {
        finishers.push(resolve)
        options?.onUpdate?.({
          content: 'Shared answer',
          metrics: { ttft_ms: 200 + index * 100, output_tokens_per_second: 20 },
        })
      })
    })
    const user = userEvent.setup()
    const { container } = render(<ChatPage />)

    await screen.findByRole('option', { name: 'reasoner' })
    await user.selectOptions(screen.getByLabelText('Concurrency'), '2')
    const composer = await screen.findByRole('textbox', { name: 'Message' })
    await user.type(composer, 'Load test')
    await user.click(screen.getByRole('button', { name: 'Send message' }))

    // Each session becomes its own full-height pane; the pane's own label
    // identifies the session, replacing the old 'Run i of n' author suffix.
    expect(await screen.findByRole('log', { name: 'Session 1' })).toBeInTheDocument()
    expect(screen.getByRole('log', { name: 'Session 2' })).toBeInTheDocument()
    expect(screen.queryByText(/Run \d+ of \d+/)).not.toBeInTheDocument()
    expect(await screen.findAllByText('Shared answer')).toHaveLength(2)
    expect(screen.queryByText(/^2 concurrent runs/)).not.toBeInTheDocument()

    await waitFor(() => expect(api.chatStream).toHaveBeenCalledTimes(2))
    expect(vi.mocked(api.chatStream).mock.calls[0]?.[1]).toEqual([{ role: 'user', content: 'Load test' }])
    expect(vi.mocked(api.chatStream).mock.calls[1]?.[1]).toEqual([{ role: 'user', content: 'Load test' }])

    await act(async () => {
      finishers[0]?.({ message: { role: 'assistant', content: 'Shared answer' }, reasoning: '', metrics: { ttft_ms: 200, output_tokens_per_second: 20, completion_tokens: 100 } })
      finishers[1]?.({ message: { role: 'assistant', content: 'Shared answer' }, reasoning: '', metrics: { ttft_ms: 300, output_tokens_per_second: 25, completion_tokens: 100 } })
    })

    // The combined rate divides total completion tokens by the shared
    // wall-clock window, so only its shape is asserted.
    expect(await screen.findByText(/^2 concurrent runs · combined \d+(\.\d+)? tok\/s over \d+(\.\d+)? s \(TTFT included\) · TTFT 200 ms – 300 ms · identical output$/)).toBeInTheDocument()

    // The sessions sit in independent panes with the combined summary in a
    // footer beneath them, and each session samples the shared prompt with
    // its own seed.
    expect(container.querySelectorAll('.chat-pane')).toHaveLength(2)
    expect(container.querySelector('.conversation.has-panes')).toBeInTheDocument()
    expect(container.querySelector('.conversation-footer .concurrency-summary')).toBeInTheDocument()
    expect(vi.mocked(api.chatStream).mock.calls[0]?.[2]?.seed).toBe(1)
    expect(vi.mocked(api.chatStream).mock.calls[1]?.[2]?.seed).toBe(2)
  })

  it('splits the conversation into N panes as soon as N sessions are selected', async () => {
    const user = userEvent.setup()
    const { container } = render(<ChatPage />)

    await screen.findByRole('option', { name: 'reasoner' })
    expect(screen.getByText('How can I help?')).toBeInTheDocument()

    // Choosing a concurrency splits the box into that many session columns
    // immediately, before any prompt is sent, previewing the fan-out.
    await user.selectOptions(screen.getByLabelText('Concurrency'), '4')
    expect(container.querySelectorAll('.chat-pane')).toHaveLength(4)
    expect(screen.getByRole('log', { name: 'Session 4' })).toBeInTheDocument()
    expect(container.querySelector('.conversation.has-panes')).toBeInTheDocument()
    expect(screen.queryByText('How can I help?')).not.toBeInTheDocument()

    // Turning the selector back down on an empty conversation returns to the
    // single empty state.
    await user.selectOptions(screen.getByLabelText('Concurrency'), '1')
    expect(container.querySelectorAll('.chat-pane')).toHaveLength(0)
    expect(screen.getByText('How can I help?')).toBeInTheDocument()
  })

  it('keeps a surviving answer in its own pane when an earlier run fails before content', async () => {
    let call = 0
    vi.mocked(api.chatStream).mockImplementation(() => (++call === 1
      ? Promise.reject(new Error('run one died'))
      : Promise.resolve({ message: { role: 'assistant', content: 'Survivor' }, reasoning: '', metrics: {} })))
    const user = userEvent.setup()
    render(<ChatPage />)

    await screen.findByRole('option', { name: 'reasoner' })
    await user.selectOptions(screen.getByLabelText('Concurrency'), '2')
    const composer = await screen.findByRole('textbox', { name: 'Message' })
    await user.type(composer, 'Fail one')
    await user.click(screen.getByRole('button', { name: 'Send message' }))

    // Removing run 1 (it failed before producing content) compacts the
    // group's array, yet run 2's answer must stay with Session 2 and the
    // column count must not collapse.
    await screen.findByText('Survivor')
    expect(within(screen.getByRole('log', { name: 'Session 2' })).getByText('Survivor')).toBeInTheDocument()
    expect(within(screen.getByRole('log', { name: 'Session 1' })).queryByText('Survivor')).not.toBeInTheDocument()
    expect(document.querySelectorAll('.chat-pane')).toHaveLength(2)
  })

  it('lists each running model name once across deployments sharing it', async () => {
    vi.mocked(api.deployments.list).mockResolvedValue([
      { id: 'a', alias: 'shared', runtime: 'vllm', status: 'running' } as never,
      { id: 'b', alias: 'shared', runtime: 'sglang', status: 'running' } as never,
      { id: 'c', alias: 'solo', runtime: 'vllm', status: 'running' } as never,
    ])
    vi.mocked(api.chatStream).mockResolvedValue({
      message: { role: 'assistant', content: 'Balanced' }, reasoning: '', metrics: {},
    })
    const user = userEvent.setup()
    render(<ChatPage />)

    // The gateway load-balances requests by model name across the
    // deployments serving it, so the dropdown is keyed by name only.
    const select = await screen.findByLabelText('Model')
    expect(within(select).getAllByRole('option').map((option) => option.textContent))
      .toEqual(['Select a running model', 'shared', 'solo'])

    await user.selectOptions(select, 'solo')
    await user.type(screen.getByRole('textbox', { name: 'Message' }), 'Route by name')
    await user.click(screen.getByRole('button', { name: 'Send message' }))
    await screen.findByText('Balanced')
    expect(vi.mocked(api.chatStream).mock.calls[0]?.[0]).toBe('solo')
  })

  it('lists the served model name once across deployments with different aliases', async () => {
    vi.mocked(api.deployments.list).mockResolvedValue([
      { id: 'a', alias: 'qwen on gx10', runtime: 'vllm', status: 'running', served_models: ['qwen3.8'] } as never,
      { id: 'b', alias: 'qwen on ws1', runtime: 'vllm', status: 'running', served_model: 'qwen3.8' } as never,
      { id: 'c', alias: 'solo', runtime: 'vllm', status: 'running' } as never,
    ])
    vi.mocked(api.chatStream).mockResolvedValue({
      message: { role: 'assistant', content: 'Balanced' }, reasoning: '', metrics: {},
    })
    const user = userEvent.setup()
    render(<ChatPage />)

    // Deployments carrying different UI aliases but the same runtime served
    // name collapse into that one name, and the request carries it so the
    // gateway can rotate the owning deployment across clusters.
    const select = await screen.findByLabelText('Model')
    expect(within(select).getAllByRole('option').map((option) => option.textContent))
      .toEqual(['Select a running model', 'qwen3.8', 'solo'])

    await user.selectOptions(select, 'qwen3.8')
    await user.type(screen.getByRole('textbox', { name: 'Message' }), 'Balance me')
    await user.click(screen.getByRole('button', { name: 'Send message' }))
    await screen.findByText('Balanced')
    expect(vi.mocked(api.chatStream).mock.calls[0]?.[0]).toBe('qwen3.8')
  })

  it('offers the alias of a deployment whose served name is another alias', async () => {
    vi.mocked(api.deployments.list).mockResolvedValue([
      { id: 'a', alias: 'alpha', runtime: 'vllm', status: 'running', served_models: ['model-two'] } as never,
      { id: 'b', alias: 'model-two', runtime: 'sglang', status: 'running', served_models: ['other-model'] } as never,
    ])
    vi.mocked(api.chatStream).mockResolvedValue({
      message: { role: 'assistant', content: 'Unambiguous' }, reasoning: '', metrics: {},
    })
    const user = userEvent.setup()
    render(<ChatPage />)

    // The gateway resolves an exact alias match before other deployments'
    // served names, so 'model-two' would route to deployment b. Deployment
    // a stays reachable through its own alias as the unambiguous selector.
    const select = await screen.findByLabelText('Model')
    expect(within(select).getAllByRole('option').map((option) => option.textContent))
      .toEqual(['Select a running model', 'model-two', 'alpha', 'other-model'])

    await user.selectOptions(select, 'alpha')
    await user.type(screen.getByRole('textbox', { name: 'Message' }), 'Reach alpha')
    await user.click(screen.getByRole('button', { name: 'Send message' }))
    await screen.findByText('Unambiguous')
    expect(vi.mocked(api.chatStream).mock.calls[0]?.[0]).toBe('alpha')
  })

  it('replays one representative answer per earlier group when concurrency returns to one', async () => {
    let call = 0
    const results: ChatStreamResult[] = [
      { message: { role: 'assistant', content: 'Branch one' }, reasoning: '', metrics: {} },
      { message: { role: 'assistant', content: 'Branch two' }, reasoning: '', metrics: {} },
      { message: { role: 'assistant', content: 'Solo answer' }, reasoning: '', metrics: {} },
    ]
    vi.mocked(api.chatStream).mockImplementation(() => Promise.resolve(results[call++]!))
    const user = userEvent.setup()
    render(<ChatPage />)

    await screen.findByRole('option', { name: 'reasoner' })
    await user.selectOptions(screen.getByLabelText('Concurrency'), '2')
    const composer = await screen.findByRole('textbox', { name: 'Message' })
    await user.type(composer, 'First')
    await user.click(screen.getByRole('button', { name: 'Send message' }))
    expect(await screen.findByText('Branch one')).toBeInTheDocument()
    await screen.findByText('Branch two')

    await user.selectOptions(screen.getByLabelText('Concurrency'), '1')
    await user.type(composer, 'Second')
    await user.click(screen.getByRole('button', { name: 'Send message' }))
    // Single-session answers show in every pane of the split history.
    expect(await screen.findAllByText('Solo answer')).toHaveLength(2)

    await waitFor(() => expect(api.chatStream).toHaveBeenCalledTimes(3))
    expect(vi.mocked(api.chatStream).mock.calls[2]?.[1]).toEqual([
      { role: 'user', content: 'First' },
      { role: 'assistant', content: 'Branch one' },
      { role: 'user', content: 'Second' },
    ])
  })

  it('keeps concurrent sessions as independent history branches on follow-up turns', async () => {
    let call = 0
    const results: ChatStreamResult[] = [
      { message: { role: 'assistant', content: 'Branch one' }, reasoning: '', metrics: {} },
      { message: { role: 'assistant', content: 'Branch two' }, reasoning: '', metrics: {} },
      { message: { role: 'assistant', content: 'Second one' }, reasoning: '', metrics: {} },
      { message: { role: 'assistant', content: 'Second two' }, reasoning: '', metrics: {} },
    ]
    vi.mocked(api.chatStream).mockImplementation(() => Promise.resolve(results[call++]!))
    const user = userEvent.setup()
    render(<ChatPage />)

    await screen.findByRole('option', { name: 'reasoner' })
    await user.selectOptions(screen.getByLabelText('Concurrency'), '2')
    const composer = await screen.findByRole('textbox', { name: 'Message' })
    await user.type(composer, 'First')
    await user.click(screen.getByRole('button', { name: 'Send message' }))

    expect(await screen.findByText('Branch one')).toBeInTheDocument()
    expect(await screen.findByText('Branch two')).toBeInTheDocument()

    await user.type(composer, 'Second')
    await user.click(screen.getByRole('button', { name: 'Send message' }))
    expect(await screen.findByText('Second two')).toBeInTheDocument()

    // Run 1 replays only its own answer; run 2 only its own.
    await waitFor(() => expect(api.chatStream).toHaveBeenCalledTimes(4))
    expect(vi.mocked(api.chatStream).mock.calls[2]?.[1]).toEqual([
      { role: 'user', content: 'First' },
      { role: 'assistant', content: 'Branch one' },
      { role: 'user', content: 'Second' },
    ])
    expect(vi.mocked(api.chatStream).mock.calls[3]?.[1]).toEqual([
      { role: 'user', content: 'First' },
      { role: 'assistant', content: 'Branch two' },
      { role: 'user', content: 'Second' },
    ])
  })

  it('follows streaming thinking text until the reader scrolls the thinking box up, then resumes at the bottom', async () => {
    let streamOptions: ChatStreamOptions | undefined
    let finishStream: ((result: ChatStreamResult) => void) | undefined
    vi.mocked(api.chatStream).mockImplementation((_model, _messages, options) => {
      streamOptions = options
      return new Promise((resolve) => {
        finishStream = resolve
        options?.onUpdate?.({ reasoning: 'First thought.' })
      })
    })
    Object.defineProperty(HTMLElement.prototype, 'scrollHeight', { configurable: true, value: 500 })
    try {
      const user = userEvent.setup()
      render(<ChatPage />)
      const composer = await screen.findByRole('textbox', { name: 'Message' })
      await user.type(composer, 'Think out loud')
      await user.click(screen.getByRole('button', { name: 'Send message' }))

      const body = await screen.findByText('First thought.')
      expect(body.scrollTop).toBe(500)

      body.scrollTop = 0
      act(() => streamOptions?.onUpdate?.({ reasoning: ' Second thought.' }))
      expect(body.scrollTop).toBe(500)

      body.scrollTop = 0
      fireEvent.scroll(body)
      act(() => streamOptions?.onUpdate?.({ reasoning: ' Third thought.' }))
      expect(body.scrollTop).toBe(0)

      body.scrollTop = 450
      fireEvent.scroll(body)
      act(() => streamOptions?.onUpdate?.({ reasoning: ' Fourth thought.' }))
      expect(body.scrollTop).toBe(500)

      await act(async () => finishStream?.({
        message: { role: 'assistant', content: '' },
        reasoning: '', metrics: {},
      }))
    } finally {
      delete (HTMLElement.prototype as unknown as Record<string, unknown>).scrollHeight
    }
  })

  it('omits the prompt-processing metric when the engine does not measure it', async () => {
    vi.mocked(api.chatStream).mockResolvedValue({
      message: { role: 'assistant', content: 'Done' },
      reasoning: '',
      metrics: { ttft_ms: 300, output_tokens_per_second: 40 },
    })
    const user = userEvent.setup()
    render(<ChatPage />)

    const composer = await screen.findByRole('textbox', { name: 'Message' })
    await user.type(composer, 'Hi')
    await user.click(screen.getByRole('button', { name: 'Send message' }))

    await screen.findByText('Done')
    expect(screen.queryByText('Prompt processing')).not.toBeInTheDocument()
    expect(screen.getByText('TTFT')).toBeInTheDocument()
    expect(screen.getByText('300 ms')).toBeInTheDocument()
    expect(screen.getByText('40.0 tok/s')).toBeInTheDocument()
    // A single session leaves the sampling seed to the engine default.
    expect(vi.mocked(api.chatStream).mock.calls[0]?.[2]?.seed).toBeUndefined()
  })

  it('marks a partial failed response and excludes it from later prompts', async () => {
    vi.mocked(api.chatStream)
      .mockImplementationOnce((_model, _messages, options) => {
        options?.onUpdate?.({ content: 'Partial answer' })
        return Promise.reject(new Error('GPU worker stopped'))
      })
      .mockResolvedValueOnce({
        message: { role: 'assistant', content: 'Recovered' },
        reasoning: '', metrics: {},
      })
    const user = userEvent.setup()
    render(<ChatPage />)

    const composer = await screen.findByRole('textbox', { name: 'Message' })
    await user.type(composer, 'First request')
    await user.click(screen.getByRole('button', { name: 'Send message' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('GPU worker stopped')
    expect(screen.getByText('Partial answer')).toBeInTheDocument()
    expect(screen.getByText('Response interrupted')).toBeInTheDocument()

    await user.type(composer, 'Second request')
    await user.click(screen.getByRole('button', { name: 'Send message' }))
    await screen.findByText('Recovered')

    expect(vi.mocked(api.chatStream).mock.calls[1]?.[1]).toEqual([
      { role: 'user', content: 'First request' },
      { role: 'user', content: 'Second request' },
    ])
  })

  it('pastes an image and sends an image-only multimodal prompt', async () => {
    vi.mocked(api.chatStream).mockResolvedValue({
      message: { role: 'assistant', content: 'I see the image.' },
      reasoning: '',
      metrics: {},
    })
    const user = userEvent.setup()
    render(<ChatPage />)

    const composer = await screen.findByRole('textbox', { name: 'Message' })
    const image = new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])], 'clipboard.png', { type: 'image/png' })
    fireEvent.paste(composer, {
      clipboardData: {
        items: [{ kind: 'file', type: 'image/png', getAsFile: () => image }],
      },
    })

    expect(await screen.findByText('clipboard.png')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Send message' })).toBeEnabled()
    await user.type(composer, '{Enter}')

    await screen.findByText('I see the image.')
    expect(vi.mocked(api.chatStream).mock.calls[0]?.[1]).toEqual([{
      role: 'user',
      content: [{
        type: 'image_url',
        image_url: { url: 'data:image/png;base64,iVBORw0KGgo=' },
      }],
    }])
    expect(screen.getByRole('img', { name: 'Attached clipboard.png' })).toBeInTheDocument()
  })

  it('uploads, removes, and preserves images in follow-up history', async () => {
    vi.mocked(api.chatStream)
      .mockResolvedValueOnce({
        message: { role: 'assistant', content: 'First answer' },
        reasoning: '', metrics: {},
      })
      .mockResolvedValueOnce({
        message: { role: 'assistant', content: 'Follow-up answer' },
        reasoning: '', metrics: {},
      })
    const user = userEvent.setup()
    render(<ChatPage />)

    const picker = await screen.findByLabelText('Choose image or video files')
    const first = new File(['RIFF\x00\x00\x00\x00WEBP'], 'first.webp', { type: 'image/webp' })
    const removed = new File([new Uint8Array([0xff, 0xd8, 0xff, 0xdb])], 'remove.jpg', { type: 'image/jpeg' })
    await user.upload(picker, [first, removed])
    const attachments = await screen.findByLabelText('Attached media')
    expect(within(attachments).getByText('first.webp')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Remove remove.jpg' }))
    expect(within(attachments).queryByText('remove.jpg')).not.toBeInTheDocument()

    const composer = screen.getByRole('textbox', { name: 'Message' })
    await user.type(composer, 'What is shown?')
    await user.click(screen.getByRole('button', { name: 'Send message' }))
    await screen.findByText('First answer')

    const firstRequest = vi.mocked(api.chatStream).mock.calls[0]?.[1]
    expect(firstRequest?.[0]).toEqual({
      role: 'user',
      content: [
        { type: 'text', text: 'What is shown?' },
        { type: 'image_url', image_url: { url: expect.stringMatching(/^data:image\/webp;base64,/) } },
      ],
    })

    await user.type(composer, 'Look closer')
    await user.click(screen.getByRole('button', { name: 'Send message' }))
    await screen.findByText('Follow-up answer')
    expect(vi.mocked(api.chatStream).mock.calls[1]?.[1]).toEqual([
      firstRequest?.[0],
      { role: 'assistant', content: 'First answer' },
      { role: 'user', content: 'Look closer' },
    ])
  })

  it('serializes overlapping attachment reads without dropping either image', async () => {
    render(<ChatPage />)

    const picker = await screen.findByLabelText('Choose image or video files')
    fireEvent.change(picker, { target: { files: [pngFile('first.png')] } })
    fireEvent.change(picker, { target: { files: [pngFile('second.png')] } })

    const attachments = await screen.findByLabelText('Attached media')
    expect(await within(attachments).findByText('first.png')).toBeInTheDocument()
    expect(await within(attachments).findByText('second.png')).toBeInTheDocument()
  })

  it('does not attach an image whose read finishes after sending', async () => {
    vi.mocked(api.chatStream).mockResolvedValue({
      message: { role: 'assistant', content: 'Done' },
      reasoning: '', metrics: {},
    })
    render(<ChatPage />)

    const composer = await screen.findByRole('textbox', { name: 'Message' })
    fireEvent.change(composer, { target: { value: 'Send immediately' } })
    fireEvent.change(screen.getByLabelText('Choose image or video files'), {
      target: { files: [pngFile('late.png')] },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))

    await screen.findByText('Done')
    await waitFor(() => expect(screen.queryByText('late.png')).not.toBeInTheDocument())
    expect(vi.mocked(api.chatStream).mock.calls[0]?.[1]).toEqual([{
      role: 'user', content: 'Send immediately',
    }])
  })

  it('keeps a pending upload when another attachment is removed', async () => {
    const user = userEvent.setup()
    render(<ChatPage />)

    const picker = await screen.findByLabelText('Choose image or video files')
    await user.upload(picker, pngFile('remove.png'))
    await screen.findByText('remove.png')

    fireEvent.change(picker, { target: { files: [pngFile('pending.png')] } })
    fireEvent.click(screen.getByRole('button', { name: 'Remove remove.png' }))

    expect(screen.queryByText('remove.png')).not.toBeInTheDocument()
    expect(await screen.findByText('pending.png')).toBeInTheDocument()
  })

  it('reuses a concurrently freed slot for the next file in a pending batch', async () => {
    const user = userEvent.setup()
    render(<ChatPage />)

    const picker = await screen.findByLabelText('Choose image or video files')
    await user.upload(picker, [pngFile('existing-1.png'), pngFile('existing-2.png'), pngFile('existing-3.png')])
    await screen.findByText('existing-3.png')

    const pendingReads: Array<() => void> = []
    class DeferredFileReader {
      result: string | ArrayBuffer | null = null
      onerror: (() => void) | null = null
      onload: (() => void) | null = null

      readAsDataURL(file: File) {
        pendingReads.push(() => {
          this.result = `data:${file.type};base64,iVBORw0KGgo=`
          this.onload?.()
        })
      }
    }
    vi.stubGlobal('FileReader', DeferredFileReader)

    fireEvent.change(picker, { target: { files: [pngFile('new-1.png'), pngFile('new-2.png')] } })
    await waitFor(() => expect(pendingReads).toHaveLength(1))
    fireEvent.click(screen.getByRole('button', { name: 'Remove existing-1.png' }))
    act(() => pendingReads[0]?.())
    await waitFor(() => expect(pendingReads).toHaveLength(2))
    act(() => pendingReads[1]?.())

    expect(await screen.findByText('new-1.png')).toBeInTheDocument()
    expect(await screen.findByText('new-2.png')).toBeInTheDocument()
    expect(screen.queryByText('existing-1.png')).not.toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('rejects unsupported and excess pasted images with an accessible error', async () => {
    const user = userEvent.setup()
    render(<ChatPage />)

    const composer = await screen.findByRole('textbox', { name: 'Message' })
    const svg = new File(['<svg/>'], 'unsafe.svg', { type: 'image/svg+xml' })
    fireEvent.paste(composer, {
      clipboardData: {
        items: [{ kind: 'file', type: 'image/svg+xml', getAsFile: () => svg }],
      },
    })
    expect(await screen.findByRole('alert')).toHaveTextContent('not a supported image or video')
    expect(screen.getByRole('button', { name: 'Send message' })).toBeDisabled()

    await user.upload(screen.getByLabelText('Choose image or video files'), new File(
      ['not really an image'], 'spoofed.png', { type: 'image/png' },
    ))
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('does not contain valid PNG image data'))

    const picker = screen.getByLabelText('Choose image or video files')
    const files = Array.from({ length: 5 }, (_, index) => pngFile(`image-${index}.png`))
    await user.upload(picker, files)
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('up to 4 images'))
    expect(within(screen.getByLabelText('Attached media')).getAllByRole('button', { name: /^Remove / })).toHaveLength(4)
  })

  it('attaches a dropped image and sends it as an image_url part', async () => {
    vi.mocked(api.chatStream).mockResolvedValue({
      message: { role: 'assistant', content: 'I see it.' },
      reasoning: '', metrics: {},
    })
    const user = userEvent.setup()
    const { container } = render(<ChatPage />)

    const composer = await screen.findByRole('textbox', { name: 'Message' })
    dropOnComposer(container, [pngFile('dropped.png')])

    expect(await screen.findByText('dropped.png')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Send message' })).toBeEnabled()
    await user.type(composer, 'What is this?')
    await user.click(screen.getByRole('button', { name: 'Send message' }))
    await screen.findByText('I see it.')
    expect(vi.mocked(api.chatStream).mock.calls[0]?.[1]).toEqual([{
      role: 'user',
      content: [
        { type: 'text', text: 'What is this?' },
        { type: 'image_url', image_url: { url: expect.stringMatching(/^data:image\/png;base64,/) } },
      ],
    }])
    expect(screen.getByRole('img', { name: 'Attached dropped.png' })).toBeInTheDocument()
  })

  it('attaches a dropped video and sends it as a video_url part', async () => {
    vi.mocked(api.chatStream).mockResolvedValue({
      message: { role: 'assistant', content: 'A short clip.' },
      reasoning: '', metrics: {},
    })
    const user = userEvent.setup()
    const { container } = render(<ChatPage />)

    const composer = await screen.findByRole('textbox', { name: 'Message' })
    dropOnComposer(container, [mp4File('demo.mp4')])

    expect(await screen.findByText('demo.mp4')).toBeInTheDocument()
    expect(container.querySelector('.composer-attachment video')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Send message' })).toBeEnabled()
    await user.type(composer, 'Describe the clip')
    await user.click(screen.getByRole('button', { name: 'Send message' }))
    await screen.findByText('A short clip.')
    expect(container.querySelector('.user-message-media video[aria-label="Attached demo.mp4"]')).toBeInTheDocument()
    expect(vi.mocked(api.chatStream).mock.calls[0]?.[1]).toEqual([{
      role: 'user',
      content: [
        { type: 'text', text: 'Describe the clip' },
        { type: 'video_url', video_url: { url: expect.stringMatching(/^data:video\/mp4;base64,/) } },
      ],
    }])
  })

  it('shows the drop hint while files are dragged over the composer', async () => {
    const { container } = render(<ChatPage />)
    const composer = container.querySelector('.composer')!
    await screen.findByRole('textbox', { name: 'Message' })

    fireEvent.dragEnter(composer, { dataTransfer: { types: ['Files'] } })
    expect(screen.getByText('Drop images or videos to attach')).toBeInTheDocument()

    fireEvent.dragLeave(composer, { dataTransfer: { types: ['Files'] } })
    expect(screen.queryByText('Drop images or videos to attach')).not.toBeInTheDocument()
  })

  it('ignores drags that do not carry files', async () => {
    const { container } = render(<ChatPage />)
    const composer = container.querySelector('.composer')!
    await screen.findByRole('textbox', { name: 'Message' })

    fireEvent.dragEnter(composer, { dataTransfer: { types: ['text/plain'] } })
    expect(screen.queryByText('Drop images or videos to attach')).not.toBeInTheDocument()
  })

  it('rejects dropped files that are not supported media', async () => {
    const { container } = render(<ChatPage />)
    await screen.findByRole('textbox', { name: 'Message' })

    dropOnComposer(container, [new File(['%PDF-1.4'], 'notes.pdf', { type: 'application/pdf' })])
    expect(await screen.findByRole('alert')).toHaveTextContent('not a supported image or video')
    expect(screen.getByRole('button', { name: 'Send message' })).toBeDisabled()

    dropOnComposer(container, [new File(['plain text'], 'fake.mp4', { type: 'video/mp4' })])
    expect(await screen.findByRole('alert')).toHaveTextContent('does not contain valid MP4 video data')
  })

  it('enforces per-kind attachment counts across a mixed dropped batch', async () => {
    const { container } = render(<ChatPage />)
    await screen.findByRole('textbox', { name: 'Message' })

    dropOnComposer(container, [
      pngFile('a.png'), pngFile('b.png'), pngFile('c.png'), pngFile('d.png'), pngFile('e.png'),
      mp4File('one.mp4'), mp4File('two.mp4'), mp4File('three.mp4'),
    ])

    const attachments = await screen.findByLabelText('Attached media')
    await waitFor(() => expect(within(attachments).getAllByRole('button', { name: /^Remove / })).toHaveLength(6))
    expect(within(attachments).queryByText('e.png')).not.toBeInTheDocument()
    expect(within(attachments).queryByText('three.mp4')).not.toBeInTheDocument()
    const alert = screen.getByRole('alert')
    expect(alert).toHaveTextContent('up to 4 images')
    expect(alert).toHaveTextContent('up to 2 videos')
  })

  it('rejects a dropped video above the per-video size limit', async () => {
    const { container } = render(<ChatPage />)
    await screen.findByRole('textbox', { name: 'Message' })

    const bytes = new Uint8Array(16 * 1024 * 1024 + 1)
    bytes.set([0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70], 0)
    dropOnComposer(container, [new File([bytes], 'large.mp4', { type: 'video/mp4' })])

    expect(await screen.findByRole('alert')).toHaveTextContent('exceeds the 16 MB per-video limit')
  })
})
