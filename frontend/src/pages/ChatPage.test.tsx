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

afterEach(() => cleanup())

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

  it('marks a partial failed response and excludes it from later prompts', async () => {
    vi.mocked(api.chatStream)
      .mockImplementationOnce((_model, _messages, options) => {
        options?.onUpdate?.({ content: 'Partial answer' })
        return Promise.reject(new Error('GPU worker stopped'))
      })
      .mockResolvedValueOnce({
        message: { role: 'assistant', content: 'Recovered' },
        reasoning: '',
        metrics: {},
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

    const picker = await screen.findByLabelText('Choose image files')
    const first = new File(['RIFF\x00\x00\x00\x00WEBP'], 'first.webp', { type: 'image/webp' })
    const removed = new File([new Uint8Array([0xff, 0xd8, 0xff, 0xdb])], 'remove.jpg', { type: 'image/jpeg' })
    await user.upload(picker, [first, removed])
    const attachments = await screen.findByLabelText('Attached images')
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

    const picker = await screen.findByLabelText('Choose image files')
    const png = (name: string) => new File(
      [new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])],
      name,
      { type: 'image/png' },
    )
    fireEvent.change(picker, { target: { files: [png('first.png')] } })
    fireEvent.change(picker, { target: { files: [png('second.png')] } })

    const attachments = await screen.findByLabelText('Attached images')
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
    const image = new File(
      [new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])],
      'late.png',
      { type: 'image/png' },
    )
    fireEvent.change(screen.getByLabelText('Choose image files'), { target: { files: [image] } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))

    await screen.findByText('Done')
    await waitFor(() => expect(screen.queryByText('late.png')).not.toBeInTheDocument())
    expect(vi.mocked(api.chatStream).mock.calls[0]?.[1]).toEqual([{
      role: 'user', content: 'Send immediately',
    }])
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
    expect(await screen.findByRole('alert')).toHaveTextContent('not a PNG, JPEG, WebP, or GIF')
    expect(screen.getByRole('button', { name: 'Send message' })).toBeDisabled()

    await user.upload(screen.getByLabelText('Choose image files'), new File(
      ['not really an image'], 'spoofed.png', { type: 'image/png' },
    ))
    expect(await screen.findByRole('alert')).toHaveTextContent('does not contain valid PNG image data')

    const picker = screen.getByLabelText('Choose image files')
    const files = Array.from({ length: 5 }, (_, index) => new File(
      [new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])],
      `image-${index}.png`,
      { type: 'image/png' },
    ))
    await user.upload(picker, files)
    expect(await screen.findByRole('alert')).toHaveTextContent('up to 4 images')
    expect(within(screen.getByLabelText('Attached images')).getAllByRole('button', { name: /^Remove / })).toHaveLength(4)
  })
})
