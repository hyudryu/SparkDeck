import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
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
})
