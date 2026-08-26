import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ImagesPage } from './ImagesPage'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('ImagesPage node pulls', () => {
  it('reports partial node failures without announcing success', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.endsWith('/api/v1/nodes')) {
        return new Response(JSON.stringify({ items: [
          { id: 'local', name: 'This device', local: true, online: true, docker_ready: true, selectable: true },
          { id: 'spark-2', name: 'Studio Spark', online: true, docker_ready: true, selectable: true },
        ] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path.endsWith('/api/v1/images/pull') && init?.method === 'POST') {
        return new Response(JSON.stringify({
          ok: false,
          image: 'vllm/vllm-openai:v1',
          node_ids: ['local', 'spark-2'],
          results: [
            { node_id: 'local', node_name: 'This device', ok: true },
            { node_id: 'spark-2', node_name: 'Studio Spark', ok: false, error: 'registry unavailable' },
          ],
        }), { status: 201, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify({ items: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }))
    const user = userEvent.setup()
    render(<ImagesPage />)

    await user.click(await screen.findByRole('checkbox', { name: /Studio Spark/ }))
    await user.type(screen.getByRole('textbox', { name: 'Container image' }), 'vllm/vllm-openai:v1')
    await user.click(screen.getByRole('button', { name: 'Pull on 2 nodes' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Studio Spark: registry unavailable')
    expect(screen.queryByText(/^Pulled /)).not.toBeInTheDocument()
  })
})
