import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ImagesPage } from './ImagesPage'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('ImagesPage node pulls', () => {
  it('shows remote-only availability after a targeted pull reloads inventory', async () => {
    let imageListCalls = 0
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.endsWith('/api/v1/nodes')) {
        return new Response(JSON.stringify({ items: [
          { id: 'local', name: 'This device', local: true, online: true, docker_ready: true, selectable: true },
          { id: 'spark-2', name: 'Studio Spark', online: true, docker_ready: true, selectable: true },
        ] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path.endsWith('/api/v1/onboarding')) {
        return new Response(JSON.stringify({ role: 'controller', node: { id: 'local', name: 'Controller', port: 7878, access_urls: [] }, controller_reachable: true }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path.endsWith('/api/v1/images/pull') && init?.method === 'POST') {
        return new Response(JSON.stringify({ ok: true, image: 'org/remote:v1', node_ids: ['spark-2'], selected_nodes: [{ id: 'spark-2', name: 'Studio Spark' }], results: [{ node_id: 'spark-2', node_name: 'Studio Spark', ok: true }] }), { status: 201, headers: { 'Content-Type': 'application/json' } })
      }
      if (path.endsWith('/api/v1/images')) {
        imageListCalls += 1
        const items = imageListCalls === 1 ? [] : [{ id: 'sha256:remote', repository: 'org/remote', tag: 'v1', node_ids: ['spark-2'], selected_nodes: [{ id: 'spark-2', name: 'Studio Spark' }] }]
        return new Response(JSON.stringify({ items }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify({}), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }))
    const user = userEvent.setup()
    render(<ImagesPage />)

    await user.click(await screen.findByRole('checkbox', { name: /Studio Spark/ }))
    await user.click(screen.getByRole('checkbox', { name: /This device/ }))
    await user.type(screen.getByRole('textbox', { name: 'Container image' }), 'org/remote:v1')
    await user.click(screen.getByRole('button', { name: 'Pull on 1 node' }))

    const availability = await screen.findByLabelText('Available on Studio Spark')
    expect(availability).toHaveTextContent('Studio Spark')
    expect(availability).not.toHaveTextContent('This device')
    expect(screen.getByRole('status')).toHaveTextContent('Pulled org/remote:v1 on Studio Spark.')
  })

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
