import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import type { PatchBuild } from '../api/types'
import { PatchedImages } from './PatchedImages'

const nodes = [{ id: 'local', name: 'Node 1', local: true, online: true }, { id: 'node-3', name: 'Node 3', online: true }]
const build: PatchBuild = { id: 'build-1', base_image: 'example/base:v1', image: 'sparkdeck/patched:v1', created_at: '2026-09-06T00:00:00Z', status: 'queued', files: [{ target: '/app/patch.py', sha256: 'abc123' }], nodes: [{ node_id: 'node-3', node_name: 'Node 3', status: 'queued', logs: [] }] }
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.useRealTimers() })

function setup(onBuilt = vi.fn()) {
  vi.spyOn(api.images, 'patchBuilds').mockResolvedValue({ items: [] })
  render(<PatchedImages images={[{ id: 'base', tags: ['example/base:v1'] }]} nodes={nodes} localLabel="This device" onBuilt={onBuilt} />)
  return onBuilt
}

async function fillForm() {
  const user = userEvent.setup()
  await user.click(screen.getByRole('button', { name: 'Create patched image' }))
  await user.selectOptions(screen.getByRole('combobox', { name: 'Available base image' }), 'example/base:v1')
  await user.type(screen.getByRole('textbox', { name: 'Patched image tag' }), 'sparkdeck/patched:v1')
  await user.type(screen.getByRole('textbox', { name: 'Container destination 1' }), '/app/patch.py')
  await user.click(screen.getByRole('checkbox', { name: /Node 3/ }))
  await user.click(screen.getByRole('checkbox', { name: /This device/ }))
  return user
}

describe('PatchedImages', () => {
  it('recovers an accepted build from history after its response is lost', async () => {
    const list = vi.spyOn(api.images, 'patchBuilds').mockResolvedValueOnce({ items: [] }).mockResolvedValue({ items: [build] })
    vi.spyOn(api.images, 'createPatchBuild').mockRejectedValue(new Error('Network response lost'))
    render(<PatchedImages images={[{ id: 'base', tags: ['example/base:v1'] }]} nodes={nodes} localLabel="This device" onBuilt={vi.fn()} />)
    const user = await fillForm()
    await user.click(screen.getByRole('button', { name: 'Build patched image' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Network response lost')
    await waitFor(() => expect(list).toHaveBeenCalledTimes(2))
    expect(await screen.findByRole('status')).toHaveTextContent('Queued')
    expect(screen.getByText('Build history')).toBeInTheDocument()
  })

  it('replaces an unavailable default node when inventory arrives', async () => {
    vi.spyOn(api.images, 'patchBuilds').mockResolvedValue({ items: [] })
    const create = vi.spyOn(api.images, 'createPatchBuild').mockResolvedValue(build)
    const onBuilt = vi.fn()
    const view = render(<PatchedImages images={[]} nodes={[]} localLabel="This device" onBuilt={onBuilt} />)
    view.rerender(<PatchedImages images={[]} nodes={[{ ...nodes[0], online: false }, nodes[1]]} localLabel="This device" onBuilt={onBuilt} />)
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Create patched image' }))
    expect(screen.getByRole('checkbox', { name: /This device/ })).not.toBeChecked()
    expect(screen.getByRole('checkbox', { name: /Node 3/ })).toBeChecked()
    await user.type(screen.getByRole('textbox', { name: 'Base image reference' }), 'example/base:v1')
    await user.type(screen.getByRole('textbox', { name: 'Patched image tag' }), 'sparkdeck/patched:v1')
    await user.type(screen.getByRole('textbox', { name: 'Container destination 1' }), '/app/__init__.py')
    await user.click(screen.getByRole('button', { name: 'Build patched image' }))
    await waitFor(() => expect(create).toHaveBeenCalledWith(expect.objectContaining({ node_ids: ['node-3'], files: [{ target: '/app/__init__.py', content: '' }] })))
  })

  it('accepts an uploaded empty package marker', async () => {
    setup()
    const create = vi.spyOn(api.images, 'createPatchBuild').mockResolvedValue(build)
    const user = await fillForm()
    const file = new File([], '__init__.py')
    Object.defineProperty(file, 'arrayBuffer', { value: async () => new ArrayBuffer(0) })
    await user.upload(screen.getByLabelText('Upload Python file 1'), file)
    await user.click(screen.getByRole('button', { name: 'Build patched image' }))
    await waitFor(() => expect(create).toHaveBeenCalledWith(expect.objectContaining({ files: [{ target: '/app/patch.py', content: '' }] })))
  })

  it('uploads Python text and submits only the selected node and explicit destination', async () => {
    setup()
    const create = vi.spyOn(api.images, 'createPatchBuild').mockResolvedValue(build)
    const user = await fillForm()
    const file = new File(['print("patched")\n'], 'patch.py', { type: 'text/x-python' })
    Object.defineProperty(file, 'arrayBuffer', { value: async () => new TextEncoder().encode('print("patched")\n').buffer })
    await user.upload(screen.getByLabelText('Upload Python file 1'), file)
    expect(await screen.findByRole('textbox', { name: 'Python contents 1' })).toHaveValue('print("patched")\n')
    await user.click(screen.getByRole('button', { name: 'Build patched image' }))
    await waitFor(() => expect(create).toHaveBeenCalledWith({ base_image: 'example/base:v1', image: 'sparkdeck/patched:v1', node_ids: ['node-3'], files: [{ target: '/app/patch.py', content: 'print("patched")\n' }] }))
  })

  it('keeps pasted code and form values after a rejected build so the user can correct and retry', async () => {
    setup()
    const create = vi.spyOn(api.images, 'createPatchBuild').mockRejectedValue(new Error('The image tag already exists on Node 3.'))
    const user = await fillForm()
    await user.type(screen.getByRole('textbox', { name: 'Python contents 1' }), 'VALUE = 42')
    await user.click(screen.getByRole('button', { name: 'Build patched image' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('image tag already exists')
    expect(screen.getByRole('textbox', { name: 'Python contents 1' })).toHaveValue('VALUE = 42')
    expect(screen.getByRole('textbox', { name: 'Patched image tag' })).toHaveValue('sparkdeck/patched:v1')
    create.mockResolvedValue(build)
    await user.click(screen.getByRole('button', { name: 'Build patched image' }))
    await waitFor(() => expect(screen.queryByRole('textbox', { name: 'Python contents 1' })).not.toBeInTheDocument())
    expect(create).toHaveBeenCalledTimes(2)
  })

  it('rejects non UTF-8 upload without replacing existing code', async () => {
    setup()
    const user = await fillForm()
    await user.type(screen.getByRole('textbox', { name: 'Python contents 1' }), 'KEEP = True')
    const file = new File(['bad'], 'bad.py')
    Object.defineProperty(file, 'arrayBuffer', { value: async () => new Uint8Array([0xff]).buffer })
    await user.upload(screen.getByLabelText('Upload Python file 1'), file)
    expect(await screen.findByRole('alert')).toHaveTextContent('UTF-8')
    expect(screen.getByRole('textbox', { name: 'Python contents 1' })).toHaveValue('KEEP = True')
  })

  it('rejects duplicate destinations and oversized pasted contents before posting', async () => {
    setup()
    const create = vi.spyOn(api.images, 'createPatchBuild')
    const user = await fillForm()
    fireEvent.change(screen.getByRole('textbox', { name: 'Python contents 1' }), { target: { value: 'a'.repeat(1024 * 1024 + 1) } })
    await user.click(screen.getByRole('button', { name: 'Build patched image' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('1 MiB')
    fireEvent.change(screen.getByRole('textbox', { name: 'Python contents 1' }), { target: { value: 'a = 1' } })
    await user.click(screen.getByRole('button', { name: 'Add patch file' }))
    await user.type(screen.getByRole('textbox', { name: 'Container destination 2' }), '/app/patch.py')
    await user.type(screen.getByRole('textbox', { name: 'Python contents 2' }), 'a = 2')
    await user.click(screen.getByRole('button', { name: 'Build patched image' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('unique absolute container path')
    expect(create).not.toHaveBeenCalled()
  })

  it('restores persisted builds, polls active nodes, and refreshes images when verification finishes', async () => {
    vi.useFakeTimers()
    const list = vi.spyOn(api.images, 'patchBuilds').mockResolvedValueOnce({ items: [build] }).mockResolvedValue({ items: [{ ...build, status: 'succeeded', nodes: [{ ...build.nodes[0], status: 'succeeded', logs: ['Verified patch SHA-256'], image_id: 'sha256:123' }] }] })
    const onBuilt = vi.fn()
    render(<PatchedImages images={[]} nodes={nodes} localLabel="This device" onBuilt={onBuilt} />)
    await act(async () => { await Promise.resolve() })
    expect(screen.getByText('Queued…')).toBeInTheDocument()
    await act(async () => { await vi.advanceTimersByTimeAsync(3000) })
    expect(screen.getByRole('status')).toHaveTextContent('hashes verified on all selected nodes')
    expect(screen.getByLabelText('Build logs for Node 3')).toHaveTextContent('Verified patch SHA-256')
    expect(onBuilt).toHaveBeenCalledOnce()
    await act(async () => { await vi.advanceTimersByTimeAsync(9000) })
    expect(list).toHaveBeenCalledTimes(2)
  })
})
