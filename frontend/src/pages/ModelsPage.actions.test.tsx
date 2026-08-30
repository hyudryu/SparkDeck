import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ModelsPage } from './ModelsPage'

const fetchMock = vi.fn<typeof fetch>()

const nodes = [
  { id: 'local', name: 'Controller', local: true, online: true, docker_ready: true, selectable: true },
  { id: 'worker-1', name: 'Node 4', online: true, docker_ready: true, selectable: true },
  { id: 'worker-2', name: 'Node 3', online: true, docker_ready: true, selectable: true },
  { id: 'worker-3', name: 'Node 2', online: true, docker_ready: true, selectable: true },
  { id: 'worker-4', name: 'Node 1', online: false, docker_ready: true, selectable: true },
]

const runningDeployment = {
  id: 'dep-1', alias: 'Chat model', runtime: 'vllm', kind: 'managed',
  model: { repository: 'org/model' }, status: 'running',
  settings: { context_length: 8192, tensor_parallel_size: 1 },
  node_ids: ['worker-1'], selected_nodes: [{ id: 'worker-1', name: 'Node 4' }],
  deployment_mode: 'single', required_node_count: 1,
  desired_state: 'running',
}

const modelCache = {
  nodes: [
    { id: 'worker-1', name: 'Node 4', online: true, models: [{ model_id: 'org/model', size_bytes: 10, revisions: ['main'] }] },
    { id: 'worker-2', name: 'Node 3', online: true, models: [{ model_id: 'org/model', size_bytes: 10, revisions: ['main'] }] },
    { id: 'worker-3', name: 'Node 2', online: true, models: [] },
    { id: 'worker-4', name: 'Node 1', online: false, models: [{ model_id: 'org/model', size_bytes: 10, revisions: ['main'] }] },
  ],
}

beforeEach(() => {
  vi.stubGlobal('fetch', fetchMock)
  fetchMock.mockClear()
  fetchMock.mockImplementation(async (input) => {
    const path = String(input)
    if (path === '/api/v1/deployments') {
      return new Response(JSON.stringify({ items: [runningDeployment] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }
    if (path === '/api/v1/nodes') {
      return new Response(JSON.stringify({ items: nodes }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }
    if (path === '/api/v1/model-cache') {
      return new Response(JSON.stringify(modelCache), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }
    if (path === '/api/v1/recipes') {
      return new Response(JSON.stringify({ items: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }
    if (path === '/api/v1/onboarding') {
      return new Response(JSON.stringify({ role: 'controller', node: { id: 'local', name: 'Controller', port: 9000, access_urls: [] } }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }
    if (path === '/api/v1/settings') {
      return new Response(JSON.stringify({ vllm_image: 'registry.example/vllm:configured' }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }
    return new Response(JSON.stringify(runningDeployment), { status: 200, headers: { 'Content-Type': 'application/json' } })
  })
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

function renderPage() {
  return render(<MemoryRouter initialEntries={['/models']}><ModelsPage /></MemoryRouter>)
}

describe('models page vLLM deployment targets', () => {
  it('allows remote-only tensor parallelism and submits the selected vLLM image', async () => {
    const user = userEvent.setup()
    renderPage()

    await screen.findByText('running')
    await user.click(screen.getByRole('button', { name: 'Create deployment' }))

    const imageInput = await screen.findByLabelText(/vLLM image/)
    expect(imageInput).toHaveValue('registry.example/vllm:configured')
    await user.clear(imageInput)
    await user.type(imageInput, 'registry.example/vllm:pinned')
    await user.type(screen.getByLabelText('Display name'), 'Remote TP')
    await user.type(screen.getByLabelText('Model repository or GGUF artifact'), 'org/remote-model')

    const controller = screen.getByRole('checkbox', { name: /Controller/ })
    const node3 = screen.getByRole('checkbox', { name: /Node 3/ })
    const node4 = screen.getByRole('checkbox', { name: /Node 4/ })
    await user.click(node3)
    await user.click(node4)
    await user.selectOptions(screen.getByLabelText(/Deployment layout/), 'sharded')

    expect(controller).toBeEnabled()
    await user.click(controller)
    expect(controller).not.toBeChecked()
    expect(node3).toBeChecked()
    expect(node4).toBeChecked()
    expect(screen.getByText(/Primary: Node 3/)).toBeInTheDocument()
    expect(screen.getByLabelText(/Tensor parallel size/)).toHaveValue(2)

    await user.click(screen.getByRole('button', { name: 'Save deployment' }))

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([input, init]) => (
        String(input) === '/api/v1/deployments' && init?.method === 'POST'
      ))
      expect(call).toBeDefined()
      const payload = JSON.parse(String(call?.[1]?.body))
      expect(payload).toMatchObject({
        model: 'org/remote-model',
        alias: 'Remote TP',
        runtime: 'vllm',
        node_ids: ['worker-2', 'worker-1'],
        deployment_mode: 'sharded',
        settings: {
          image: 'registry.example/vllm:pinned',
          tensor_parallel_size: 2,
        },
      })
    })
  })
})

describe('models page llama.cpp pull targets', () => {
  it('lets llama.cpp bookmark creation pick several nodes without pinning the controller', async () => {
    const user = userEvent.setup()
    renderPage()

    await screen.findByText('running')
    await user.click(screen.getByRole('button', { name: 'Create deployment' }))

    const runtimeSelect = await screen.findByLabelText('Runtime')
    await user.selectOptions(runtimeSelect, 'llama.cpp')

    // Llama server replicas run on any prepared node, so the controller is
    // no longer locked in as the only selection.
    const controllerCheckbox = screen.getByRole('checkbox', { name: /Controller/ })
    expect(controllerCheckbox).toBeChecked()
    expect(controllerCheckbox).toBeEnabled()
    const remoteCheckbox = screen.getByRole('checkbox', { name: /Node 4/ })
    expect(remoteCheckbox).toBeEnabled()
    await user.click(remoteCheckbox)
    expect(remoteCheckbox).toBeChecked()
  })

  it('offers a Hub download seed when launching a saved llama deployment on several nodes', async () => {
    const user = userEvent.setup()
    const savedLlama = {
      id: 'dep-llama', alias: 'Local GGUF', runtime: 'llama.cpp', kind: 'managed',
      model: { repository: 'org/model', artifact: 'FP16/model-F16.gguf' },
      status: 'saved', settings: {}, node_ids: ['local', 'worker-1'],
      deployment_mode: 'replicated', required_node_count: 2,
    }
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (path === '/api/v1/deployments/dep-llama/prepare/preflight' && init?.method === 'POST') {
        return new Response(JSON.stringify({
          enabled: true, model_id: 'org/model', revision: 'main',
          source: null, sources: [], download: null, download_error: null,
          targets: [
            { node_id: 'local', node_name: 'Controller', eligible: false, has_required_weights: false, has_model_cache: false, download_eligible: true },
            { node_id: 'worker-1', node_name: 'Node 4', eligible: false, has_required_weights: false, has_model_cache: false, download_eligible: true },
          ],
          node_ids: ['local', 'worker-1'], eligible: true, action: 'download',
          download_node_id: 'local', download_node_ids: ['local', 'worker-1'],
          transfer_target_node_ids: [], reason: null,
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      const body = path === '/api/v1/deployments' ? { items: [savedLlama] }
        : path === '/api/v1/nodes' ? { items: nodes }
          : path === '/api/v1/model-cache' ? { nodes: [] }
            : path === '/api/v1/recipes' ? { items: [] } : {}
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })

    renderPage()
    await user.click(await screen.findByRole('button', { name: 'Launch' }))

    const dialog = await screen.findByRole('dialog', { name: 'Launch Local GGUF' })
    const seedSelect = await screen.findByLabelText(/Hub download seed/)
    expect(seedSelect).toBeEnabled()
    expect(within(dialog).getByRole('option', { name: 'Node 4' })).toBeInTheDocument()
    expect(within(dialog).getByRole('option', { name: 'Automatic' })).toBeInTheDocument()
  })
})

describe('models page running actions', () => {
  it('clones a persisted deployment and shows the generated copy name', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (path === '/api/v1/deployments/dep-1/clone' && init?.method === 'POST') {
        return new Response(JSON.stringify({
          ...runningDeployment,
          id: 'dep-copy',
          alias: '(Copy) Chat model',
          status: 'saved',
          desired_state: 'stopped',
        }), { status: 201, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/deployments') return new Response(JSON.stringify({ items: [runningDeployment] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path === '/api/v1/nodes') return new Response(JSON.stringify({ items: nodes }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path === '/api/v1/model-cache') return new Response(JSON.stringify(modelCache), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path === '/api/v1/recipes') return new Response(JSON.stringify({ items: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path === '/api/v1/onboarding') return new Response(JSON.stringify({ role: 'controller', node: { id: 'local', name: 'Controller', port: 9000, access_urls: [] } }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path === '/api/v1/settings') return new Response(JSON.stringify({}), { status: 200, headers: { 'Content-Type': 'application/json' } })
      return new Response(JSON.stringify(runningDeployment), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    renderPage()

    await user.click(await screen.findByRole('button', { name: 'Clone Chat model' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/deployments/dep-1/clone',
      expect.objectContaining({ method: 'POST' }),
    ))
    expect(await screen.findByText('(Copy) Chat model')).toBeInTheDocument()
    await waitFor(() => expect(fetchMock.mock.calls.filter(([input]) => (
      String(input) === '/api/v1/deployments'
    )).length).toBeGreaterThanOrEqual(2))
    // The post-clone list is deliberately stale in this fixture. The accepted
    // deployment guard keeps the optimistic clone visible until the API lists it.
    expect(screen.getByText('(Copy) Chat model')).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent(
      'Cloned Chat model as (Copy) Chat model.',
    )
  })

  it('refreshes an external clone from registered to its probed endpoint status', async () => {
    const user = userEvent.setup()
    const external = {
      id: 'external-1', alias: 'Hosted model', runtime: 'vllm', kind: 'external',
      model: { repository: 'org/model' }, status: 'running', settings: {},
      desired_state: 'running',
    }
    const registeredClone = {
      ...external,
      id: 'external-copy', alias: '(Copy) Hosted model', status: 'registered',
    }
    let cloned = false
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (path === '/api/v1/deployments/external-1/clone' && init?.method === 'POST') {
        cloned = true
        return new Response(JSON.stringify(registeredClone), { status: 201, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/deployments') {
        return new Response(JSON.stringify({
          items: cloned ? [external, { ...registeredClone, status: 'running' }] : [external],
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/nodes') return new Response(JSON.stringify({ items: nodes }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path === '/api/v1/model-cache') return new Response(JSON.stringify(modelCache), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path === '/api/v1/recipes') return new Response(JSON.stringify({ items: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path === '/api/v1/onboarding') return new Response(JSON.stringify({ role: 'controller', node: { id: 'local', name: 'Controller', port: 9000, access_urls: [] } }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path === '/api/v1/settings') return new Response(JSON.stringify({}), { status: 200, headers: { 'Content-Type': 'application/json' } })
      return new Response(JSON.stringify(external), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    renderPage()

    await user.click(await screen.findByRole('button', { name: 'Clone Hosted model' }))

    const cloneName = await screen.findByText('(Copy) Hosted model')
    const cloneRow = cloneName.closest('[role="row"]')
    expect(cloneRow).not.toBeNull()
    await waitFor(() => expect(within(cloneRow as HTMLElement).getByText('running')).toBeInTheDocument())
    expect(fetchMock.mock.calls.filter(([input]) => (
      String(input) === '/api/v1/deployments'
    )).length).toBeGreaterThanOrEqual(2)
  })

  it('shows the running nodes in a status tooltip', async () => {
    renderPage()

    const badge = await screen.findByText('running')
    fireEvent.mouseOver(badge)

    const tooltip = await screen.findByRole('tooltip')
    expect(tooltip).toHaveTextContent('Running on')
    expect(tooltip).toHaveTextContent('Node 4')
  })

  it('keeps the running replica while launching on an additional node', async () => {
    const user = userEvent.setup()
    renderPage()

    await screen.findByText('Chat model')
    await user.click(screen.getByRole('button', { name: 'More actions for Chat model' }))
    await user.click(screen.getByRole('menuitem', { name: 'Launch on additional nodes…' }))

    const dialog = await screen.findByRole('dialog', { name: 'Add nodes to Chat model' })
    // The node that is already running is locked into the selection.
    expect(within(dialog).getByRole('checkbox', { name: /Node 4/ })).toBeDisabled()
    // Nodes without cached weights stay unavailable.
    expect(within(dialog).getByRole('checkbox', { name: /Node 2/ })).toBeDisabled()
    // A cached-but-offline node reports its real status, not a bogus
    // "weights not cached" reason, and stays unselectable.
    expect(within(dialog).getByRole('checkbox', { name: /Node 1/ })).toBeDisabled()
    expect(within(dialog).getByText(/Offline/)).toBeInTheDocument()
    await user.click(within(dialog).getByRole('checkbox', { name: /Node 3/ }))
    await user.click(within(dialog).getByRole('button', { name: 'Launch on 1 node' }))

    await waitFor(() => {
      const start = fetchMock.mock.calls.find(([path, init]) => (
        String(path).endsWith('/deployments/dep-1/start') && init?.method === 'POST'
      ))
      expect(start).toBeDefined()
      expect(JSON.parse(String(start?.[1]?.body))).toEqual({ additional_node_ids: ['worker-2'] })
    })
  })

  it('stops the deployment from the split-button main action', async () => {
    const user = userEvent.setup()
    renderPage()

    await screen.findByText('Chat model')
    await user.click(screen.getByRole('button', { name: 'Stop' }))

    await waitFor(() => {
      const stop = fetchMock.mock.calls.find(([path, init]) => (
        String(path).endsWith('/deployments/dep-1/stop') && init?.method === 'POST'
      ))
      expect(stop).toBeDefined()
    })
  })

  it('shows controls and loading progress for a discovered external container', async () => {
    const user = userEvent.setup()
    const external = {
      id: 'container:kimi-vllm', alias: 'Kimi vLLM', runtime: 'vllm', kind: 'external',
      model: { repository: 'org/model' }, status: 'starting', settings: {},
      controllable: true, logs_available: true, removable: true,
      launch_phase: 'loading', launch_message: 'loading checkpoint shards 7/48',
    }
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path === '/api/v1/deployments') return new Response(JSON.stringify({ items: [external] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path === '/api/v1/nodes') return new Response(JSON.stringify({ items: nodes }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path === '/api/v1/model-cache') return new Response(JSON.stringify(modelCache), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path === '/api/v1/recipes') return new Response(JSON.stringify({ items: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path === '/api/v1/onboarding') return new Response(JSON.stringify({ role: 'controller', node: { id: 'local', name: 'Controller', port: 9000, access_urls: [] } }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      if (path === '/api/v1/settings') return new Response(JSON.stringify({}), { status: 200, headers: { 'Content-Type': 'application/json' } })
      return new Response(JSON.stringify(external), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    renderPage()

    expect(await screen.findByText('loading checkpoint shards 7/48')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Logs for Kimi vLLM' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Clone Kimi vLLM' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Stop' }))

    await waitFor(() => expect(fetchMock.mock.calls.some(([path, init]) => (
      String(path).endsWith('/deployments/container%3Akimi-vllm/stop') && init?.method === 'POST'
    ))).toBe(true))
  })

  it('falls back to a plain start button when stopped', async () => {
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path === '/api/v1/deployments') {
        return new Response(JSON.stringify({
          items: [{ ...runningDeployment, status: 'stopped', desired_state: 'stopped' }],
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/nodes') {
        return new Response(JSON.stringify({ items: nodes }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/model-cache') {
        return new Response(JSON.stringify(modelCache), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/recipes') {
        return new Response(JSON.stringify({ items: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/onboarding') {
        return new Response(JSON.stringify({ role: 'controller', node: { id: 'local', name: 'Controller', port: 9000, access_urls: [] } }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/settings') {
        return new Response(JSON.stringify({}), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify(runningDeployment), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    renderPage()

    await screen.findByText('Chat model')
    expect(screen.queryByRole('button', { name: 'More actions for Chat model' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Start' })).toBeInTheDocument()
  })

  it('shows the controller tooltip and no add-nodes action for standalone deployments', async () => {
    const standalone = {
      id: 'dep-2', alias: 'Local runner', runtime: 'vllm', kind: 'managed',
      model: { repository: 'org/model' }, status: 'running',
      settings: {}, desired_state: 'running',
    }
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path === '/api/v1/deployments') {
        return new Response(JSON.stringify({ items: [standalone] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/nodes') {
        return new Response(JSON.stringify({ items: nodes }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/model-cache') {
        return new Response(JSON.stringify(modelCache), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/recipes') {
        return new Response(JSON.stringify({ items: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/onboarding') {
        return new Response(JSON.stringify({ role: 'controller', node: { id: 'local', name: 'Controller', port: 9000, access_urls: [] } }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/settings') {
        return new Response(JSON.stringify({}), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify(standalone), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    renderPage()

    const badge = await screen.findByText('running')
    fireEvent.mouseOver(badge)

    const tooltip = await screen.findByRole('tooltip')
    expect(tooltip).toHaveTextContent('Running on')
    expect(tooltip).toHaveTextContent('This device')
    // No cluster to grow: the standalone card keeps a plain Stop button.
    expect(screen.queryByRole('button', { name: 'More actions for Local runner' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Stop' })).toBeInTheDocument()
  })
})

describe('deployment creator model and quantization pickers', () => {
  const CACHED_SHA = 'a'.repeat(40)
  const NEW_SHA = 'b'.repeat(40)
  const ggufCache = {
    nodes: [
      {
        id: 'local', name: 'Controller', online: true,
        models: [{
          model_id: 'org/gguf', size_bytes: 10,
          revision_refs: { main: CACHED_SHA },
          snapshot_files: { [CACHED_SHA]: ['Llama-3.2-1B-Q4_K_M.gguf'] },
        }],
      },
    ],
  }
  const ggufCatalog = (revision: string) => ({
    model: {
      id: 'org/gguf',
      revision,
      quantizations: [
        {
          name: 'Q4_K_M',
          files: [{ filename: 'Llama-3.2-1B-Q4_K_M.gguf', size_bytes: 807 }],
          weight_size_bytes: 807,
        },
        {
          name: 'Q8_0',
          files: [{ filename: 'Llama-3.2-1B-Q8_0.gguf', size_bytes: 1200 }],
          weight_size_bytes: 1200,
        },
      ],
    },
    aggregates: [],
  })

  function mockGgufCluster(revision = CACHED_SHA, catalogOverrides: Record<string, unknown> = {}) {
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path === '/api/v1/deployments') {
        return new Response(JSON.stringify({ items: [runningDeployment] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/nodes') {
        return new Response(JSON.stringify({ items: nodes }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/model-cache') {
        return new Response(JSON.stringify(ggufCache), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/catalog/models/org%2Fgguf') {
        return new Response(JSON.stringify({ ...ggufCatalog(revision), ...catalogOverrides }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/catalog/models/org%2Fother') {
        return new Response(JSON.stringify({
          model: {
            id: 'org/other',
            revision: 'c'.repeat(40),
            quantizations: [{
              name: 'R1',
              files: [{ filename: 'other-R1.gguf', size_bytes: 900 }],
              weight_size_bytes: 900,
            }],
          },
          aggregates: [],
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/recipes') {
        return new Response(JSON.stringify({ items: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/onboarding') {
        return new Response(JSON.stringify({ role: 'controller', node: { id: 'local', name: 'Controller', port: 9000, access_urls: [] } }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/settings') {
        return new Response(JSON.stringify({}), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify(runningDeployment), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
  }

  it('keeps the picked cached model visible in the picker', async () => {
    const user = userEvent.setup()
    renderPage()

    await screen.findByText('running')
    await user.click(screen.getByRole('button', { name: 'Create deployment' }))

    const picker = await screen.findByLabelText('Or pick a model already on the cluster')
    await user.selectOptions(picker, 'org/model')

    // The picker reflects the selection instead of snapping back to the
    // placeholder, and the repository field receives the model id.
    expect(picker).toHaveValue('org/model')
    expect(screen.getByLabelText('Model repository or GGUF artifact')).toHaveValue('org/model')
  })

  it('lists repository quantizations and GGUF artifacts with downloaded marks', async () => {
    const user = userEvent.setup()
    mockGgufCluster()
    renderPage()

    await screen.findByText('running')
    await user.click(screen.getByRole('button', { name: 'Create deployment' }))

    await user.selectOptions(await screen.findByLabelText('Runtime'), 'llama.cpp')
    await user.type(screen.getByLabelText('Model repository or GGUF artifact'), 'org/gguf')

    // The field help text inside the label extends the accessible name, so
    // match by a name prefix.
    const quantSelect = await screen.findByRole('combobox', { name: /Quantization/ })
    expect(within(quantSelect).getByRole('option', { name: 'Q4_K_M · 807 B · ✓ Downloaded' })).toBeInTheDocument()
    expect(within(quantSelect).getByRole('option', { name: 'Q8_0 · 1.2 KB' })).toBeInTheDocument()

    const artifactSelect = screen.getByRole('combobox', { name: /GGUF artifact/ })
    expect(within(artifactSelect).getByRole('option', { name: 'Llama-3.2-1B-Q4_K_M.gguf · 807 B · ✓ Downloaded' })).toBeInTheDocument()

    // Picking a quantization selects its artifact, and picking an artifact
    // carries its quantization back.
    await user.selectOptions(quantSelect, 'Q4_K_M')
    expect(artifactSelect).toHaveValue('Llama-3.2-1B-Q4_K_M.gguf')
    await user.selectOptions(artifactSelect, 'Llama-3.2-1B-Q8_0.gguf')
    expect(quantSelect).toHaveValue('Q8_0')

    // Clearing the quantization clears the linked artifact, so an
    // incompatible combination cannot be saved.
    await user.selectOptions(quantSelect, '')
    expect(quantSelect).toHaveValue('')
    expect(artifactSelect).toHaveValue('')
  })

  it('adopts the canonical repository id after a Hub rename redirect', async () => {
    const user = userEvent.setup()
    const redirected = ggufCatalog(CACHED_SHA)
    mockGgufCluster(CACHED_SHA, { model: { ...redirected.model, id: 'org/gguf-renamed' } })
    renderPage()

    await screen.findByText('running')
    await user.click(screen.getByRole('button', { name: 'Create deployment' }))

    await user.selectOptions(await screen.findByLabelText('Runtime'), 'llama.cpp')
    await user.type(screen.getByLabelText('Model repository or GGUF artifact'), 'org/gguf')

    // The listing resolved for the typed query is trusted even though the
    // Hub answered with the canonical id, and the form adopts it.
    const quantSelect = await screen.findByRole('combobox', { name: /Quantization/ })
    await waitFor(() => {
      expect(screen.getByLabelText('Model repository or GGUF artifact')).toHaveValue('org/gguf-renamed')
    })
    // The dropdowns survive the adoption; the cached files are keyed by the
    // old repository id, so the downloaded mark honestly disappears.
    expect(within(quantSelect).getByRole('option', { name: 'Q4_K_M · 807 B' })).toBeInTheDocument()
    expect(within(quantSelect).queryByRole('option', { name: /✓ Downloaded/ })).not.toBeInTheDocument()
  })

  it('clears repository-derived selections when the model id changes', async () => {
    const user = userEvent.setup()
    mockGgufCluster()
    renderPage()

    await screen.findByText('running')
    await user.click(screen.getByRole('button', { name: 'Create deployment' }))

    await user.selectOptions(await screen.findByLabelText('Runtime'), 'llama.cpp')
    const modelInput = screen.getByLabelText('Model repository or GGUF artifact')
    await user.type(modelInput, 'org/gguf')

    const quantSelect = await screen.findByRole('combobox', { name: /Quantization/ })
    await user.selectOptions(quantSelect, 'Q8_0')

    const artifactSelect = screen.getByRole('combobox', { name: /GGUF artifact/ })
    expect(artifactSelect).toHaveValue('Llama-3.2-1B-Q8_0.gguf')

    await user.clear(screen.getByLabelText('Model repository or GGUF artifact'))
    await user.type(screen.getByLabelText('Model repository or GGUF artifact'), 'org/other')

    // The other repository's listing does not carry the previously picked
    // artifact or quantization, so both reset instead of saving a broken
    // combination.
    const dialog = screen.getByRole('dialog')
    await within(dialog).findByRole('option', { name: 'R1 · 900 B' })
    // The fields swap between text inputs and dropdowns while the listing
    // loads, so re-query them instead of reusing the old elements.
    expect(within(dialog).getByRole('combobox', { name: /Quantization/ })).toHaveValue('')
    expect(within(dialog).getByRole('combobox', { name: /GGUF artifact/ })).toHaveValue('')
  })

  it('keeps a failed repository lookup without resurrecting the previous one', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path === '/api/v1/deployments') {
        return new Response(JSON.stringify({ items: [runningDeployment] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/nodes') {
        return new Response(JSON.stringify({ items: nodes }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/model-cache') {
        return new Response(JSON.stringify(ggufCache), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/catalog/models/org%2Fgguf') {
        return new Response(JSON.stringify(ggufCatalog(CACHED_SHA)), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/catalog/models/org%2Fbroken') {
        return new Response(JSON.stringify({ detail: 'boom' }), { status: 500, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/recipes') {
        return new Response(JSON.stringify({ items: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/onboarding') {
        return new Response(JSON.stringify({ role: 'controller', node: { id: 'local', name: 'Controller', port: 9000, access_urls: [] } }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/settings') {
        return new Response(JSON.stringify({}), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify(runningDeployment), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    renderPage()

    await screen.findByText('running')
    await user.click(screen.getByRole('button', { name: 'Create deployment' }))

    await user.selectOptions(await screen.findByLabelText('Runtime'), 'llama.cpp')
    await user.type(screen.getByLabelText('Model repository or GGUF artifact'), 'org/gguf')
    await screen.findByRole('combobox', { name: /Quantization/ })

    // The failed lookup for the new repository must keep the new id in the
    // field, must not rewrite it back to the previous repository via the
    // redirect adoption, and must not keep showing the old dropdowns.
    await user.clear(screen.getByLabelText('Model repository or GGUF artifact'))
    await user.type(screen.getByLabelText('Model repository or GGUF artifact'), 'org/broken')
    await waitFor(() => {
      expect(screen.getByLabelText('Model repository or GGUF artifact')).toHaveValue('org/broken')
    })
    await new Promise((resolve) => setTimeout(resolve, 900))
    expect(screen.getByLabelText('Model repository or GGUF artifact')).toHaveValue('org/broken')
    expect(screen.queryByRole('combobox', { name: /Quantization/ })).not.toBeInTheDocument()
    expect(screen.getByLabelText('Quantization (optional)')).toHaveValue('')
  })

  it('clears linked artifact picks when a repository lists no GGUF files', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path === '/api/v1/deployments') {
        return new Response(JSON.stringify({ items: [runningDeployment] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/nodes') {
        return new Response(JSON.stringify({ items: nodes }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/model-cache') {
        return new Response(JSON.stringify(ggufCache), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/catalog/models/org%2Fgguf') {
        return new Response(JSON.stringify(ggufCatalog(CACHED_SHA)), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/catalog/models/org%2Fplain') {
        return new Response(JSON.stringify({
          model: { id: 'org/plain', revision: 'c'.repeat(40), quantizations: [] },
          aggregates: [],
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/recipes') {
        return new Response(JSON.stringify({ items: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/onboarding') {
        return new Response(JSON.stringify({ role: 'controller', node: { id: 'local', name: 'Controller', port: 9000, access_urls: [] } }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/settings') {
        return new Response(JSON.stringify({}), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify(runningDeployment), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    renderPage()

    await screen.findByText('running')
    await user.click(screen.getByRole('button', { name: 'Create deployment' }))

    await user.selectOptions(await screen.findByLabelText('Runtime'), 'llama.cpp')
    await user.type(screen.getByLabelText('Model repository or GGUF artifact'), 'org/gguf')
    const quantSelect = await screen.findByRole('combobox', { name: /Quantization/ })
    await user.selectOptions(quantSelect, 'Q8_0')

    // The new repository lists no GGUF artifacts at all, so the artifact
    // picked from the previous repository is cleared instead of being
    // exposed through the manual fallback input.
    await user.clear(screen.getByLabelText('Model repository or GGUF artifact'))
    await user.type(screen.getByLabelText('Model repository or GGUF artifact'), 'org/plain')
    // The help text inside the label extends the accessible name, so match
    // by a name prefix (anchored to skip the repository field's label).
    await waitFor(() => {
      expect(screen.getByLabelText(/^GGUF artifact/)).toHaveValue('')
    })
    expect(screen.getByLabelText('Quantization (optional)')).toHaveValue('')
  })

  it('marks a sharded quantization downloaded only when one node holds every shard', async () => {
    const user = userEvent.setup()
    const shardCache = {
      nodes: [
        {
          id: 'local', name: 'Controller', online: true,
          models: [{
            model_id: 'org/gguf', size_bytes: 10,
            revision_refs: { main: CACHED_SHA },
            snapshot_files: { [CACHED_SHA]: ['shard-00001-of-00002.gguf'] },
          }],
        },
        {
          id: 'worker-1', name: 'Node 4', online: true,
          models: [{
            model_id: 'org/gguf', size_bytes: 10,
            revision_refs: { main: CACHED_SHA },
            snapshot_files: { [CACHED_SHA]: ['shard-00002-of-00002.gguf'] },
          }],
        },
      ],
    }
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path === '/api/v1/deployments') {
        return new Response(JSON.stringify({ items: [runningDeployment] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/nodes') {
        return new Response(JSON.stringify({ items: nodes }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/model-cache') {
        return new Response(JSON.stringify(shardCache), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/catalog/models/org%2Fgguf') {
        return new Response(JSON.stringify({
          model: {
            id: 'org/gguf',
            revision: CACHED_SHA,
            quantizations: [{
              name: 'Q4_K_M',
              files: [
                { filename: 'shard-00001-of-00002.gguf', size_bytes: 400 },
                { filename: 'shard-00002-of-00002.gguf', size_bytes: 407 },
              ],
              weight_size_bytes: 807,
            }],
          },
          aggregates: [],
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/recipes') {
        return new Response(JSON.stringify({ items: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/onboarding') {
        return new Response(JSON.stringify({ role: 'controller', node: { id: 'local', name: 'Controller', port: 9000, access_urls: [] } }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/settings') {
        return new Response(JSON.stringify({}), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify(runningDeployment), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    renderPage()

    await screen.findByText('running')
    await user.click(screen.getByRole('button', { name: 'Create deployment' }))

    await user.selectOptions(await screen.findByLabelText('Runtime'), 'llama.cpp')
    await user.type(screen.getByLabelText('Model repository or GGUF artifact'), 'org/gguf')

    // Shards split across two nodes do not add up to one usable copy, so
    // the quantization must not claim a downloaded state.
    const quantSelect = await screen.findByRole('combobox', { name: /Quantization/ })
    expect(within(quantSelect).getByRole('option', { name: 'Q4_K_M · 807 B' })).toBeInTheDocument()
    expect(within(quantSelect).queryByRole('option', { name: /✓ Downloaded/ })).not.toBeInTheDocument()
  })

  it('hides externally managed bundles from the cached-model picker', async () => {
    const user = userEvent.setup()
    const cacheWithBundle = {
      nodes: [
        {
          id: 'local', name: 'Controller', online: true,
          models: [
            { model_id: 'org/model', size_bytes: 10 },
            { model_id: 'comfy/bundle', size_bytes: 999, externally_managed: true, transferable: false },
            { model_id: 'org/both', size_bytes: 999, externally_managed: true, transferable: false },
          ],
        },
        {
          id: 'worker-1', name: 'Node 4', online: true,
          models: [
            { model_id: 'org/both', size_bytes: 10 },
          ],
        },
      ],
    }
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path === '/api/v1/deployments') {
        return new Response(JSON.stringify({ items: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/nodes') {
        return new Response(JSON.stringify({ items: nodes }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/model-cache') {
        return new Response(JSON.stringify(cacheWithBundle), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/recipes') {
        return new Response(JSON.stringify({ items: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/onboarding') {
        return new Response(JSON.stringify({ role: 'controller', node: { id: 'local', name: 'Controller', port: 9000, access_urls: [] } }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/settings') {
        return new Response(JSON.stringify({}), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify(runningDeployment), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    renderPage()

    await screen.findByText('Create deployment')
    await user.click(screen.getByRole('button', { name: 'Create deployment' }))

    const picker = await screen.findByLabelText('Or pick a model already on the cluster')
    // Normal cache entries are offered (including one that also exists as
    // an external bundle), while external-only bundles are not.
    expect(within(picker).getByRole('option', { name: /org\/model/ })).toBeInTheDocument()
    expect(within(picker).getByRole('option', { name: /org\/both/ })).toBeInTheDocument()
    expect(within(picker).queryByRole('option', { name: /comfy\/bundle/ })).not.toBeInTheDocument()
  })

  it('keeps saved artifact settings when opening the editor after a creator pick', async () => {
    const user = userEvent.setup()
    const savedLlama = {
      id: 'dep-llama', alias: 'Saved GGUF', runtime: 'llama.cpp', kind: 'managed',
      model: { repository: 'org/gguf', artifact: 'Llama-3.2-1B-Q4_K_M.gguf', quantization: 'Q4_K_M' },
      settings: {},
      status: 'saved', node_ids: ['local'],
      deployment_mode: 'single', required_node_count: 1,
    }
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path === '/api/v1/deployments') {
        return new Response(JSON.stringify({ items: [savedLlama] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/nodes') {
        return new Response(JSON.stringify({ items: nodes }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/model-cache') {
        return new Response(JSON.stringify(ggufCache), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/catalog/models/org%2Fgguf') {
        return new Response(JSON.stringify(ggufCatalog(CACHED_SHA)), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/recipes') {
        return new Response(JSON.stringify({ items: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/onboarding') {
        return new Response(JSON.stringify({ role: 'controller', node: { id: 'local', name: 'Controller', port: 9000, access_urls: [] } }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/settings') {
        return new Response(JSON.stringify({}), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify(savedLlama), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    renderPage()

    await screen.findByText('Saved GGUF')

    // Prime the provenance refs with the exact names the saved deployment
    // also uses, then load the editor: the saved settings must survive.
    await user.click(screen.getByRole('button', { name: 'Create deployment' }))
    await user.selectOptions(await screen.findByLabelText('Runtime'), 'llama.cpp')
    await user.type(screen.getByLabelText('Model repository or GGUF artifact'), 'org/gguf')
    const quantSelect = await screen.findByRole('combobox', { name: /Quantization/ })
    await user.selectOptions(quantSelect, 'Q4_K_M')
    await user.click(screen.getByRole('button', { name: 'Close dialog' }))

    await user.click(screen.getByRole('button', { name: 'Edit Saved GGUF' }))
    const dialog = await screen.findByRole('dialog', { name: 'Edit Saved GGUF' })
    // The saved quantization and artifact survive the listing load.
    const dialogQuant = await within(dialog).findByRole('combobox', { name: /Quantization/ })
    expect(dialogQuant).toHaveValue('Q4_K_M')
    const dialogArtifact = await within(dialog).findByRole('combobox', { name: /^GGUF artifact/ })
    expect(dialogArtifact).toHaveValue('Llama-3.2-1B-Q4_K_M.gguf')
  })

  it('starts local-path entry from an empty field after a repository pick', async () => {
    const user = userEvent.setup()
    mockGgufCluster()
    renderPage()

    await screen.findByText('running')
    await user.click(screen.getByRole('button', { name: 'Create deployment' }))

    await user.selectOptions(await screen.findByLabelText('Runtime'), 'llama.cpp')
    await user.type(screen.getByLabelText('Model repository or GGUF artifact'), 'org/gguf')
    const quantSelect = await screen.findByRole('combobox', { name: /Quantization/ })
    await user.selectOptions(quantSelect, 'Q8_0')
    const artifactSelect = screen.getByRole('combobox', { name: /^GGUF artifact/ })
    expect(artifactSelect).toHaveValue('Llama-3.2-1B-Q8_0.gguf')

    await user.selectOptions(artifactSelect, '__enter-local-path__')

    // The revealed text field starts empty: the repository artifact must
    // not stay prefilled and silently keep being used.
    const artifactInput = screen.getByLabelText(/^GGUF artifact/)
    expect(artifactInput).toHaveValue('')
    await user.type(artifactInput, '/tmp/local-model.gguf')
    expect(artifactInput).toHaveValue('/tmp/local-model.gguf')

    // Switching back returns to the repository dropdown.
    await user.click(screen.getByRole('button', { name: 'Choose from the repository files…' }))
    expect(screen.getByRole('combobox', { name: /^GGUF artifact/ })).toBeInTheDocument()
  })

  it('keeps deep-link artifact picks across a canonical repository redirect', async () => {
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      if (path === '/api/v1/deployments') {
        return new Response(JSON.stringify({ items: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/nodes') {
        return new Response(JSON.stringify({ items: nodes }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/model-cache') {
        return new Response(JSON.stringify(ggufCache), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/catalog/models/org%2Fold-gguf') {
        return new Response(JSON.stringify({
          model: { ...ggufCatalog(CACHED_SHA).model, id: 'org/new-gguf' },
          aggregates: [],
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/recipes') {
        return new Response(JSON.stringify({ items: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/onboarding') {
        return new Response(JSON.stringify({ role: 'controller', node: { id: 'local', name: 'Controller', port: 9000, access_urls: [] } }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/settings') {
        return new Response(JSON.stringify({}), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify(runningDeployment), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    render(
      <MemoryRouter initialEntries={['/models?model=org/old-gguf&runtime=llama.cpp&quantization=Q4_K_M&artifact=Llama-3.2-1B-Q4_K_M.gguf']}>
        <ModelsPage />
      </MemoryRouter>,
    )

    const dialog = await screen.findByRole('dialog', { name: 'Create deployment' })
    const modelInput = await within(dialog).findByLabelText('Model repository or GGUF artifact')

    // The Hub answers the old id with its canonical name; the form adopts
    // it, but the prepopulated artifact and quantization must survive the
    // rename because the listing still describes the same files.
    await waitFor(() => {
      expect(modelInput).toHaveValue('org/new-gguf')
    })
    const artifactSelect = await within(dialog).findByRole('combobox', { name: /^GGUF artifact/ })
    expect(artifactSelect).toHaveValue('Llama-3.2-1B-Q4_K_M.gguf')
    expect(within(dialog).getByRole('combobox', { name: /Quantization/ })).toHaveValue('Q4_K_M')
  })

  it('withholds downloaded marks when the listing resolves a newer revision', async () => {
    const user = userEvent.setup()
    mockGgufCluster(NEW_SHA)
    renderPage()

    await screen.findByText('running')
    await user.click(screen.getByRole('button', { name: 'Create deployment' }))

    await user.selectOptions(await screen.findByLabelText('Runtime'), 'llama.cpp')
    await user.type(screen.getByLabelText('Model repository or GGUF artifact'), 'org/gguf')

    // The cache only holds the older revision's files, so the current
    // listing's quantizations must not claim a downloaded state.
    const quantSelect = await screen.findByRole('combobox', { name: /Quantization/ })
    expect(within(quantSelect).getByRole('option', { name: 'Q4_K_M · 807 B' })).toBeInTheDocument()
    expect(within(quantSelect).queryByRole('option', { name: /✓ Downloaded/ })).not.toBeInTheDocument()
  })
})
