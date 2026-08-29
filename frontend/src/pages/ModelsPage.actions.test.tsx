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
      return new Response(JSON.stringify({}), { status: 200, headers: { 'Content-Type': 'application/json' } })
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

  function mockGgufCluster(revision = CACHED_SHA) {
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
        return new Response(JSON.stringify(ggufCatalog(revision)), { status: 200, headers: { 'Content-Type': 'application/json' } })
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
