import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DeploymentPage } from './DeploymentPage'

const fetchMock = vi.fn<typeof fetch>()
const nodes = ['local', 'worker-1', 'worker-2', 'worker-3'].map((id, index) => ({
  id, name: id === 'local' ? 'Controller' : `Node ${index + 1}`,
  online: true, docker_ready: true, selectable: true,
}))

const detail = {
  id: 'dep-1', alias: 'Reasoning server', runtime: 'vllm', kind: 'managed',
  model: { repository: 'org/model', quantization: 'awq' }, model_revision: 'rev-123', status: 'stopped',
  settings: { quantization: 'awq', dtype: 'bfloat16' },
  node_ids: ['local'], deployment_mode: 'sharded', desired_state: 'stopped',
  editable: true, edit_reason: null,
  extra_args: [
    '--enable-prefix-caching', '--speculative-config',
    '{"method":"ngram","foo":true}', 'C:\\models\\foo', '', "owner's-model",
  ],
  launch_controls: {
    context_window: 8192, max_concurrency: 4, tensor_parallel_size: 2,
    pipeline_parallel_size: 1, thinking_mode: 'default',
    speculative_method: 'ngram', draft_sample_method: 'greedy',
  },
  gpu_memory_utilization: 0.9, gpu_memory_gb: null, image: null,
}

beforeEach(() => {
  vi.stubGlobal('fetch', fetchMock)
  fetchMock.mockClear()
  fetchMock.mockImplementation(async (input, init) => {
    const path = String(input)
    if (path === '/api/v1/nodes') {
      return new Response(JSON.stringify({ items: nodes }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }
    if (path === '/api/v1/runtime-flags/preview' && init?.method === 'POST') {
      const body = JSON.parse(String(init.body))
      return new Response(JSON.stringify({
        flags: body.extra_args,
        command_flags: body.extra_args.map(quoteArgForTest).join(' '),
        environment: body.environment,
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }
    if (path === '/api/v1/deployments/dep-1/settings' && init?.method === 'PUT') {
      return new Response(JSON.stringify(detail), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }
    if (path === '/api/v1/deployments/dep-1/start' && init?.method === 'POST') {
      return new Response(JSON.stringify(detail), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }
    return new Response(JSON.stringify(detail), { status: 200, headers: { 'Content-Type': 'application/json' } })
  })
})

const quoteArgForTest = (arg: string) => (/[^A-Za-z0-9_./:=+-]/.test(arg) ? `'${arg}'` : arg)

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

function renderPage() {
  return render(<MemoryRouter initialEntries={['/models/dep-1']}><Routes>
    <Route path="/models/:deploymentId" element={<DeploymentPage />} />
    <Route path="/models" element={<h1>Models destination</h1>} />
  </Routes></MemoryRouter>)
}

describe('deployment object page', () => {
  it('polls the detail while a stop is in flight and updates once stopped', async () => {
    let detailRequests = 0
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (path === '/api/v1/nodes') {
        return new Response(JSON.stringify({ items: nodes }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/deployments/dep-1' && (!init?.method || init.method === 'GET')) {
        detailRequests += 1
        const body = detailRequests === 1
          ? { ...detail, status: 'stopping', desired_state: 'stopped', editable: false, controllable: true }
          : detail
        return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify(detail), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    renderPage()

    expect(await screen.findByRole('button', { name: 'Stopping…' })).toBeDisabled()

    await waitFor(() => expect(detailRequests).toBeGreaterThanOrEqual(2), { timeout: 5000 })
    expect(await screen.findByRole('button', { name: 'Run' })).toBeInTheDocument()
  })

  it('only enables Save while deployment settings differ from the saved values', async () => {
    const user = userEvent.setup()
    renderPage()

    const maxConcurrency = await screen.findByLabelText('Max concurrency')
    const save = screen.getByRole('button', { name: 'Save' })
    expect(save).toBeDisabled()

    await user.clear(maxConcurrency)
    await user.type(maxConcurrency, '6')
    expect(save).toBeEnabled()

    await user.click(save)
    await screen.findByText('Deployment settings saved. They will be applied on the next run.')
    expect(save).toBeDisabled()

    await user.clear(maxConcurrency)
    await user.type(maxConcurrency, '7')
    expect(save).toBeEnabled()
  })

  it('offers KV cache dtype choices and saves the selected value', async () => {
    const user = userEvent.setup()
    renderPage()

    const kvCacheDtype = await screen.findByRole('combobox', { name: 'KV cache dtype' })
    expect(kvCacheDtype).toHaveValue('')
    expect(within(kvCacheDtype).getByRole('option', { name: 'Auto / unset' })).toBeInTheDocument()
    expect(within(kvCacheDtype).getByRole('option', { name: 'fp8' })).toBeInTheDocument()
    expect(within(kvCacheDtype).getByRole('option', { name: 'fp8_e4m3' })).toBeInTheDocument()
    expect(within(kvCacheDtype).getByRole('option', { name: 'nvfp4_ds_mla' })).toBeInTheDocument()

    await user.selectOptions(kvCacheDtype, 'fp8_e4m3')
    await waitFor(() => expect(fetchMock.mock.calls.some(([input, init]) => (
      String(input) === '/api/v1/runtime-flags/preview'
      && JSON.parse(String(init?.body)).launch_controls.kv_cache_dtype === 'fp8_e4m3'
    ))).toBe(true))
    await user.click(screen.getByRole('button', { name: 'Save' }))

    const saveCall = await waitFor(() => {
      const call = fetchMock.mock.calls.find(([input, init]) => (
        String(input) === '/api/v1/deployments/dep-1/settings' && init?.method === 'PUT'
      ))
      expect(call).toBeDefined()
      return call
    })
    expect(JSON.parse(String(saveCall?.[1]?.body)).launch_controls.kv_cache_dtype).toBe('fp8_e4m3')
  })

  it('preserves a custom KV cache dtype as a selectable current value', async () => {
    fetchMock.mockImplementation(async (input) => {
      const body = String(input) === '/api/v1/nodes'
        ? { items: nodes }
        : { ...detail, launch_controls: { ...detail.launch_controls, kv_cache_dtype: 'future_dtype' } }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    renderPage()

    const kvCacheDtype = await screen.findByRole('combobox', { name: 'KV cache dtype' })
    expect(kvCacheDtype).toHaveValue('future_dtype')
    expect(within(kvCacheDtype).getByRole('option', { name: 'future_dtype (current)' })).toBeInTheDocument()
  })

  it('uses SGLang KV choices and hides the unsupported llama.cpp control', async () => {
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      const body = path === '/api/v1/nodes'
        ? { items: nodes }
        : { ...detail, runtime: 'sglang', launch_controls: { ...detail.launch_controls, kv_cache_dtype: 'bf16' } }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    const rendered = renderPage()

    const sglangKv = await screen.findByRole('combobox', { name: 'KV cache dtype' })
    expect(sglangKv).toHaveValue('bf16')
    expect(within(sglangKv).getByRole('option', { name: 'fp4_mx_block16' })).toBeInTheDocument()
    expect(within(sglangKv).queryByRole('option', { name: 'nvfp4_ds_mla' })).not.toBeInTheDocument()

    rendered.unmount()
    fetchMock.mockImplementation(async (input) => {
      const path = String(input)
      const body = path === '/api/v1/nodes'
        ? { items: nodes }
        : { ...detail, runtime: 'llama.cpp' }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    renderPage()
    await screen.findByRole('heading', { name: 'Reasoning server' })
    expect(screen.queryByLabelText('KV cache dtype')).not.toBeInTheDocument()
  })

  it('shows saved flags and saves before running, then returns to Models', async () => {
    const user = userEvent.setup()
    const { container } = renderPage()

    expect(await screen.findByRole('heading', { name: 'Reasoning server' })).toBeInTheDocument()
    expect(container.querySelector('.page')).toContainElement(screen.getByRole('heading', { name: 'Reasoning server' }))
    expect(screen.getByLabelText('Tensor parallel size')).toHaveValue(2)
    expect(screen.getByLabelText('Pipeline parallel size')).toHaveValue(1)
    expect(screen.getByLabelText('Speculative method')).toHaveValue('ngram')
    expect(screen.getByLabelText('Draft sample method')).toHaveValue('greedy')
    expect(screen.getByLabelText(/Runtime flags/)).toHaveValue(
      `--enable-prefix-caching --speculative-config '{"method":"ngram","foo":true}' 'C:\\models\\foo' '' 'owner'\\''s-model'`,
    )
    const finalFlags = await screen.findByLabelText(/Final runtime flags/)
    await waitFor(() => expect((finalFlags as HTMLTextAreaElement).value).toContain('--speculative-config'))
    const previewBody = JSON.parse(String(fetchMock.mock.calls.find(
      ([input]) => String(input) === '/api/v1/runtime-flags/preview',
    )?.[1]?.body))
    expect(previewBody.managed).toBe(true)
    expect(previewBody.model_revision).toBe('rev-123')
    expect(previewBody.quantization).toBe('awq')
    expect(previewBody.dtype).toBe('bfloat16')
    expect(previewBody).not.toHaveProperty('gpu_memory_gb')
    await user.clear(screen.getByLabelText('Max concurrency'))
    await user.type(screen.getByLabelText('Max concurrency'), '6')
    await user.clear(screen.getByLabelText('Tensor parallel size'))
    await user.type(screen.getByLabelText('Tensor parallel size'), '4')
    await user.selectOptions(screen.getByLabelText('Speculative method'), 'dspark')
    await user.selectOptions(screen.getByLabelText('Draft sample method'), 'probabilistic')
    await user.click(screen.getByRole('button', { name: 'Run' }))
    expect(await screen.findByRole('dialog', { name: 'Start Reasoning server' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Start on 4 nodes' }))

    expect(await screen.findByRole('heading', { name: 'Models destination' })).toBeInTheDocument()
    const mutationCalls = fetchMock.mock.calls.filter(([input, init]) => (
      (init?.method === 'PUT' || init?.method === 'POST')
      && !String(input).includes('/runtime-flags/preview')
    ))
    expect(mutationCalls.map(([input]) => String(input))).toEqual([
      '/api/v1/deployments/dep-1/settings',
      '/api/v1/deployments/dep-1/start',
    ])
    const saved = JSON.parse(String(mutationCalls[0][1]?.body))
    expect(saved.launch_controls.max_concurrency).toBe(6)
    expect(saved.launch_controls.tensor_parallel_size).toBe(4)
    expect(saved.launch_controls.pipeline_parallel_size).toBe(1)
    expect(saved.launch_controls.speculative_method).toBe('dspark')
    expect(saved.launch_controls.draft_sample_method).toBe('probabilistic')
    expect(saved.extra_args).toEqual([
      '--enable-prefix-caching', '--speculative-config',
      '{"method":"ngram","foo":true}', 'C:\\models\\foo', '', "owner's-model",
    ])
    expect(JSON.parse(String(mutationCalls[1][1]?.body))).toEqual({
      node_ids: ['local', 'worker-1', 'worker-2', 'worker-3'],
    })
  })

  it('starts single-node tensor parallel deployments on one physical node', async () => {
    const user = userEvent.setup()
    const single = {
      ...detail,
      deployment_mode: 'single',
      launch_controls: { ...detail.launch_controls, tensor_parallel_size: 4 },
    }
    fetchMock.mockImplementation(async (input) => {
      if (String(input) === '/api/v1/nodes') {
        return new Response(JSON.stringify({ items: nodes }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify(single), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    renderPage()

    expect(await screen.findByLabelText('Tensor parallel size')).toHaveValue(4)
    await user.click(screen.getByRole('button', { name: 'Run' }))
    expect(await screen.findByRole('button', { name: 'Start on 1 node' })).toBeInTheDocument()
    expect(screen.getByText(/single-node layout runs TP4 on one physical node/)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Start on 1 node' }))

    const startCall = fetchMock.mock.calls.find(([input]) => String(input).endsWith('/start'))
    expect(JSON.parse(String(startCall?.[1]?.body))).toEqual({ node_ids: ['local'] })
  })

  it('uses the persisted node count for replicated deployments', async () => {
    const user = userEvent.setup()
    const replicated = {
      ...detail,
      deployment_mode: 'replicated',
      required_node_count: 3,
      node_ids: ['local', 'worker-1', 'worker-2'],
      launch_controls: { ...detail.launch_controls, tensor_parallel_size: 1 },
    }
    fetchMock.mockImplementation(async (input) => {
      if (String(input) === '/api/v1/nodes') {
        return new Response(JSON.stringify({ items: nodes }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify(replicated), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    renderPage()

    expect(await screen.findByLabelText('Tensor parallel size')).toHaveValue(1)
    await user.click(screen.getByRole('button', { name: 'Run' }))
    const start = await screen.findByRole('button', { name: 'Start on 3 nodes' })
    expect(screen.getByText(/replicated layout runs on exactly 3 nodes/)).toBeInTheDocument()
    await user.click(start)

    const startCall = fetchMock.mock.calls.find(([input]) => String(input).endsWith('/start'))
    expect(JSON.parse(String(startCall?.[1]?.body))).toEqual({
      node_ids: ['local', 'worker-1', 'worker-2'],
    })
  })

  it('preserves a valid saved multi-rank-per-node sharded topology', async () => {
    const user = userEvent.setup()
    const sharded = {
      ...detail,
      node_ids: ['local', 'worker-1'],
      required_node_count: 2,
      launch_controls: {
        ...detail.launch_controls,
        tensor_parallel_size: 4,
        pipeline_parallel_size: 1,
      },
    }
    fetchMock.mockImplementation(async (input) => {
      if (String(input) === '/api/v1/nodes') {
        return new Response(JSON.stringify({ items: nodes }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify(sharded), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    renderPage()

    expect(await screen.findByLabelText('Tensor parallel size')).toHaveValue(4)
    await user.click(screen.getByRole('button', { name: 'Run' }))
    expect(await screen.findByRole('button', { name: 'Start on 2 nodes' })).toBeInTheDocument()
    expect(screen.getByText(/TP4 is distributed across exactly 2 nodes/)).toBeInTheDocument()
  })

  it('recomputes the run topology from the server-persisted node set after a trimming save', async () => {
    const user = userEvent.setup()
    const fourNode = {
      ...detail,
      model_id: 'org/model',
      node_ids: ['local', 'worker-1', 'worker-2', 'worker-3'],
      launch_controls: {
        ...detail.launch_controls,
        tensor_parallel_size: 4,
        pipeline_parallel_size: 1,
      },
    }
    // The backend answers a TP 4 -> 2 save with the trimmed two-node topology.
    const trimmed = {
      ...fourNode,
      node_ids: ['local', 'worker-1'],
      launch_controls: { ...fourNode.launch_controls, tensor_parallel_size: 2 },
    }
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (path === '/api/v1/nodes') {
        return new Response(JSON.stringify({ items: nodes }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/runtime-flags/preview' && init?.method === 'POST') {
        const body = JSON.parse(String(init.body))
        return new Response(JSON.stringify({
          flags: body.extra_args,
          command_flags: body.extra_args.map(quoteArgForTest).join(' '),
          environment: body.environment,
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path === '/api/v1/deployments/dep-1/settings' && init?.method === 'PUT') {
        return new Response(JSON.stringify(trimmed), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify(fourNode), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    renderPage()

    expect(await screen.findByLabelText('Tensor parallel size')).toHaveValue(4)
    await user.clear(screen.getByLabelText('Tensor parallel size'))
    await user.type(screen.getByLabelText('Tensor parallel size'), '2')
    await user.click(screen.getByRole('button', { name: 'Save' }))
    await screen.findByText('Deployment settings saved. They will be applied on the next run.')

    // The page resource reflects the trimmed topology immediately.
    expect(await screen.findByText(/on 2 nodes/)).toBeInTheDocument()

    // Editing back to TP4 must act on the persisted two-node layout rather
    // than the stale four-node one loaded before the save.
    await user.clear(screen.getByLabelText('Tensor parallel size'))
    await user.type(screen.getByLabelText('Tensor parallel size'), '4')
    await user.click(screen.getByRole('button', { name: 'Run' }))
    const start = await screen.findByRole('button', { name: 'Start on 2 nodes' })
    await user.click(start)

    const startCall = fetchMock.mock.calls.find(([input]) => String(input).endsWith('/start'))
    expect(JSON.parse(String(startCall?.[1]?.body))).toEqual({
      node_ids: ['local', 'worker-1'],
    })
  })

  it('keeps a running deployment read-only', async () => {
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({
      ...detail, status: 'running', desired_state: 'running', editable: false,
      edit_reason: 'Stop this deployment before changing its launch settings.',
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    renderPage()

    expect(await screen.findByText('Stop this deployment before changing its launch settings.')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('button', { name: 'Run' })).toBeDisabled())
    expect(screen.getByLabelText(/Runtime flags/)).toBeDisabled()
  })

  it('edits an inspected external deployment and still allows stopping it', async () => {
    const user = userEvent.setup()
    fetchMock.mockImplementation(async () => {
      const external = {
        ...detail,
        id: 'container:kimi-vllm', kind: 'external', managed: false,
        status: 'starting', desired_state: 'running', editable: true,
        controllable: true,
        edit_reason: null,
        environment: { VLLM_CACHE_ROOT: '/cache/vllm', NCCL_DEBUG: 'INFO' },
        extra_args: ['--max-model-len', '400000', '--enable-prefix-caching'],
      }
      return new Response(JSON.stringify(external), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    render(<MemoryRouter initialEntries={['/models/container%3Akimi-vllm']}><Routes>
      <Route path="/models/:deploymentId" element={<DeploymentPage />} />
      <Route path="/models" element={<h1>Models destination</h1>} />
    </Routes></MemoryRouter>)

    expect(await screen.findByLabelText(/Runtime environment variables/)).toHaveValue(
      'VLLM_CACHE_ROOT=/cache/vllm\nNCCL_DEBUG=INFO',
    )
    expect(screen.getByLabelText(/Runtime flags/)).toHaveValue(
      '--max-model-len 400000 --enable-prefix-caching',
    )
    expect(screen.queryByLabelText('GPU memory reserve (GB)')).not.toBeInTheDocument()
    await user.clear(screen.getByLabelText('Max concurrency'))
    await user.type(screen.getByLabelText('Max concurrency'), '6')
    await user.click(screen.getByRole('button', { name: 'Save' }))
    await screen.findByText('Deployment settings saved and applied.')
    await user.click(screen.getByRole('button', { name: 'Stop' }))

    expect(await screen.findByRole('heading', { name: 'Models destination' })).toBeInTheDocument()
    const mutations = fetchMock.mock.calls.filter(([, init]) => init?.method === 'PUT' || init?.method === 'POST')
    expect(mutations.map(([input]) => String(input))).toEqual([
      '/api/v1/deployments/container%3Akimi-vllm/settings',
      '/api/v1/deployments/container%3Akimi-vllm/stop',
    ])
    expect(JSON.parse(String(mutations[0][1]?.body)).launch_controls.max_concurrency).toBe(6)
  })

  it('preserves literal backslashes in double-quoted flags', async () => {
    const user = userEvent.setup()
    renderPage()

    const flags = await screen.findByLabelText(/Runtime flags/)
    fireEvent.change(flags, { target: { value: '--regex "\\d+" --windows-path "C:\\models\\foo"' } })
    await user.click(screen.getByRole('button', { name: 'Save' }))

    const save = fetchMock.mock.calls.find(([, init]) => init?.method === 'PUT')
    expect(save).toBeDefined()
    expect(JSON.parse(String(save?.[1]?.body)).extra_args).toEqual([
      '--regex', '\\d+', '--windows-path', 'C:\\models\\foo',
    ])
  })

  it('does not save or start when Run encounters invalid form values', async () => {
    const user = userEvent.setup()
    renderPage()

    const utilization = await screen.findByLabelText('GPU memory utilization')
    fireEvent.change(utilization, { target: { value: '2' } })
    expect(utilization).toBeInvalid()
    await user.click(screen.getByRole('button', { name: 'Run' }))

    const mutationCalls = fetchMock.mock.calls.filter(([, init]) => init?.method === 'PUT' || init?.method === 'POST')
    expect(mutationCalls).toEqual([])
  })
})

describe('env-file backed deployment page', () => {
  const envFileDetail = {
    ...detail,
    id: 'container:vllm-dspark', kind: 'external', managed: false,
    status: 'stopped', desired_state: 'stopped', editable: true, controllable: true,
    edit_mode: 'env-file',
    launch_controls: { context_window: 262144 },
    extra_args: ['--served-model-name', 'org/old'],
    settings_env: {
      path: '/opt/stack/.env.dspark', mtime: 1234.5,
      entries: [
        { key: 'SERVED_MODEL_NAME', value: 'org/old', enabled: true, line: 2 },
        { key: 'MAX_MODEL_LEN', value: '262144', enabled: true, line: 3 },
        { key: 'MAX_NUM_SEQS', value: '32', enabled: false, line: 4 },
        { key: 'API_TOKEN', value: null, enabled: true, line: 5, redacted: true },
      ],
      field_mapping: {
        context_window: 'MAX_MODEL_LEN',
        max_concurrency: 'MAX_NUM_SEQS',
        served_model_name: 'SERVED_MODEL_NAME',
      },
    },
  }

  function renderEnvFilePage(putResponse?: unknown) {
    fetchMock.mockImplementation(async (input, init) => {
      const path = String(input)
      if (path === '/api/v1/nodes') {
        return new Response(JSON.stringify({ items: nodes }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (path.endsWith('/settings') && init?.method === 'PUT') {
        return new Response(JSON.stringify(putResponse ?? { ...envFileDetail, restart_required: true }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify(envFileDetail), { status: 200, headers: { 'Content-Type': 'application/json' } })
    })
    return render(<MemoryRouter initialEntries={['/models/container%3Avllm-dspark']}><Routes>
      <Route path="/models/:deploymentId" element={<DeploymentPage />} />
      <Route path="/models" element={<h1>Models destination</h1>} />
    </Routes></MemoryRouter>)
  }

  it('renders the env entries table and served name input, without previewing flags', async () => {
    renderEnvFilePage()

    expect(await screen.findByLabelText('Served model name')).toHaveValue('org/old')
    expect(screen.getByText('MAX_MODEL_LEN')).toBeInTheDocument()
    expect(screen.getByLabelText('MAX_NUM_SEQS value')).toHaveValue('32')
    expect(screen.getByLabelText('Enable MAX_NUM_SEQS')).not.toBeChecked()
    // Secret rows load masked and overwrite-only.
    expect(screen.getByLabelText('API_TOKEN value')).toHaveValue('')
    expect(screen.getByLabelText('API_TOKEN value')).toHaveAttribute('placeholder', '••••••')
    // Controls without a backing env variable are disabled with a hint.
    expect(screen.getByLabelText(/GPU memory utilization/)).toBeDisabled()
    // GPU utilization, KV dtype, and max batched tokens are all unmapped here.
    expect(screen.getAllByText(/no variable backing this control/)).toHaveLength(3)
    expect(screen.getByLabelText('Context window')).toBeEnabled()
    // Flags are script-generated and the editor is read-only in this mode.
    expect(screen.getByLabelText(/Runtime flags/)).toBeDisabled()
    expect(screen.queryByLabelText(/Final runtime flags/)).not.toBeInTheDocument()
    await waitFor(() => expect(screen.getByLabelText('Served model name')).toBeInTheDocument())
    expect(fetchMock.mock.calls.some(([input]) => String(input) === '/api/v1/runtime-flags/preview')).toBe(false)
  })

  it('saves dirty env rows, mapped controls, and the served name', async () => {
    const user = userEvent.setup()
    renderEnvFilePage()

    const servedName = await screen.findByLabelText('Served model name')
    await user.clear(servedName)
    await user.type(servedName, 'org/new')
    const contextValue = screen.getByLabelText('MAX_MODEL_LEN value')
    await user.clear(contextValue)
    await user.type(contextValue, '131072')
    await user.click(screen.getByLabelText('Enable MAX_NUM_SEQS'))
    await user.type(screen.getByLabelText('API_TOKEN value'), 'newsecret')
    await user.click(screen.getByRole('button', { name: 'Add variable' }))
    await user.type(screen.getByLabelText('new variable value'), '3')
    await user.type(screen.getByLabelText('New variable name'), 'MTP_NUM_TOKENS')
    await user.click(screen.getByRole('button', { name: 'Save' }))

    expect(await screen.findByText('Saved to settings file. Stop and start the deployment to apply changes.')).toBeInTheDocument()
    const saveCall = fetchMock.mock.calls.find(([input, init]) => String(input).endsWith('/settings') && init?.method === 'PUT')
    expect(saveCall).toBeDefined()
    const body = JSON.parse(String(saveCall?.[1]?.body))
    expect(body).toEqual({
      launch_controls: { context_window: 262144, max_concurrency: null },
      served_model_name: 'org/new',
      environment: {
        MAX_MODEL_LEN: { value: '131072', enabled: true },
        MAX_NUM_SEQS: { value: '32', enabled: true },
        API_TOKEN: { value: 'newsecret', enabled: true },
        MTP_NUM_TOKENS: { value: '3', enabled: true },
      },
      env_file_mtime: 1234.5,
    })
    expect(fetchMock.mock.calls.some(([input]) => String(input) === '/api/v1/runtime-flags/preview')).toBe(false)
  })

  it('sends null for deleted rows and keeps untouched secrets out of the save', async () => {
    const user = userEvent.setup()
    renderEnvFilePage()

    await screen.findByLabelText('Served model name')
    await user.click(screen.getByLabelText('Delete MAX_MODEL_LEN'))
    await user.click(screen.getByLabelText('Enable API_TOKEN'))
    await user.click(screen.getByRole('button', { name: 'Save' }))

    const saveCall = fetchMock.mock.calls.find(([input, init]) => String(input).endsWith('/settings') && init?.method === 'PUT')
    const body = JSON.parse(String(saveCall?.[1]?.body))
    expect(body.environment).toEqual({
      MAX_MODEL_LEN: null,
      // Toggled but never typed: the redacted value stays on disk.
      API_TOKEN: { value: null, enabled: false },
    })
  })
})
