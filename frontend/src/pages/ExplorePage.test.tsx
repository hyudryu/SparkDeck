import { act, cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ExplorePage } from './ExplorePage'

const communityAccess = vi.hoisted(() => ({ signedIn: true, sharingEnabled: true, loading: false, enabled: true, reload: () => undefined }))

vi.mock('../hooks/useCommunityAccess', () => ({
  communityAccessHint: (signedIn: boolean) => signedIn
    ? 'Enable telemetry under Settings → Community Features to see community data.'
    : 'Sign in under Settings → Community Features to see community data.',
  useCommunityAccess: () => communityAccess,
}))

const gib = 1024 ** 3

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

beforeEach(() => {
  Object.assign(communityAccess, { signedIn: true, sharingEnabled: true, loading: false, enabled: true })
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('ExplorePage model rows', () => {
  it('pulls once to the first selected node and queues Virtual NAS fan-out', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input, init) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models?')) return json({ items: [{
        id: 'org/pull-model', name: 'pull-model', downloads: 10, likes: 2,
        weight_size_bytes: 8 * gib, runtime_compatibility: [{ runtime: 'vllm', supported: true }],
      }], total: 1 })
      if (path.includes('/api/v1/catalog/models/')) return json({ model: {
        id: 'org/pull-model', name: 'pull-model', revision: 'a'.repeat(40),
        runtime_compatibility: [{ runtime: 'vllm', supported: true }],
      }, aggregates: [] })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [
        { id: 'local', name: 'Controller', local: true, online: true, docker_ready: true, selectable: true },
        { id: 'worker-1', name: 'Node 4', online: true, docker_ready: true, selectable: true },
      ] })
      if (path === '/api/v1/storage/preparations' && init?.method === 'POST') return json({
        workflow_id: 'workflow-1', job_ids: ['download-1', 'transfer-1'], jobs: [
          { id: 'download-1', kind: 'download', model_id: 'org/pull-model', source_node_id: 'huggingface', source_node_name: 'Hugging Face', target_node_id: 'local', target_node_name: 'Controller', status: 'queued', bytes_total: 100, bytes_transferred: 0, created_at: 1 },
          { id: 'transfer-1', kind: 'transfer', model_id: 'org/pull-model', source_node_id: 'local', source_node_name: 'Controller', target_node_id: 'worker-1', target_node_name: 'Node 4', status: 'queued', depends_on_job_id: 'download-1', bytes_total: 100, bytes_transferred: 0, created_at: 1 },
        ],
      }, 202)
      return json({ items: [], availability: 'not_configured', evidence_policy: {} })
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)
    await user.click(await screen.findByRole('button', { name: 'Expand org/pull-model' }))
    await user.click(await screen.findByRole('button', { name: 'Pull weights for org/pull-model' }))

    const dialog = screen.getByRole('dialog', { name: 'Pull org/pull-model' })
    expect(within(dialog).getByRole('checkbox', { name: /Controller/ })).toBeChecked()
    await user.click(within(dialog).getByRole('checkbox', { name: /Node 4/ }))
    expect(within(dialog).getByText(/Download seed:/).closest('p')).toHaveTextContent('Controller')
    await user.click(within(dialog).getByRole('button', { name: 'Pull to 2 nodes' }))

    expect(await screen.findByText(/Queued 1 Hugging Face download and 1 Virtual NAS transfer/)).toBeInTheDocument()
    const preparationCall = fetchMock.mock.calls.find(([input, init]) => (
      String(input) === '/api/v1/storage/preparations' && init?.method === 'POST'
    ))
    expect(JSON.parse(String(preparationCall?.[1]?.body))).toEqual({
      model_id: 'org/pull-model',
      revision: 'a'.repeat(40),
      node_ids: ['local', 'worker-1'],
      download_node_id: 'local',
    })
  })

  it('does not render a separate community evidence panel on Hugging Face rows', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({
        items: [{
          id: 'org/model', author: 'org', name: 'model', downloads: 1000, likes: 12,
          parameter_count: 7_000_000_000, weight_size_bytes: 14 * gib,
          runtime_compatibility: [],
          community: { model_id: 'org/model', tensor_parallel_size: 1, prompt_tokens_bucket: 1000, inference_tokens_per_second: 31.25, sample_count: 14 },
        }],
        total: 1,
      })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [{
        id: 'local', name: 'Spark', online: true, docker_ready: true, selectable: true,
        stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] },
      }] })
      return json({ items: [], availability: 'not_configured', evidence_policy: {} })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)

    const row = await screen.findByRole('button', { name: 'Expand org/model' })
    expect(row).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByText('Sampled from other SparkDeck users')).not.toBeInTheDocument()

    await user.click(row)

    expect(screen.queryByLabelText('Community inference-speed estimate for org/model')).not.toBeInTheDocument()
    expect(screen.getByText('TP1 31.3 tok/s')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Deploy org/model' })).toHaveAttribute('href', '/models?model=org%2Fmodel&runtime=vllm')
  })

  it('offers an unquantized Hugging Face GGUF with controller-local fit and an overridable runtime filter', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models?')) return json({
        items: [{
          id: 'org/model-GGUF', name: 'model-GGUF', downloads: 100, likes: 5,
          weight_size_bytes: 12 * gib,
          runtime_compatibility: [
            { runtime: 'vllm', supported: true },
            { runtime: 'llama.cpp', supported: true },
            { runtime: 'sglang', supported: false },
          ],
          local_deployment_ids: ['existing-vllm'],
        }],
        total: 1,
      })
      if (path.includes('/api/v1/catalog/models/')) return json({ model: {
        id: 'org/model-GGUF', name: 'model-GGUF',
        runtime_compatibility: [
          { runtime: 'vllm', supported: false },
          { runtime: 'llama.cpp', supported: true },
          { runtime: 'sglang', supported: false },
        ],
        quantizations: [{
          name: 'unknown', files: [{ filename: 'model.gguf', size_bytes: 6 * gib }],
          artifacts: [{
            filename: 'model.gguf',
            files: [{ filename: 'model.gguf', size_bytes: 6 * gib }],
            weight_size_bytes: 6 * gib,
            sharded: false,
          }],
        }],
      }, aggregates: [] })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [
        { id: 'local', name: 'Controller', local: true, online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 8 * 1024 }] } },
        { id: 'worker', name: 'Worker', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 16 * 1024 }] } },
      ] })
      return json({ items: [], availability: 'not_configured', evidence_policy: {} })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)
    const row = await screen.findByRole('button', { name: 'Expand org/model-GGUF' })
    expect(within(row).getByText('12 GB').closest('.catalog-model-size')).toHaveClass('fit-easy')
    await user.selectOptions(screen.getByRole('combobox', { name: 'Runtime' }), 'vllm')
    await user.click(screen.getByRole('button', { name: 'Expand org/model-GGUF' }))

    const deploymentType = await screen.findByRole('combobox', { name: 'Deployment type for org/model-GGUF' })
    expect(deploymentType).toHaveValue('vllm')
    expect(within(deploymentType).getAllByRole('option')).toHaveLength(3)
    await screen.findByText('model.gguf')
    expect(within(deploymentType).getByRole('option', { name: 'vLLM' })).toBeEnabled()
    await user.selectOptions(deploymentType, 'llama.cpp')
    expect(deploymentType).toHaveValue('llama.cpp')
    const selectedRow = screen.getByRole('button', { name: 'Collapse org/model-GGUF' })
    expect(within(selectedRow).getByText('6.0 GB').closest('.catalog-model-size')).toHaveClass('fit-tight')
    expect(screen.getByText(/Tight fit · 6\.0 GB/)).toBeInTheDocument()
    expect(screen.getByText(/Llama server deployments run on the controller/)).toHaveTextContent('8.0 GB on the controller node')
    expect(screen.getByRole('combobox', { name: 'GGUF artifact for org/model-GGUF' })).toHaveValue('unknown\u0000model.gguf')
    expect(screen.getByRole('link', { name: 'Deploy org/model-GGUF' })).toHaveAttribute(
      'href', '/models?model=org%2Fmodel-GGUF&runtime=llama.cpp&artifact=model.gguf',
    )
  })

  it('preserves a compatible row runtime selection when model details arrive', async () => {
    const user = userEvent.setup()
    let resolveDetails: ((response: Response) => void) | undefined
    const pendingDetails = new Promise<Response>((resolve) => { resolveDetails = resolve })
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models?')) return json({ items: [{
        id: 'org/async-details', name: 'async-details', weight_size_bytes: 8 * gib,
        downloads: 1, likes: 0, runtime_compatibility: [
          { runtime: 'vllm', supported: true },
          { runtime: 'llama.cpp', supported: true },
          { runtime: 'sglang', supported: false },
        ],
        quantizations: [{
          name: 'Q4_K_M', weight_size_bytes: 8 * gib,
          files: [{ filename: 'search.gguf', size_bytes: 8 * gib }],
          artifacts: [{
            filename: 'search.gguf', weight_size_bytes: 8 * gib,
            files: [{ filename: 'search.gguf', size_bytes: 8 * gib }],
          }],
        }],
      }], total: 1 })
      if (path.includes('/api/v1/catalog/models/')) return pendingDetails
      if (path.endsWith('/api/v1/nodes')) return json({ items: [{
        id: 'local', name: 'Controller', local: true, online: true, docker_ready: true, selectable: true,
        stats: { gpus: [{ index: 0, mem_total_mib: 16 * 1024 }] },
      }] })
      return json({ items: [], availability: 'not_configured', evidence_policy: {} })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)
    expect(await screen.findByRole('button', { name: 'Expand org/async-details' })).toBeInTheDocument()
    await user.selectOptions(screen.getByRole('combobox', { name: 'Runtime' }), 'vllm')
    await user.click(screen.getByRole('button', { name: 'Expand org/async-details' }))
    expect(screen.getByText(/Loading available quantizations/)).toBeInTheDocument()
    const deploymentType = screen.getByRole('combobox', { name: 'Deployment type for org/async-details' })
    await user.selectOptions(deploymentType, 'llama.cpp')
    expect(deploymentType).toHaveValue('llama.cpp')

    await act(async () => resolveDetails?.(json({ model: {
      id: 'org/async-details', name: 'async-details', runtime_compatibility: [
        { runtime: 'vllm', supported: true },
        { runtime: 'llama.cpp', supported: true },
        { runtime: 'sglang', supported: false },
      ],
      quantizations: [{
        name: 'Q5_K_M', weight_size_bytes: 9 * gib,
        files: [{ filename: 'detail-only.gguf', size_bytes: 9 * gib }],
        artifacts: [{
          filename: 'detail-only.gguf', weight_size_bytes: 9 * gib,
          files: [{ filename: 'detail-only.gguf', size_bytes: 9 * gib }],
        }],
      }],
    }, aggregates: [] })))

    expect(await screen.findByText('detail-only.gguf')).toBeInTheDocument()
    expect(deploymentType).toHaveValue('llama.cpp')
  })

  it('uses controller capacity for Llama-only models when filtering all runtimes by fit', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({ items: [
        {
          id: 'org/llama-too-large', name: 'llama-too-large', weight_size_bytes: 12 * gib,
          downloads: 30, likes: 3, runtime_compatibility: [
            { runtime: 'vllm', supported: false },
            { runtime: 'llama.cpp', supported: true },
            { runtime: 'sglang', supported: false },
          ],
        },
        {
          id: 'org/llama-small', name: 'llama-small', weight_size_bytes: 6 * gib,
          downloads: 20, likes: 2, runtime_compatibility: [
            { runtime: 'vllm', supported: false },
            { runtime: 'llama.cpp', supported: true },
            { runtime: 'sglang', supported: false },
          ],
        },
        {
          id: 'org/vllm-model', name: 'vllm-model', weight_size_bytes: 12 * gib,
          downloads: 10, likes: 1, runtime_compatibility: [
            { runtime: 'vllm', supported: true },
            { runtime: 'llama.cpp', supported: false },
            { runtime: 'sglang', supported: true },
          ],
        },
      ], total: 3 })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [
        { id: 'local', name: 'Controller', local: true, online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 8 * 1024 }] } },
        { id: 'worker', name: 'Worker', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 16 * 1024 }] } },
      ] })
      return json({ items: [], availability: 'not_configured', evidence_policy: {} })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)
    expect(await screen.findByRole('button', { name: 'Expand org/llama-too-large' })).toBeInTheDocument()

    await user.click(screen.getByRole('checkbox', { name: /Only what fits/ }))

    expect(screen.queryByRole('button', { name: 'Expand org/llama-too-large' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Expand org/llama-small' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Expand org/vllm-model' })).toBeInTheDocument()
  })

  it('filters Llama models by their default GGUF artifact size instead of model-level weights', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({ items: [
        {
          id: 'org/small-artifact', name: 'small-artifact', weight_size_bytes: 30 * gib,
          downloads: 20, likes: 2, runtime_compatibility: [
            { runtime: 'vllm', supported: false },
            { runtime: 'llama.cpp', supported: true },
            { runtime: 'sglang', supported: false },
          ],
          quantizations: [{
            name: 'Q4_K_M', weight_size_bytes: 8 * gib,
            files: [{ filename: 'small-q4_k_m.gguf', size_bytes: 8 * gib }],
            artifacts: [{
              filename: 'small-q4_k_m.gguf', weight_size_bytes: 8 * gib,
              files: [{ filename: 'small-q4_k_m.gguf', size_bytes: 8 * gib }],
            }],
          }],
        },
        {
          id: 'org/large-artifact', name: 'large-artifact', weight_size_bytes: 8 * gib,
          downloads: 10, likes: 1, runtime_compatibility: [
            { runtime: 'vllm', supported: false },
            { runtime: 'llama.cpp', supported: true },
            { runtime: 'sglang', supported: false },
          ],
          quantizations: [{
            name: 'F16', weight_size_bytes: 30 * gib,
            files: [{ filename: 'large-f16.gguf', size_bytes: 30 * gib }],
            artifacts: [{
              filename: 'large-f16.gguf', weight_size_bytes: 30 * gib,
              files: [{ filename: 'large-f16.gguf', size_bytes: 30 * gib }],
            }],
          }],
        },
      ], total: 2 })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [
        { id: 'local', name: 'Controller', local: true, online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 16 * 1024 }] } },
        { id: 'worker', name: 'Worker', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 64 * 1024 }] } },
      ] })
      return json({ items: [], availability: 'not_configured', evidence_policy: {} })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)
    expect(await screen.findByRole('button', { name: 'Expand org/small-artifact' })).toBeInTheDocument()
    await user.selectOptions(screen.getByRole('combobox', { name: 'Runtime' }), 'llama.cpp')
    expect(await screen.findByText('16 GB controller memory for Llama server')).toBeInTheDocument()

    await user.click(screen.getByRole('checkbox', { name: /Only what fits/ }))

    expect(screen.getByRole('button', { name: 'Expand org/small-artifact' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Expand org/large-artifact' })).not.toBeInTheDocument()
  })

  it('allows an active fit filter to be cleared after switching to Llama without controller telemetry', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({ items: [{
        id: 'org/mixed-runtime', name: 'mixed-runtime', weight_size_bytes: 8 * gib,
        downloads: 1, likes: 0, runtime_compatibility: [
          { runtime: 'vllm', supported: true },
          { runtime: 'llama.cpp', supported: true },
          { runtime: 'sglang', supported: false },
        ],
        quantizations: [{
          name: 'Q4_K_M', weight_size_bytes: 8 * gib,
          files: [{ filename: 'mixed-q4_k_m.gguf', size_bytes: 8 * gib }],
          artifacts: [{
            filename: 'mixed-q4_k_m.gguf', weight_size_bytes: 8 * gib,
            files: [{ filename: 'mixed-q4_k_m.gguf', size_bytes: 8 * gib }],
          }],
        }],
      }], total: 1 })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [
        { id: 'local', name: 'Controller', local: true, online: true, docker_ready: false, selectable: false, stats: { gpus: [{ index: 0, mem_total_mib: 8 * 1024 }] } },
        { id: 'worker', name: 'Worker', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 16 * 1024 }] } },
      ] })
      return json({ items: [], availability: 'not_configured', evidence_policy: {} })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)
    expect(await screen.findByRole('button', { name: 'Expand org/mixed-runtime' })).toBeInTheDocument()
    const fits = screen.getByRole('checkbox', { name: /Only what fits/ })
    await user.click(fits)
    expect(fits).toBeChecked()

    await user.selectOptions(screen.getByRole('combobox', { name: 'Runtime' }), 'llama.cpp')

    expect(await screen.findByText('Controller memory unavailable')).toBeInTheDocument()
    expect(fits).toBeChecked()
    expect(fits).toBeEnabled()
    expect(await screen.findByText('No models found')).toBeInTheDocument()

    await user.click(fits)
    expect(fits).not.toBeChecked()
    expect(await screen.findByRole('button', { name: 'Expand org/mixed-runtime' })).toBeInTheDocument()
  })

  it('color codes cluster fit and filters fitting models largest first', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({ items: [
        { id: 'org/easy', name: 'easy', parameter_count: 100_000_000_000, weight_size_bytes: 200 * gib, downloads: 30, likes: 3, runtime_compatibility: [], community: { model_id: 'org/easy', tensor_parallel_size: 1, prompt_tokens_bucket: 1000, inference_tokens_per_second: 40, sample_count: 10 } },
        { id: 'zai-org/GLM-5.3-Flash', name: 'GLM-5.3-Flash', parameter_count: 321_300_000_000, weight_size_bytes: 306 * gib, downloads: 20, likes: 2, runtime_compatibility: [] },
        { id: 'org/tight', name: 'tight', parameter_count: 400_000_000_000, weight_size_bytes: 400 * gib, downloads: 15, likes: 1, runtime_compatibility: [] },
        { id: 'org/large', name: 'large', parameter_count: 600_000_000_000, weight_size_bytes: 600 * gib, downloads: 10, likes: 1, runtime_compatibility: [] },
        { id: 'org/unknown', name: 'unknown', downloads: 5, likes: 0, runtime_compatibility: [] },
      ], total: 5 })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [
        { id: 'local', name: 'Spark One', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] } },
        { id: 'node-2', name: 'Spark Two', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] } },
        { id: 'node-3', name: 'Spark Three', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] } },
        { id: 'node-4', name: 'Spark Four', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] } },
        { id: 'offline', name: 'Offline', online: false, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 500 * 1024 }] } },
      ] })
      return json({ items: [], availability: 'not_configured', evidence_policy: {} })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)

    expect(await screen.findByRole('button', { name: 'Expand org/easy' })).toBeInTheDocument()
    expect(screen.getByText('200 GB').closest('.catalog-model-size')).toHaveClass('fit-easy')
    expect(screen.getByText('306 GB').closest('.catalog-model-size')).toHaveClass('fit-easy')
    expect(screen.getByText('400 GB').closest('.catalog-model-size')).toHaveClass('fit-tight')
    expect(screen.getByText('600 GB').closest('.catalog-model-size')).toHaveClass('fit-no-fit')
    expect(screen.getByText('512 GB aggregate sharded memory across 4 measured nodes')).toBeInTheDocument()

    // Minimum node counts pool the largest measured nodes first.
    expect(screen.getByText('200 GB').closest('.catalog-model-size')).toHaveTextContent('Fits easily · 2+ nodes')
    expect(screen.getByText('306 GB').closest('.catalog-model-size')).toHaveTextContent('Fits easily · 3+ nodes')
    expect(screen.getByText('400 GB').closest('.catalog-model-size')).toHaveTextContent('Tight fit · 4+ nodes')
    expect(screen.getByText('600 GB').closest('.catalog-model-size')).not.toHaveTextContent('node')
    expect(screen.getByText('600 GB').closest('.catalog-model-size')).not.toHaveTextContent('Fits on')

    await user.click(screen.getByRole('button', { name: 'Expand zai-org/GLM-5.3-Flash' }))
    const fitDetails = screen.getByText(/Fit assumes a sharded deployment/)
    expect(fitDetails).toHaveTextContent('512 GB aggregate memory across 4 measured nodes')
    expect(fitDetails).toHaveTextContent('replicated deployments still require the full model weights')
    expect(fitDetails.closest('.catalog-model-details')).toHaveTextContent('Fits easily · 306 GB · Fits on 3+ nodes')
    expect(screen.getByRole('link', { name: 'Deploy zai-org/GLM-5.3-Flash' })).toHaveAttribute(
      'href', '/models?model=zai-org%2FGLM-5.3-Flash&runtime=vllm&layout=sharded',
    )

    await user.click(screen.getByRole('checkbox', { name: /Only what fits/ }))

    const visibleRows = screen.getAllByRole('button', { name: /^(Expand|Collapse) (org|zai-org)\// })
    expect(visibleRows.map((row) => row.getAttribute('aria-label'))).toEqual([
      'Expand org/tight',
      'Collapse zai-org/GLM-5.3-Flash',
      'Expand org/easy',
    ])
    expect(screen.queryByRole('button', { name: 'Expand org/large' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Expand org/unknown' })).not.toBeInTheDocument()

    expect(screen.queryByRole('checkbox', { name: /Only with community data/ })).not.toBeInTheDocument()
  })

  it('reports a one-node minimum when a single node holds the weights alone', async () => {
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({ items: [{
        id: 'org/compact', name: 'compact', weight_size_bytes: 80 * gib,
        downloads: 1, likes: 0, runtime_compatibility: [],
      }], total: 1 })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [
        { id: 'local', name: 'Spark One', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] } },
      ] })
      return json({ items: [], availability: 'not_configured', evidence_policy: {} })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)

    expect(await screen.findByRole('button', { name: 'Expand org/compact' })).toBeInTheDocument()
    expect(screen.getByText('80 GB').closest('.catalog-model-size')).toHaveTextContent('Fits easily · 1 node')
  })

  it('always counts the controller toward the sharded minimum node count', async () => {
    // 200 GB of weights fit on two 128 GB workers alone, but every sharded
    // deployment includes the 8 GB controller, so three nodes are required.
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({ items: [{
        id: 'org/small-controller', name: 'small-controller', weight_size_bytes: 200 * gib,
        downloads: 1, likes: 0, runtime_compatibility: [],
      }], total: 1 })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [
        { id: 'local', name: 'Controller', local: true, online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 8 * 1024 }] } },
        { id: 'node-2', name: 'Worker Two', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] } },
        { id: 'node-3', name: 'Worker Three', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] } },
      ] })
      return json({ items: [], availability: 'not_configured', evidence_policy: {} })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)

    expect(await screen.findByRole('button', { name: 'Expand org/small-controller' })).toBeInTheDocument()
    expect(screen.getByText('200 GB').closest('.catalog-model-size')).toHaveTextContent('Tight fit · 3+ nodes')
  })

  it('does not pool worker memory when the controller cannot join a sharded deployment', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({ items: [{
        id: 'org/worker-pool-only', name: 'worker-pool-only', weight_size_bytes: 200 * gib,
        downloads: 1, likes: 0, runtime_compatibility: [],
      }], total: 1 })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [
        { id: 'local', name: 'Controller', local: true, online: true, docker_ready: false, selectable: false, stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] } },
        { id: 'node-2', name: 'Worker Two', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] } },
        { id: 'node-3', name: 'Worker Three', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 128 * 1024 }] } },
      ] })
      return json({ items: [], availability: 'not_configured', evidence_policy: {} })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)

    const expand = await screen.findByRole('button', { name: 'Expand org/worker-pool-only' })
    expect(screen.getByText('200 GB').closest('.catalog-model-size')).toHaveClass('fit-no-fit')
    expect(screen.getByText('128 GB largest per-node memory across 2 measured nodes')).toBeInTheDocument()
    await user.click(expand)
    expect(screen.getByRole('link', { name: 'Deploy org/worker-pool-only' })).toHaveAttribute(
      'href', '/models?model=org%2Fworker-pool-only&runtime=vllm',
    )

    await user.click(screen.getByRole('checkbox', { name: /Only what fits/ }))
    expect(screen.queryByRole('button', { name: 'Expand org/worker-pool-only' })).not.toBeInTheDocument()
  })

  it('lists aggregated benchmark models in the community tab without claiming live tracking', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({ detail: 'Hugging Face unavailable' }, 503)
      if (path.endsWith('/api/v1/nodes')) return json({ items: [] })
      return json({
        items: [{ model_id: 'community/model', tensor_parallel_size: 1, prompt_tokens_bucket: 1000, inference_tokens_per_second: 18.5, sample_count: 22 }],
        availability: 'local',
        evidence_policy: {},
      })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)
    await user.click(await screen.findByRole('tab', { name: 'Community Run Models' }))

    const row = await screen.findByRole('button', { name: 'Expand community/model' })
    expect(screen.queryByText('Hugging Face unavailable')).not.toBeInTheDocument()
    expect(screen.getByText('Based on aggregated benchmark samples—not live session tracking.')).toBeInTheDocument()
    expect(screen.queryByRole('checkbox', { name: /Only with community data/ })).not.toBeInTheDocument()
    await user.click(row)
    const quantizations = (await screen.findByText('Available quantizations and artifacts')).closest<HTMLElement>('.catalog-quantizations')!
    expect(within(quantizations).getByText(/^TP1 18\.5 tok\/s/)).toBeInTheDocument()
  })

  it('does not carry a hidden Hugging Face Llama fit filter into the community tab', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({ items: [{
        id: 'org/community-mixed', name: 'community-mixed', weight_size_bytes: 12 * gib,
        downloads: 1, likes: 0, runtime_compatibility: [
          { runtime: 'vllm', supported: true },
          { runtime: 'llama.cpp', supported: true },
          { runtime: 'sglang', supported: false },
        ],
        quantizations: [{
          name: 'Q4_K_M', weight_size_bytes: 12 * gib,
          files: [{ filename: 'community-mixed-q4_k_m.gguf', size_bytes: 12 * gib }],
          artifacts: [{
            filename: 'community-mixed-q4_k_m.gguf', weight_size_bytes: 12 * gib,
            files: [{ filename: 'community-mixed-q4_k_m.gguf', size_bytes: 12 * gib }],
          }],
        }],
      }], total: 1 })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [
        { id: 'local', name: 'Controller', local: true, online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 8 * 1024 }] } },
        { id: 'worker', name: 'Worker', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 16 * 1024 }] } },
      ] })
      return json({
        items: [{
          model_id: 'org/community-mixed', quantization: 'Q4_K_M', tensor_parallel_size: 1, prompt_tokens_bucket: 1000,
          inference_tokens_per_second: 20, sample_count: 10, unique_cluster_count: 2,
          weight_size_bytes: 12 * gib,
        }],
        availability: 'available', evidence_policy: {},
      })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)
    expect(await screen.findByRole('button', { name: 'Expand org/community-mixed' })).toBeInTheDocument()
    await user.selectOptions(screen.getByRole('combobox', { name: 'Runtime' }), 'llama.cpp')
    await user.click(screen.getByRole('checkbox', { name: /Only what fits/ }))
    expect(await screen.findByText('No models found')).toBeInTheDocument()

    await user.click(screen.getByRole('tab', { name: 'Community Run Models' }))

    expect(await screen.findByText('24 GB aggregate sharded memory across 2 measured nodes')).toBeInTheDocument()
    expect(await screen.findByRole('button', {
      name: 'Expand org/community-mixed',
    })).toBeInTheDocument()
  })

  it('groups community Llama quantizations into one model row', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({ items: [{
        id: 'org/community-llama', name: 'community-llama', weight_size_bytes: 20 * gib,
        downloads: 1, likes: 0, runtime_compatibility: [
          { runtime: 'vllm', supported: false },
          { runtime: 'llama.cpp', supported: true },
          { runtime: 'sglang', supported: false },
        ],
        quantizations: [
          {
            name: 'Q4_K_M', weight_size_bytes: 8 * gib,
            files: [{ filename: 'community-q4_k_m.gguf', size_bytes: 8 * gib }],
            artifacts: [{
              filename: 'community-q4_k_m.gguf', weight_size_bytes: 8 * gib,
              files: [{ filename: 'community-q4_k_m.gguf', size_bytes: 8 * gib }],
            }],
          },
          {
            name: 'Q8_0', weight_size_bytes: 20 * gib,
            files: [{ filename: 'community-q8_0.gguf', size_bytes: 20 * gib }],
            artifacts: [{
              filename: 'community-q8_0.gguf', weight_size_bytes: 20 * gib,
              files: [{ filename: 'community-q8_0.gguf', size_bytes: 20 * gib }],
            }],
          },
        ],
      }], total: 1 })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [
        { id: 'local', name: 'Controller', local: true, online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 16 * 1024 }] } },
        { id: 'worker', name: 'Worker', online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 64 * 1024 }] } },
      ] })
      return json({
        items: [
          { model_id: 'org/community-llama', quantization: 'Q4_K_M', tensor_parallel_size: 1, prompt_tokens_bucket: 1000, inference_tokens_per_second: 30, sample_count: 10, unique_cluster_count: 2 },
          { model_id: 'org/community-llama', quantization: 'Q8_0', tensor_parallel_size: 1, prompt_tokens_bucket: 1000, inference_tokens_per_second: 20, sample_count: 20, unique_cluster_count: 2 },
        ],
        availability: 'available', evidence_policy: {},
      })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)
    await user.click(await screen.findByRole('tab', { name: 'Community Run Models' }))
    const row = await screen.findByRole('button', { name: 'Expand org/community-llama' })
    expect(within(row).getByText('20.0–30.0 tok/s')).toBeInTheDocument()

    await user.click(screen.getByRole('checkbox', { name: /Only what fits/ }))

    const fittingRow = screen.getByRole('button', { name: 'Expand org/community-llama' })
    expect(within(fittingRow).getByText('8.0 GB')).toBeInTheDocument()
    await user.click(fittingRow)
    expect(screen.getByRole('combobox', { name: 'GGUF artifact for org/community-llama' })).toHaveValue('Q4_K_M\u0000community-q4_k_m.gguf')
    expect(screen.getByRole('link', { name: 'Deploy org/community-llama' })).toHaveAttribute(
      'href', '/models?model=org%2Fcommunity-llama&runtime=llama.cpp&quantization=Q4_K_M&artifact=community-q4_k_m.gguf',
    )
  })

  it('filters and sorts grouped non-Llama models by their largest fitting quantization', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({ items: [
        {
          id: 'org/grouped-vllm', name: 'grouped-vllm', weight_size_bytes: 20 * gib,
          downloads: 2, likes: 0, runtime_compatibility: [{ runtime: 'vllm', supported: true }],
        },
        {
          id: 'org/twelve-gib', name: 'twelve-gib', weight_size_bytes: 12 * gib,
          downloads: 1, likes: 0, runtime_compatibility: [{ runtime: 'vllm', supported: true }],
        },
      ], total: 2 })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [
        { id: 'local', name: 'Controller', local: true, online: true, docker_ready: true, selectable: true, stats: { gpus: [{ index: 0, mem_total_mib: 16 * 1024 }] } },
      ] })
      return json({
        items: [
          { model_id: 'org/grouped-vllm', quantization: 'Q8_0', tensor_parallel_size: 1, prompt_tokens_bucket: 1000, inference_tokens_per_second: 20, sample_count: 20, weight_size_bytes: 20 * gib },
          { model_id: 'org/grouped-vllm', quantization: 'Q4_K_M', tensor_parallel_size: 1, prompt_tokens_bucket: 1000, inference_tokens_per_second: 30, sample_count: 10, weight_size_bytes: 8 * gib },
          { model_id: 'org/twelve-gib', quantization: 'FP16', tensor_parallel_size: 1, prompt_tokens_bucket: 1000, inference_tokens_per_second: 15, sample_count: 10, weight_size_bytes: 12 * gib },
        ],
        availability: 'available', evidence_policy: {},
      })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)
    await user.click(await screen.findByRole('tab', { name: 'Community Run Models' }))
    await user.click(screen.getByRole('checkbox', { name: /Only what fits/ }))

    const rows = screen.getAllByRole('button', { name: /^Expand org\// })
    expect(rows.map((row) => row.getAttribute('aria-label'))).toEqual([
      'Expand org/twelve-gib',
      'Expand org/grouped-vllm',
    ])
    expect(within(rows[1]).getByText('8.0 GB')).toBeInTheDocument()
    await user.click(rows[1])
    expect(screen.getByRole('link', { name: 'Deploy org/grouped-vllm' })).toHaveAttribute(
      'href', '/models?model=org%2Fgrouped-vllm&runtime=vllm&quantization=Q4_K_M',
    )
  })

  it('pages large community result sets instead of rendering every model at once', async () => {
    const user = userEvent.setup()
    const items = Array.from({ length: 120 }, (_, index) => ({
      model_id: `community/model-${index}`,
      tensor_parallel_size: 1,
      prompt_tokens_bucket: 1000,
      inference_tokens_per_second: 20 + index,
      sample_count: 10,
    }))
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({ items: [], total: 0 })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [] })
      return json({ items, availability: 'available', evidence_policy: {} })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)
    await user.click(await screen.findByRole('tab', { name: 'Community Run Models' }))

    expect(await screen.findAllByRole('button', { name: /^Expand community\/model-/ })).toHaveLength(50)
    const loadMore = screen.getByRole('button', { name: 'Load more community models (70 remaining)' })
    await user.click(loadMore)

    expect(screen.getAllByRole('button', { name: /^Expand community\/model-/ })).toHaveLength(100)
    expect(screen.getByRole('button', { name: 'Load more community models (20 remaining)' })).toBeInTheDocument()
  })

  it('groups repository quantizations, shows a speed range, and labels each quantization speed', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models?')) return json({ items: [{
        id: 'RadixArk/Qwen3.8-27B', name: 'Qwen3.8-27B', parameter_count: 27_000_000_000,
        weight_size_bytes: 16 * gib, downloads: 9000, likes: 400, runtime_compatibility: [],
      }], total: 1 })
      if (path.includes('/api/v1/catalog/models/')) return json({ model: {
        id: 'RadixArk/Qwen3.8-27B', name: 'Qwen3.8-27B', runtime_compatibility: [],
        quantizations: [
          { name: 'NVFP4', weight_size_bytes: 16 * gib, files: [{ filename: 'model-nvfp4.safetensors', size_bytes: 16 * gib }] },
          { name: 'Q4_K_M', weight_size_bytes: 15 * gib, files: [{ filename: 'qwen3.8-q4_k_m.gguf', size_bytes: 15 * gib }] },
        ],
      }, aggregates: [] })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [] })
      return json({
        items: [
          { model_id: 'RadixArk/Qwen3.8-27B', quantization: 'NVFP4', tensor_parallel_size: 1, prompt_tokens_bucket: 1000, inference_tokens_per_second: 52.4, sample_count: 30, unique_cluster_count: 7, parameter_count: 27_000_000_000, weight_size_bytes: 16 * gib },
          { model_id: 'RadixArk/Qwen3.8-27B', quantization: 'Q4_K_M', tensor_parallel_size: 1, prompt_tokens_bucket: 1000, inference_tokens_per_second: 31.2, sample_count: 20, unique_cluster_count: 4, parameter_count: 27_000_000_000, weight_size_bytes: 15 * gib },
          { model_id: 'RadixArk/Qwen3.8-27B', quantization: 'NVFP4', tensor_parallel_size: 1, prompt_tokens_bucket: 400, inference_tokens_per_second: 60.1, sample_count: 30, unique_cluster_count: 3, parameter_count: 27_000_000_000, weight_size_bytes: 16 * gib },
          { model_id: 'RadixArk/Qwen3.8-27B', quantization: 'NVFP4', tensor_parallel_size: 4, prompt_tokens_bucket: 1000, inference_tokens_per_second: 83.7, sample_count: 18, unique_cluster_count: 5, parameter_count: 27_000_000_000, weight_size_bytes: 16 * gib },
        ],
        availability: 'available', evidence_policy: {},
      })
    }))

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)
    await user.click(await screen.findByRole('tab', { name: 'Community Run Models' }))

    const header = document.querySelector('.catalog-model-header')
    expect(header).toHaveTextContent('Output speed')
    expect(header).toHaveTextContent('Max contributors')
    expect(screen.queryByText('Downloads')).not.toBeInTheDocument()
    expect(screen.queryByText('Likes')).not.toBeInTheDocument()
    const row = await screen.findByRole('button', { name: 'Expand RadixArk/Qwen3.8-27B' })
    expect(within(row).getByText('31.2–83.7 tok/s')).toBeInTheDocument()
    expect(within(row).getByText('30')).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: 'Expand RadixArk/Qwen3.8-27B' })).toHaveLength(1)

    await user.click(row)
    expect(await screen.findByText('Available quantizations and artifacts')).toBeInTheDocument()
    expect(screen.getByText('model-nvfp4.safetensors')).toBeInTheDocument()
    expect(screen.getByText('qwen3.8-q4_k_m.gguf')).toBeInTheDocument()
    expect(screen.getByText(/TP1 52\.4 tok\/s · TP4 83\.7 tok\/s/)).toBeInTheDocument()
    const modelArticle = row.closest('article')!
    expect(within(modelArticle).getByText(/^TP1 52\.4 tok\/s/)).toBeInTheDocument()
    expect(within(modelArticle).getByText(/^TP1 31\.2 tok\/s/)).toBeInTheDocument()
    expect(within(modelArticle).queryByText('60.1 tok/s')).not.toBeInTheDocument()
    expect(within(modelArticle).getByRole('link', { name: 'Deploy RadixArk/Qwen3.8-27B' })).toHaveAttribute(
      'href', '/models?model=RadixArk%2FQwen3.8-27B&runtime=vllm&quantization=NVFP4',
    )

    const deploymentType = within(modelArticle).getByRole('combobox', { name: 'Deployment type for RadixArk/Qwen3.8-27B' })
    expect(within(deploymentType).getAllByRole('option')).toHaveLength(3)
    await user.selectOptions(deploymentType, 'llama.cpp')
    const artifactSelect = within(modelArticle).getByRole('combobox', { name: 'GGUF artifact for RadixArk/Qwen3.8-27B' })
    expect(artifactSelect).toHaveValue('Q4_K_M\u0000qwen3.8-q4_k_m.gguf')
    expect(within(artifactSelect).getByRole('option', { name: /Q4_K_M · TP1 31\.2 tok\/s/ })).toBeInTheDocument()
    expect(within(modelArticle).getByRole('link', { name: 'Deploy RadixArk/Qwen3.8-27B' })).toHaveAttribute(
      'href', '/models?model=RadixArk%2FQwen3.8-27B&runtime=llama.cpp&quantization=Q4_K_M&artifact=qwen3.8-q4_k_m.gguf',
    )
  })

  it('locks community features behind sign-in and telemetry opt-in', async () => {
    Object.assign(communityAccess, { signedIn: false, sharingEnabled: false, enabled: false })
    const user = userEvent.setup()
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({
        items: [{
          id: 'org/model', author: 'org', name: 'model', downloads: 1000, likes: 12,
          parameter_count: 7_000_000_000, weight_size_bytes: 14 * gib,
          runtime_compatibility: [],
          community: { model_id: 'org/model', tensor_parallel_size: 1, prompt_tokens_bucket: 1000, inference_tokens_per_second: 31.25, sample_count: 14 },
        }],
        total: 1,
      })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [] })
      return json({ items: [], availability: 'not_configured', evidence_policy: {} })
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)

    const tab = await screen.findByRole('tab', { name: 'Community Run Models' })
    expect(tab).toBeDisabled()
    expect(tab).toHaveAttribute('title', 'Sign in under Settings → Community Features to see community data.')
    expect(screen.queryByRole('checkbox', { name: /Only with community data/ })).not.toBeInTheDocument()

    await user.click(await screen.findByRole('button', { name: 'Expand org/model' }))

    expect(screen.queryByLabelText('Community inference-speed estimate for org/model')).not.toBeInTheDocument()
    expect(screen.queryByText('31.3 tok/s')).not.toBeInTheDocument()
    expect(fetchMock.mock.calls.some(([input]) => String(input).endsWith('/api/v1/community/aggregates'))).toBe(false)
  })

  it('does not request hosted aggregates for a signed-in user who opted out', async () => {
    Object.assign(communityAccess, { signedIn: true, sharingEnabled: false, enabled: false })
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = String(input)
      if (path.includes('/api/v1/catalog/models')) return json({ items: [], total: 0 })
      if (path.endsWith('/api/v1/nodes')) return json({ items: [] })
      throw new Error(`Unexpected request: ${path}`)
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<MemoryRouter><ExplorePage /></MemoryRouter>)

    const tab = await screen.findByRole('tab', { name: 'Community Run Models' })
    expect(tab).toBeDisabled()
    expect(tab).toHaveAttribute('title', 'Enable telemetry under Settings → Community Features to see community data.')
    expect(fetchMock.mock.calls.some(([input]) => String(input).endsWith('/api/v1/community/aggregates'))).toBe(false)
  })
})
