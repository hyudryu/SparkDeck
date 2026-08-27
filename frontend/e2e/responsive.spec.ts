import { expect, test } from '@playwright/test'

const routes = ['/', '/dashboard', '/explore', '/models', '/cluster', '/switch', '/chat', '/compare', '/benchmarks', '/usage', '/images', '/storage', '/settings', '/logs']

test.beforeEach(async ({ page }) => {
  let settings = { theme: 'light', default_runtime: 'vllm', default_context_length: 8192, community_api_url: '' }
  await page.route((url) => url.pathname.startsWith('/api/'), async (route) => {
    const path = new URL(route.request().url()).pathname
    let body: unknown = {}
    if (path.endsWith('/routeros/presence')) body = { detected: false, nodes: [{ node_id: 'local', node_name: 'This device', detected: false, configured: false, connected: false }] }
    else if (path.endsWith('/routeros')) body = { detected: false, nodes: [{ node_id: 'local', node_name: 'This device', detected: false, configured: false, connected: false, discovery: [], health: [], interfaces: [] }] }
    else if (path.includes('/catalog/models')) body = { items: [], total: 0, next_cursor: null }
    else if (path === '/api/token-stats') body = {
      models: { 'org/test-model': { input: 1500, cached: 500, output: 750, requests: 12, gen_time_s: 25 } },
      total: { input: 1500, cached: 500, output: 750, requests: 12 },
      groups: [{
        key: 'model:org/test-model', label: 'Test workload', merge_group: null, route_target: 'org/test-model', models: ['org/test-model'],
        members: [{ model: 'org/test-model', alias: 'Test workload', merge_group: null, routed_to: null }],
        stats: { input: 1500, input_miss: 1000, cached: 500, measured_cached: 500, estimated_cached: 0, output: 750, requests: 12, gen_tokens: 750, gen_time_s: 25 },
        speed: { tokens: 750, active_time_s: 25, tok_s: 30, legacy: false }, total_cost: 1.25, cost_estimated: false,
      }],
    }
    else if (path === '/api/token-stats/hourly') body = [{ hour: '2026-08-26T11', input: 600, cached: 200, output: 250, requests: 2 }]
    else if (path === '/api/token-stats/daily') body = [{ date: '2026-08-26', input: 600, cached: 200, output: 250, requests: 2 }]
    else if (path.endsWith('/stats')) body = { cpu_pct: 24, cpu_temp_c: 52, gpus: [{ index: 0, name: 'Test GPU', util: 48, mem_used_mib: 8192, mem_total_mib: 32768, temp: 61 }], active_requests: { 'test-model': { connections: 2, output_tok_s: 18.4, queued: 3 } }, ts: 1787724000 }
    else if (path.endsWith('/inference-queue')) body = { 'dep-1': { model: 'test-model', running: 2, queued: 3, oldest_wait_seconds: 1.5 } }
    else if (path.endsWith('/deployments')) body = { items: [{ id: 'dep-1', alias: 'Test model', runtime: 'vllm', kind: 'managed', model: { repository: 'org/test-model' }, status: 'running', settings: {} }] }
    else if (path.endsWith('/nodes')) body = { items: [{ id: 'local', name: 'This device', local: true, online: true, docker_ready: true, fabric_ready: true, selectable: true }, { id: 'spark-2', name: 'Studio Spark', local: false, online: true, docker_ready: true, fabric_ready: true, selectable: true }] }
    else if (path.endsWith('/onboarding')) body = { role: 'controller', node: { id: 'local', name: 'Studio controller', port: 7878, access_urls: ['https://controller.tailnet.ts.net:7878'] }, controller_reachable: true, join_code: 'PAIR-123', instructions: [] }
    else if (path.endsWith('/benchmark-models')) body = { items: [{
      model_id: 'nvidia/Qwen3.5-35B-A3B-NVFP4', run_count: 12,
      best_prompt_tokens_per_second: 5980, best_generation_tokens_per_second: 174,
      context_windows: [4096, 16384, 32768], tensor_parallel_sizes: [1, 2],
      latest_at: '2026-08-27T12:00:00Z',
    }] }
    else if (path.includes('/benchmark-models/')) body = {
      model_id: 'nvidia/Qwen3.5-35B-A3B-NVFP4',
      points: [4096, 16384, 32768].flatMap((context, contextIndex) => [1, 2, 5, 10].map((concurrency, index) => ({
        context_window_size: context, concurrency, tensor_parallel_size: 1,
        prompt_tokens_per_second: 5980 - contextIndex * 900 - index * 280,
        generation_tokens_per_second: 98 + index * 25 - contextIndex * 12,
        sample_count: 3,
      }))).concat([{ context_window_size: 16384, concurrency: 1, tensor_parallel_size: 2, prompt_tokens_per_second: 6200, generation_tokens_per_second: 112, sample_count: 2 }]),
    }
    else if (path.includes('/benchmarks')) body = { items: [], total: 0, limit: 100, offset: 0 }
    else if (path.endsWith('/community/sync')) body = { consent: true, pairing: { status: 'paired' }, outbox: { pending: 1, synced: 4 } }
    else if (path.endsWith('/community/aggregates')) body = { items: [], availability: 'not_configured', evidence_policy: { minimum_samples: 10, exact_match_dimensions: ['model_id', 'context_window_size'], metric: 'inference_tokens_per_second' } }
    else if (path.endsWith('/images')) body = { items: [{ id: 'sha256:remote', repository: 'org/remote-runtime', tag: 'v1', size: 2147483648, runtimes: ['vllm'], node_ids: ['spark-2'], selected_nodes: [{ id: 'spark-2', name: 'Studio Spark' }] }] }
    else if (path.endsWith('/storage/settings')) body = { enabled: true }
    else if (path.endsWith('/storage/transfers')) body = { id: 'job-new', status: 'queued' }
    else if (path.endsWith('/storage')) body = {
      enabled: true,
      nodes: [
        { id: 'local', name: 'This device', online: true, total_size: 2000000000000, models: [{ model_id: 'org/test-model', size_bytes: 16000000000, revision: 'main', file_count: 8 }] },
        { id: 'spark-2', name: 'Studio Spark', online: true, total_size: 2000000000000, models: [] },
        { id: 'spark-3', name: 'Archive Spark', online: true, total_size: 2000000000000, models: [{ model_id: 'org/test-model', size_bytes: 16000000000 }] },
      ],
      jobs: [{ id: 'job-1', model_id: 'org/queued-model', source_node_id: 'local', source_node_name: 'This device', target_node_id: 'spark-2', target_node_name: 'Studio Spark', status: 'running', bytes_total: 1000, bytes_transferred: 300, created_at: 1787724000 }],
      instructions: ['Keep source and target nodes online until the transfer completes.'],
    }
    else if (path.endsWith('/logs') || path.endsWith('/server-logs')) body = { entries: [] }
    else if (path.endsWith('/settings')) {
      if (route.request().method() === 'PUT') settings = route.request().postDataJSON() as typeof settings
      body = settings
    }
    await route.fulfill({ json: body })
  })
})

test('keeps every primary route within the viewport', async ({ page }) => {
  for (const route of routes) {
    await page.goto(route)
    await expect(page.locator('main')).toBeVisible()
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
    expect(overflow, `${route} has horizontal overflow`).toBeLessThanOrEqual(1)
  }
})

test('renders lifetime and analysis usage without horizontal overflow', async ({ page }) => {
  await page.goto('/usage')
  await expect(page.getByRole('heading', { name: 'Usage stats', exact: true })).toBeVisible()
  await expect(page.getByRole('table', { name: 'Lifetime model usage' })).toContainText('Test workload')
  await expect(page.getByRole('table', { name: 'Lifetime model usage' })).toContainText('$1.25')
  const trend = page.getByRole('img', { name: 'Daily model token trend for the last 30 days' })
  await expect(trend).toBeVisible()
  await expect(trend).toContainText('All models')
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  expect(overflow).toBeLessThanOrEqual(1)
})

test('uses Dashboard as home and keeps Explore on its own route', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible()
  await expect(page.getByLabel('System overview').getByText('52.0°C')).toBeVisible()
  await expect(page.getByText('2 active · 3 queued requests')).toBeVisible()
  await expect(page.getByText('Test model')).toBeVisible()
  await page.goto('/explore')
  await expect(page).toHaveURL(/\/explore$/)
  await expect(page.getByRole('heading', { name: 'Find the right model for your hardware' })).toBeVisible()
})

test('keeps a saved theme after reload', async ({ page }) => {
  await page.goto('/settings')
  await page.getByRole('combobox', { name: 'Appearance' }).selectOption('dark')
  await page.getByRole('button', { name: 'Save settings' }).click()
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')
  await page.reload()
  await expect(page.getByRole('combobox', { name: 'Appearance' })).toHaveValue('dark')
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')
})

test('renders the benchmark explorer dialog in dark mode without overflow', async ({ page }, testInfo) => {
  await page.route('**/api/v1/settings', async (route) => route.fulfill({ json: { theme: 'dark', hf_token_configured: false, community_api_url: '' } }))
  await page.goto('/benchmarks')
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')
  await page.getByRole('button', { name: /nvidia\/Qwen3.5-35B-A3B-NVFP4/ }).click()
  const dialog = page.getByRole('dialog', { name: 'nvidia/Qwen3.5-35B-A3B-NVFP4' })
  await expect(dialog).toBeVisible()
  await expect(dialog.getByRole('img', { name: /^Prompt throughput/ })).toBeVisible()
  await expect(dialog.getByRole('img', { name: /^Text generation throughput/ })).toBeVisible()
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  expect(overflow).toBeLessThanOrEqual(1)
  if (testInfo.project.name === 'desktop-1440') {
    await page.screenshot({ path: testInfo.outputPath('benchmark-explorer-dark.png'), fullPage: true })
  }
})

test('persists the navigation theme toggle after reload', async ({ page }) => {
  await page.goto('/')
  const sidebar = page.getByRole('complementary', { name: 'Primary navigation' })
  if ((page.viewportSize()?.width ?? 0) <= 768) {
    await page.getByRole('button', { name: 'Open navigation' }).click()
  }
  await sidebar.getByRole('button', { name: 'Switch to dark mode' }).click()
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')

  await page.reload()
  if ((page.viewportSize()?.width ?? 0) <= 768) {
    await page.getByRole('button', { name: 'Open navigation' }).click()
  }
  await expect(sidebar.getByRole('button', { name: 'Switch to light mode' })).toBeVisible()
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')
  await expect(sidebar).not.toContainText('Local service')
})

test('uses a drawer on mobile and a persistent sidebar on desktop', async ({ page }) => {
  await page.goto('/')
  const menu = page.getByRole('button', { name: 'Open navigation' })
  const sidebar = page.getByRole('complementary', { name: 'Primary navigation' })

  if ((page.viewportSize()?.width ?? 0) <= 768) {
    await expect(menu).toBeVisible()
    await menu.click()
    await expect(sidebar).toHaveClass(/drawer-open/)
    await expect(sidebar.getByRole('link', { name: 'Benchmarks' })).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(sidebar).not.toHaveClass(/drawer-open/)
  } else {
    await expect(menu).toBeHidden()
    await expect(sidebar).toBeVisible()
    await expect(sidebar.getByRole('link', { name: 'Benchmarks' })).toBeVisible()
  }
})

test('renames remote cluster nodes without horizontal overflow', async ({ page }) => {
  let remoteName = 'Studio Spark'
  await page.route('**/api/v1/nodes/*', async (route) => {
    const request = route.request()
    if (request.method() !== 'PATCH') return route.fallback()
    remoteName = (request.postDataJSON() as { name: string }).name
    await route.fulfill({ json: { id: 'spark-2', name: remoteName, online: true, docker_ready: true, fabric_ready: true, selectable: true } })
  })
  await page.route('**/api/v1/nodes', async (route) => {
    await route.fulfill({ json: { items: [{ id: 'local', name: 'This device', local: true, online: true, docker_ready: true, fabric_ready: true, selectable: true }, { id: 'spark-2', name: remoteName, online: true, docker_ready: true, fabric_ready: true, selectable: true }] } })
  })
  await page.goto('/cluster')

  await page.getByRole('button', { name: 'Edit name for Studio Spark' }).click()
  const input = page.getByRole('textbox', { name: 'New name for Studio Spark' })
  await input.fill('Render Spark')
  await page.getByRole('button', { name: 'Save' }).click()
  await expect(page.getByRole('button', { name: 'Edit name for Render Spark' })).toBeVisible()
  await expect(page.getByRole('status').filter({ hasText: 'Renamed Studio Spark to Render Spark.' })).toBeVisible()
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  expect(overflow).toBeLessThanOrEqual(1)
})

test('keeps community sharing disclosure and estimates clear on every viewport', async ({ page }) => {
  let consent = false
  await page.addInitScript(() => {
    const payload = btoa(JSON.stringify({ exp: 4_102_444_800, email: 'test@example.com' }))
      .replaceAll('+', '-').replaceAll('/', '_').replaceAll('=', '')
    localStorage.setItem('sparkdeck.cognito.id_token', `test.${payload}.signature`)
  })
  await page.route('**/api/v1/community/sync', (route) => route.fulfill({ json: { consent, pairing: { status: 'paired' }, outbox: { pending: 0, synced: 4 } } }))
  await page.route('**/api/v1/community/consent', async (route) => {
    consent = Boolean((await route.request().postDataJSON()).enabled)
    await route.fulfill({ json: {} })
  })
  await page.route('**/api/v1/community/aggregates', (route) => route.fulfill({ json: {
    items: [{ model_id: 'org/test-model', context_window_size: 8192, inference_tokens_per_second: 37.2, sample_count: 12 }],
    availability: 'available',
    evidence_policy: { minimum_samples: 10, exact_match_dimensions: ['model_id', 'context_window_size'], metric: 'inference_tokens_per_second' },
  } }))
  await page.route('**/api/v1/catalog/models*', (route) => route.fulfill({ json: {
    items: [{ id: 'org/test-model', author: 'org', downloads: 1200, likes: 42, runtime_compatibility: [], community: { model_id: 'org/test-model', context_window_size: 8192, inference_tokens_per_second: 37.2, sample_count: 12 } }],
    total: 1,
    next_cursor: null,
  } }))

  await page.goto('/benchmarks')
  await expect(page.getByRole('heading', { name: 'Community estimates' })).toBeVisible()
  await page.getByRole('button', { name: 'Review & enable' }).click()
  const dialog = page.getByRole('dialog', { name: 'Enable community sharing?' })
  await expect(dialog).toContainText('Benchmark JSON: model identifier, context-window size, measured inference tok/s')
  await expect(dialog).toContainText('Never in benchmark JSON: prompts or outputs')
  await dialog.getByRole('button', { name: /I understand, enable sharing/ }).click()
  await expect(page.getByText('37.2 tok/s')).toBeVisible()

  await page.goto('/explore')
  await page.getByRole('button', { name: 'Expand org/test-model' }).click()
  const estimate = page.getByLabel('Community inference-speed estimate for org/test-model')
  await expect(estimate).toContainText('37.2 tok/s')
  await expect(estimate).toContainText('8,192-token context window')
  await expect(estimate).toContainText('estimate, not a guarantee')
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  expect(overflow).toBeLessThanOrEqual(1)
})

test('offers node targets for image pulls and deployments', async ({ page }) => {
  await page.goto('/images')
  await expect(page.getByLabel('Available on Studio Spark')).toContainText('Studio Spark')
  await expect(page.getByLabel('Available on Studio Spark')).not.toContainText('This device')
  await expect(page.getByRole('checkbox', { name: /This device/ })).toBeChecked()
  await page.getByRole('checkbox', { name: /Studio Spark/ }).check()
  await expect(page.getByText('Targets:').locator('..')).toContainText('This device, Studio Spark')

  await page.goto('/models')
  await page.getByRole('button', { name: 'Add model' }).click()
  await expect(page.getByRole('checkbox', { name: /This device/ })).toBeChecked()
  await expect(page.getByRole('checkbox', { name: /This device/ })).toBeEnabled()
  await page.getByRole('checkbox', { name: /Studio Spark/ }).check()
  await page.getByRole('checkbox', { name: /This device/ }).uncheck()
  await expect(page.getByText('Target:').locator('..')).toContainText('Studio Spark')
})

test('keeps storage inventory and transfer controls touch friendly', async ({ page }) => {
  await page.goto('/storage')
  await expect(page.getByRole('heading', { name: 'Storage', exact: true })).toBeVisible()
  await expect(page.getByRole('table', { name: 'Model storage inventory' })).toContainText('org/test-model')
  await expect(page.getByRole('progressbar', { name: 'Transfer org/queued-model progress' })).toHaveAttribute('value', '30')

  const target = page.getByRole('checkbox', { name: /Studio Spark/ })
  await expect(target).toBeEnabled()
  await target.check()
  await expect(page.getByRole('checkbox', { name: /Archive Spark/ })).toBeDisabled()
  await page.getByRole('button', { name: 'Queue transfer' }).click()
  await expect(page.locator('main').getByRole('status')).toContainText('Queued org/test-model for transfer to Studio Spark.')

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  expect(overflow).toBeLessThanOrEqual(1)
})
