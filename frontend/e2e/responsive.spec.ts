import { expect, test } from '@playwright/test'

const routes = ['/', '/models', '/chat', '/compare', '/benchmarks', '/images', '/settings', '/logs']

test.beforeEach(async ({ page }) => {
  await page.route(/^http:\/\/127\.0\.0\.1:4173\/api\//, async (route) => {
    const path = new URL(route.request().url()).pathname
    let body: unknown = {}
    if (path.includes('/catalog/models')) body = { items: [], total: 0, next_cursor: null }
    else if (path.endsWith('/deployments')) body = { items: [] }
    else if (path.includes('/benchmarks')) body = { items: [], total: 0, limit: 100, offset: 0 }
    else if (path.endsWith('/community/sync')) body = { consent: false, pairing: { status: 'not_paired' }, outbox: {} }
    else if (path.endsWith('/community/aggregates')) body = { items: [], availability: 'not_configured' }
    else if (path.endsWith('/images')) body = []
    else if (path.endsWith('/logs') || path.endsWith('/server-logs')) body = { entries: [] }
    else if (path.endsWith('/settings')) body = { theme: 'light', default_runtime: 'vllm', default_context_length: 8192 }
    await route.fulfill({ json: body })
  })
})

test('keeps every primary route within the viewport', async ({ page }) => {
  for (const route of routes) {
    await page.goto(`http://127.0.0.1:4173/static/app/#${route}`)
    await expect(page.locator('main')).toBeVisible()
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
    expect(overflow, `${route} has horizontal overflow`).toBeLessThanOrEqual(1)
  }
})

test('uses a drawer on mobile and a persistent sidebar on desktop', async ({ page }) => {
  await page.goto('http://127.0.0.1:4173/static/app/#/')
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
