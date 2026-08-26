import { defineConfig } from '@playwright/test'

const viewports = [
  { name: 'mobile-320', width: 320, height: 720 },
  { name: 'mobile-390', width: 390, height: 844 },
  { name: 'tablet-768', width: 768, height: 1024 },
  { name: 'desktop-1440', width: 1440, height: 960 },
]

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  retries: 0,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:4173/',
    colorScheme: 'light',
  },
  projects: viewports.map(({ name, width, height }) => ({
    name,
    use: { viewport: { width, height } },
  })),
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 4173 --base /',
    url: 'http://127.0.0.1:4173/',
    reuseExistingServer: true,
  },
})
