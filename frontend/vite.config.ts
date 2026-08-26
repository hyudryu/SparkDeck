import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

function commandOutput(command: string, args: string[]) {
  try {
    return execFileSync(command, args, {
      cwd: fileURLToPath(new URL('..', import.meta.url)),
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim()
  } catch {
    return ''
  }
}

function buildVersion() {
  const explicitVersion = process.env.SPARKDECK_VERSION?.trim()
  if (explicitVersion) return explicitVersion

  if (process.env.GITHUB_REF_TYPE === 'tag') {
    const githubTag = process.env.GITHUB_REF_NAME?.trim()
    if (githubTag) return githubTag
  }

  const exactTag = commandOutput('git', ['describe', '--tags', '--exact-match', 'HEAD'])
  if (exactTag) return exactTag

  const revision = (process.env.GITHUB_SHA?.trim() || commandOutput('git', ['rev-parse', '--short=8', 'HEAD'])).slice(0, 8)
  return revision ? `dev-${revision}` : 'development'
}

export default defineConfig({
  base: '/static/app/',
  plugins: [react()],
  define: {
    __SPARKDECK_VERSION__: JSON.stringify(buildVersion()),
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:7878',
      '/v1': 'http://127.0.0.1:7878',
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    assetsDir: 'assets',
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    css: true,
    exclude: ['e2e/**', 'node_modules/**', 'dist/**'],
  },
})
