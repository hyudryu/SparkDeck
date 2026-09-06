import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { UpdateBanner } from './UpdateBanner'
import { api } from '../api/client'
import type { SystemUpdateOverview } from '../api/types'

vi.mock('../api/client', () => ({
  api: {
    updates: {
      overview: vi.fn(),
    },
  },
}))

const overviewMock = vi.mocked(api.updates.overview)

afterEach(cleanup)

beforeEach(() => {
  localStorage.clear()
  overviewMock.mockReset()
  overviewMock.mockResolvedValue({ repository: 'hyudryu/SparkDeck', can_update: false } as SystemUpdateOverview)
})

function availableOverview(revision = 'abc1234567890'): SystemUpdateOverview {
  return {
    repository: 'hyudryu/SparkDeck',
    current_revision: '0000000000000',
    target: { branch: 'main', revision },
    up_to_date: false,
    can_update: true,
    blockers: [],
    nodes: [],
  } as SystemUpdateOverview
}

function renderBanner(initialPath = '/dashboard') {
  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <Routes>
        <Route path="*" element={<UpdateBanner />} />
        <Route path="/settings" element={<p>settings page reached</p>} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('UpdateBanner', () => {
  it('shows an installable update with a link to the settings update section', async () => {
    overviewMock.mockResolvedValue(availableOverview())
    renderBanner()

    expect(await screen.findByRole('status')).toHaveTextContent(/Update available/)
    const link = screen.getByRole('link', { name: /software update settings/i })
    expect(link).toHaveAttribute('href', '/settings#software-update')
  })

  it('stays hidden when the cluster is up to date', async () => {
    let resolve!: (value: SystemUpdateOverview) => void
    overviewMock.mockReturnValue(new Promise<SystemUpdateOverview>((r) => { resolve = r }))
    renderBanner()
    expect(screen.queryByRole('status')).not.toBeInTheDocument()

    resolve({
      repository: 'hyudryu/SparkDeck',
      up_to_date: true,
      can_update: false,
    } as SystemUpdateOverview)
    // Give the resolved state a chance to render; the banner must stay gone.
    await waitFor(() => expect(overviewMock).toHaveResolved())
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
  })

  it('stays hidden when an update exists but cannot be installed', async () => {
    let resolve!: (value: SystemUpdateOverview) => void
    overviewMock.mockReturnValue(new Promise<SystemUpdateOverview>((r) => { resolve = r }))
    renderBanner()
    expect(screen.queryByRole('status')).not.toBeInTheDocument()

    resolve({ ...availableOverview(), can_update: false, blockers: ['node offline'] })
    await waitFor(() => expect(overviewMock).toHaveResolved())
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
  })

  it('dismissing hides the banner and remembers the revision', async () => {
    const user = userEvent.setup()
    overviewMock.mockResolvedValue(availableOverview('aaa111'))
    const { unmount } = renderBanner()

    await user.click(await screen.findByRole('button', { name: /dismiss update notification/i }))
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
    expect(localStorage.getItem('sparkdeck.update-banner.dismissed-revision')).toBe('aaa111')
    unmount()

    // The dismissed revision stays dismissed across remounts.
    overviewMock.mockResolvedValue(availableOverview('aaa111'))
    renderBanner()
    expect(screen.queryByRole('status')).not.toBeInTheDocument()

    // A newer revision re-shows the banner.
    overviewMock.mockResolvedValue(availableOverview('bbb222'))
    renderBanner()
    expect(await screen.findByRole('status')).toHaveTextContent(/Update available/)
  })

  it('navigates to the settings update section when clicked', async () => {
    const user = userEvent.setup()
    overviewMock.mockResolvedValue(availableOverview())
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <Routes>
          <Route path="/dashboard" element={<UpdateBanner />} />
          <Route path="/settings" element={<p>settings page reached</p>} />
        </Routes>
      </MemoryRouter>,
    )

    await user.click(await screen.findByRole('link', { name: /software update settings/i }))
    expect(await screen.findByText('settings page reached')).toBeInTheDocument()
  })
})
