import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { SplitButton } from './ui'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

function renderSplitButton() {
  const onSelect = vi.fn()
  render(
    <SplitButton
      label="Stop"
      onMainAction={() => {}}
      toggleAriaLabel="More actions for Chat model"
      items={[{ key: 'additional', label: 'Launch on additional nodes…', onSelect }]}
    />,
  )
  return onSelect
}

describe('SplitButton', () => {
  it('moves keyboard focus into the menu and back to the toggle on close', async () => {
    const user = userEvent.setup()
    renderSplitButton()

    await user.click(screen.getByRole('button', { name: 'More actions for Chat model' }))
    const item = screen.getByRole('menuitem', { name: 'Launch on additional nodes…' })
    expect(item).toHaveFocus()

    fireEvent.keyDown(item, { key: 'Escape' })
    expect(screen.queryByRole('menuitem')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'More actions for Chat model' })).toHaveFocus()
  })

  it('anchors the menu below the caret when there is room', async () => {
    const user = userEvent.setup()
    renderSplitButton()

    await user.click(screen.getByRole('button', { name: 'More actions for Chat model' }))
    expect(screen.getByRole('menu')).toHaveStyle({ top: '4px' })
  })

  it('flips the menu above the caret near the bottom of the viewport', async () => {
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
      top: 700, bottom: 730, left: 10, right: 110, width: 100, height: 30,
      x: 10, y: 700, toJSON: () => ({}),
    } as DOMRect)
    const user = userEvent.setup()
    renderSplitButton()

    await user.click(screen.getByRole('button', { name: 'More actions for Chat model' }))
    // Below the caret only 768 - 730 = 38px remain, less than the menu needs,
    // so the menu anchors above it instead.
    expect(screen.getByRole('menu')).toHaveStyle({ bottom: '72px' })
  })
})
