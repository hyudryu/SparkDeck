import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { MarkdownContent } from './MarkdownContent'

afterEach(() => cleanup())

describe('MarkdownContent', () => {
  it('renders remote Markdown images as inert text', () => {
    const { container } = render(<MarkdownContent content="![private chart](https://example.test/track?conversation=secret)" />)

    expect(container.querySelector('img')).not.toBeInTheDocument()
    expect(screen.getByText('Image blocked: private chart')).toBeInTheDocument()
    expect(document.body.innerHTML).not.toContain('example.test')
    expect(document.body.innerHTML).not.toContain('conversation=secret')
  })
})
