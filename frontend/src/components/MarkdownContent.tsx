import ReactMarkdown from 'react-markdown'
import type { Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'

const components: Components = {
  a: ({ children, href }) => <a href={href} target="_blank" rel="noreferrer">{children}</a>,
  img: ({ alt }) => <span className="markdown-image-blocked">Image blocked{alt ? `: ${alt}` : ''}</span>,
}

export function MarkdownContent({ content }: { content: string }) {
  return <ReactMarkdown components={components} remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
}
