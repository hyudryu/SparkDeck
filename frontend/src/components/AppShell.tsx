import { useEffect, useRef, useState, type ReactNode } from 'react'
import {
  Boxes,
  ChevronLeft,
  ClipboardList,
  Cpu,
  GitCompareArrows,
  Images,
  Menu,
  MessageSquareText,
  Search,
  Settings,
  X,
} from 'lucide-react'
import { NavLink, useLocation } from 'react-router-dom'

const navigation = [
  { to: '/', label: 'Explore', icon: Search, end: true },
  { to: '/models', label: 'Models', icon: Cpu },
  { to: '/chat', label: 'Chat', icon: MessageSquareText },
  { to: '/compare', label: 'Compare', icon: GitCompareArrows },
  { to: '/benchmarks', label: 'Benchmarks', icon: ClipboardList },
  { to: '/images', label: 'Images', icon: Images },
  { to: '/settings', label: 'Settings', icon: Settings },
  { to: '/logs', label: 'Logs', icon: Boxes },
]

export function AppShell({ children }: { children: ReactNode }) {
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem('sparkdeck.sidebar') === 'collapsed')
  const firstLinkRef = useRef<HTMLAnchorElement>(null)
  const openerRef = useRef<HTMLButtonElement>(null)
  const location = useLocation()
  const current = navigation.find((item) => (item.end ? location.pathname === '/' : location.pathname.startsWith(item.to)))

  useEffect(() => setDrawerOpen(false), [location.pathname])

  useEffect(() => {
    if (!drawerOpen) return
    document.body.classList.add('drawer-active')
    firstLinkRef.current?.focus()
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setDrawerOpen(false)
        openerRef.current?.focus()
      }
    }
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.body.classList.remove('drawer-active')
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [drawerOpen])

  const toggleCollapsed = () => {
    setCollapsed((value) => {
      localStorage.setItem('sparkdeck.sidebar', value ? 'expanded' : 'collapsed')
      return !value
    })
  }

  return (
    <div className={`app-shell ${collapsed ? 'sidebar-collapsed' : ''}`}>
      <a className="skip-link" href="#main-content">Skip to content</a>
      <aside className={`sidebar ${drawerOpen ? 'drawer-open' : ''}`} aria-label="Primary navigation">
        <div className="brand-row">
          <NavLink to="/" className="brand" aria-label="SparkDeck home">
            <span className="brand-mark" aria-hidden="true"><span /><span /><span /></span>
            <span className="brand-name">SparkDeck</span>
          </NavLink>
          <button className="icon-button mobile-close" onClick={() => setDrawerOpen(false)} aria-label="Close navigation">
            <X size={19} />
          </button>
        </div>
        <nav className="nav-list">
          {navigation.map(({ to, label, icon: Icon, end }, index) => (
            <NavLink
              ref={index === 0 ? firstLinkRef : undefined}
              key={to}
              to={to}
              end={end}
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              title={collapsed ? label : undefined}
            >
              <Icon size={19} aria-hidden="true" />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="local-status">
            <span className="status-dot status-running" aria-hidden="true" />
            <span>Local service</span>
          </div>
          <button className="collapse-button" onClick={toggleCollapsed} aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}>
            <ChevronLeft size={17} aria-hidden="true" />
            <span>Collapse</span>
          </button>
        </div>
      </aside>
      {drawerOpen && <button className="drawer-backdrop" onClick={() => setDrawerOpen(false)} aria-label="Close navigation" />}
      <div className="app-stage">
        <header className="mobile-appbar">
          <button ref={openerRef} className="icon-button" onClick={() => setDrawerOpen(true)} aria-label="Open navigation" aria-expanded={drawerOpen}>
            <Menu size={20} />
          </button>
          <span className="mobile-title">{current?.label ?? 'SparkDeck'}</span>
          <span className="mobile-brand-mark" aria-hidden="true" />
        </header>
        <main id="main-content" className="main-content" tabIndex={-1}>{children}</main>
      </div>
    </div>
  )
}
