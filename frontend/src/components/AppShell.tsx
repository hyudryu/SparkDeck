import { useEffect, useRef, useState, type ReactNode } from 'react'
import {
  Activity,
  Boxes,
  ChevronLeft,
  ClipboardList,
  Cpu,
  GitCompareArrows,
  HardDrive,
  Images,
  LayoutDashboard,
  Menu,
  MessageSquareText,
  Moon,
  Network,
  Search,
  Settings,
  Sun,
  X,
} from 'lucide-react'
import { NavLink, useLocation } from 'react-router-dom'
import { api } from '../api/client'
import { persistTheme, storedTheme } from '../theme'

const navigation = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/explore', label: 'Explore', icon: Search },
  { to: '/models', label: 'Models', icon: Cpu },
  { to: '/cluster', label: 'Cluster', icon: Network },
  { to: '/chat', label: 'Chat', icon: MessageSquareText },
  { to: '/compare', label: 'Compare', icon: GitCompareArrows },
  { to: '/benchmarks', label: 'Benchmarks', icon: ClipboardList },
  { to: '/usage', label: 'Usage', icon: Activity },
  { to: '/images', label: 'Images', icon: Images },
  { to: '/storage', label: 'Storage', icon: HardDrive },
  { to: '/settings', label: 'Settings', icon: Settings },
  { to: '/logs', label: 'Logs', icon: Boxes },
]

type ResolvedTheme = 'light' | 'dark'

function resolvedTheme(): ResolvedTheme {
  return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'
}

export function AppShell({ children }: { children: ReactNode }) {
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem('sparkdeck.sidebar') === 'collapsed')
  const [theme, setTheme] = useState<ResolvedTheme>(resolvedTheme)
  const [themeSyncing, setThemeSyncing] = useState(false)
  const [themeStatus, setThemeStatus] = useState('')
  const firstLinkRef = useRef<HTMLAnchorElement>(null)
  const openerRef = useRef<HTMLButtonElement>(null)
  const themeInteractedRef = useRef(false)
  const location = useLocation()
  const current = navigation.find((item) => (item.end ? location.pathname === '/' : location.pathname.startsWith(item.to)))

  useEffect(() => setDrawerOpen(false), [location.pathname])

  useEffect(() => {
    const controller = new AbortController()
    api.settings.get(controller.signal).then((settings) => {
      if (themeInteractedRef.current) return
      persistTheme(settings.theme ?? storedTheme())
      setTheme(resolvedTheme())
    }).catch(() => {
      // The locally stored preference remains authoritative while offline.
    })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    const observer = new MutationObserver(() => setTheme(resolvedTheme()))
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    return () => observer.disconnect()
  }, [])

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

  const toggleTheme = async () => {
    const next: ResolvedTheme = theme === 'dark' ? 'light' : 'dark'
    themeInteractedRef.current = true
    persistTheme(next)
    setTheme(next)
    setThemeSyncing(true)
    setThemeStatus('')
    try {
      const settings = await api.settings.get()
      const saved = await api.settings.update({ ...settings, theme: next })
      persistTheme(saved.theme ?? next)
      setThemeStatus(`${next === 'dark' ? 'Dark' : 'Light'} mode saved.`)
    } catch {
      setThemeStatus(`${next === 'dark' ? 'Dark' : 'Light'} mode saved on this device; server sync failed.`)
    } finally {
      setThemeSyncing(false)
    }
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
          <button
            className="theme-toggle"
            onClick={() => void toggleTheme()}
            disabled={themeSyncing}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            title={collapsed ? `${theme === 'dark' ? 'Light' : 'Dark'} mode` : undefined}
          >
            {theme === 'dark' ? <Sun size={17} aria-hidden="true" /> : <Moon size={17} aria-hidden="true" />}
            <span>{theme === 'dark' ? 'Light mode' : 'Dark mode'}</span>
          </button>
          <span className="sr-only" role="status" aria-live="polite">{themeStatus}</span>
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
