import { useEffect, useRef, useState, type ReactNode } from 'react'
import {
  Activity,
  Boxes,
  Cable,
  ChevronLeft,
  ClipboardList,
  Cpu,
  Fan,
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
  { to: '/switch', label: 'Switch', icon: Cable },
  { to: '/fan-control', label: 'Fan Control', icon: Fan },
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

export function AppShell({
  children,
  controllerAvailable = true,
  nodeName: entryNodeName,
}: {
  children: ReactNode
  controllerAvailable?: boolean
  nodeName?: string
}) {
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem('sparkdeck.sidebar') === 'collapsed')
  const [theme, setTheme] = useState<ResolvedTheme>(resolvedTheme)
  const [themeSyncing, setThemeSyncing] = useState(false)
  const [themeStatus, setThemeStatus] = useState('')
  const [switchDetected, setSwitchDetected] = useState(false)
  const [fanControlAvailable, setFanControlAvailable] = useState(false)
  const firstLinkRef = useRef<HTMLAnchorElement>(null)
  const openerRef = useRef<HTMLButtonElement>(null)
  const themeInteractedRef = useRef(false)
  const location = useLocation()
  const current = navigation.find((item) => (item.end ? location.pathname === '/' : location.pathname.startsWith(item.to)))

  useEffect(() => setDrawerOpen(false), [location.pathname])

  useEffect(() => {
    if (!controllerAvailable) return
    let disposed = false
    let controller: AbortController | undefined
    let latestRequest = 0
    const refreshSettings = () => {
      controller?.abort()
      const requestController = new AbortController()
      const requestId = ++latestRequest
      controller = requestController
      const isFresh = () => !disposed && requestId === latestRequest && !requestController.signal.aborted
      api.settings.get(requestController.signal).then((settings) => {
        if (!isFresh()) return
        if (!themeInteractedRef.current) {
          persistTheme(settings.theme ?? storedTheme())
          setTheme(resolvedTheme())
        }
      }).catch(() => {
        // The locally stored preference remains authoritative while offline.
      })
    }
    refreshSettings()
    window.addEventListener('sparkdeck:node-name-changed', refreshSettings)
    return () => {
      disposed = true
      latestRequest += 1
      controller?.abort()
      window.removeEventListener('sparkdeck:node-name-changed', refreshSettings)
    }
  }, [controllerAvailable])

  useEffect(() => {
    if (!controllerAvailable) return
    let disposed = false
    let controller: AbortController | undefined
    let latestRequest = 0
    const refreshPresence = () => {
      controller?.abort()
      const requestController = new AbortController()
      const requestId = ++latestRequest
      controller = requestController
      const isFresh = () => !disposed && requestId === latestRequest && !requestController.signal.aborted
      api.fanControl.get(requestController.signal)
        .then((overview) => {
          if (isFresh()) setFanControlAvailable(Boolean(overview.available && overview.nodes.length > 0))
        })
        .catch(() => {
          if (isFresh()) setFanControlAvailable(false)
        })
    }
    refreshPresence()
    const interval = window.setInterval(refreshPresence, 15_000)
    window.addEventListener('sparkdeck:fan-control-presence-changed', refreshPresence)
    return () => {
      disposed = true
      latestRequest += 1
      controller?.abort()
      window.clearInterval(interval)
      window.removeEventListener('sparkdeck:fan-control-presence-changed', refreshPresence)
    }
  }, [controllerAvailable])

  useEffect(() => {
    if (!controllerAvailable) return
    let disposed = false
    let controller: AbortController | undefined
    let latestRequest = 0
    const refreshPresence = () => {
      controller?.abort()
      const requestController = new AbortController()
      const requestId = ++latestRequest
      controller = requestController
      api.routeros.presence(requestController.signal)
        .then((presence) => {
          if (
            !disposed
            && requestId === latestRequest
            && !requestController.signal.aborted
          ) {
            setSwitchDetected(Boolean(presence.detected))
          }
        })
        .catch(() => {
          if (
            !disposed
            && requestId === latestRequest
            && !requestController.signal.aborted
          ) {
            setSwitchDetected(false)
          }
        })
    }
    refreshPresence()
    const interval = window.setInterval(refreshPresence, 15_000)
    window.addEventListener('sparkdeck:routeros-presence-changed', refreshPresence)
    return () => {
      disposed = true
      latestRequest += 1
      controller?.abort()
      window.clearInterval(interval)
      window.removeEventListener('sparkdeck:routeros-presence-changed', refreshPresence)
    }
  }, [controllerAvailable])

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
    if (!controllerAvailable) {
      setThemeStatus(`${next === 'dark' ? 'Dark' : 'Light'} mode applied locally. Controller unavailable.`)
      return
    }
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
          {drawerOpen && <button className="icon-button mobile-close" onClick={() => setDrawerOpen(false)} aria-label="Close navigation">
            <X size={19} />
          </button>}
        </div>
        <nav className="nav-list">
          {navigation.map(({ to, label, icon: Icon, end }, index) => {
            if (to === '/fan-control' && !fanControlAvailable) return null
            if (to === '/switch' && !switchDetected) {
              return <span
                key={to}
                className="nav-link nav-link-disabled"
                role="link"
                aria-disabled="true"
                tabIndex={0}
                title="Switch is not detected"
              >
                <Icon size={19} aria-hidden="true" />
                <span>{label}</span>
              </span>
            }
            return <NavLink
              ref={index === 0 ? firstLinkRef : undefined}
              key={to}
              to={to}
              end={end}
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              title={collapsed ? label : undefined}
            >
              <Icon size={19} aria-hidden="true" />
              <span>{label}</span>
              {/* The leading space separates the node name in the accessible
                  name: JSX strips inter-element whitespace and jsdom has no
                  layout-based word spacing. */}
              {to === '/' && entryNodeName && <span className="nav-node-name">{' '}{entryNodeName}</span>}
            </NavLink>
          })}
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
