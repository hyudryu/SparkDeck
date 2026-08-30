import { useCallback, useEffect, useRef, useState } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { RefreshCw } from 'lucide-react'
import { api } from './api/client'
import type { OnboardingStatus } from './api/types'
import { AuthProvider } from './auth/AuthContext'
import { AppShell } from './components/AppShell'
import { useConfirmDialog } from './components/useConfirmDialog'
import { Button, LoadingState, Panel } from './components/ui'
import { BenchmarksPage } from './pages/BenchmarksPage'
import { ChatPage } from './pages/ChatPage'
import { ClusterPage } from './pages/ClusterPage'
import { ComparePage } from './pages/ComparePage'
import { DashboardPage } from './pages/DashboardPage'
import { ExplorePage } from './pages/ExplorePage'
import { ImagesPage } from './pages/ImagesPage'
import { LogsPage } from './pages/LogsPage'
import { ModelsPage } from './pages/ModelsPage'
import { DeploymentPage } from './pages/DeploymentPage'
import { SettingsPage } from './pages/SettingsPage'
import { StoragePage } from './pages/StoragePage'
import { UsagePage } from './pages/UsagePage'
import { SwitchPage } from './pages/SwitchPage'
import { FanControlPage } from './pages/FanControlPage'

export default function App() {
  const [onboarding, setOnboarding] = useState<OnboardingStatus>()
  const [connectionError, setConnectionError] = useState<string>()
  const [connectionVersion, setConnectionVersion] = useState(0)
  const [leavingCluster, setLeavingCluster] = useState(false)
  const retryTimer = useRef<number | undefined>(undefined)
  const { confirm, confirmationDialog } = useConfirmDialog()
  const retryConnection = useCallback(() => setConnectionVersion((value) => value + 1), [])

  useEffect(() => {
    const controller = new AbortController()
    window.clearTimeout(retryTimer.current)
    api.onboarding.get(controller.signal).then((status) => {
      if (controller.signal.aborted) return
      setOnboarding(status)
      setConnectionError(undefined)
      if (status.role === 'worker') {
        retryTimer.current = window.setTimeout(
          retryConnection,
          status.controller_reachable === false ? 5_000 : 15_000,
        )
      }
    }).catch((reason: unknown) => {
      if (controller.signal.aborted) return
      setConnectionError(reason instanceof Error ? reason.message : 'Could not check the controller connection')
      retryTimer.current = window.setTimeout(retryConnection, 5_000)
    })
    return () => {
      controller.abort()
      window.clearTimeout(retryTimer.current)
    }
  }, [connectionVersion, retryConnection])

  useEffect(() => {
    window.addEventListener('sparkdeck:node-name-changed', retryConnection)
    return () => window.removeEventListener('sparkdeck:node-name-changed', retryConnection)
  }, [retryConnection])

  const controllerUnavailable = onboarding?.role === 'worker' && onboarding.controller_reachable === false
  const controllerAvailable = Boolean(onboarding) && !controllerUnavailable && !connectionError

  const leaveCluster = async () => {
    if (!await confirm({
      title: 'Leave this cluster?',
      message: 'This node will stop using the unavailable controller and become its own standalone controller.',
      confirmLabel: 'Leave cluster',
      danger: true,
    })) return
    setLeavingCluster(true)
    setConnectionError(undefined)
    try {
      const status = await api.onboarding.leave()
      window.clearTimeout(retryTimer.current)
      setOnboarding(status)
    } catch (reason) {
      setConnectionError(reason instanceof Error ? reason.message : 'Could not leave the cluster')
    } finally {
      setLeavingCluster(false)
    }
  }

  return (
    <AppShell controllerAvailable={controllerAvailable} nodeName={onboarding?.node?.name}>
      {!onboarding && !connectionError && <LoadingState label="Connecting to controller" />}
      {(controllerUnavailable || connectionError) && (
        <Panel className="empty-state controller-connection-state" role="alert">
          <h2>Controller unavailable</h2>
          <p>{controllerUnavailable
            ? `This node cannot reach ${onboarding?.controller_url || 'its controller'}. SparkDeck will retry automatically.`
            : connectionError}</p>
          <Button type="button" variant="primary" onClick={retryConnection}>
            <RefreshCw size={16} /> Retry now
          </Button>
          {controllerUnavailable && (
            <Button type="button" variant="danger" disabled={leavingCluster} onClick={() => void leaveCluster()}>
              {leavingCluster ? 'Leaving…' : 'Leave cluster'}
            </Button>
          )}
        </Panel>
      )}
      {controllerAvailable && <AuthProvider>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/explore" element={<ExplorePage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/models/:deploymentId" element={<DeploymentPage />} />
          <Route path="/cluster" element={<ClusterPage />} />
          <Route path="/switch" element={<SwitchPage />} />
          <Route path="/fan-control" element={<FanControlPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/benchmarks" element={<BenchmarksPage />} />
          <Route path="/usage" element={<UsagePage />} />
          <Route path="/images" element={<ImagesPage />} />
          <Route path="/storage" element={<StoragePage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/logs" element={<LogsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AuthProvider>}
      {confirmationDialog}
    </AppShell>
  )
}
