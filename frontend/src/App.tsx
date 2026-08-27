import { Navigate, Route, Routes } from 'react-router-dom'
import { AuthProvider } from './auth/AuthContext'
import { AppShell } from './components/AppShell'
import { BenchmarksPage } from './pages/BenchmarksPage'
import { ChatPage } from './pages/ChatPage'
import { ClusterPage } from './pages/ClusterPage'
import { ComparePage } from './pages/ComparePage'
import { DashboardPage } from './pages/DashboardPage'
import { ExplorePage } from './pages/ExplorePage'
import { ImagesPage } from './pages/ImagesPage'
import { LogsPage } from './pages/LogsPage'
import { ModelsPage } from './pages/ModelsPage'
import { SettingsPage } from './pages/SettingsPage'
import { StoragePage } from './pages/StoragePage'
import { UsagePage } from './pages/UsagePage'
import { SwitchPage } from './pages/SwitchPage'

export default function App() {
  return (
    <AuthProvider>
      <AppShell>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/explore" element={<ExplorePage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/cluster" element={<ClusterPage />} />
          <Route path="/switch" element={<SwitchPage />} />
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
      </AppShell>
    </AuthProvider>
  )
}
