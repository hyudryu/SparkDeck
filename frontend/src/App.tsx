import { Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { BenchmarksPage } from './pages/BenchmarksPage'
import { ChatPage } from './pages/ChatPage'
import { ComparePage } from './pages/ComparePage'
import { DashboardPage } from './pages/DashboardPage'
import { ExplorePage } from './pages/ExplorePage'
import { ImagesPage } from './pages/ImagesPage'
import { LogsPage } from './pages/LogsPage'
import { ModelsPage } from './pages/ModelsPage'
import { SettingsPage } from './pages/SettingsPage'

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/explore" element={<ExplorePage />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/compare" element={<ComparePage />} />
        <Route path="/benchmarks" element={<BenchmarksPage />} />
        <Route path="/images" element={<ImagesPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="/logs" element={<LogsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  )
}
