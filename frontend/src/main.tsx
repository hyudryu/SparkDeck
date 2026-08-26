import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import './styles.css'
import { applyTheme, storedTheme } from './theme'

applyTheme(storedTheme())
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  if (storedTheme() === 'system') applyTheme('system')
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
)
