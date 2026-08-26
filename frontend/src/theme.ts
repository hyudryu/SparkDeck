import type { AppSettings } from './api/types'

export type ThemePreference = NonNullable<AppSettings['theme']>

export const THEME_STORAGE_KEY = 'sparkdeck.theme'

export function isThemePreference(value: string | null | undefined): value is ThemePreference {
  return value === 'system' || value === 'light' || value === 'dark'
}

export function storedTheme(): ThemePreference {
  const stored = localStorage.getItem(THEME_STORAGE_KEY)
  return isThemePreference(stored) ? stored : 'system'
}

export function applyTheme(theme: ThemePreference) {
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
  document.documentElement.dataset.theme = theme === 'dark' || (theme === 'system' && prefersDark) ? 'dark' : 'light'
}

export function persistTheme(theme: ThemePreference) {
  localStorage.setItem(THEME_STORAGE_KEY, theme)
  applyTheme(theme)
}
