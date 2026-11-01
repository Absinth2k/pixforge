/**
 * Design tokens come from ../design/tokens.json (single source of truth, see docs/design.md).
 * They are applied as CSS custom properties on <html>; Tailwind maps them via @theme in
 * styles.css, so components use classes like `bg-panel text-text2 border-line`.
 */
import tokens from '../../../design/tokens.json'

export type ThemeName = 'dark' | 'light'

const kebab = (s: string): string => s.replace(/[A-Z]/g, (c) => '-' + c.toLowerCase())

export function applyTheme(name: ThemeName): void {
  const root = document.documentElement
  const colors = tokens.color[name] as Record<string, string>
  for (const [k, v] of Object.entries(colors)) root.style.setProperty('--' + kebab(k), v)
  root.dataset.theme = name
  root.style.colorScheme = name
}

export function systemTheme(): ThemeName {
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

export const sizes = tokens.size
