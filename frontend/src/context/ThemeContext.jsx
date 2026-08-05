import { createContext, useContext, useState, useEffect } from 'react'

const ThemeContext = createContext({
  theme: 'dark',
  c: null,       // will be populated by provider
  toggle: () => {},
  setTheme: () => {},
})

export const dark = {
  bg:           '#09090b',
  bgMid:        '#0c0c0f',
  surface:      '#111113',
  surfaceHigh:  '#18181b',
  surfaceHigher:'#1e1e22',
  border:       '#27272a',
  text:         '#fafafa',
  textMuted:    '#a1a1aa',
  textDim:      '#71717a',
  primary:      '#a78bfa',
  primaryBg:    'rgba(167,139,250,0.1)',
  primaryBorder:'rgba(167,139,250,0.25)',
  primaryDark:  '#7c3aed',
  green:        '#34d399',
  greenBg:      'rgba(52,211,153,0.1)',
  greenBorder:  'rgba(52,211,153,0.25)',
  red:          '#ef4444',
  redBg:        'rgba(239,68,68,0.1)',
  redBorder:    'rgba(239,68,68,0.25)',
  amber:        '#f59e0b',
  amberBg:      'rgba(245,158,11,0.1)',
}

export const light = {
  bg:           '#f4f4f8',
  bgMid:        '#ebebf2',
  surface:      '#ffffff',
  surfaceHigh:  '#f1f1f8',
  surfaceHigher:'#e8e8f2',
  border:       '#dddde8',
  text:         '#09090b',
  textMuted:    '#52525b',
  textDim:      '#71717a',
  primary:      '#7c3aed',
  primaryBg:    'rgba(124,58,237,0.08)',
  primaryBorder:'rgba(124,58,237,0.25)',
  primaryDark:  '#6d28d9',
  green:        '#059669',
  greenBg:      'rgba(5,150,105,0.08)',
  greenBorder:  'rgba(5,150,105,0.25)',
  red:          '#dc2626',
  redBg:        'rgba(220,38,38,0.08)',
  redBorder:    'rgba(220,38,38,0.25)',
  amber:        '#d97706',
  amberBg:      'rgba(217,119,6,0.1)',
}

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(() => localStorage.getItem('ax-theme') || 'dark')

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
    document.documentElement.classList.toggle('light', theme === 'light')
    localStorage.setItem('ax-theme', theme)
  }, [theme])

  const c = theme === 'dark' ? dark : light
  const toggle = () => setTheme(t => t === 'dark' ? 'light' : 'dark')

  return (
    <ThemeContext.Provider value={{ theme, setTheme, toggle, c }}>
      {children}
    </ThemeContext.Provider>
  )
}

export const useTheme = () => useContext(ThemeContext)
