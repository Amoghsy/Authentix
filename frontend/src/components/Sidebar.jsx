import { Link, useLocation } from 'react-router-dom'
import { useTheme } from '../context/ThemeContext'

const navLinks = [
  { label: 'Home',      to: '/',         icon: 'home' },
  { label: 'Detect', to: '/detect',   icon: 'videocam' },
  { label: 'Analysis',  to: '/analysis', icon: 'analytics' },
  { label: 'Reports',   to: '/reports',  icon: 'description' },
  { label: 'History',   to: '/history',  icon: 'history' },
  { label: 'About',     to: '/about',    icon: 'info' },
  { label: 'Settings',  to: '/settings', icon: 'tune' },
]

export default function Sidebar() {
  const location = useLocation()
  const { c, theme, toggle } = useTheme()

  return (
    <nav
      className="hidden md:flex flex-col h-screen w-64 shrink-0 py-6 transition-colors duration-300"
      style={{
        background: c.bgMid,
        borderRight: `1px solid ${c.border}`,
      }}
    >
      {/* Logo */}
      <div className="px-6 mb-8">
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-full flex items-center justify-center overflow-hidden"
            style={{ background: 'linear-gradient(135deg,#7c3aed,#a78bfa)', border: `1px solid ${c.border}` }}
          >
            <span className="material-symbols-outlined text-white text-lg">shield</span>
          </div>
          <div>
            <h2
              className="font-bold tracking-tight text-lg leading-none"
              style={{ color: c.primary }}
            >
              Authentix
            </h2>
            <span
              className="text-xs uppercase tracking-wider font-semibold"
              style={{ color: c.textDim }}
            >
              Deepfake Guard
            </span>
          </div>
        </div>
      </div>

      {/* Nav links */}
      <div className="flex-1 px-4 space-y-1 overflow-y-auto">
        {navLinks.map((link) => {
          const isActive = location.pathname === link.to
          return (
            <Link
              key={link.label}
              to={link.to}
              className="flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200"
              style={isActive ? {
                background: c.surfaceHigh,
                color: c.primary,
                fontWeight: 600,
                borderRight: `2px solid ${c.primary}`,
              } : {
                color: c.textDim,
              }}
              onMouseEnter={e => { if (!isActive) { e.currentTarget.style.background = c.surfaceHigh; e.currentTarget.style.color = c.textMuted } }}
              onMouseLeave={e => { if (!isActive) { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = c.textDim } }}
            >
              <span
                className="material-symbols-outlined text-xl"
                style={isActive ? { fontVariationSettings: "'FILL' 1" } : {}}
              >
                {link.icon}
              </span>
              <span>{link.label}</span>
            </Link>
          )
        })}
      </div>

      {/* Theme toggle + CTA */}
      <div className="px-6 mt-auto space-y-3">
        {/* Theme toggle */}
        <button
          onClick={toggle}
          className="w-full flex items-center justify-between px-4 py-2.5 rounded-lg transition-colors duration-200"
          style={{ background: c.surface, border: `1px solid ${c.border}`, color: c.textMuted }}
          onMouseEnter={e => e.currentTarget.style.borderColor = c.primary}
          onMouseLeave={e => e.currentTarget.style.borderColor = c.border}
        >
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-base" style={{ color: c.primary }}>
              {theme === 'dark' ? 'light_mode' : 'dark_mode'}
            </span>
            <span className="text-xs font-medium">
              {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
            </span>
          </div>
          {/* Toggle pill */}
          <div
            className="w-10 h-5 rounded-full relative transition-colors duration-300 flex items-center"
            style={{ background: theme === 'dark' ? c.primary : c.border }}
          >
            <div
              className="w-4 h-4 rounded-full absolute transition-all duration-300"
              style={{
                background: '#fff',
                left: theme === 'dark' ? '22px' : '2px',
                boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
              }}
            />
          </div>
        </button>

        {/* New Scan */}
        <Link
          to="/detect"
          className="w-full py-3 rounded-lg font-semibold flex items-center justify-center gap-2 transition-colors duration-200"
          style={{ background: c.primary, color: '#fff' }}
          onMouseEnter={e => e.currentTarget.style.opacity = '0.9'}
          onMouseLeave={e => e.currentTarget.style.opacity = '1'}
        >
          <span className="material-symbols-outlined text-lg">add_circle</span>
          New Scan
        </Link>
      </div>
    </nav>
  )
}
