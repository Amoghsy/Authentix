import { Link, useLocation } from 'react-router-dom'

const navLinks = [
  { label: 'Detect',   to: '/detect',   icon: 'videocam' },
  { label: 'Analysis', to: '/analysis', icon: 'analytics' },
  { label: 'Reports',  to: '/reports',  icon: 'description' },
  { label: 'History',  to: '/history',  icon: 'history' },
]

export default function Sidebar() {
  const location = useLocation()

  return (
    <nav
      className="hidden md:flex flex-col h-screen w-64 shrink-0 py-6"
      style={{
        backgroundColor: '#0a0a0d',
        borderRight: '1px solid #1f1f23',
        fontFamily: 'Geist, sans-serif',
      }}
    >
      {/* Brand */}
      <div className="px-5 mb-8">
        <div className="flex items-center gap-3">
          <div
            className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0"
            style={{ background: 'linear-gradient(135deg, #7c3aed, #a78bfa)' }}
          >
            <span className="material-symbols-outlined text-white" style={{ fontSize: '18px' }}>shield</span>
          </div>
          <div>
            <h2 className="font-bold tracking-tight text-sm leading-none text-violet-400">
              Authentix
            </h2>
            <span className="text-[10px] uppercase tracking-wider font-medium text-zinc-600">
              Deepfake Guard
            </span>
          </div>
        </div>
      </div>

      {/* Nav links */}
      <div className="flex-1 px-3 space-y-0.5 overflow-y-auto">
        {navLinks.map((link) => {
          const isActive = location.pathname === link.to
          return (
            <Link
              key={link.label}
              to={link.to}
              className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150"
              style={isActive ? {
                background: 'rgba(167,139,250,0.12)',
                color: '#a78bfa',
              } : {
                color: '#71717a',
              }}
              onMouseEnter={e => {
                if (!isActive) {
                  e.currentTarget.style.background = 'rgba(255,255,255,0.04)'
                  e.currentTarget.style.color = '#a1a1aa'
                }
              }}
              onMouseLeave={e => {
                if (!isActive) {
                  e.currentTarget.style.background = 'transparent'
                  e.currentTarget.style.color = '#71717a'
                }
              }}
            >
              <span
                className="material-symbols-outlined shrink-0"
                style={{
                  fontSize: '20px',
                  fontVariationSettings: isActive ? "'FILL' 1" : "'FILL' 0",
                }}
              >
                {link.icon}
              </span>
              <span className="truncate">{link.label}</span>

              {/* Active pip */}
              {isActive && (
                <span
                  className="ml-auto w-1.5 h-1.5 rounded-full shrink-0"
                  style={{ background: '#a78bfa' }}
                />
              )}
            </Link>
          )
        })}
      </div>

      {/* Bottom CTA */}
      <div className="px-4 mt-4 pt-4" style={{ borderTop: '1px solid #1f1f23' }}>
        <Link
          to="/detect"
          className="flex items-center justify-center gap-2 w-full py-2.5 rounded-lg text-sm font-semibold transition-all duration-200 hover:opacity-90 active:scale-[0.98]"
          style={{ background: '#7c3aed', color: '#fff' }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>add_circle</span>
          New Scan
        </Link>
      </div>
    </nav>
  )
}
