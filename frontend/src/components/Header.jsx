/**
 * Header.jsx — Shared top navigation bar.
 *
 * Props:
 *   mode    'default' | 'signInOnly' | 'none'
 *           default   → shows avatar dropdown (signed-in) or Sign In button (signed-out)
 *           signInOnly → shows only a Sign In button (no avatar dropdown)
 *           none      → shows only the logo, no account UI
 *
 *   sticky  boolean (default false)
 *           true  → position sticky (used inside AppLayout's flex column — no offset needed)
 *           false → position fixed top-0 left-0 w-full (standalone full-page use)
 */
import { useState, useRef, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function Header({ mode = 'default', sticky = false }) {
  const { user, signOut } = useAuth()
  const navigate = useNavigate()
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const dropdownRef = useRef(null)

  /* ── close dropdown on outside click ── */
  useEffect(() => {
    const handler = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setDropdownOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const handleSignOut = async () => {
    setDropdownOpen(false)
    await signOut()
    navigate('/')
  }

  const positionClass = sticky
    ? 'sticky top-0 z-40 w-full'
    : 'fixed top-0 left-0 w-full z-50'

  return (
    <header
      className={`${positionClass} bg-[#09090b]/90 backdrop-blur-md border-b border-zinc-800 shrink-0`}
      style={{ fontFamily: 'Geist, sans-serif' }}
    >
      <div className="flex items-center justify-between px-6 py-3 w-full">

        {/* Logo — only shown on full-page (non-sticky) headers */}
        {!sticky ? (
          <Link to="/" className="text-xl font-bold tracking-tighter text-violet-400">
            Authentix
          </Link>
        ) : (
          /* Spacer so account section stays right-aligned inside AppLayout */
          <div />
        )}

        {/* ── Account section ── */}
        {mode === 'none' && null}

        {mode === 'signInOnly' && (
          <Link to="/profile">
            <button
              className="text-xs font-semibold px-4 py-2 rounded-lg transition-all duration-200 hover:opacity-90 active:scale-95"
              style={{
                background: '#7c3aed',
                color: '#ffffff',
                boxShadow: '0 0 18px rgba(124,58,237,0.45)',
              }}
            >
              Sign In
            </button>
          </Link>
        )}

        {mode === 'default' && (
          <div className="flex items-center gap-4 relative" ref={dropdownRef}>
            {user ? (
              <>
                {/* Avatar trigger */}
                <button
                  onClick={() => setDropdownOpen(o => !o)}
                  className="flex items-center gap-2 group focus:outline-none"
                >
                  <div className="relative">
                    <img
                      alt={user.name}
                      className="w-8 h-8 rounded-full border-2 object-cover transition-all group-hover:border-violet-400"
                      style={{ borderColor: dropdownOpen ? '#a78bfa' : '#3f3f46' }}
                      src={user.avatar}
                    />
                    <span
                      className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full border-2"
                      style={{ background: '#34d399', borderColor: '#09090b' }}
                    />
                  </div>
                  <span className="hidden md:block text-xs font-medium text-zinc-300 group-hover:text-violet-300 transition-colors">
                    {user.name.split(' ')[0]}
                  </span>
                  <span
                    className="material-symbols-outlined text-sm text-zinc-500 group-hover:text-violet-300 transition-transform duration-200"
                    style={{ transform: dropdownOpen ? 'rotate(180deg)' : 'rotate(0deg)', fontSize: '18px' }}
                  >
                    expand_more
                  </span>
                </button>

                {/* Dropdown panel */}
                {dropdownOpen && (
                  <div
                    className="absolute right-0 top-12 w-56 rounded-xl border border-zinc-800 shadow-2xl overflow-hidden z-50"
                    style={{ background: '#111113' }}
                  >
                    {/* Identity row */}
                    <div className="flex items-center gap-3 px-4 py-3.5 border-b border-zinc-800">
                      <img
                        src={user.avatar}
                        alt={user.name}
                        className="w-9 h-9 rounded-full border border-zinc-700 object-cover"
                      />
                      <div className="min-w-0">
                        <p className="text-xs font-semibold text-zinc-100 truncate">{user.name}</p>
                        <p className="text-[10px] text-zinc-500 truncate mt-0.5">{user.email}</p>
                      </div>
                    </div>

                    {/* Navigation items */}
                    <div className="py-1">
                      {[
                        { icon: 'person', label: 'My Profile', to: '/profile' },
                        { icon: 'history', label: 'Activity History', to: '/history' },
                        { icon: 'description', label: 'Reports', to: '/reports' },
                      ].map(item => (
                        <Link
                          key={item.label}
                          to={item.to}
                          onClick={() => setDropdownOpen(false)}
                          className="flex items-center gap-3 px-4 py-2.5 text-xs text-zinc-300 hover:bg-zinc-800/70 hover:text-white transition-colors"
                        >
                          <span className="material-symbols-outlined text-base text-zinc-500" style={{ fontSize: '16px' }}>
                            {item.icon}
                          </span>
                          {item.label}
                        </Link>
                      ))}
                    </div>

                    {/* Sign out */}
                    <div className="border-t border-zinc-800 py-1">
                      <button
                        onClick={handleSignOut}
                        className="w-full flex items-center gap-3 px-4 py-2.5 text-xs text-red-400 hover:bg-zinc-800/70 hover:text-red-300 transition-colors text-left"
                      >
                        <span className="material-symbols-outlined text-base" style={{ fontSize: '16px' }}>
                          logout
                        </span>
                        Sign Out
                      </button>
                    </div>
                  </div>
                )}
              </>
            ) : (
              /* Not signed in */
              <Link to="/profile" className="flex items-center gap-2 group">
                <span
                  className="text-xs font-semibold px-3 py-1.5 rounded-lg transition-colors hover:text-white"
                  style={{ border: '1px solid #27272a', background: '#121215', color: '#a1a1aa' }}
                >
                  Sign In
                </span>
              </Link>
            )}
          </div>
        )}
      </div>
    </header>
  )
}
