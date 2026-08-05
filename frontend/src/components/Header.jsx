import { Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

/**
 * Shared top navbar.
 * @param {'default' | 'signInOnly' | 'none'} mode - Navbar account control mode.
 */
export default function Header({ mode = 'default' }) {
  const { user } = useAuth()

  return (
    <header className="fixed top-0 w-full z-50 bg-[#09090b]/80 backdrop-blur-md border-b border-zinc-800 tracking-tight" style={{ fontFamily: 'Geist, sans-serif' }}>
      <div className="flex justify-between items-center px-6 py-3 max-w-7xl mx-auto">
        {/* Logo */}
        <Link to="/" className="text-xl font-bold tracking-tighter text-violet-400">
          Authentix
        </Link>

        {/* Right side options */}
        {mode === 'signInOnly' && (
          <Link to="/profile">
            <button
              className="text-xs font-semibold px-4 py-2 rounded-lg transition-all duration-200 hover:opacity-90 active:scale-95"
              style={{ background: '#7c3aed', color: '#ffffff', boxShadow: '0 0 15px rgba(124, 58, 237, 0.4)' }}
            >
              Sign In
            </button>
          </Link>
        )}

        {mode === 'default' && (
          <div className="flex items-center gap-4">
            {user ? (
              <Link to="/profile" className="flex items-center gap-2 group">
                <div className="relative">
                  <img
                    alt={user.name}
                    className="w-8 h-8 rounded-full border-2 object-cover transition-all group-hover:border-violet-400"
                    style={{ borderColor: '#27272a' }}
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
              </Link>
            ) : (
              <Link to="/profile" className="flex items-center gap-2 group">
                <span className="text-xs font-semibold px-3 py-1.5 rounded transition-colors hover:text-white" style={{ border: '1px solid #27272a', background: '#121215', color: '#a1a1aa' }}>
                  Sign In
                </span>
                <div
                  className="w-8 h-8 rounded-full border flex items-center justify-center transition-colors hover:border-violet-500"
                  style={{ background: '#18181b', borderColor: '#27272a' }}
                >
                  <span className="material-symbols-outlined text-sm text-zinc-400">person</span>
                </div>
              </Link>
            )}
          </div>
        )}
      </div>
    </header>
  )
}
