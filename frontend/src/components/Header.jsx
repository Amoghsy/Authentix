import { Link, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function Header() {
  const location = useLocation()
  const { user } = useAuth()

  const navLinks = [
    {},
    {},
    {},
  ]

  return (
    <header className="fixed top-0 w-full z-50 bg-[#09090b]/80 backdrop-blur-md border-b border-zinc-800 tracking-tight" style={{ fontFamily: 'Geist, sans-serif' }}>
      <div className="flex justify-between items-center px-6 py-3 max-w-7xl mx-auto">
        <div className="flex items-center gap-8">
          <Link to="/" className="text-xl font-bold tracking-tighter text-violet-400">
            Authentix
          </Link>
          <nav className="hidden md:flex gap-6">
            {navLinks.map((link) => (
              <Link
                key={link.label}
                to={link.to}
                className={`text-sm font-medium transition-colors ${location.pathname === link.to
                    ? 'text-violet-400'
                    : 'text-zinc-400 hover:text-violet-300'
                  }`}
              >
                {link.label}
              </Link>
            ))}
          </nav>
        </div>

        <div className="flex items-center gap-4">
          {user ? (
            /* Signed in — show avatar linking to profile */
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
            /* Not signed in — sign in link + avatar placeholder */
            <>


              <Link to="/profile">
                <div
                  className="w-8 h-8 rounded-full border flex items-center justify-center transition-colors hover:border-violet-500"
                  style={{ background: '#18181b', borderColor: '#27272a' }}
                >
                  <span className="material-symbols-outlined text-sm text-zinc-400">person</span>
                </div>
              </Link>
            </>
          )}
        </div>
      </div>
    </header>
  )
}
