import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { useTheme } from '../context/ThemeContext'
import Header from '../components/Header'

/* ── helpers ── */
const GOOGLE_AVATAR = 'https://lh3.googleusercontent.com/aida-public/AB6AXuBsWmh5HNIBRWOv2hgSWlpvlzMVr-Dg_JCAlQxmB5WlN1-exda5KUzYKaV6hhKXjMLMcMfUrfCLYwodo-KrRodmJAaWYDuVwsKYrb8QAbuxpOlGbw2aRcFKOmQXb2ABk6HXfeZ3Lj5RyBLSZm9cWoVMj-UNSiQ-ZlHjjKaacdbAQuf4DbXwukEvtTlSpOYmy7yDPQRmdtqy9X3MUwC0BAde_sAmTBzGrH5kIk6S_4yDeBqMnUJ7WBKDU5PCnUlzPPx3P4-nPr3aIRg'
const GITHUB_AVATAR  = 'https://avatars.githubusercontent.com/u/9919?v=4'

/* ── signed-out view ── */
function SignInView({ c }) {
  const { signIn } = useAuth()
  const navigate   = useNavigate()

  const [tab,      setTab]      = useState('signin')   // 'signin' | 'signup'
  const [email,    setEmail]    = useState('')
  const [password, setPassword] = useState('')
  const [name,     setName]     = useState('')
  const [loading,  setLoading]  = useState(null)
  const [error,    setError]    = useState('')

  const mockSignIn = async (provider, userData) => {
    setLoading(provider)
    await new Promise(r => setTimeout(r, 900))
    signIn(userData)
    navigate('/')
  }

  const handleEmail = async (e) => {
    e.preventDefault()
    setError('')
    if (!email || !password) { setError('Please fill in all fields.'); return }
    if (tab === 'signup' && !name) { setError('Please enter your name.'); return }
    setLoading('email')
    await new Promise(r => setTimeout(r, 800))
    signIn({
      name:     tab === 'signup' ? name : 'Alex Chen',
      email,
      avatar:   `https://api.dicebear.com/7.x/initials/svg?seed=${encodeURIComponent(tab === 'signup' ? name : 'Alex Chen')}&backgroundColor=7c3aed&textColor=ffffff`,
      provider: 'email',
    })
    navigate('/')
  }

  const input = (props) => (
    <input
      {...props}
      className="w-full px-4 py-3 rounded-lg text-sm outline-none transition-colors"
      style={{
        background:  c.surfaceHigh,
        border:      `1px solid ${c.border}`,
        color:       c.text,
        caretColor:  c.primary,
      }}
      onFocus={e  => e.target.style.borderColor = c.primary}
      onBlur={e   => e.target.style.borderColor = c.border}
    />
  )

  return (
    <div className="w-full max-w-sm mx-auto">
      {/* Logo */}
      <div className="text-center mb-8">
        <div
          className="w-14 h-14 rounded-2xl mx-auto mb-4 flex items-center justify-center"
          style={{ background: 'linear-gradient(135deg,#7c3aed,#a78bfa)' }}
        >
          <span className="material-symbols-outlined text-white text-2xl">shield</span>
        </div>
        <h1 className="text-2xl font-black tracking-tight" style={{ color: c.text }}>
          {tab === 'signup' ? 'Create account' : 'Welcome back'}
        </h1>
        <p className="text-sm mt-1" style={{ color: c.textDim }}>
          {tab === 'signup' ? 'Start detecting deepfakes today.' : 'Sign in to your Authentix account.'}
        </p>
      </div>

      {/* OAuth buttons */}
      <div className="flex flex-col gap-3 mb-6">
        <button
          onClick={() => mockSignIn('google', { name: 'Alex Chen',   email: 'alex@gmail.com',   avatar: GOOGLE_AVATAR, provider: 'google' })}
          disabled={!!loading}
          className="w-full flex items-center justify-center gap-3 py-3 rounded-lg font-semibold text-sm transition-all duration-200 hover:scale-[1.02] active:scale-95 disabled:opacity-60"
          style={{ background: c.surface, border: `1px solid ${c.border}`, color: c.text }}
        >
          {loading === 'google' ? (
            <span className="material-symbols-outlined animate-spin text-base">autorenew</span>
          ) : (
            /* Google coloured G */
            <svg width="18" height="18" viewBox="0 0 48 48">
              <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
              <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
              <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
              <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.18 1.48-4.97 2.31-8.16 2.31-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
              <path fill="none" d="M0 0h48v48H0z"/>
            </svg>
          )}
          Continue with Google
        </button>

        <button
          onClick={() => mockSignIn('github', { name: 'Authentix-544',   email: 'jatin@github.com',  avatar: GITHUB_AVATAR,  provider: 'github' })}
          disabled={!!loading}
          className="w-full flex items-center justify-center gap-3 py-3 rounded-lg font-semibold text-sm transition-all duration-200 hover:scale-[1.02] active:scale-95 disabled:opacity-60"
          style={{ background: '#24292e', border: `1px solid #444`, color: '#fff' }}
        >
          {loading === 'github' ? (
            <span className="material-symbols-outlined animate-spin text-base">autorenew</span>
          ) : (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="#fff">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
            </svg>
          )}
          Continue with GitHub
        </button>
      </div>

      {/* Divider */}
      <div className="flex items-center gap-3 mb-6">
        <div className="flex-1 h-px" style={{ background: c.border }} />
        <span className="text-xs" style={{ color: c.textDim }}>or continue with email</span>
        <div className="flex-1 h-px" style={{ background: c.border }} />
      </div>

      {/* Tab switch */}
      <div
        className="flex p-1 rounded-lg mb-5"
        style={{ background: c.surfaceHigh, border: `1px solid ${c.border}` }}
      >
        {['signin', 'signup'].map(t => (
          <button
            key={t}
            onClick={() => { setTab(t); setError('') }}
            className="flex-1 py-2 rounded-md text-xs font-semibold capitalize transition-all duration-200"
            style={tab === t
              ? { background: c.primary, color: '#fff' }
              : { color: c.textMuted }}
          >
            {t === 'signin' ? 'Sign In' : 'Sign Up'}
          </button>
        ))}
      </div>

      {/* Email form */}
      <form onSubmit={handleEmail} className="flex flex-col gap-3">
        {tab === 'signup' && input({
          placeholder: 'Full name',
          value:       name,
          onChange:    e => setName(e.target.value),
          type:        'text',
        })}
        {input({
          placeholder: 'Email address',
          value:       email,
          onChange:    e => setEmail(e.target.value),
          type:        'email',
        })}
        {input({
          placeholder: tab === 'signin' ? 'Password' : 'Create password',
          value:       password,
          onChange:    e => setPassword(e.target.value),
          type:        'password',
        })}

        {error && (
          <p className="text-xs px-1" style={{ color: c.red }}>{error}</p>
        )}

        {tab === 'signin' && (
          <div className="text-right">
            <button type="button" className="text-xs transition-colors hover:underline" style={{ color: c.primary }}>
              Forgot password?
            </button>
          </div>
        )}

        <button
          type="submit"
          disabled={!!loading}
          className="w-full py-3 rounded-lg font-semibold text-sm transition-all duration-200 hover:opacity-90 active:scale-95 disabled:opacity-60 mt-1 flex items-center justify-center gap-2"
          style={{ background: c.primary, color: '#fff' }}
        >
          {loading === 'email'
            ? <><span className="material-symbols-outlined animate-spin text-base">autorenew</span> Signing in…</>
            : tab === 'signin' ? 'Sign In' : 'Create Account'
          }
        </button>
      </form>

      <p className="text-xs text-center mt-6" style={{ color: c.textDim }}>
        By continuing, you agree to our{' '}
        <Link to="#" className="hover:underline" style={{ color: c.primary }}>Terms</Link>
        {' '}and{' '}
        <Link to="#" className="hover:underline" style={{ color: c.primary }}>Privacy Policy</Link>.
      </p>
    </div>
  )
}

/* ── signed-in profile view ── */
function ProfileView({ c }) {
  const { user, signOut } = useAuth()
  const navigate = useNavigate()

  const stats = [
    { label: 'Total Scans',   value: '128', icon: 'folder_open' },
    { label: 'Fakes Found',   value: '74',  icon: 'warning'     },
    { label: 'Real Verified', value: '54',  icon: 'verified'    },
  ]

  const providerIcon = { google: '🔵 Google', github: '⚫ GitHub', email: '✉️ Email' }

  return (
    <div className="w-full max-w-lg mx-auto">
      {/* Avatar & info */}
      <div className="flex flex-col items-center text-center mb-8">
        <div className="relative mb-4">
          <img
            src={user.avatar}
            alt={user.name}
            className="w-24 h-24 rounded-full object-cover border-4"
            style={{ borderColor: c.primary }}
          />
          <span
            className="absolute bottom-1 right-1 w-4 h-4 rounded-full border-2"
            style={{ background: '#34d399', borderColor: c.bg }}
          />
        </div>
        <h2 className="text-2xl font-black tracking-tight" style={{ color: c.text }}>{user.name}</h2>
        <p className="text-sm mt-1" style={{ color: c.textDim }}>{user.email}</p>
        <span
          className="mt-2 text-xs px-3 py-1 rounded-full font-medium"
          style={{ background: c.primaryBg, color: c.primary, border: `1px solid ${c.primaryBorder}` }}
        >
          {providerIcon[user.provider] || '✉️ Email'}
        </span>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-3 mb-6">
        {stats.map(s => (
          <div
            key={s.label}
            className="flex flex-col items-center p-4 rounded-xl gap-1"
            style={{ background: c.surface, border: `1px solid ${c.border}` }}
          >
            <span className="material-symbols-outlined text-lg" style={{ color: c.primary }}>{s.icon}</span>
            <p className="text-2xl font-black" style={{ color: c.text }}>{s.value}</p>
            <p className="text-xs text-center" style={{ color: c.textDim }}>{s.label}</p>
          </div>
        ))}
      </div>

      {/* Quick links */}
      <div
        className="rounded-xl overflow-hidden mb-5"
        style={{ border: `1px solid ${c.border}` }}
      >
        {[
          { icon: 'folder_open',  label: 'My Reports',    to: '/reports'  },
          { icon: 'history',      label: 'Activity Log',  to: '/history'  },
          { icon: 'tune',         label: 'Settings',      to: '/settings' },
        ].map((item, i, arr) => (
          <Link
            key={item.label}
            to={item.to}
            className="flex items-center justify-between px-5 py-4 transition-colors"
            style={{
              background:   c.surface,
              borderBottom: i < arr.length - 1 ? `1px solid ${c.border}` : 'none',
              color:        c.text,
            }}
            onMouseEnter={e => e.currentTarget.style.background = c.surfaceHigh}
            onMouseLeave={e => e.currentTarget.style.background = c.surface}
          >
            <div className="flex items-center gap-3">
              <span className="material-symbols-outlined text-lg" style={{ color: c.primary }}>{item.icon}</span>
              <span className="text-sm font-medium">{item.label}</span>
            </div>
            <span className="material-symbols-outlined text-base" style={{ color: c.textDim }}>chevron_right</span>
          </Link>
        ))}
      </div>

      {/* Sign out */}
      <button
        onClick={() => { signOut(); navigate('/') }}
        className="w-full py-3 rounded-xl font-semibold text-sm transition-all duration-200 flex items-center justify-center gap-2 hover:opacity-90"
        style={{
          color:      c.red,
          background: c.redBg,
          border:     `1px solid ${c.redBorder}`,
        }}
      >
        <span className="material-symbols-outlined text-base">logout</span>
        Sign Out
      </button>
    </div>
  )
}

/* ── main page ── */
export default function Profile() {
  const { c }    = useTheme()
  const { user } = useAuth()

  return (
    <div
      className="min-h-screen flex flex-col"
      style={{ backgroundColor: c.bg, color: c.text, fontFamily: 'Geist, sans-serif' }}
    >
      <Header />

      <main className="flex-1 flex items-center justify-center px-4 py-24">
        <div
          className="w-full max-w-sm md:max-w-lg p-8 rounded-2xl relative"
          style={{
            background:    c.surface,
            border:        `1px solid ${c.border}`,
            boxShadow:     '0 25px 60px rgba(0,0,0,0.35)',
          }}
        >
          {/* Subtle violet glow top */}
          <div
            className="absolute inset-x-0 top-0 h-px rounded-t-2xl"
            style={{ background: `linear-gradient(to right, transparent, ${c.primary}, transparent)` }}
          />

          {user ? <ProfileView c={c} /> : <SignInView c={c} />}
        </div>
      </main>
    </div>
  )
}
