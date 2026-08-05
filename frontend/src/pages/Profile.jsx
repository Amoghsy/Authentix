import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { useTheme } from '../context/ThemeContext'
import { useDetection } from '../context/DetectionContext'
import Header from '../components/Header'

/* ── signed-out view ── */
function SignInView({ c }) {
  const { signIn, signUp } = useAuth()
  const navigate = useNavigate()

  const [tab, setTab] = useState('signin') // 'signin' | 'signup'
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [name, setName] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const validatePassword = (pwd) => {
    const missing = []
    if (pwd.length < 8) missing.push('at least 8 characters')
    if (!/[A-Z]/.test(pwd)) missing.push('one uppercase letter')
    if (!/[a-z]/.test(pwd)) missing.push('one lowercase letter')
    if (!/[0-9]/.test(pwd)) missing.push('one digit')
    if (!/[!@#$%^&*()_+\-=[\]{}|;':",./\<>?]/.test(pwd)) missing.push('one special character (!@#$...)')
    return missing
  }

  const handleEmail = async (e) => {
    e.preventDefault()
    setError('')
    if (!email || !password) {
      setError('Please fill in all fields.')
      return
    }
    if (tab === 'signup' && !name) {
      setError('Please enter your name.')
      return
    }
    if (tab === 'signup') {
      const missing = validatePassword(password)
      if (missing.length > 0) {
        setError(`Password must contain: ${missing.join(', ')}.`)
        return
      }
    }
    setLoading(true)

    try {
      if (tab === 'signin') {
        await signIn(email, password)
      } else {
        await signUp(name, email, password)
      }
      navigate('/detect')
    } catch (err) {
      setError(err.message || 'Authentication failed. Please check your credentials.')
    } finally {
      setLoading(false)
    }
  }

  const input = (props) => (
    <input
      {...props}
      className="w-full px-4 py-3 rounded-lg text-sm outline-none transition-colors"
      style={{
        background: c.surfaceHigh,
        border: `1px solid ${c.border}`,
        color: c.text,
        caretColor: c.primary,
      }}
      onFocus={e => e.target.style.borderColor = c.primary}
      onBlur={e => e.target.style.borderColor = c.border}
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
          value: name,
          onChange: e => setName(e.target.value),
          type: 'text',
        })}
        {input({
          placeholder: 'Email address',
          value: email,
          onChange: e => setEmail(e.target.value),
          type: 'email',
        })}
        {input({
          placeholder: tab === 'signin' ? 'Password' : 'Password',
          value: password,
          onChange: e => setPassword(e.target.value),
          type: 'password',
        })}
        {tab === 'signup' && (
          <p className="text-xs px-1" style={{ color: '#71717a' }}>
            Must have ≥8 chars, 1 uppercase, 1 lowercase, 1 digit, 1 special character (e.g. <code>Abc@1234</code>)
          </p>
        )}

        {error && (
          <p className="text-xs px-1 text-left leading-relaxed" style={{ color: c.red }}>{error}</p>
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full py-3 rounded-lg font-semibold text-sm transition-all duration-200 hover:opacity-90 active:scale-95 disabled:opacity-60 mt-1 flex items-center justify-center gap-2"
          style={{ background: c.primary, color: '#fff' }}
        >
          {loading
            ? <><div className="w-4 h-4 border-2 border-t-transparent rounded-full animate-spin" /> Authenticating…</>
            : tab === 'signin' ? 'Sign In' : 'Create Account'
          }
        </button>
      </form>
    </div>
  )
}

/* ── signed-in profile view ── */
function ProfileView({ c }) {
  const { user, signOut, updateUserProfile } = useAuth()
  const { history } = useDetection()
  const navigate = useNavigate()

  // Profile update fields
  const [editMode, setEditMode] = useState(false)
  const [updateName, setUpdateName] = useState(user?.name || '')
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [updating, setUpdating] = useState(false)
  const [updateError, setUpdateError] = useState('')
  const [updateSuccess, setUpdateSuccess] = useState('')

  // Calculate real stats from database
  const totalScans = history.length
  const fakes = history.filter(item => item.label === 'FAKE').length
  const reals = history.filter(item => item.label === 'REAL').length

  const stats = [
    { label: 'Total Scans', value: totalScans.toString(), icon: 'folder_open' },
    { label: 'Fakes Found', value: fakes.toString(), icon: 'warning' },
    { label: 'Real Verified', value: reals.toString(), icon: 'verified' },
  ]

  const handleUpdate = async (e) => {
    e.preventDefault()
    setUpdateError('')
    setUpdateSuccess('')
    setUpdating(true)

    try {
      await updateUserProfile({
        name: updateName,
        currentPassword: currentPassword || undefined,
        newPassword: newPassword || undefined
      })
      setUpdateSuccess('Profile updated successfully!')
      setCurrentPassword('')
      setNewPassword('')
      setTimeout(() => setEditMode(false), 1500)
    } catch (err) {
      setUpdateError(err.message || 'Failed to update profile.')
    } finally {
      setUpdating(false)
    }
  }

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
        <p className="text-sm mt-1" style={{ color: c.textDim }}>{user.email} • {user.role}</p>
      </div>

      {/* Edit Profile Form */}
      {editMode ? (
        <form onSubmit={handleUpdate} className="flex flex-col gap-3 mb-6 p-4 rounded-xl border" style={{ background: c.surfaceHigh, borderColor: c.border }}>
          <h3 className="text-sm font-bold uppercase tracking-wider mb-2" style={{ color: c.text }}>Edit Profile</h3>
          
          <div className="space-y-1.5 text-left">
            <label className="text-xs font-semibold" style={{ color: c.textDim }}>Display Name</label>
            <input
              type="text"
              placeholder="Display name"
              value={updateName}
              onChange={e => setUpdateName(e.target.value)}
              className="w-full px-3 py-2 rounded-lg text-xs outline-none"
              style={{ background: c.surface, border: `1px solid ${c.border}`, color: c.text }}
            />
          </div>

          <div className="space-y-1.5 text-left">
            <label className="text-xs font-semibold" style={{ color: c.textDim }}>Current Password (required for changes)</label>
            <input
              type="password"
              placeholder="••••••••"
              value={currentPassword}
              onChange={e => setCurrentPassword(e.target.value)}
              className="w-full px-3 py-2 rounded-lg text-xs outline-none"
              style={{ background: c.surface, border: `1px solid ${c.border}`, color: c.text }}
            />
          </div>

          <div className="space-y-1.5 text-left">
            <label className="text-xs font-semibold" style={{ color: c.textDim }}>New Password (optional)</label>
            <input
              type="password"
              placeholder="••••••••"
              value={newPassword}
              onChange={e => setNewPassword(e.target.value)}
              className="w-full px-3 py-2 rounded-lg text-xs outline-none"
              style={{ background: c.surface, border: `1px solid ${c.border}`, color: c.text }}
            />
          </div>

          {updateError && <p className="text-xs px-1 text-left" style={{ color: c.red }}>{updateError}</p>}
          {updateSuccess && <p className="text-xs px-1 text-left" style={{ color: c.green }}>{updateSuccess}</p>}

          <div className="flex gap-2.5 mt-2">
            <button
              type="submit"
              disabled={updating}
              className="flex-1 py-2 rounded-lg font-semibold text-xs text-white flex items-center justify-center gap-1.5"
              style={{ background: c.primary }}
            >
              {updating && <div className="w-3.5 h-3.5 border-2 border-t-transparent rounded-full animate-spin" />}
              Save Changes
            </button>
            <button
              type="button"
              onClick={() => { setEditMode(false); setUpdateError(''); setUpdateSuccess('') }}
              className="flex-1 py-2 rounded-lg font-semibold text-xs border"
              style={{ borderColor: c.border, color: c.text }}
            >
              Cancel
            </button>
          </div>
        </form>
      ) : (
        <>
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
                <p className="text-xs text-center text-zinc-500" style={{ color: c.textDim }}>{s.label}</p>
              </div>
            ))}
          </div>

          {/* Quick links */}
          <div
            className="rounded-xl overflow-hidden mb-5"
            style={{ border: `1px solid ${c.border}` }}
          >
            {[
              { icon: 'folder_open', label: 'My Reports', to: '/reports' },
              { icon: 'history', label: 'Activity Log', to: '/history' },
            ].map((item, i, arr) => (
              <Link
                key={item.label}
                to={item.to}
                className="flex items-center justify-between px-5 py-4 transition-colors"
                style={{
                  background: c.surface,
                  borderBottom: i < arr.length - 1 ? `1px solid ${c.border}` : 'none',
                  color: c.text,
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

            <button
              onClick={() => setEditMode(true)}
              className="w-full flex items-center justify-between px-5 py-4 transition-colors text-left"
              style={{ background: c.surface, color: c.text }}
              onMouseEnter={e => e.currentTarget.style.background = c.surfaceHigh}
              onMouseLeave={e => e.currentTarget.style.background = c.surface}
            >
              <div className="flex items-center gap-3">
                <span className="material-symbols-outlined text-lg" style={{ color: c.primary }}>edit</span>
                <span className="text-sm font-medium">Edit Profile Settings</span>
              </div>
              <span className="material-symbols-outlined text-base" style={{ color: c.textDim }}>chevron_right</span>
            </button>
          </div>
        </>
      )}

      {/* Sign out */}
      <button
        onClick={() => { signOut(); navigate('/') }}
        className="w-full py-3 rounded-xl font-semibold text-sm transition-all duration-200 flex items-center justify-center gap-2 hover:opacity-90 animate-fade-in"
        style={{
          color: c.red,
          background: c.redBg,
          border: `1px solid ${c.redBorder}`,
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
  const { c } = useTheme()
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
            background: c.surface,
            border: `1px solid ${c.border}`,
            boxShadow: '0 25px 60px rgba(0,0,0,0.35)',
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
