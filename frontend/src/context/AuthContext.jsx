import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import * as api from '../lib/api'

const AuthContext = createContext({
  user: null,
  loading: true,
  signIn: async () => {},
  signUp: async () => {},
  signOut: async () => {},
  updateUserProfile: async () => {},
})

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  // Fetch the current user profile from backend
  const fetchUserProfile = useCallback(async () => {
    try {
      const profile = await api.getMe()
      setUser({
        uuid: profile.uuid,
        name: profile.name || 'Anonymous User',
        email: profile.email,
        role: profile.role,
        avatar: `https://api.dicebear.com/7.x/initials/svg?seed=${encodeURIComponent(profile.name || profile.email)}&backgroundColor=7c3aed&textColor=ffffff`,
        provider: 'email',
        createdAt: profile.created_at,
        lastLogin: profile.last_login
      })
    } catch {
      api.clearAuthTokens()
      setUser(null)
    } finally {
      setLoading(false)
    }
  }, [])

  // Restore session on mount
  useEffect(() => {
    const accessToken = localStorage.getItem('ax_access_token')
    if (accessToken) {
      fetchUserProfile()
    } else {
      setLoading(false)
    }

    // Handle token expiration events dispatched by api.js
    const handleAuthExpired = () => {
      setUser(null)
    }
    window.addEventListener('auth-expired', handleAuthExpired)
    return () => window.removeEventListener('auth-expired', handleAuthExpired)
  }, [fetchUserProfile])

  // Login handler
  const signIn = async (email, password) => {
    setLoading(true)
    try {
      await api.login(email, password)
      await fetchUserProfile()
    } catch (err) {
      setLoading(false)
      throw err
    }
  }

  // Register handler
  const signUp = async (name, email, password) => {
    setLoading(true)
    try {
      await api.register(name, email, password)
      await fetchUserProfile()
    } catch (err) {
      setLoading(false)
      throw err
    }
  }

  // Logout handler
  const signOut = async () => {
    setLoading(true)
    try {
      await api.logout()
    } finally {
      setUser(null)
      setLoading(false)
    }
  }

  // Profile update handler
  const updateUserProfile = async ({ name, currentPassword, newPassword }) => {
    const profile = await api.updateProfile({
      name,
      current_password: currentPassword,
      new_password: newPassword
    })
    setUser(prev => prev ? {
      ...prev,
      name: profile.name || prev.name,
      avatar: `https://api.dicebear.com/7.x/initials/svg?seed=${encodeURIComponent(profile.name || profile.email)}&backgroundColor=7c3aed&textColor=ffffff`
    } : null)
    return profile
  }

  return (
    <AuthContext.Provider value={{ user, loading, signIn, signUp, signOut, updateUserProfile }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
