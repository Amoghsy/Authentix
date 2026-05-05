import { createContext, useContext, useState } from 'react'

const AuthContext = createContext({
  user: null,
  signIn: () => {},
  signOut: () => {},
})

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)

  const signIn = (userData) => setUser(userData)
  const signOut = () => setUser(null)

  return (
    <AuthContext.Provider value={{ user, signIn, signOut }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
