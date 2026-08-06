/**
 * api.js — Authentix API client
 * Connects directly to the backend through the Vite proxy.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api'

export function clearAuthTokens() {
  localStorage.removeItem('ax_access_token')
  localStorage.removeItem('ax_refresh_token')
}

/**
 * Decodes a JWT token payload locally to inspect expiry
 */
function isTokenExpired(token) {
  try {
    const base64Url = token.split('.')[1]
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/')
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split('')
        .map(c => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    )
    const payload = JSON.parse(jsonPayload)
    const exp = payload.exp
    const now = Date.now() / 1000
    // Return true if token is expired or expires in the next 10 seconds
    return exp - now < 10
  } catch {
    return true
  }
}

/**
 * Requests a new access token using the stored refresh token
 */
export async function refreshTokens() {
  const refreshToken = localStorage.getItem('ax_refresh_token')
  if (!refreshToken) {
    clearAuthTokens()
    throw new Error('Session expired')
  }

  const res = await fetch(`${API_BASE}/auth/refresh`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh_token: refreshToken }),
  })

  if (!res.ok) {
    clearAuthTokens()
    window.dispatchEvent(new CustomEvent('auth-expired'))
    throw new Error('Session expired')
  }

  const data = await res.json()
  localStorage.setItem('ax_access_token', data.access_token)
  localStorage.setItem('ax_refresh_token', data.refresh_token)
  return data.access_token
}

/**
 * Returns a valid, non-expired access token. Refreshes if needed.
 */
export async function getValidAccessToken() {
  const accessToken = localStorage.getItem('ax_access_token')
  const refreshToken = localStorage.getItem('ax_refresh_token')

  if (!accessToken || !refreshToken) {
    return null
  }

  if (isTokenExpired(accessToken)) {
    try {
      return await refreshTokens()
    } catch {
      return null
    }
  }

  return accessToken
}

/**
 * Wrapper around fetch that automatically appends the Bearer token
 * and handles token refreshing if a 401 is encountered.
 */
export async function fetchWithAuth(url, options = {}) {
  options.headers = options.headers || {}

  const token = await getValidAccessToken()
  if (token) {
    options.headers['Authorization'] = `Bearer ${token}`
  }

  let res = await fetch(url, options)

  // If unauthorized, attempt to refresh and try again once
  if (res.status === 401 && token) {
    try {
      const newToken = await refreshTokens()
      options.headers['Authorization'] = `Bearer ${newToken}`
      res = await fetch(url, options)
    } catch {
      window.dispatchEvent(new CustomEvent('auth-expired'))
      throw new Error('Unauthorized')
    }
  }

  return res
}

/* ────────────────────────────────────────────────────────────────────────── */
/*                                DETECTION API                               */
/* ────────────────────────────────────────────────────────────────────────── */

/**
 * Upload a video file to the backend for deepfake detection.
 * Requires a valid access token in headers.
 */
export async function detectVideo(file, onProgress) {
  const token = await getValidAccessToken()
  const formData = new FormData()
  formData.append('file', file) // Backend expects 'file' parameter

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()

    if (onProgress) {
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          onProgress(Math.round((e.loaded / e.total) * 100))
        }
      })
    }

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText))
        } catch {
          reject(new Error('Invalid JSON response from server'))
        }
      } else {
        let detail = `Server error (${xhr.status})`
        try {
          const err = JSON.parse(xhr.responseText)
          detail = err.detail || detail
        } catch { /* ignore */ }
        reject(new Error(detail))
      }
    })

    xhr.addEventListener('error', () => reject(new Error('Network error — is the backend running?')))
    xhr.addEventListener('timeout', () => reject(new Error('Request timed out — video may be too large')))

    xhr.timeout = 300_000 // 5 minutes
    xhr.open('POST', `${API_BASE}/analyze`) // Backend path is /api/analyze

    if (token) {
      xhr.setRequestHeader('Authorization', `Bearer ${token}`)
    }

    xhr.send(formData)
  })
}

/* ────────────────────────────────────────────────────────────────────────── */
/*                             AUTHENTICATION API                             */
/* ────────────────────────────────────────────────────────────────────────── */

function formatApiError(err, defaultMsg) {
  if (err && err.detail) {
    if (typeof err.detail === 'string') {
      return err.detail
    }
    if (Array.isArray(err.detail)) {
      return err.detail.map(d => {
        const field = d.loc && d.loc.length > 0 ? d.loc[d.loc.length - 1] : ''
        return `${field ? field + ': ' : ''}${d.msg}`
      }).join('; ')
    }
  }
  return defaultMsg
}

export async function login(email, password) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(formatApiError(err, 'Login failed'))
  }

  const data = await res.json()
  localStorage.setItem('ax_access_token', data.access_token)
  localStorage.setItem('ax_refresh_token', data.refresh_token)
  return data
}

export async function register(name, email, password) {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, email, password }),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(formatApiError(err, 'Registration failed'))
  }

  const data = await res.json()
  localStorage.setItem('ax_access_token', data.access_token)
  localStorage.setItem('ax_refresh_token', data.refresh_token)
  return data
}

export async function getMe() {
  const res = await fetchWithAuth(`${API_BASE}/auth/me`)
  if (!res.ok) {
    throw new Error('Failed to fetch user profile')
  }
  return res.json()
}

export async function updateProfile({ name, current_password, new_password }) {
  const payload = {}
  if (name) payload.name = name
  if (current_password) payload.current_password = current_password
  if (new_password) payload.new_password = new_password

  const res = await fetchWithAuth(`${API_BASE}/auth/profile`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(formatApiError(err, 'Failed to update profile'))
  }
  return res.json()
}

export async function logout() {
  const refreshToken = localStorage.getItem('ax_refresh_token')
  if (refreshToken) {
    try {
      await fetchWithAuth(`${API_BASE}/auth/logout`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken }),
      })
    } catch { /* ignore and clean local state */ }
  }
  clearAuthTokens()
}

/* ────────────────────────────────────────────────────────────────────────── */
/*                                 HISTORY API                                */
/* ────────────────────────────────────────────────────────────────────────── */

export async function fetchHistory(skip = 0, limit = 50) {
  const res = await fetchWithAuth(`${API_BASE}/history?skip=${skip}&limit=${limit}`)
  if (!res.ok) {
    throw new Error('Failed to fetch analysis history')
  }
  return res.json()
}

export async function deleteHistoryItem(analysisId) {
  const res = await fetchWithAuth(`${API_BASE}/history/${analysisId}`, {
    method: 'DELETE',
  })
  if (!res.ok) {
    throw new Error('Failed to delete history record')
  }
  return res.json()
}

export async function fetchReport(analysisId) {
  const res = await fetchWithAuth(`${API_BASE}/report/${analysisId}`)
  if (!res.ok) {
    throw new Error('Failed to fetch analysis report details')
  }
  return res.json()
}

/* ────────────────────────────────────────────────────────────────────────── */
/*                                SYSTEM STATUS                               */
/* ────────────────────────────────────────────────────────────────────────── */

/**
 * Health check — tries GET /health (proxied directly)
 */
export async function healthCheck() {
  try {
    const res = await fetch('/health')
    return res.ok
  } catch {
    return false
  }
}

/**
 * Fetch system and AI model versions — GET /api/version
 */
export async function fetchVersion() {
  try {
    const res = await fetch(`${API_BASE}/version`)
    if (!res.ok) return null
    return res.json()
  } catch {
    return null
  }
}
