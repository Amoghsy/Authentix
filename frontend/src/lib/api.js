/**
 * api.js — Authentix API client
 *
 * All backend calls go through the Vite dev proxy:
 *   /api/detect  →  http://127.0.0.1:8000/detect
 */

const API_BASE = '/api'

/**
 * Upload a video file to the backend for deepfake detection.
 * @param {File} file — The video file (mp4, mov, etc.)
 * @param {(progress: number) => void} [onProgress] — optional upload progress callback (0–100)
 * @returns {Promise<object>} — The FinalDetectionResponse from the backend
 */
export async function detectVideo(file, onProgress) {
  const formData = new FormData()
  formData.append('video_file', file)

  // Use XMLHttpRequest for upload progress tracking
  if (onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()

      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          onProgress(Math.round((e.loaded / e.total) * 100))
        }
      })

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
      xhr.open('POST', `${API_BASE}/detect`)
      xhr.send(formData)
    })
  }

  // Simple fetch fallback (no progress)
  const res = await fetch(`${API_BASE}/detect`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    let detail = `Server error (${res.status})`
    try {
      const err = await res.json()
      detail = err.detail || detail
    } catch { /* ignore */ }
    throw new Error(detail)
  }

  return res.json()
}

/**
 * Health check — tries GET / and sees if the backend responds.
 */
export async function healthCheck() {
  try {
    const res = await fetch(`${API_BASE}/`, { redirect: 'manual' })
    return res.status < 500
  } catch {
    return false
  }
}
