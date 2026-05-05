import { createContext, useContext, useState, useCallback, useEffect } from 'react'

const DetectionContext = createContext({
  result: null,
  history: [],
  isAnalyzing: false,
  progress: 0,
  error: null,
  setResult: () => {},
  addToHistory: () => {},
  clearResult: () => {},
  setIsAnalyzing: () => {},
  setProgress: () => {},
  setError: () => {},
})

const HISTORY_KEY = 'ax-scan-history'
const MAX_HISTORY = 50

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveHistory(history) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, MAX_HISTORY)))
  } catch { /* quota exceeded — ignore */ }
}

export function DetectionProvider({ children }) {
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState(loadHistory)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState(null)

  const addToHistory = useCallback((scanResult, fileName) => {
    const entry = {
      ...scanResult,
      fileName: fileName || 'unknown.mp4',
      scannedAt: new Date().toISOString(),
    }
    setHistory(prev => {
      const next = [entry, ...prev].slice(0, MAX_HISTORY)
      saveHistory(next)
      return next
    })
  }, [])

  const clearResult = useCallback(() => {
    setResult(null)
    setError(null)
    setProgress(0)
  }, [])

  // Persist history on changes
  useEffect(() => {
    saveHistory(history)
  }, [history])

  return (
    <DetectionContext.Provider
      value={{
        result, history, isAnalyzing, progress, error,
        setResult, addToHistory, clearResult,
        setIsAnalyzing, setProgress, setError,
      }}
    >
      {children}
    </DetectionContext.Provider>
  )
}

export const useDetection = () => useContext(DetectionContext)
