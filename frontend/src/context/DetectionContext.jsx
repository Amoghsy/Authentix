import { createContext, useContext, useState, useCallback, useEffect } from 'react'
import * as api from '../lib/api'
import { useAuth } from './AuthContext'

const DetectionContext = createContext({
  result: null,
  history: [],
  isAnalyzing: false,
  progress: 0,
  error: null,
  setResult: () => {},
  clearResult: () => {},
  setIsAnalyzing: () => {},
  setProgress: () => {},
  setError: () => {},
  loadHistory: async () => {},
  deleteHistoryItem: async () => {},
})

/**
 * Standardizes flat (database history) and nested (direct API response) structures
 * into a single unified frontend format.
 */
export function normalizeResult(raw) {
  if (!raw) return null

  const id = raw.analysis_id || raw.analysis_uuid
  const prediction = raw.prediction || raw.final_prediction || "Authentic"
  const label = (prediction === "Deepfake" || prediction === "FAKE") ? "FAKE" : "REAL"
  const confidence = raw.confidence !== undefined ? raw.confidence : 0
  const riskLevel = raw.risk_level || "Low"
  const processingTime = raw.processing_time_ms || 0
  const reasoning = Array.isArray(raw.reasoning)
    ? raw.reasoning
    : typeof raw.reasoning === 'string'
      ? JSON.parse(raw.reasoning)
      : []
  const createdAt = raw.created_at || raw.timestamp || new Date().toISOString()

  let videoScore = 0
  let videoIsFake = false
  let audioScore = 0
  let audioIsFake = false
  let lipSyncScore = 0
  let lipSyncIsFake = false
  let fusionScore = 0

  if (raw.video && typeof raw.video === 'object') {
    // Nested structure (direct scan response)
    videoScore = raw.video.score
    videoIsFake = raw.video.is_fake
    audioScore = raw.audio.score
    audioIsFake = raw.audio.is_fake
    lipSyncScore = raw.lip_sync.score
    lipSyncIsFake = raw.lip_sync.is_fake
    fusionScore = raw.fusion ? raw.fusion.fusion_score : confidence
  } else {
    // Flat structure (history record from database)
    videoScore = raw.video_score !== undefined ? raw.video_score : 0
    videoIsFake = raw.video_prediction === "Deepfake" || raw.video_prediction === "FAKE"
    audioScore = raw.audio_score !== undefined ? raw.audio_score : 0
    audioIsFake = raw.audio_prediction === "Deepfake" || raw.audio_prediction === "FAKE"
    lipSyncScore = raw.lip_sync_score !== undefined ? raw.lip_sync_score : 0
    lipSyncIsFake = lipSyncScore >= 0.5
    fusionScore = raw.fusion_score !== undefined ? raw.fusion_score : confidence
  }

  // Derive a file name from original video path or fallback
  let fileName = raw.fileName || 'uploaded_video.mp4'
  if (raw.original_video_path) {
    const parts = raw.original_video_path.split(/[\\/]/)
    fileName = parts[parts.length - 1]
  }

  // Check if audio track was absent from the video
  let audioNoTrack = false
  if (raw.audio && raw.audio.metadata && raw.audio.metadata.no_audio) {
    audioNoTrack = true
  }

  return {
    id,
    label, // "FAKE" or "REAL"
    prediction, // "Deepfake" or "Authentic"
    confidence, // float (0 to 1)
    riskLevel,
    processingTime,
    reasoning,
    createdAt,
    fileName,
    video: {
      score: videoScore,
      isFake: videoIsFake,
      version: raw.video_model_version || "1.0.2"
    },
    audio: {
      score: audioScore,
      isFake: audioIsFake,
      noTrack: audioNoTrack,
      version: raw.audio_model_version || "2.1.0"
    },
    lipSync: {
      score: lipSyncScore,
      isFake: lipSyncIsFake,
      version: raw.sync_model_version || "1.0.0"
    },
    fusion: {
      score: fusionScore
    }
  }
}

export function DetectionProvider({ children }) {
  const { user } = useAuth()
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState(null)

  // Fetch history list from backend
  const loadHistory = useCallback(async () => {
    if (!user) {
      setHistory([])
      return
    }
    try {
      const data = await api.fetchHistory(0, 50)
      const records = data.records || []
      setHistory(records.map(r => normalizeResult(r)))
    } catch (err) {
      console.error('Failed to load history:', err)
    }
  }, [user])

  // Delete history item
  const deleteHistoryItem = useCallback(async (analysisId) => {
    try {
      await api.deleteHistoryItem(analysisId)
      setHistory(prev => prev.filter(item => item.id !== analysisId))
      if (result && result.id === analysisId) {
        setResult(null)
      }
    } catch (err) {
      console.error('Failed to delete history item:', err)
      throw err
    }
  }, [result])

  const clearResult = useCallback(() => {
    setResult(null)
    setError(null)
    setProgress(0)
  }, [])

  // Auto-fetch history on login status changes
  useEffect(() => {
    if (user) {
      loadHistory()
    } else {
      setHistory([])
    }
  }, [user, loadHistory])

  return (
    <DetectionContext.Provider
      value={{
        result, history, isAnalyzing, progress, error,
        setResult, clearResult, setIsAnalyzing, setProgress, setError,
        loadHistory, deleteHistoryItem
      }}
    >
      {children}
    </DetectionContext.Provider>
  )
}

export const useDetection = () => useContext(DetectionContext)
