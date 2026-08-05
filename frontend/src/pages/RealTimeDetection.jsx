import { useState, useRef, useCallback, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import Sidebar from '../components/Sidebar'
import Header from '../components/Header'
import { useDetection, normalizeResult } from '../context/DetectionContext'
import { useAuth } from '../context/AuthContext'
import { detectVideo } from '../lib/api'

export default function RealTimeDetection() {
  const navigate = useNavigate()
  const fileInputRef = useRef(null)
  const dropZoneRef = useRef(null)

  const { user, loading: authLoading } = useAuth()
  const {
    setResult, loadHistory,
    isAnalyzing, setIsAnalyzing,
    progress, setProgress,
    error, setError,
  } = useDetection()

  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [phase, setPhase] = useState('idle') // idle | uploading | processing | done | error

  // If user logs out during this page, clean file select
  useEffect(() => {
    if (!user) {
      setSelectedFile(null)
      setPreview(null)
    }
  }, [user])

  const handleFileSelect = useCallback((file) => {
    if (!file) return
    const validTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm']
    if (!validTypes.includes(file.type) && !file.name.match(/\.(mp4|mov|avi|webm)$/i)) {
      setError('Please select a valid video file (MP4, MOV, AVI, or WebM)')
      return
    }
    setSelectedFile(file)
    setError(null)
    setPhase('idle')

    // Generate video thumbnail preview
    const url = URL.createObjectURL(file)
    setPreview(url)
  }, [setError])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer?.files?.[0]
    handleFileSelect(file)
  }, [handleFileSelect])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setDragOver(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setDragOver(false)
  }, [])

  const startAnalysis = useCallback(async () => {
    if (!selectedFile || isAnalyzing) return

    setIsAnalyzing(true)
    setError(null)
    setProgress(0)
    setPhase('uploading')

    try {
      const rawResult = await detectVideo(selectedFile, (pct) => {
        setProgress(pct)
        if (pct >= 100) {
          setPhase('processing')
        }
      })

      const normalized = normalizeResult(rawResult)
      setPhase('done')
      setResult(normalized)
      
      // Refresh user's database history in context
      loadHistory()

      // Navigate to results page after brief animation delay
      setTimeout(() => {
        navigate('/analysis')
      }, 800)
    } catch (err) {
      setPhase('error')
      setError(err.message || 'Analysis failed. Is the backend server running?')
    } finally {
      setIsAnalyzing(false)
    }
  }, [selectedFile, isAnalyzing, setIsAnalyzing, setError, setProgress, setResult, loadHistory, navigate])

  const resetState = useCallback(() => {
    setSelectedFile(null)
    setPreview(null)
    setError(null)
    setProgress(0)
    setPhase('idle')
    if (preview) URL.revokeObjectURL(preview)
  }, [preview, setError, setProgress])

  const phaseLabel = {
    idle: 'Ready to Scan',
    uploading: `Uploading… ${progress}%`,
    processing: 'Running AI Pipeline…',
    done: 'Analysis Complete ✓',
    error: 'Error',
  }

  const phaseColor = {
    idle: '#34d399',
    uploading: '#a78bfa',
    processing: '#f59e0b',
    done: '#34d399',
    error: '#ef4444',
  }

  if (authLoading) {
    return (
      <div className="flex h-screen w-screen items-center justify-center" style={{ backgroundColor: '#09090b', color: '#fafafa' }}>
        <div className="w-10 h-10 rounded-full border-4 border-t-transparent animate-spin" style={{ borderColor: '#a78bfa' }} />
      </div>
    )
  }

  // Enforce authentication
  if (!user) {
    return (
      <div className="flex h-screen overflow-hidden" style={{ backgroundColor: '#09090b', color: '#fafafa', fontFamily: 'Geist, sans-serif' }}>
        <Sidebar />
        <div className="flex-1 flex flex-col">
          <Header />
          <main className="flex-1 flex items-center justify-center p-6 pt-20">
            <div className="max-w-md w-full p-8 rounded-2xl text-center space-y-6" style={{ background: '#121215', border: '1px solid #27272a', boxShadow: '0 25px 50px rgba(0,0,0,0.5)' }}>
              <div className="w-16 h-16 rounded-full mx-auto flex items-center justify-center" style={{ background: 'rgba(167,139,250,0.1)', border: '1px solid rgba(167,139,250,0.2)' }}>
                <span className="material-symbols-outlined text-3xl" style={{ color: '#a78bfa' }}>lock</span>
              </div>
              <div>
                <h2 className="text-2xl font-bold tracking-tight mb-2">Authentication Required</h2>
                <p className="text-sm" style={{ color: '#a1a1aa' }}>
                  You must sign in or create an account to use the multimodal fusion deepfake detection engine.
                </p>
              </div>
              <button
                onClick={() => navigate('/profile')}
                className="w-full py-3 rounded-lg font-semibold tracking-wide transition-all duration-200"
                style={{ background: '#a78bfa', color: '#0a0012' }}
              >
                Go to Login / Profile
              </button>
            </div>
          </main>
        </div>
      </div>
    )
  }

  return (
    <div
      className="flex h-screen overflow-hidden"
      style={{ backgroundColor: '#09090b', color: '#fafafa', fontFamily: 'Geist, sans-serif' }}
    >
      <Sidebar />

      <div className="flex-1 flex flex-col h-full overflow-hidden">
        <Header />
        <main className="flex-1 flex flex-col h-full relative overflow-hidden pt-14">
        {/* Mobile top bar */}
        <header
          className="md:hidden fixed top-0 w-full z-50 flex justify-between items-center px-6 py-3 border-b border-zinc-800"
          style={{ background: 'rgba(9,9,11,0.8)', backdropFilter: 'blur(12px)' }}
        >
          <span className="text-xl font-bold tracking-tighter text-violet-400">Authentix</span>
        </header>

        {/* Scrollable canvas */}
        <div className="flex-1 overflow-y-auto p-6 md:p-10 mt-16 md:mt-0">
          <div className="max-w-6xl mx-auto h-full flex flex-col lg:flex-row gap-8">

            {/* Left column: Video preview + controls */}
            <div className="flex-1 flex flex-col gap-6">
              {/* Page title + status */}
              <div className="flex justify-between items-end">
                <div>
                  <h1 className="text-3xl font-bold tracking-tight mb-1" style={{ color: '#fafafa' }}>
                    Deepfake Detection
                  </h1>
                  <p className="text-sm" style={{ color: '#a1a1aa' }}>
                    Upload a video to analyze fake or real media via multimodal fusion.
                  </p>
                </div>
                <div
                  className="flex items-center gap-2 px-3 py-1.5 rounded-full"
                  style={{ background: '#121215', border: '1px solid #27272a' }}
                >
                  <span
                    className={`w-2 h-2 rounded-full ${phase === 'processing' || phase === 'uploading' ? 'animate-pulse' : ''}`}
                    style={{ backgroundColor: phaseColor[phase] }}
                  />
                  <span
                    className="text-xs font-medium uppercase tracking-wide"
                    style={{ color: phaseColor[phase] }}
                  >
                    {phaseLabel[phase]}
                  </span>
                </div>
              </div>

              {/* Video preview / drop zone */}
              <div
                ref={dropZoneRef}
                className={`relative w-full overflow-hidden flex items-center justify-center group cursor-pointer transition-all duration-300 ${dragOver ? 'scale-[1.01]' : ''}`}
                style={{
                  aspectRatio: '16/9',
                  borderRadius: '0.75rem',
                  background: '#09090b',
                  border: `2px ${dragOver ? 'solid' : selectedFile ? 'solid' : 'dashed'} ${dragOver ? '#a78bfa' : selectedFile ? '#27272a' : '#3f3f46'}`,
                  boxShadow: dragOver ? '0 0 40px rgba(167,139,250,0.15)' : '0 25px 50px -12px rgba(0,0,0,0.8)',
                }}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => { if (!isAnalyzing) fileInputRef.current?.click() }}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/mp4,video/quicktime,video/x-msvideo,video/webm,.mp4,.mov,.avi,.webm"
                  className="hidden"
                  onChange={(e) => handleFileSelect(e.target.files?.[0])}
                />

                {preview ? (
                  <video
                    src={preview}
                    className="absolute inset-0 w-full h-full object-cover opacity-70 group-hover:opacity-90 transition-opacity duration-500"
                    muted
                    loop
                    autoPlay
                    playsInline
                  />
                ) : (
                  <div className="absolute inset-0 w-full h-full flex items-center justify-center"
                    style={{ background: 'radial-gradient(ellipse at center, #121215 0%, #09090b 100%)' }}
                  />
                )}

                {/* Scanning overlay */}
                {(phase === 'processing') && (
                  <div
                    className="absolute w-full h-1/4 top-1/3 border-y pointer-events-none"
                    style={{
                      background: 'linear-gradient(to bottom, transparent, rgba(167,139,250,0.15), transparent)',
                      borderColor: 'rgba(167,139,250,0.3)',
                      animation: 'scan 2s ease-in-out infinite',
                    }}
                  />
                )}

                {/* Corner markers */}
                <div className="absolute top-4 left-4 w-6 h-6 border-t-2 border-l-2" style={{ borderColor: 'rgba(167,139,250,0.5)' }} />
                <div className="absolute top-4 right-4 w-6 h-6 border-t-2 border-r-2" style={{ borderColor: 'rgba(167,139,250,0.5)' }} />
                <div className="absolute bottom-4 left-4 w-6 h-6 border-b-2 border-l-2" style={{ borderColor: 'rgba(167,139,250,0.5)' }} />
                <div className="absolute bottom-4 right-4 w-6 h-6 border-b-2 border-r-2" style={{ borderColor: 'rgba(167,139,250,0.5)' }} />

                {/* Centre content */}
                {!preview && (
                  <div className="relative z-10 flex flex-col items-center gap-3 pointer-events-none">
                    <div
                      className="w-16 h-16 rounded-full flex items-center justify-center"
                      style={{
                        border: '1px solid #27272a',
                        background: 'rgba(9,9,11,0.6)',
                        backdropFilter: 'blur(4px)',
                      }}
                    >
                      <span className="material-symbols-outlined text-3xl" style={{ color: '#a78bfa' }}>
                        upload_file
                      </span>
                    </div>
                    <div className="text-center">
                      <p className="text-sm font-semibold" style={{ color: '#fafafa' }}>
                        Drop video here or click to browse
                      </p>
                      <p className="text-xs mt-1" style={{ color: '#71717a' }}>
                        MP4, MOV, AVI, WebM • Max 200MB
                      </p>
                    </div>
                  </div>
                )}

                {/* Upload progress bar */}
                {phase === 'uploading' && (
                  <div className="absolute bottom-0 left-0 right-0 h-1.5 z-20" style={{ background: '#121215' }}>
                    <div
                      className="h-full transition-all duration-300 ease-out"
                      style={{
                        width: `${progress}%`,
                        background: 'linear-gradient(to right, #7c3aed, #a78bfa)',
                        boxShadow: '0 0 12px rgba(167,139,250,0.5)',
                      }}
                    />
                  </div>
                )}

                {/* Processing spinner overlay */}
                {phase === 'processing' && (
                  <div className="absolute inset-0 z-10 flex flex-col items-center justify-center"
                    style={{ background: 'rgba(9,9,11,0.7)', backdropFilter: 'blur(6px)' }}
                  >
                    <div className="w-14 h-14 rounded-full border-4 border-t-transparent animate-spin mb-4"
                      style={{ borderColor: '#a78bfa', borderTopColor: 'transparent' }}
                    />
                    <p className="text-sm font-semibold" style={{ color: '#fafafa' }}>
                      Executing Modality Predictors…
                    </p>
                    <p className="text-xs mt-1" style={{ color: '#71717a' }}>
                      Video Classifier • Audio Spectrogram • Lip-Sync Preprocessing • Fusion Engine
                    </p>
                  </div>
                )}

                {/* Done checkmark overlay */}
                {phase === 'done' && (
                  <div className="absolute inset-0 z-10 flex flex-col items-center justify-center"
                    style={{ background: 'rgba(9,9,11,0.7)', backdropFilter: 'blur(6px)' }}
                  >
                    <span className="material-symbols-outlined text-5xl mb-2"
                      style={{ color: '#34d399', fontVariationSettings: "'FILL' 1" }}
                    >
                      check_circle
                    </span>
                    <p className="text-sm font-semibold" style={{ color: '#34d399' }}>
                      Analysis Complete — Redirecting…
                    </p>
                  </div>
                )}

                {/* Overlay stats */}
                {(phase === 'processing' || phase === 'uploading') && (
                  <div
                    className="absolute top-4 left-1/2 -translate-x-1/2 flex gap-4 text-xs font-mono px-4 py-1.5 rounded-md z-20"
                    style={{
                      color: '#a1a1aa',
                      background: 'rgba(9,9,11,0.8)',
                      backdropFilter: 'blur(12px)',
                      border: '1px solid #27272a',
                    }}
                  >
                    <span>MODALITIES: 3</span>
                    <span>PIPELINE: v1.0</span>
                    <span>ENGINE: Fusion</span>
                  </div>
                )}
              </div>

              {/* Error display */}
              {error && (
                <div
                  className="flex items-center gap-3 p-4 rounded-lg"
                  style={{
                    background: 'rgba(239,68,68,0.08)',
                    border: '1px solid rgba(239,68,68,0.25)',
                  }}
                >
                  <span className="material-symbols-outlined text-lg" style={{ color: '#ef4444', fontVariationSettings: "'FILL' 1" }}>error</span>
                  <div className="flex-1">
                    <p className="text-sm font-medium" style={{ color: '#ef4444' }}>{error}</p>
                    <p className="text-xs mt-0.5" style={{ color: '#a1a1aa' }}>
                      Please make sure the backend server is running and your file is valid.
                    </p>
                  </div>
                  <button onClick={() => setError(null)} style={{ color: '#71717a' }}>
                    <span className="material-symbols-outlined text-base">close</span>
                  </button>
                </div>
              )}

              {/* Control buttons */}
              <div className="flex gap-4">
                <button
                  id="btn-start-analysis"
                  onClick={startAnalysis}
                  disabled={!selectedFile || isAnalyzing}
                  className="flex-1 py-4 font-bold tracking-wide flex items-center justify-center gap-2 transition-all duration-200 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed"
                  style={{
                    background: selectedFile && !isAnalyzing ? '#a78bfa' : '#3f3f46',
                    color: selectedFile && !isAnalyzing ? '#0a0012' : '#71717a',
                  }}
                  onMouseEnter={e => { if (selectedFile && !isAnalyzing) e.currentTarget.style.background = '#c4b5fd' }}
                  onMouseLeave={e => { if (selectedFile && !isAnalyzing) e.currentTarget.style.background = '#a78bfa' }}
                >
                  {isAnalyzing ? (
                    <>
                      <div className="w-5 h-5 rounded-full border-2 border-t-transparent animate-spin"
                        style={{ borderColor: '#0a0012', borderTopColor: 'transparent' }}
                      />
                      Analyzing…
                    </>
                  ) : (
                    <>
                      <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>play_arrow</span>
                      {selectedFile ? 'Start Analysis' : 'Select a Video First'}
                    </>
                  )}
                </button>
                <button
                  id="btn-reset"
                  onClick={resetState}
                  disabled={isAnalyzing}
                  className="flex-1 py-4 font-bold tracking-wide flex items-center justify-center gap-2 transition-colors rounded-lg disabled:opacity-40"
                  style={{ background: 'transparent', border: '1px solid #27272a', color: '#fafafa' }}
                  onMouseEnter={e => e.currentTarget.style.background = '#18181b'}
                  onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                >
                  <span className="material-symbols-outlined">restart_alt</span>
                  Reset
                </button>
              </div>

              {/* File info */}
              {selectedFile && (
                <div
                  className="flex items-center gap-3 p-4 rounded-lg"
                  style={{ background: '#121215', border: '1px solid #27272a' }}
                >
                  <span className="material-symbols-outlined text-xl" style={{ color: '#a78bfa' }}>movie</span>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate" style={{ color: '#fafafa' }}>{selectedFile.name}</p>
                    <p className="text-xs" style={{ color: '#71717a' }}>
                      {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • {selectedFile.type || 'video'}
                    </p>
                  </div>
                  {!isAnalyzing && (
                    <button
                      onClick={(e) => { e.stopPropagation(); resetState() }}
                      className="p-1.5 rounded-md transition-colors"
                      style={{ color: '#71717a' }}
                      onMouseEnter={e => { e.currentTarget.style.color = '#ef4444'; e.currentTarget.style.background = 'rgba(239,68,68,0.1)' }}
                      onMouseLeave={e => { e.currentTarget.style.color = '#71717a'; e.currentTarget.style.background = 'transparent' }}
                    >
                      <span className="material-symbols-outlined text-base">delete</span>
                    </button>
                  )}
                </div>
              )}
            </div>

            {/* Right column: Modalities & Status info */}
            <div className="w-full lg:w-80 flex flex-col gap-6 pt-12 lg:pt-0">
              {/* Quick upload zone (alternative) */}
              <div
                className="p-6 flex flex-col items-center justify-center text-center gap-4 cursor-pointer transition-all duration-200 min-h-48 rounded-xl"
                style={{
                  background: '#121215',
                  border: `1px dashed ${dragOver ? '#a78bfa' : '#27272a'}`,
                }}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => { if (!isAnalyzing) fileInputRef.current?.click() }}
                onMouseEnter={e => e.currentTarget.style.background = '#18181b'}
                onMouseLeave={e => e.currentTarget.style.background = '#121215'}
              >
                <div
                  className="w-12 h-12 rounded-full flex items-center justify-center"
                  style={{ background: '#09090b', border: '1px solid #27272a', color: '#a78bfa' }}
                >
                  <span className="material-symbols-outlined text-2xl">upload_file</span>
                </div>
                <div>
                  <h3 className="text-sm font-semibold mb-1" style={{ color: '#fafafa' }}>Upload Video</h3>
                  <p className="text-xs" style={{ color: '#a1a1aa' }}>
                    Drag and drop MP4 or MOV files here to analyze pre-recorded footage.
                  </p>
                </div>
                <button
                  className="mt-2 text-xs font-semibold px-4 py-2 rounded-md transition-colors"
                  style={{ color: '#a78bfa', border: '1px solid rgba(167,139,250,0.3)' }}
                  onMouseEnter={e => e.currentTarget.style.background = 'rgba(167,139,250,0.1)'}
                  onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                  onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click() }}
                >
                  Select File
                </button>
              </div>

              {/* Modality pipeline panel */}
              <div
                className="p-5 rounded-xl flex flex-col gap-4"
                style={{ background: '#121215', border: '1px solid #27272a' }}
              >
                <h3 className="text-sm font-bold uppercase tracking-wider" style={{ color: '#fafafa' }}>
                  Detection Modalities
                </h3>
                <div className="space-y-4">
                  {[
                    { icon: 'movie', label: 'Video Modality', sub: 'CNN & temporal face classification', version: 'v1.0.2' },
                    { icon: 'graphic_eq', label: 'Audio Modality', sub: 'Librosa spectral sound analysis', version: 'v2.1.0' },
                    { icon: 'record_voice_over', label: 'Lip-Sync Modality', sub: 'Active lip A/V sync detection', version: 'v1.0.0' },
                  ].map((m) => (
                    <div key={m.label} className="flex items-center justify-between">
                      <div className="flex items-center gap-2.5 text-sm" style={{ color: '#a1a1aa' }}>
                        <span className="material-symbols-outlined text-lg" style={{ color: '#a78bfa' }}>{m.icon}</span>
                        <div>
                          <span className="text-xs font-bold block" style={{ color: '#d4d4d8' }}>{m.label}</span>
                          <p className="text-[10px] leading-tight" style={{ color: '#52525b' }}>{m.sub}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <span className="text-xs font-mono block" style={{ color: '#34d399' }}>READY</span>
                        <span className="text-[9px] font-mono" style={{ color: '#52525b' }}>{m.version}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Backend status */}
              <div
                className="p-4 rounded-xl flex items-center gap-3"
                style={{ background: '#121215', border: '1px solid #27272a' }}
              >
                <div className="w-8 h-8 rounded-lg flex items-center justify-center"
                  style={{ background: 'rgba(52,211,153,0.1)' }}
                >
                  <span className="material-symbols-outlined text-sm" style={{ color: '#34d399' }}>dns</span>
                </div>
                <div>
                  <p className="text-xs font-medium" style={{ color: '#fafafa' }}>Backend API Status</p>
                  <p className="text-xs" style={{ color: '#71717a' }}>localhost:8000 • FastAPI</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      <style>{`
        @keyframes scan {
          0%, 100% { transform: translateY(-30px); opacity: 0.4; }
          50% { transform: translateY(30px); opacity: 1; }
        }
      `}</style>
      </div>
    </div>
  )
}
