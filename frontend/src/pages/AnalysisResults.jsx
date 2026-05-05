import { useNavigate } from 'react-router-dom'
import Sidebar from '../components/Sidebar'
import { useDetection } from '../context/DetectionContext'

/* ── helpers ── */
const card = {
  background: '#111113',
  border: '1px solid #27272a',
  borderRadius: '0.5rem',
}

function MetricBar({ label, value, color = '#ef4444' }) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100)
  return (
    <div>
      <div className="flex justify-between items-center mb-1.5">
        <span className="text-sm" style={{ color: '#d4d4d8' }}>{label}</span>
        <span className="text-sm font-semibold" style={{ color }}>{value.toFixed(4)}</span>
      </div>
      <div className="h-1 w-full rounded-full" style={{ background: '#27272a' }}>
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  )
}

/* ── page ── */
export default function AnalysisResults() {
  const { result } = useDetection()
  const navigate = useNavigate()

  // If no result available, show empty state
  if (!result) {
    return (
      <div
        className="flex h-screen overflow-hidden"
        style={{ backgroundColor: '#09090b', color: '#fafafa', fontFamily: 'Geist, sans-serif' }}
      >
        <Sidebar />
        <main className="flex-1 overflow-y-auto p-6 md:p-8 flex items-center justify-center" style={{ backgroundColor: '#09090b' }}>
          <div className="flex flex-col items-center gap-6 text-center max-w-md">
            <div
              className="w-20 h-20 rounded-full flex items-center justify-center"
              style={{ background: '#121215', border: '1px solid #27272a' }}
            >
              <span className="material-symbols-outlined text-4xl" style={{ color: '#3f3f46' }}>analytics</span>
            </div>
            <div>
              <h2 className="text-2xl font-bold mb-2" style={{ color: '#fafafa' }}>No Analysis Results</h2>
              <p className="text-sm" style={{ color: '#71717a' }}>
                Upload and analyze a video first to see the detection results here.
              </p>
            </div>
            <button
              onClick={() => navigate('/detect')}
              className="flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-colors"
              style={{ background: '#a78bfa', color: '#0a0012' }}
              onMouseEnter={e => e.currentTarget.style.background = '#c4b5fd'}
              onMouseLeave={e => e.currentTarget.style.background = '#a78bfa'}
            >
              <span className="material-symbols-outlined text-lg">upload_file</span>
              Go to Detection
            </button>
          </div>
        </main>
      </div>
    )
  }

  const isFake = result.label === 'FAKE'
  const confidence = (result.confidence * 100).toFixed(1)
  const accentColor = isFake ? '#ef4444' : '#34d399'
  const accentGlow = isFake ? 'rgba(239,68,68,0.4)' : 'rgba(52,211,153,0.4)'
  const accentBgGlow = isFake ? 'rgba(239,68,68,0.07)' : 'rgba(52,211,153,0.07)'
  const gradientBg = isFake
    ? 'linear-gradient(to right, #dc2626, #ef4444)'
    : 'linear-gradient(to right, #059669, #34d399)'

  // Build explanation from backend data
  const explanationLines = result.explanation || []
  const attribution = result.attribution || 'N/A'

  // Build signal breakdown
  const breakdown = result.breakdown || {}
  const signalMetrics = [
    { label: 'CNN Classifier', value: breakdown.cnn ?? 0, icon: 'memory' },
    { label: 'Temporal Analysis', value: breakdown.temporal ?? 0, icon: 'timeline' },
    { label: 'FFT Spectral', value: breakdown.fft ?? 0, icon: 'waves' },
    { label: 'DeepFace', value: breakdown.deepface ?? 0, icon: 'face' },
    { label: 'Landmark', value: breakdown.landmark ?? 0, icon: 'pin_drop' },
    { label: 'Lip-Sync', value: breakdown.lip_sync ?? 0, icon: 'record_voice_over' },
  ]

  // Heuristic issues
  const issues = result.issues || []

  // Features
  const videoFeatures = result.features?.video || {}
  const audioFeatures = result.features?.audio || {}

  return (
    <div
      className="flex h-screen overflow-hidden"
      style={{ backgroundColor: '#09090b', color: '#fafafa', fontFamily: 'Geist, sans-serif' }}
    >
      <Sidebar />

      <main className="flex-1 overflow-y-auto p-6 md:p-8" style={{ backgroundColor: '#09090b' }}>
        <div className="max-w-7xl mx-auto flex flex-col gap-6">

          {/* ── Page header ── */}
          <div className="flex justify-between items-end">
            <div>
              <h1 className="text-2xl font-bold tracking-tight" style={{ color: '#fafafa' }}>
                Analysis Results
              </h1>
              <p className="text-xs mt-1" style={{ color: '#71717a' }}>
                Video ID: {result.video_id}
              </p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => navigate('/detect')}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors"
                style={{ background: '#a78bfa', color: '#0a0012' }}
                onMouseEnter={e => e.currentTarget.style.background = '#c4b5fd'}
                onMouseLeave={e => e.currentTarget.style.background = '#a78bfa'}
              >
                <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>add</span>
                New Scan
              </button>
            </div>
          </div>

          {/* ── Row 1: Verdict + Detection Summary ── */}
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">

            {/* Verdict — 3/5 */}
            <div className="lg:col-span-3 p-6 flex flex-col gap-4 relative overflow-hidden" style={card}>
              {/* Glow */}
              <div className="absolute inset-0 pointer-events-none" style={{ background: `radial-gradient(ellipse at 20% 40%, ${accentBgGlow} 0%, transparent 70%)` }} />

              <div className="relative z-10">
                {/* Alert badge */}
                <div className="flex items-center gap-2 mb-3">
                  <span
                    className="material-symbols-outlined text-sm"
                    style={{ color: accentColor, fontVariationSettings: "'FILL' 1" }}
                  >
                    {isFake ? 'warning' : 'verified'}
                  </span>
                  <span className="text-xs font-bold uppercase tracking-widest" style={{ color: accentColor }}>
                    {isFake ? 'Critical Alert' : 'Media Verified'}
                  </span>
                </div>

                {/* Verdict label */}
                <div
                  className="text-7xl font-black tracking-tight leading-none mb-3"
                  style={{ color: accentColor, textShadow: `0 0 40px ${accentGlow}` }}
                >
                  {result.label}
                </div>

                <p className="text-sm leading-relaxed mb-6" style={{ color: '#a1a1aa', maxWidth: '420px' }}>
                  {isFake
                    ? 'High probability of synthetic manipulation detected. Immediate review recommended.'
                    : 'This media appears authentic based on multimodal analysis across 9 AI models.'
                  }
                </p>

                {/* Confidence bar */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: '#71717a' }}>
                      Confidence Level
                    </span>
                    <span className="text-lg font-bold" style={{ color: accentColor }}>{confidence}%</span>
                  </div>
                  <div className="h-2 w-full rounded-full overflow-hidden" style={{ background: '#1e1e22' }}>
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{ width: `${confidence}%`, background: gradientBg }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Detection Summary — 2/5 */}
            <div className="lg:col-span-2 p-6 flex flex-col gap-5" style={card}>
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-base" style={{ color: '#a78bfa' }}>manage_search</span>
                <h3 className="text-sm font-semibold" style={{ color: '#fafafa' }}>Detection Summary</h3>
              </div>

              <div>
                <p className="text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: '#71717a' }}>Attribution</p>
                <span
                  className="text-xs font-semibold px-3 py-1 rounded"
                  style={{ background: '#7c3aed', color: '#ede9fe' }}
                >
                  {attribution}
                </span>
              </div>

              {issues.length > 0 && (
                <div>
                  <p className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: '#71717a' }}>Issues Detected</p>
                  <ul className="space-y-2.5">
                    {issues.map((item, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="material-symbols-outlined shrink-0" style={{ fontSize: '14px', color: '#ef4444', marginTop: '2px' }}>close</span>
                        <span className="text-xs leading-relaxed" style={{ color: '#d4d4d8' }}>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {issues.length === 0 && (
                <div className="flex items-center gap-2 p-3 rounded" style={{ background: 'rgba(52,211,153,0.08)', border: '1px solid rgba(52,211,153,0.2)' }}>
                  <span className="material-symbols-outlined text-sm" style={{ color: '#34d399', fontVariationSettings: "'FILL' 1" }}>check_circle</span>
                  <span className="text-xs" style={{ color: '#34d399' }}>No anomalies detected by heuristic analysis</span>
                </div>
              )}
            </div>
          </div>

          {/* ── Row 2: Model Prediction + Heuristic Analysis + Fusion Algorithm ── */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

            {/* Model Prediction */}
            <div className="p-6 flex flex-col gap-4" style={card}>
              <div className="flex justify-between items-center">
                <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: '#71717a' }}>Model Prediction</p>
                <span className="material-symbols-outlined text-base" style={{ color: '#52525b' }}>settings</span>
              </div>

              <div className="flex flex-col items-center justify-center py-2">
                <span
                  className="text-6xl font-black tracking-tight"
                  style={{ color: '#a78bfa', textShadow: '0 0 30px rgba(167,139,250,0.3)' }}
                >
                  {result.model_score?.toFixed(4) ?? '—'}
                </span>
                <span className="text-xs mt-1" style={{ color: '#71717a' }}>Combined Model Score</span>
              </div>

              <div className="space-y-2">
                <div
                  className="flex justify-between items-center px-3 py-2 rounded"
                  style={{ background: '#0c0c0f', border: '1px solid #27272a' }}
                >
                  <span className="text-xs" style={{ color: '#71717a' }}>Video Model</span>
                  <span className="text-xs font-semibold font-mono" style={{ color: '#fafafa' }}>
                    {result.video_model_score?.toFixed(4) ?? '—'}
                  </span>
                </div>
                <div
                  className="flex justify-between items-center px-3 py-2 rounded"
                  style={{ background: '#0c0c0f', border: '1px solid #27272a' }}
                >
                  <span className="text-xs" style={{ color: '#71717a' }}>Audio Model</span>
                  <span className="text-xs font-semibold font-mono" style={{ color: '#fafafa' }}>
                    {result.audio_model_score?.toFixed(4) ?? '—'}
                  </span>
                </div>
              </div>
            </div>

            {/* Heuristic Analysis */}
            <div className="p-6 flex flex-col gap-4" style={card}>
              <div className="flex justify-between items-center">
                <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: '#71717a' }}>Heuristic Analysis</p>
                <span className="material-symbols-outlined text-base" style={{ color: '#52525b' }}>search</span>
              </div>

              <div className="flex flex-col gap-4 mt-1">
                <MetricBar
                  label="Video Heuristic"
                  value={result.heuristic_analysis?.video_heur ?? 0}
                  color={result.heuristic_analysis?.video_heur >= 0.5 ? '#ef4444' : '#34d399'}
                />
                <MetricBar
                  label="Audio Heuristic"
                  value={result.heuristic_analysis?.audio_heur ?? 0}
                  color={result.heuristic_analysis?.audio_heur >= 0.5 ? '#ef4444' : '#34d399'}
                />
                <MetricBar
                  label="Aggregated Heuristic"
                  value={result.heuristic_score ?? 0}
                  color={result.heuristic_score >= 0.5 ? '#ef4444' : '#34d399'}
                />
              </div>
            </div>

            {/* Fusion Algorithm */}
            <div className="p-6 flex flex-col gap-4 justify-between" style={card}>
              <div className="flex justify-between items-center">
                <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: '#71717a' }}>Fusion Algorithm</p>
                <span className="material-symbols-outlined text-base" style={{ color: '#52525b' }}>device_hub</span>
              </div>

              <div
                className="flex-1 flex flex-col items-center justify-center rounded py-4 gap-2"
                style={{ background: '#0c0c0f', border: '1px solid #27272a' }}
              >
                <code className="text-xs font-mono" style={{ color: '#71717a' }}>
                  0.65 × model<sup>0.85</sup> + 0.25 × lip_sync + 0.10 × deepface
                </code>
                <p className="text-xs" style={{ color: '#52525b' }}>Non-linear fusion with signal injection</p>
              </div>

              <div className="flex justify-between items-end">
                <span className="text-xs" style={{ color: '#71717a' }}>Final Computed Score</span>
                <span
                  className="text-3xl font-black"
                  style={{ color: accentColor, textShadow: `0 0 20px ${accentGlow}` }}
                >
                  {result.final_score?.toFixed(4) ?? '—'}
                </span>
              </div>
            </div>
          </div>

          {/* ── Row 3: Signal Breakdown ── */}
          <div>
            <h2 className="text-base font-semibold mb-4" style={{ color: '#fafafa' }}>
              Signal Breakdown —  AI Models
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {signalMetrics.map((m) => {
                const signalColor = m.value >= 0.5 ? '#ef4444' : '#34d399'
                const pct = Math.round(m.value * 100)
                return (
                  <div
                    key={m.label}
                    className="p-5 flex flex-col gap-3"
                    style={{
                      ...card,
                      borderTop: `2px solid ${signalColor}`,
                    }}
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className="material-symbols-outlined text-base"
                        style={{ color: signalColor }}
                      >
                        {m.icon}
                      </span>
                      <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: '#71717a' }}>
                        {m.label}
                      </span>
                    </div>

                    <div className="flex items-center gap-3">
                      <span className="text-2xl font-black font-mono" style={{ color: signalColor }}>
                        {m.value.toFixed(4)}
                      </span>
                      <span className="text-xs px-2 py-0.5 rounded font-medium"
                        style={{
                          color: signalColor,
                          background: m.value >= 0.5 ? 'rgba(239,68,68,0.1)' : 'rgba(52,211,153,0.1)',
                          border: `1px solid ${m.value >= 0.5 ? 'rgba(239,68,68,0.25)' : 'rgba(52,211,153,0.25)'}`,
                        }}
                      >
                        {m.value >= 0.5 ? 'SUSPICIOUS' : 'NORMAL'}
                      </span>
                    </div>

                    <div className="h-1 w-full rounded-full" style={{ background: '#27272a' }}>
                      <div
                        className="h-full rounded-full transition-all duration-700"
                        style={{ width: `${pct}%`, background: signalColor }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* ── Row 4: Features + AI Reasoning ── */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pb-6">

            {/* Extracted Features */}
            <div className="p-6 flex flex-col gap-4" style={card}>
              <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: '#71717a' }}>Extracted Features</p>
              <div className="space-y-3">
                <p className="text-xs font-bold uppercase tracking-wider" style={{ color: '#a78bfa' }}>Video Features</p>
                {Object.entries(videoFeatures).map(([key, val]) => (
                  <div
                    key={key}
                    className="flex items-center justify-between p-3 rounded"
                    style={{ background: '#0c0c0f', border: '1px solid #27272a' }}
                  >
                    <span className="text-xs capitalize" style={{ color: '#a1a1aa' }}>{key.replace(/_/g, ' ')}</span>
                    <span className="text-xs font-mono font-semibold" style={{ color: '#fafafa' }}>
                      {typeof val === 'number' ? val.toFixed(6) : val}
                    </span>
                  </div>
                ))}

                <p className="text-xs font-bold uppercase tracking-wider pt-2" style={{ color: '#a78bfa' }}>Audio Features</p>
                {Object.entries(audioFeatures).map(([key, val]) => (
                  <div
                    key={key}
                    className="flex items-center justify-between p-3 rounded"
                    style={{ background: '#0c0c0f', border: '1px solid #27272a' }}
                  >
                    <span className="text-xs capitalize" style={{ color: '#a1a1aa' }}>{key.replace(/_/g, ' ')}</span>
                    <span className="text-xs font-mono font-semibold" style={{ color: '#fafafa' }}>
                      {typeof val === 'number' ? val.toFixed(6) : val}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* AI Reasoning */}
            <div className="p-6 flex flex-col gap-4" style={card}>
              <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: '#71717a' }}>AI Reasoning</p>
              {explanationLines.length > 0 ? (
                <ul className="space-y-4">
                  {explanationLines.map((line, idx) => (
                    <li key={idx} className="flex items-start gap-3">
                      <span
                        className="material-symbols-outlined shrink-0 mt-0.5"
                        style={{
                          fontSize: '16px',
                          color: line.toLowerCase().includes('normal') || line.toLowerCase().includes('authentic') || line.toLowerCase().includes('pass')
                            ? '#34d399' : '#a78bfa',
                        }}
                      >
                        {line.toLowerCase().includes('normal') || line.toLowerCase().includes('authentic') || line.toLowerCase().includes('pass')
                          ? 'check_circle' : 'info'}
                      </span>
                      <div>
                        <p className="text-sm leading-relaxed" style={{ color: '#d4d4d8' }}>{line}</p>
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm" style={{ color: '#71717a' }}>No explanation data available.</p>
              )}

              {/* Score summary */}
              <div className="mt-auto pt-4 space-y-2" style={{ borderTop: '1px solid #27272a' }}>
                <div className="flex justify-between text-xs">
                  <span style={{ color: '#71717a' }}>Pretrained Video Score</span>
                  <span className="font-mono font-semibold" style={{ color: '#fafafa' }}>
                    {result.features?.pretrained_video_score?.toFixed(4) ?? '—'}
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span style={{ color: '#71717a' }}>Pretrained Audio Score</span>
                  <span className="font-mono font-semibold" style={{ color: '#fafafa' }}>
                    {result.features?.pretrained_audio_score?.toFixed(4) ?? '—'}
                  </span>
                </div>
              </div>
            </div>
          </div>

        </div>
      </main>
    </div>
  )
}
