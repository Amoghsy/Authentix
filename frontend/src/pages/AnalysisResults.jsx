import { useNavigate } from 'react-router-dom'
import AppLayout from '../components/AppLayout'
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
        <span className="text-sm font-mono font-semibold" style={{ color }}>{value.toFixed(4)}</span>
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
      <AppLayout mainClassName="p-6 md:p-8">
        <div className="flex items-center justify-center h-full">
          <div className="flex flex-col items-center gap-6 text-center max-w-md">
            <div
              className="w-20 h-20 rounded-full flex items-center justify-center"
              style={{ background: '#111113', border: '1px solid #1f1f23' }}
            >
              <span className="material-symbols-outlined text-4xl" style={{ color: '#3f3f46' }}>analytics</span>
            </div>
            <div>
              <h2 className="text-2xl font-bold mb-2">No Analysis Results</h2>
              <p className="text-sm" style={{ color: '#71717a' }}>
                Upload and analyze a video first to see the detection results here.
              </p>
            </div>
            <button
              onClick={() => navigate('/detect')}
              className="flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all duration-200 hover:opacity-90 active:scale-95"
              style={{ background: '#7c3aed', color: '#fff' }}
            >
              <span className="material-symbols-outlined text-lg">upload_file</span>
              Go to Detection
            </button>
          </div>
        </div>
      </AppLayout>
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

  const explanationLines = result.reasoning || []

  // Modality configurations
  const modalitySignals = [
    {
      label: 'Video Modality Feed',
      value: result.video?.score ?? 0,
      isFake: result.video?.isFake,
      noTrack: false,
      icon: 'movie',
      version: result.video?.version || '1.0.2',
      desc: 'Artifacts & temporal face classification'
    },
    {
      label: 'Audio Modality Feed',
      value: result.audio?.score ?? 0.5,
      isFake: result.audio?.isFake,
      noTrack: result.audio?.noTrack ?? false,
      icon: 'graphic_eq',
      version: result.audio?.version || '2.1.0',
      desc: 'Mel-spectrogram sound analysis'
    },
    {
      label: 'Lip-Sync A/V Modality',
      value: result.lipSync?.score ?? 0,
      isFake: result.lipSync?.isFake,
      noTrack: false,
      icon: 'record_voice_over',
      version: result.lipSync?.version || '1.0.0',
      desc: 'A/V lips matching synchronization'
    }
  ]

  return (
    <AppLayout mainClassName="p-6 md:p-8">
      <div className="max-w-7xl mx-auto flex flex-col gap-6">

          {/* ── Page header ── */}
          <div className="flex justify-between items-end">
            <div>
              <h1 className="text-2xl font-bold tracking-tight" style={{ color: '#fafafa' }}>
                Analysis Results
              </h1>
              <p className="text-xs mt-1" style={{ color: '#71717a' }}>
                Session ID: {result.id} • File: {result.fileName}
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
                    {isFake ? 'Deepfake Alert' : 'Media Verified'}
                  </span>
                </div>

                {/* Verdict label */}
                <div
                  className="text-7xl font-black tracking-tight leading-none mb-3"
                  style={{ color: accentColor, textShadow: `0 0 40px ${accentGlow}` }}
                >
                  {result.prediction}
                </div>

                <p className="text-sm leading-relaxed mb-6" style={{ color: '#a1a1aa', maxWidth: '420px' }}>
                  {isFake
                    ? 'Our multimodal neural network detected significant signals of synthesis or manipulation.'
                    : 'Multimodal fusion has determined the uploaded video is highly likely to be authentic.'
                  }
                </p>

                {/* Confidence bar */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: '#71717a' }}>
                      Fusion Confidence Level
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
                <span className="material-symbols-outlined text-base" style={{ color: '#a78bfa' }}>analytics</span>
                <h3 className="text-sm font-semibold" style={{ color: '#fafafa' }}>Execution stats</h3>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-widest mb-1" style={{ color: '#71717a' }}>Risk Level</p>
                  <span
                    className="text-xs font-bold px-2.5 py-1 rounded inline-block text-center"
                    style={{
                      color: accentColor,
                      background: isFake ? 'rgba(239,68,68,0.1)' : 'rgba(52,211,153,0.1)',
                      border: `1px solid ${isFake ? 'rgba(239,68,68,0.2)' : 'rgba(52,211,153,0.2)'}`
                    }}
                  >
                    {result.riskLevel}
                  </span>
                </div>
                <div>
                  <p className="text-xs font-semibold uppercase tracking-widest mb-1" style={{ color: '#71717a' }}>Processing speed</p>
                  <span className="text-sm font-mono font-semibold" style={{ color: '#fafafa' }}>
                    {(result.processingTime / 1000).toFixed(2)}s
                  </span>
                </div>
              </div>

              <div style={{ borderTop: '1px solid #27272a', paddingTop: '15px' }}>
                <p className="text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: '#71717a' }}>File analyzed</p>
                <div className="flex items-center gap-2">
                  <span className="material-symbols-outlined text-base" style={{ color: '#a1a1aa' }}>movie</span>
                  <span className="text-xs font-mono truncate max-w-[200px]" style={{ color: '#d4d4d8' }}>{result.fileName}</span>
                </div>
              </div>
            </div>
          </div>

          {/* ── Row 2: Modality Breakdown ── */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {modalitySignals.map((m) => {
              const signalColor = m.noTrack ? '#71717a' : (m.isFake ? '#ef4444' : '#34d399')
              return (
                <div key={m.label} className="p-6 flex flex-col justify-between h-48" style={card}>
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                      <span className="material-symbols-outlined text-base animate-pulse" style={{ color: m.noTrack ? '#52525b' : '#a78bfa' }}>{m.icon}</span>
                      <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: '#71717a' }}>{m.label}</p>
                    </div>
                    <span className="text-[10px] font-mono text-zinc-600">{m.version}</span>
                  </div>

                  {m.noTrack ? (
                    /* No audio track — show muted state */
                    <div className="flex flex-col gap-2 py-2">
                      <div className="flex items-center gap-2">
                        <span className="material-symbols-outlined text-base" style={{ color: '#52525b' }}>volume_off</span>
                        <span className="text-sm font-semibold" style={{ color: '#52525b' }}>No Audio Track</span>
                      </div>
                      <p className="text-xs" style={{ color: '#3f3f46' }}>Video has no audio stream. Audio modality skipped — fusion uses Video + Lip-Sync only.</p>
                    </div>
                  ) : (
                    <div className="flex items-center justify-between py-2">
                      <span
                        className="text-4xl font-black tracking-tight font-mono"
                        style={{ color: '#a78bfa' }}
                      >
                        {m.value.toFixed(4)}
                      </span>
                      <span
                        className="text-[10px] px-2 py-0.5 rounded font-bold uppercase"
                        style={{
                          color: signalColor,
                          background: m.isFake ? 'rgba(239,68,68,0.1)' : 'rgba(52,211,153,0.1)',
                          border: `1px solid ${m.isFake ? 'rgba(239,68,68,0.2)' : 'rgba(52,211,153,0.2)'}`
                        }}
                      >
                        {m.isFake ? 'FAKE' : 'AUTHENTIC'}
                      </span>
                    </div>
                  )}

                  <MetricBar
                    label={m.noTrack ? 'Not analyzed (no audio)' : 'Modality scoring weight'}
                    value={m.noTrack ? 0 : m.value}
                    color={signalColor}
                  />
                </div>
              )
            })}
          </div>

          {/* ── Row 3: Multimodal Fusion Reasoning & Graph ── */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-4 pb-6">

            {/* AI Reasoning — 7/12 */}
            <div className="md:col-span-8 p-6 flex flex-col gap-4" style={card}>
              <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: '#71717a' }}>Explainable AI Reasoning</p>
              {explanationLines.length > 0 ? (
                <ul className="space-y-4">
                  {explanationLines.map((line, idx) => {
                    const isNormal = line.toLowerCase().includes('normal') || line.toLowerCase().includes('authentic') || line.toLowerCase().includes('pass') || line.toLowerCase().includes('aligned')
                    return (
                      <li key={idx} className="flex items-start gap-3">
                        <span
                          className="material-symbols-outlined shrink-0 mt-0.5"
                          style={{
                            fontSize: '16px',
                            color: isNormal ? '#34d399' : '#a78bfa',
                          }}
                        >
                          {isNormal ? 'check_circle' : 'info'}
                        </span>
                        <div>
                          <p className="text-sm leading-relaxed" style={{ color: '#d4d4d8' }}>{line}</p>
                        </div>
                      </li>
                    )
                  })}
                </ul>
              ) : (
                <p className="text-sm" style={{ color: '#71717a' }}>No explanation data available.</p>
              )}
            </div>

            {/* Fusion Score Panel — 4/12 */}
            <div className="md:col-span-4 p-6 flex flex-col justify-between" style={card}>
              <div>
                <p className="text-xs font-semibold uppercase tracking-widest mb-4" style={{ color: '#71717a' }}>Decision Fusion Engine</p>
                <div
                  className="p-4 flex flex-col items-center justify-center rounded py-6 gap-2"
                  style={{ background: '#0c0c0f', border: '1px solid #27272a' }}
                >
                  <code className="text-xs font-mono text-zinc-400">
                    Non-linear Scoring Calibrator
                  </code>
                  <p className="text-[10px] text-center" style={{ color: '#71717a' }}>
                    Weights Video, Audio, and Lip-Sync modality predictions via non-linear Consensus Calibration
                  </p>
                </div>
              </div>

              <div className="flex justify-between items-end border-t border-zinc-800 pt-4 mt-4">
                <div>
                  <span className="text-xs block" style={{ color: '#71717a' }}>Unified Score</span>
                  <span className="text-[10px]" style={{ color: '#52525b' }}>Scale: 0.00 (Real) to 1.00 (Fake)</span>
                </div>
                <span
                  className="text-4xl font-black font-mono leading-none"
                  style={{ color: accentColor, textShadow: `0 0 20px ${accentGlow}` }}
                >
                  {(result.fusion?.score ?? result.confidence).toFixed(4)}
                </span>
              </div>
            </div>
          </div>

        </div>
    </AppLayout>
  )
}
