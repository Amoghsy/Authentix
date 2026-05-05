import { useState } from 'react'
import Sidebar from '../components/Sidebar'
import { useTheme } from '../context/ThemeContext'

function Section({ title, icon, children, c }) {
  return (
    <div className="rounded-xl overflow-hidden transition-colors duration-300"
      style={{ border: `1px solid ${c.border}` }}>
      <div className="flex items-center gap-3 px-6 py-4"
        style={{ background: c.surfaceHigh, borderBottom: `1px solid ${c.border}` }}>
        <span className="material-symbols-outlined text-lg" style={{ color: c.primary }}>{icon}</span>
        <h2 className="text-sm font-bold uppercase tracking-widest" style={{ color: c.text }}>{title}</h2>
      </div>
      <div className="px-6 py-5 flex flex-col gap-5" style={{ background: c.surface }}>
        {children}
      </div>
    </div>
  )
}

function Row({ label, sub, children, c }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <div>
        <p className="text-sm font-medium" style={{ color: c.text }}>{label}</p>
        {sub && <p className="text-xs mt-0.5" style={{ color: c.textDim }}>{sub}</p>}
      </div>
      {children}
    </div>
  )
}

function Toggle({ value, onChange, c }) {
  return (
    <button
      onClick={() => onChange(!value)}
      className="w-11 h-6 rounded-full relative transition-colors duration-300 flex items-center shrink-0"
      style={{ background: value ? c.primary : c.border }}
    >
      <div className="w-5 h-5 rounded-full absolute transition-all duration-300"
        style={{ background: '#fff', left: value ? '22px' : '2px', boxShadow: '0 1px 4px rgba(0,0,0,0.25)' }} />
    </button>
  )
}

function Slider({ value, onChange, c }) {
  return (
    <div className="flex items-center gap-3 w-48">
      <input
        type="range" min="0" max="100" value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="flex-1 h-1.5 rounded-full appearance-none cursor-pointer"
        style={{ accentColor: c.primary, background: c.border }}
      />
      <span className="text-xs font-mono w-8 text-right" style={{ color: c.primary }}>{value}%</span>
    </div>
  )
}

export default function Settings() {
  const { c, theme, toggle, setTheme } = useTheme()

  // Detection toggles
  const [heuristics, setHeuristics] = useState({
    facial: true, audio: true, temporal: true, lighting: false, pupil: true,
  })
  const [sensitivity, setSensitivity] = useState(72)
  const [audioWeight, setAudioWeight] = useState(30)

  // Notifications
  const [notifs, setNotifs] = useState({
    fakeAlert: true, reportReady: true, systemHealth: false, weeklyDigest: true,
  })

  // Account
  const [autoExport, setAutoExport] = useState(false)
  const [retentionDays, setRetentionDays] = useState(30)

  const Divider = () => (
    <div style={{ borderTop: `1px solid ${c.border}` }} />
  )

  return (
    <div className="flex h-screen overflow-hidden transition-colors duration-300"
      style={{ backgroundColor: c.bg, color: c.text, fontFamily: 'Geist, sans-serif' }}>
      <Sidebar />

      <main className="flex-1 overflow-y-auto p-6 md:p-8">
        <div className="max-w-3xl mx-auto flex flex-col gap-6">

          {/* Page header */}
          <div>
            <h1 className="text-2xl font-bold tracking-tight" style={{ color: c.text }}>Settings</h1>
            <p className="text-sm mt-1" style={{ color: c.textDim }}>
              Configure Authentix detection behaviour, appearance, and notifications.
            </p>
          </div>

          {/* ── Appearance ── */}
          <Section title="Appearance" icon="palette" c={c}>
            <Row label="Theme" sub="Switch between dark and light mode" c={c}>
              <div className="flex items-center gap-2 p-1 rounded-lg" style={{ background: c.surfaceHigh, border: `1px solid ${c.border}` }}>
                {['dark', 'light'].map(t => (
                  <button
                    key={t}
                    onClick={() => setTheme(t)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold capitalize transition-all duration-200"
                    style={theme === t
                      ? { background: c.primary, color: '#fff', boxShadow: '0 2px 8px rgba(0,0,0,0.2)' }
                      : { color: c.textMuted }}
                  >
                    <span className="material-symbols-outlined text-sm">
                      {t === 'dark' ? 'dark_mode' : 'light_mode'}
                    </span>
                    {t}
                  </button>
                ))}
              </div>
            </Row>

            <Divider />

            {/* Theme preview */}
            <div className="flex gap-3">
              {[
                { name: 'Dark', bg: '#09090b', surface: '#111113', accent: '#a78bfa' },
                { name: 'Light', bg: '#f4f4f8', surface: '#ffffff', accent: '#7c3aed' },
              ].map(p => (
                <button
                  key={p.name}
                  onClick={() => setTheme(p.name.toLowerCase())}
                  className="flex-1 rounded-xl overflow-hidden transition-all duration-200"
                  style={{
                    border: `2px solid ${theme === p.name.toLowerCase() ? c.primary : c.border}`,
                    boxShadow: theme === p.name.toLowerCase() ? `0 0 0 3px ${c.primaryBg}` : 'none',
                  }}
                >
                  {/* Mini preview */}
                  <div className="p-3" style={{ background: p.bg }}>
                    <div className="flex gap-1.5 mb-2">
                      <div className="w-2 h-2 rounded-full" style={{ background: p.accent }} />
                      <div className="w-12 h-2 rounded" style={{ background: p.surface }} />
                    </div>
                    <div className="w-full h-6 rounded" style={{ background: p.surface }} />
                    <div className="mt-1.5 w-3/4 h-1.5 rounded" style={{ background: p.accent, opacity: 0.4 }} />
                  </div>
                  <div className="py-2 text-center text-xs font-semibold" style={{ background: c.surfaceHigh, color: c.textMuted }}>
                    {p.name}
                    {theme === p.name.toLowerCase() && (
                      <span className="ml-1.5 text-xs" style={{ color: c.primary }}>✓</span>
                    )}
                  </div>
                </button>
              ))}
            </div>
          </Section>

          {/* ── Detection ── */}
          <Section title="Detection Engine" icon="manage_search" c={c}>
            <Row label="Detection Sensitivity"
              sub="Higher sensitivity catches more fakes but may increase false positives." c={c}>
              <Slider value={sensitivity} onChange={setSensitivity} c={c} />
            </Row>

            <Divider />

            <Row label="Audio Analysis Weight"
              sub={`Audio contributes ${audioWeight}% to the final fusion score.`} c={c}>
              <Slider value={audioWeight} onChange={setAudioWeight} c={c} />
            </Row>

            <Divider />

            <div>
              <p className="text-sm font-medium mb-4" style={{ color: c.text }}>Active Heuristics</p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {[
                  { key: 'facial',   icon: 'face',       label: 'Facial Artifacts',    sub: 'GAN blending & edge distortion' },
                  { key: 'audio',    icon: 'graphic_eq', label: 'Audio Sync Drift',    sub: 'Lip-sync and vocal mismatch' },
                  { key: 'temporal', icon: 'blur_on',    label: 'Temporal Noise',      sub: 'Frame-to-frame inconsistencies' },
                  { key: 'lighting', icon: 'lightbulb',  label: 'Lighting Analysis',   sub: 'Shadow & reflection consistency' },
                  { key: 'pupil',    icon: 'visibility', label: 'Pupil Reflection',    sub: 'Unnatural light reflections' },
                ].map(h => (
                  <div
                    key={h.key}
                    className="flex items-center justify-between p-3 rounded-lg"
                    style={{ background: c.surfaceHigh, border: `1px solid ${c.border}` }}
                  >
                    <div className="flex items-center gap-2.5">
                      <span className="material-symbols-outlined text-base" style={{ color: c.primary }}>{h.icon}</span>
                      <div>
                        <p className="text-xs font-semibold" style={{ color: c.text }}>{h.label}</p>
                        <p className="text-xs" style={{ color: c.textDim }}>{h.sub}</p>
                      </div>
                    </div>
                    <Toggle
                      value={heuristics[h.key]}
                      onChange={v => setHeuristics(prev => ({ ...prev, [h.key]: v }))}
                      c={c}
                    />
                  </div>
                ))}
              </div>
            </div>
          </Section>

          {/* ── Notifications ── */}
          <Section title="Notifications" icon="notifications" c={c}>
            {[
              { key: 'fakeAlert',    label: 'Fake Media Alert',     sub: 'Notify immediately when a fake is detected with high confidence.' },
              { key: 'reportReady',  label: 'Report Ready',         sub: 'Send a notification when a scan report is fully processed.' },
              { key: 'systemHealth', label: 'System Health Alerts', sub: 'Notify on model degradation or service interruptions.' },
              { key: 'weeklyDigest', label: 'Weekly Digest',        sub: 'Summary of all scans, detections, and model performance.' },
            ].map((n, i, arr) => (
              <div key={n.key}>
                <Row label={n.label} sub={n.sub} c={c}>
                  <Toggle value={notifs[n.key]} onChange={v => setNotifs(p => ({ ...p, [n.key]: v }))} c={c} />
                </Row>
                {i < arr.length - 1 && <Divider />}
              </div>
            ))}
          </Section>

          {/* ── Data & Privacy ── */}
          <Section title="Data & Privacy" icon="lock" c={c}>
            <Row label="Auto-Export Reports"
              sub="Automatically export PDF report after each scan completes." c={c}>
              <Toggle value={autoExport} onChange={setAutoExport} c={c} />
            </Row>

            <Divider />

            <Row label="Scan Data Retention"
              sub={`Reports older than ${retentionDays} days are automatically purged.`} c={c}>
              <div className="flex items-center gap-2">
                {[7, 14, 30, 90].map(d => (
                  <button
                    key={d}
                    onClick={() => setRetentionDays(d)}
                    className="px-3 py-1 rounded text-xs font-semibold transition-colors"
                    style={retentionDays === d
                      ? { background: c.primary, color: '#fff' }
                      : { background: c.surfaceHigh, border: `1px solid ${c.border}`, color: c.textMuted }}
                  >
                    {d}d
                  </button>
                ))}
              </div>
            </Row>

            <Divider />

            <Row label="Clear All Scan Data"
              sub="Permanently delete all reports and scan history from this device." c={c}>
              <button
                className="px-4 py-2 rounded-lg text-xs font-bold transition-colors"
                style={{ color: c.red, border: `1px solid ${c.redBorder}`, background: c.redBg }}
                onMouseEnter={e => e.currentTarget.style.background = c.red.replace(')', ',0.2)').replace('rgb', 'rgba')}
                onMouseLeave={e => e.currentTarget.style.background = c.redBg}
              >
                Clear Data
              </button>
            </Row>
          </Section>

          {/* Save button */}
          <div className="flex justify-end pb-6">
            <button
              className="px-6 py-3 rounded-lg font-semibold text-sm transition-all duration-200 flex items-center gap-2 hover:opacity-90"
              style={{ background: c.primary, color: '#fff', boxShadow: `0 4px 15px ${c.primaryBg}` }}
            >
              <span className="material-symbols-outlined text-base">save</span>
              Save Settings
            </button>
          </div>

        </div>
      </main>
    </div>
  )
}
