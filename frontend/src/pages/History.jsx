import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Sidebar from '../components/Sidebar'
import { useTheme } from '../context/ThemeContext'
import { useDetection } from '../context/DetectionContext'

const TYPE_META = {
  FAKE: { label: 'Fake Detected', colorKey: 'red', icon: 'warning' },
  REAL: { label: 'Verified Real', colorKey: 'green', icon: 'verified' },
}

export default function History() {
  const { c } = useTheme()
  const { history, setResult } = useDetection()
  const navigate = useNavigate()
  const [filter, setFilter] = useState('ALL')
  const [search, setSearch] = useState('')

  const filters = ['ALL', 'FAKE', 'REAL']
  const filterLabels = { ALL: 'All', FAKE: 'Fakes', REAL: 'Authentic' }

  const getColor = (colorKey) => c[colorKey] || c.textMuted

  // Group history by date
  const grouped = history.reduce((acc, item) => {
    const dateStr = item.scannedAt
      ? new Date(item.scannedAt).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })
      : 'Unknown Date'
    if (!acc[dateStr]) acc[dateStr] = []
    acc[dateStr].push(item)
    return acc
  }, {})

  const filteredGroups = Object.entries(grouped)
    .map(([date, items]) => ({
      date,
      items: items.filter(item => {
        const matchType = filter === 'ALL' || item.label === filter
        const matchSearch = search === '' ||
          item.video_id?.toLowerCase().includes(search.toLowerCase()) ||
          item.fileName?.toLowerCase().includes(search.toLowerCase()) ||
          item.label?.toLowerCase().includes(search.toLowerCase())
        return matchType && matchSearch
      }),
    }))
    .filter(g => g.items.length > 0)

  const totalEvents = history.length
  const fakes = history.filter(i => i.label === 'FAKE').length
  const reals = history.filter(i => i.label === 'REAL').length

  const handleViewResult = (item) => {
    setResult(item)
    navigate('/analysis')
  }

  return (
    <div
      className="flex h-screen overflow-hidden transition-colors duration-300"
      style={{ backgroundColor: c.bg, color: c.text, fontFamily: 'Geist, sans-serif' }}
    >
      <Sidebar />

      <main className="flex-1 overflow-y-auto p-6 md:p-8">
        <div className="max-w-4xl mx-auto flex flex-col gap-6">

          {/* Header */}
          <div className="flex justify-between items-end">
            <div>
              <h1 className="text-2xl font-bold tracking-tight" style={{ color: c.text }}>Activity History</h1>
              <p className="text-sm mt-1" style={{ color: c.textDim }}>
                Full chronological log of all detection scans.
              </p>
            </div>
            <button
              onClick={() => navigate('/detect')}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-colors"
              style={{ background: c.primary, color: '#fff' }}
            >
              <span className="material-symbols-outlined text-base">add</span>
              New Scan
            </button>
          </div>

          {/* Quick stats */}
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {[
              { label: 'Total Scans', value: totalEvents, icon: 'timeline', color: c.primary },
              { label: 'Fakes Detected', value: fakes, icon: 'warning', color: c.red },
              { label: 'Real Verified', value: reals, icon: 'verified', color: c.green },
            ].map(s => (
              <div
                key={s.label}
                className="flex items-center gap-3 p-4 rounded-xl transition-colors"
                style={{ background: c.surface, border: `1px solid ${c.border}` }}
              >
                <div
                  className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0"
                  style={{ background: s.color + '18' }}
                >
                  <span className="material-symbols-outlined text-base" style={{ color: s.color }}>{s.icon}</span>
                </div>
                <div>
                  <p className="text-xl font-black leading-none" style={{ color: c.text }}>{s.value}</p>
                  <p className="text-xs mt-0.5" style={{ color: c.textDim }}>{s.label}</p>
                </div>
              </div>
            ))}
          </div>

          {/* Search + type filter */}
          <div className="flex flex-col sm:flex-row gap-3">
            <div
              className="flex items-center gap-2 flex-1 px-4 py-2.5 rounded-lg"
              style={{ background: c.surface, border: `1px solid ${c.border}` }}
            >
              <span className="material-symbols-outlined text-base" style={{ color: c.textDim }}>search</span>
              <input
                type="text"
                placeholder="Search by video ID, filename…"
                value={search}
                onChange={e => setSearch(e.target.value)}
                className="flex-1 bg-transparent outline-none text-sm"
                style={{ color: c.text, caretColor: c.primary }}
              />
              {search && (
                <button onClick={() => setSearch('')} style={{ color: c.textDim }}>
                  <span className="material-symbols-outlined text-base">close</span>
                </button>
              )}
            </div>
            <div className="flex gap-1.5 flex-wrap">
              {filters.map(f => (
                <button
                  key={f}
                  onClick={() => setFilter(f)}
                  className="px-3 py-2 rounded-lg text-xs font-semibold transition-colors whitespace-nowrap"
                  style={filter === f
                    ? { background: c.primary, color: '#fff' }
                    : { background: c.surface, border: `1px solid ${c.border}`, color: c.textMuted }
                  }
                >
                  {filterLabels[f]}
                </button>
              ))}
            </div>
          </div>

          {/* Timeline */}
          {history.length === 0 ? (
            <div
              className="flex flex-col items-center justify-center py-20 rounded-xl gap-3"
              style={{ background: c.surface, border: `1px solid ${c.border}` }}
            >
              <span className="material-symbols-outlined text-4xl" style={{ color: c.border }}>history</span>
              <p className="text-sm" style={{ color: c.textDim }}>No scan history yet. Upload a video to get started!</p>
              <button
                onClick={() => navigate('/detect')}
                className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold mt-2"
                style={{ background: c.primary, color: '#fff' }}
              >
                Start Detection
              </button>
            </div>
          ) : filteredGroups.length === 0 ? (
            <div
              className="flex flex-col items-center justify-center py-20 rounded-xl gap-3"
              style={{ background: c.surface, border: `1px solid ${c.border}` }}
            >
              <span className="material-symbols-outlined text-4xl" style={{ color: c.border }}>history</span>
              <p className="text-sm" style={{ color: c.textDim }}>No events match your search.</p>
            </div>
          ) : (
            <div className="flex flex-col gap-8 pb-8">
              {filteredGroups.map(group => (
                <div key={group.date}>
                  {/* Date header */}
                  <div className="flex items-center gap-3 mb-4">
                    <span className="text-xs font-bold uppercase tracking-widest" style={{ color: c.textDim }}>
                      {group.date}
                    </span>
                    <div className="flex-1 h-px" style={{ background: c.border }} />
                    <span
                      className="text-xs px-2 py-0.5 rounded-full font-medium"
                      style={{ background: c.surfaceHigh, color: c.textDim }}
                    >
                      {group.items.length} scan{group.items.length !== 1 ? 's' : ''}
                    </span>
                  </div>

                  {/* Event items */}
                  <div className="relative">
                    {/* Vertical line */}
                    <div
                      className="absolute left-[23px] top-0 bottom-0 w-px"
                      style={{ background: c.border }}
                    />

                    <div className="flex flex-col gap-3">
                      {group.items.map((item, idx) => {
                        const meta = TYPE_META[item.label] || TYPE_META.REAL
                        const iconColor = getColor(meta.colorKey)
                        const time = item.scannedAt
                          ? new Date(item.scannedAt).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
                          : ''

                        return (
                          <div key={idx} className="flex gap-4 items-start">
                            {/* Icon dot */}
                            <div
                              className="w-12 h-12 rounded-full flex items-center justify-center shrink-0 relative z-10"
                              style={{
                                background: iconColor + '15',
                                border: `2px solid ${iconColor}40`,
                              }}
                            >
                              <span
                                className="material-symbols-outlined text-lg"
                                style={{
                                  color: iconColor,
                                  fontVariationSettings: "'FILL' 1",
                                }}
                              >
                                {meta.icon}
                              </span>
                            </div>

                            {/* Event card */}
                            <div
                              className="flex-1 p-4 rounded-xl transition-colors duration-200 cursor-pointer"
                              style={{ background: c.surface, border: `1px solid ${c.border}` }}
                              onMouseEnter={e => e.currentTarget.style.borderColor = iconColor + '60'}
                              onMouseLeave={e => e.currentTarget.style.borderColor = c.border}
                              onClick={() => handleViewResult(item)}
                            >
                              <div className="flex items-start justify-between gap-3 flex-wrap">
                                <div className="flex items-center gap-2 flex-wrap">
                                  <p className="text-sm font-semibold" style={{ color: c.text }}>
                                    {item.label === 'FAKE' ? 'Deepfake Detected' : 'Authentic Media Verified'}
                                  </p>
                                  <span
                                    className="text-xs px-2 py-0.5 rounded-full font-medium"
                                    style={{
                                      color: iconColor,
                                      background: iconColor + '18',
                                      border: `1px solid ${iconColor}30`,
                                    }}
                                  >
                                    {meta.label}
                                  </span>
                                  <span
                                    className="text-xs font-mono px-2 py-0.5 rounded"
                                    style={{ color: c.textDim, background: c.surfaceHigh }}
                                  >
                                    {item.video_id?.slice(0, 8)}…
                                  </span>
                                </div>
                                <div className="flex items-center gap-3 shrink-0">
                                  <span className="text-xs font-mono font-semibold" style={{ color: iconColor }}>
                                    {(item.confidence * 100).toFixed(1)}%
                                  </span>
                                  <span className="text-xs" style={{ color: c.textDim }}>{time}</span>
                                </div>
                              </div>

                              <p className="text-xs mt-2 leading-relaxed" style={{ color: c.textDim }}>
                                {item.fileName} • Final score: {item.final_score?.toFixed(4)}
                              </p>

                              <div className="flex items-center gap-1.5 mt-3">
                                <span className="material-symbols-outlined text-xs" style={{ color: c.textDim }}>upload_file</span>
                                <span className="text-xs" style={{ color: c.textDim }}>Video Upload</span>
                              </div>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

        </div>
      </main>
    </div>
  )
}
