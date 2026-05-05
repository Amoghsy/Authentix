import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Sidebar from '../components/Sidebar'
import { useTheme } from '../context/ThemeContext'
import { useDetection } from '../context/DetectionContext'

export default function Reports() {
  const { c } = useTheme()
  const { history, setResult } = useDetection()
  const navigate = useNavigate()
  const [filter, setFilter] = useState('ALL')
  const [search, setSearch] = useState('')

  const filtered = history.filter(r => {
    const matchVerdict = filter === 'ALL' || r.label === filter
    const matchSearch = search === '' ||
      r.video_id?.toLowerCase().includes(search.toLowerCase()) ||
      r.fileName?.toLowerCase().includes(search.toLowerCase())
    return matchVerdict && matchSearch
  })

  const totalScans = history.length
  const fakes = history.filter(r => r.label === 'FAKE').length
  const reals = history.filter(r => r.label === 'REAL').length
  const avgScore = history.length > 0
    ? (history.reduce((a, r) => a + (r.confidence || 0), 0) / history.length * 100).toFixed(1)
    : '0.0'

  const statCards = [
    { icon: 'folder_open', label: 'Total Scans', value: totalScans.toString(), sub: 'All time', color: '#a78bfa' },
    { icon: 'warning', label: 'Fakes Detected', value: fakes.toString(), sub: totalScans > 0 ? `${((fakes / totalScans) * 100).toFixed(1)}% of scans` : '—', color: '#ef4444' },
    { icon: 'verified_user', label: 'Real Verified', value: reals.toString(), sub: totalScans > 0 ? `${((reals / totalScans) * 100).toFixed(1)}% of scans` : '—', color: '#34d399' },
    { icon: 'speed', label: 'Avg Confidence', value: `${avgScore}%`, sub: 'Across all scans', color: '#f59e0b' },
  ]

  const handleViewResult = (item) => {
    setResult(item)
    navigate('/analysis')
  }

  const handleExportAll = () => {
    const dataStr = JSON.stringify(history, null, 2)
    const blob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `authentix-reports-${new Date().toISOString().slice(0, 10)}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div
      className="flex h-screen overflow-hidden transition-colors duration-300"
      style={{ backgroundColor: c.bg, color: c.text, fontFamily: 'Geist, sans-serif' }}
    >
      <Sidebar />

      <main className="flex-1 overflow-y-auto p-6 md:p-8">
        <div className="max-w-7xl mx-auto flex flex-col gap-6">

          {/* Header */}
          <div className="flex justify-between items-end">
            <div>
              <h1 className="text-2xl font-bold tracking-tight" style={{ color: c.text }}>Reports</h1>
              <p className="text-sm mt-1" style={{ color: c.textDim }}>Full history of all deepfake detection scans.</p>
            </div>
            <button
              onClick={handleExportAll}
              disabled={history.length === 0}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-colors disabled:opacity-40"
              style={{ background: c.primary, color: '#fff' }}
            >
              <span className="material-symbols-outlined text-base">download</span>
              Export All
            </button>
          </div>

          {/* Stat cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {statCards.map((s) => (
              <div
                key={s.label}
                className="p-5 rounded-xl flex flex-col gap-2 transition-colors duration-300"
                style={{ background: c.surface, border: `1px solid ${c.border}` }}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: c.textDim }}>
                    {s.label}
                  </span>
                  <div
                    className="w-8 h-8 rounded-lg flex items-center justify-center"
                    style={{ background: s.color + '18' }}
                  >
                    <span className="material-symbols-outlined text-base" style={{ color: s.color }}>{s.icon}</span>
                  </div>
                </div>
                <p className="text-3xl font-black tracking-tight" style={{ color: c.text }}>{s.value}</p>
                <p className="text-xs" style={{ color: c.textDim }}>{s.sub}</p>
              </div>
            ))}
          </div>

          {/* Search + Filter */}
          <div className="flex flex-col sm:flex-row gap-3">
            <div
              className="flex items-center gap-2 flex-1 px-4 py-2.5 rounded-lg"
              style={{ background: c.surface, border: `1px solid ${c.border}` }}
            >
              <span className="material-symbols-outlined text-base" style={{ color: c.textDim }}>search</span>
              <input
                type="text"
                placeholder="Search by video ID or filename…"
                value={search}
                onChange={e => setSearch(e.target.value)}
                className="flex-1 bg-transparent outline-none text-sm"
                style={{ color: c.text, caretColor: c.primary }}
              />
            </div>
            <div className="flex gap-2">
              {['ALL', 'FAKE', 'REAL'].map(f => (
                <button
                  key={f}
                  onClick={() => setFilter(f)}
                  className="px-4 py-2 rounded-lg text-xs font-bold tracking-wider transition-colors"
                  style={filter === f
                    ? { background: c.primary, color: '#fff' }
                    : { background: c.surface, border: `1px solid ${c.border}`, color: c.textMuted }
                  }
                >
                  {f}
                </button>
              ))}
            </div>
          </div>

          {/* Table */}
          {history.length === 0 ? (
            <div
              className="flex flex-col items-center justify-center py-20 rounded-xl gap-3"
              style={{ background: c.surface, border: `1px solid ${c.border}` }}
            >
              <span className="material-symbols-outlined text-4xl" style={{ color: c.border }}>description</span>
              <p className="text-sm" style={{ color: c.textDim }}>No scan reports yet.</p>
              <button
                onClick={() => navigate('/detect')}
                className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold mt-2"
                style={{ background: c.primary, color: '#fff' }}
              >
                Start Detection
              </button>
            </div>
          ) : (
            <div
              className="rounded-xl overflow-hidden transition-colors duration-300"
              style={{ border: `1px solid ${c.border}` }}
            >
              {/* Table header */}
              <div
                className="grid grid-cols-12 px-5 py-3 text-xs font-semibold uppercase tracking-widest"
                style={{ background: c.surfaceHigh, color: c.textDim, borderBottom: `1px solid ${c.border}` }}
              >
                <span className="col-span-3">Video ID</span>
                <span className="col-span-3">Date & Time</span>
                <span className="col-span-2 text-center">Verdict</span>
                <span className="col-span-1 text-center">Confidence</span>
                <span className="col-span-2">Filename</span>
                <span className="col-span-1 text-center">Actions</span>
              </div>

              {/* Rows */}
              {filtered.length === 0 ? (
                <div
                  className="py-16 text-center text-sm"
                  style={{ background: c.surface, color: c.textDim }}
                >
                  No scans match your filter.
                </div>
              ) : (
                filtered.map((r, i) => {
                  const isFake = r.label === 'FAKE'
                  const dateStr = r.scannedAt
                    ? new Date(r.scannedAt).toLocaleString('en-US', {
                        month: 'short', day: 'numeric', year: 'numeric',
                        hour: '2-digit', minute: '2-digit',
                      })
                    : '—'
                  return (
                    <div
                      key={r.video_id + '-' + i}
                      className="grid grid-cols-12 px-5 py-4 items-center transition-colors duration-150 cursor-pointer"
                      style={{
                        background: c.surface,
                        borderBottom: i < filtered.length - 1 ? `1px solid ${c.border}` : 'none',
                      }}
                      onMouseEnter={e => e.currentTarget.style.background = c.surfaceHigh}
                      onMouseLeave={e => e.currentTarget.style.background = c.surface}
                      onClick={() => handleViewResult(r)}
                    >
                      {/* Video ID */}
                      <div className="col-span-3 flex items-center gap-2">
                        <span
                          className="w-2 h-2 rounded-full shrink-0"
                          style={{ background: isFake ? c.red : c.green }}
                        />
                        <span className="text-sm font-mono font-semibold truncate" style={{ color: c.text }}>
                          {r.video_id?.slice(0, 12)}…
                        </span>
                      </div>

                      {/* Date */}
                      <div className="col-span-3">
                        <span className="text-xs" style={{ color: c.textDim }}>{dateStr}</span>
                      </div>

                      {/* Verdict badge */}
                      <div className="col-span-2 flex justify-center">
                        <span
                          className="px-3 py-1 rounded text-xs font-bold"
                          style={{
                            color: isFake ? c.red : c.green,
                            background: isFake ? c.redBg : c.greenBg,
                            border: `1px solid ${isFake ? c.redBorder : c.greenBorder}`,
                          }}
                        >
                          {r.label}
                        </span>
                      </div>

                      {/* Confidence */}
                      <div className="col-span-1 text-center">
                        <span
                          className="text-sm font-semibold"
                          style={{ color: isFake ? c.red : c.green }}
                        >
                          {(r.confidence * 100).toFixed(1)}%
                        </span>
                      </div>

                      {/* Filename */}
                      <div className="col-span-2">
                        <span className="text-xs truncate block" style={{ color: c.textMuted }}>
                          {r.fileName || '—'}
                        </span>
                      </div>

                      {/* Actions */}
                      <div className="col-span-1 flex justify-center gap-2">
                        <button
                          title="View details"
                          className="w-7 h-7 rounded flex items-center justify-center transition-colors"
                          style={{ color: c.textDim }}
                          onMouseEnter={e => { e.currentTarget.style.color = c.primary; e.currentTarget.style.background = c.primaryBg }}
                          onMouseLeave={e => { e.currentTarget.style.color = c.textDim; e.currentTarget.style.background = 'transparent' }}
                          onClick={(e) => { e.stopPropagation(); handleViewResult(r) }}
                        >
                          <span className="material-symbols-outlined text-base">open_in_new</span>
                        </button>
                      </div>
                    </div>
                  )
                })
              )}
            </div>
          )}

          {/* Count hint */}
          {history.length > 0 && (
            <p className="text-xs text-center pb-4" style={{ color: c.textDim }}>
              Showing {filtered.length} of {history.length} scans
            </p>
          )}
        </div>
      </main>
    </div>
  )
}
