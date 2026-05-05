import { Link } from 'react-router-dom'
import Header from '../components/Header'

export default function About() {
  return (
    <div
      className="min-h-screen flex flex-col"
      style={{ backgroundColor: '#09090b', color: '#fafafa', fontFamily: 'Geist, sans-serif' }}
    >
      <Header />

      {/* ── Hero ── */}
      <section className="flex flex-col items-center text-center px-6 pt-36 pb-20">
        <h1
          className="text-5xl md:text-6xl font-black tracking-tight leading-tight max-w-2xl"
          style={{ color: '#fafafa' }}
        >
          Precision in{' '}
          <span style={{ color: '#a78bfa' }}>Detection</span>.
        </h1>
        <p
          className="mt-5 text-base md:text-lg leading-relaxed max-w-xl"
          style={{ color: '#a1a1aa' }}
        >
          Authentix is a multimodal, explainable AI system engineered to detect synthetic media with
          high precision in dark, high-contrast environments.
        </p>
      </section>

      {/* ── What is deepfake detection ── */}
      <section className="flex-1 px-6 md:px-16 lg:px-24 pb-24">
        <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">

          {/* Left: text */}
          <div>
            <h2
              className="text-2xl font-bold mb-6"
              style={{ color: '#fafafa' }}
            >
              What is deepfake detection?
            </h2>
            <p className="text-sm leading-relaxed mb-5" style={{ color: '#a1a1aa' }}>
              As generative AI evolves, distinguishing synthetic media from authentic reality becomes
              increasingly complex. Deepfake detection is the algorithmic process of analyzing visual
              and auditory data to identify manipulated patterns, artifacts, and anomalies introduced
              by neural networks.
            </p>
            <p className="text-sm leading-relaxed" style={{ color: '#a1a1aa' }}>
              Authentix utilizes an ensemble approach, analyzing spectral inconsistencies, biometric
              discrepancies, and spatial artifacts that remain invisible to the human eye.
            </p>

            {/* Feature pills */}
            <div className="flex flex-wrap gap-3 mt-8">
              {[
                { icon: 'visibility', label: 'Visual Analysis' },
                { icon: 'graphic_eq', label: 'Audio Analysis' },
                { icon: 'psychology', label: 'Explainable AI' },
                { icon: 'speed', label: 'Real-Time' },
              ].map((f) => (
                <div
                  key={f.label}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium"
                  style={{ background: '#121215', border: '1px solid #27272a', color: '#d4d4d8' }}
                >
                  <span className="material-symbols-outlined text-sm" style={{ color: '#a78bfa' }}>{f.icon}</span>
                  {f.label}
                </div>
              ))}
            </div>
          </div>

          {/* Right: face wireframe card */}
          <div
            className="relative rounded-xl overflow-hidden"
            style={{ border: '1px solid #27272a', background: '#0c0c0f' }}
          >
            {/* Gradient overlay */}
            <div
              className="absolute inset-0 z-10 pointer-events-none"
              style={{ background: 'linear-gradient(to top, #09090b 0%, transparent 60%)' }}
            />

            <img
              src="https://lh3.googleusercontent.com/aida-public/AB6AXuAiHcniNaHG8cW8e-97c5Q0d8_YVMn0nAfDEDCrMXrJfVibNSBq8j7ZD-o5CnjsT1LS8Xqr1eB7VaV0fQg_pSr0HElTpqvZkm7pDOphFDhRUBBxJh4H4V5gS6qkLaI7VqYm6H2H0XN8O7hLVb0kVipqKqY0mFw0tqHxnVFV5HJR6Q1LvwG7V2Jp8Tn9yH7J0EBU0JVx_4wC-Tgc-HrLIxrP2-7r2AYmf98QeSNX8HCDijjnuuTNQiHjJDuH5Uj2VgaZtT5lHVbr"
              alt="3D face wireframe scan"
              className="w-full h-full object-cover opacity-80"
              style={{ minHeight: '360px' }}
              onError={(e) => {
                // fallback: show a dark placeholder
                e.currentTarget.style.display = 'none'
              }}
            />

            {/* Status badges */}
            <div className="absolute bottom-5 left-5 z-20 flex gap-2">
              <span
                className="text-xs font-semibold px-3 py-1.5 rounded"
                style={{ background: '#7c3aed', color: '#ede9fe' }}
              >
                Analysis Active
              </span>
              <span
                className="text-xs font-semibold px-3 py-1.5 rounded"
                style={{ background: '#065f46', color: '#bbf7d0' }}
              >
                Artifacts Detected
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* ── How it works ── */}
      <section className="px-6 md:px-16 lg:px-24 pb-24">
        <div className="max-w-6xl mx-auto">
          <h2
            className="text-xl font-bold mb-8 text-center"
            style={{ color: '#fafafa' }}
          >
            How Authentix Works
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              {
                step: '01',
                icon: 'input',
                title: 'Ingest Media',
                desc: 'Upload a video file or connect a live stream. Authentix accepts MP4, MOV, WebM, and RTSP feeds.',
              },
              {
                step: '02',
                icon: 'hub',
                title: 'Multi-Model Analysis',
                desc: 'Ensemble of GAN detectors, audio-visual sync analyzers, and biometric validators run in parallel.',
              },
              {
                step: '03',
                icon: 'verified',
                title: 'Explainable Verdict',
                desc: 'Receive a confidence-scored result with a detailed breakdown of every heuristic that was triggered.',
              },
            ].map((s) => (
              <div
                key={s.step}
                className="p-6 rounded-xl flex flex-col gap-4"
                style={{ background: '#111113', border: '1px solid #27272a' }}
              >
                <div className="flex items-center gap-3">
                  <span
                    className="text-3xl font-black"
                    style={{ color: '#1e1e22' }}
                  >
                    {s.step}
                  </span>
                  <div
                    className="w-9 h-9 rounded-lg flex items-center justify-center"
                    style={{ background: 'rgba(167,139,250,0.1)', border: '1px solid rgba(167,139,250,0.2)' }}
                  >
                    <span className="material-symbols-outlined text-lg" style={{ color: '#a78bfa' }}>{s.icon}</span>
                  </div>
                </div>
                <h3 className="text-sm font-semibold" style={{ color: '#fafafa' }}>{s.title}</h3>
                <p className="text-xs leading-relaxed" style={{ color: '#71717a' }}>{s.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer
        className="px-6 md:px-16 lg:px-24 py-6 flex flex-col md:flex-row justify-between items-center gap-4"
        style={{ borderTop: '1px solid #18181b' }}
      >
        <span className="font-bold text-sm" style={{ color: '#fafafa' }}>Authentix</span>
        <span className="text-xs" style={{ color: '#52525b' }}>
          © 2024 Authentix Systems. All rights reserved.
        </span>
        <div className="flex gap-5">
          {['Privacy Policy', 'Terms of Service', 'Security Documentation'].map((l) => (
            <Link
              key={l}
              to="#"
              className="text-xs transition-colors hover:text-violet-400"
              style={{ color: '#71717a' }}
            >
              {l}
            </Link>
          ))}
        </div>
      </footer>
    </div>
  )
}
