import { Suspense, lazy, useEffect, useState, useRef } from 'react'
import { Link } from 'react-router-dom'
import Header from '../components/Header'
import { Banner } from '../components/ui/banner'

const Spline = lazy(() => import('@splinetool/react-spline'))

/* ── Animated counter hook ── */
function useCounter(target, duration = 2000) {
  const [val, setVal] = useState(0)
  const ref = useRef(null)
  const started = useRef(false)

  useEffect(() => {
    const observer = new IntersectionObserver(([e]) => {
      if (e.isIntersecting && !started.current) {
        started.current = true
        const start = performance.now()
        const tick = (now) => {
          const t = Math.min((now - start) / duration, 1)
          const ease = 1 - Math.pow(1 - t, 3)
          setVal(Math.round(ease * target))
          if (t < 1) requestAnimationFrame(tick)
        }
        requestAnimationFrame(tick)
      }
    }, { threshold: 0.3 })
    if (ref.current) observer.observe(ref.current)
    return () => observer.disconnect()
  }, [target, duration])

  return [val, ref]
}

/* ── Floating particle component ── */
function Particles() {
  const particles = Array.from({ length: 40 }, (_, i) => ({
    id: i,
    x: Math.random() * 100,
    y: Math.random() * 100,
    size: Math.random() * 3 + 1,
    dur: Math.random() * 20 + 15,
    delay: Math.random() * -20,
    opacity: Math.random() * 0.4 + 0.1,
  }))

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none z-0">
      {particles.map(p => (
        <div
          key={p.id}
          className="absolute rounded-full"
          style={{
            left: `${p.x}%`,
            top: `${p.y}%`,
            width: p.size,
            height: p.size,
            background: p.id % 3 === 0 ? '#a78bfa' : p.id % 3 === 1 ? '#7c3aed' : '#34d399',
            opacity: p.opacity,
            animation: `float-particle ${p.dur}s ease-in-out ${p.delay}s infinite`,
          }}
        />
      ))}
    </div>
  )
}

/* ── Typewriter effect ── */
function Typewriter({ words, speed = 100, pause = 2000 }) {
  const [text, setText] = useState('')
  const [wordIdx, setWordIdx] = useState(0)
  const [charIdx, setCharIdx] = useState(0)
  const [deleting, setDeleting] = useState(false)

  useEffect(() => {
    const word = words[wordIdx]
    const timeout = setTimeout(() => {
      if (!deleting) {
        setText(word.slice(0, charIdx + 1))
        if (charIdx + 1 === word.length) {
          setTimeout(() => setDeleting(true), pause)
        } else {
          setCharIdx(c => c + 1)
        }
      } else {
        setText(word.slice(0, charIdx))
        if (charIdx === 0) {
          setDeleting(false)
          setWordIdx(i => (i + 1) % words.length)
        } else {
          setCharIdx(c => c - 1)
        }
      }
    }, deleting ? speed / 2 : speed)
    return () => clearTimeout(timeout)
  }, [charIdx, deleting, wordIdx, words, speed, pause])

  return (
    <span>
      {text}
      <span className="inline-block w-0.5 h-[1em] ml-0.5 align-middle" style={{ background: '#a78bfa', animation: 'blink-cursor 0.8s steps(2) infinite' }} />
    </span>
  )
}

/* ── Stats section ── */
function StatsSection() {
  const stats = [
    { label: 'AI Models', target: 9, suffix: '', icon: 'memory', color: '#a78bfa' },
    { label: 'Accuracy', target: 97, suffix: '%', icon: 'verified', color: '#34d399' },
    { label: 'Signals Analyzed', target: 15, suffix: '+', icon: 'analytics', color: '#f59e0b' },
    { label: 'Latency', target: 2, suffix: 's', icon: 'speed', color: '#ec4899' },
  ]

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full max-w-4xl mx-auto">
      {stats.map((s, i) => {
        const [val, ref] = useCounter(s.target, 1800)
        return (
          <div
            key={s.label}
            ref={ref}
            className="ax-fade-up flex flex-col items-center gap-2 p-6 rounded-xl relative overflow-hidden group"
            style={{
              background: 'rgba(18,18,21,0.6)',
              border: '1px solid #27272a',
              backdropFilter: 'blur(12px)',
              animationDelay: `${0.8 + i * 0.15}s`,
            }}
          >
            <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
              style={{ background: `radial-gradient(circle at center, ${s.color}10, transparent 70%)` }}
            />
            <span className="material-symbols-outlined text-2xl transition-transform duration-300 group-hover:scale-110" style={{ color: s.color }}>{s.icon}</span>
            <span className="text-3xl font-black tracking-tight" style={{ color: '#fafafa' }}>
              {val}{s.suffix}
            </span>
            <span className="text-xs font-medium uppercase tracking-widest" style={{ color: '#71717a' }}>{s.label}</span>
          </div>
        )
      })}
    </div>
  )
}

/* ── Pipeline visual ── */
function PipelineStrip() {
  const steps = [
    { icon: 'upload_file', label: 'Upload' },
    { icon: 'movie', label: 'Frames' },
    { icon: 'graphic_eq', label: 'Audio' },
    { icon: 'memory', label: 'AI Models' },
    { icon: 'device_hub', label: 'Fusion' },
    { icon: 'verified', label: 'Verdict' },
  ]

  return (
    <div className="w-full max-w-3xl mx-auto ax-fade-up" style={{ animationDelay: '1.4s' }}>
      <p className="text-xs font-bold uppercase tracking-[0.2em] text-center mb-5" style={{ color: '#52525b' }}>Detection Pipeline</p>
      <div className="flex items-center justify-between relative">
        {/* Connecting line */}
        <div className="absolute top-1/2 left-8 right-8 h-px" style={{ background: 'linear-gradient(to right, transparent, rgba(167,139,250,0.5), transparent)', boxShadow: '0 0 8px rgba(167,139,250,0.3)', animation: 'pipeline-line-glow 3s ease-in-out infinite' }} />
        {steps.map((s, i) => (
          <div
            key={s.label}
            className="relative z-10 flex flex-col items-center gap-2 group"
            style={{ animation: `pipeline-pop 0.5s ease-out ${1.6 + i * 0.12}s both` }}
          >
            <div
              className="w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300 group-hover:scale-125"
              style={{
                background: '#121215',
                border: '1px solid #3f3f46',
                boxShadow: '0 0 8px rgba(167,139,250,0.15)',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.borderColor = '#a78bfa'
                e.currentTarget.style.boxShadow = '0 0 25px rgba(167,139,250,0.5), 0 0 50px rgba(124,58,237,0.2)'
                e.currentTarget.style.background = '#1a1a2e'
              }}
              onMouseLeave={e => {
                e.currentTarget.style.borderColor = '#3f3f46'
                e.currentTarget.style.boxShadow = '0 0 8px rgba(167,139,250,0.15)'
                e.currentTarget.style.background = '#121215'
              }}
            >
              <span className="material-symbols-outlined text-base" style={{ color: '#a78bfa' }}>{s.icon}</span>
            </div>
            <span className="text-xs font-medium" style={{ color: '#52525b' }}>{s.label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════ */
export default function Home() {
  const [loaded, setLoaded] = useState(false)
  useEffect(() => { setLoaded(true) }, [])

  return (
    <div className="min-h-screen overflow-hidden relative" style={{ backgroundColor: '#09090b', color: '#fafafa', fontFamily: 'Geist, sans-serif' }}>

      {/* ── Animated background layers ── */}
      <div className="absolute inset-0 z-0">
        {/* Gradient mesh */}
        <div className="absolute inset-0" style={{ background: 'radial-gradient(ellipse at 20% 50%, rgba(124,58,237,0.12) 0%, transparent 50%), radial-gradient(ellipse at 80% 20%, rgba(167,139,250,0.1) 0%, transparent 50%), radial-gradient(ellipse at 50% 80%, rgba(52,211,153,0.06) 0%, transparent 50%)' }} />

        {/* Large orbiting glows — brighter */}
        <div className="absolute w-[700px] h-[700px] rounded-full pointer-events-none" style={{ top: '5%', left: '55%', background: 'radial-gradient(circle, rgba(167,139,250,0.15), rgba(124,58,237,0.05) 50%, transparent 70%)', animation: 'orbit-glow 18s ease-in-out infinite', filter: 'blur(40px)' }} />
        <div className="absolute w-[600px] h-[600px] rounded-full pointer-events-none" style={{ bottom: '5%', left: '5%', background: 'radial-gradient(circle, rgba(52,211,153,0.12), rgba(16,185,129,0.04) 50%, transparent 70%)', animation: 'orbit-glow 22s ease-in-out infinite reverse', filter: 'blur(40px)' }} />
        <div className="absolute w-[500px] h-[500px] rounded-full pointer-events-none" style={{ top: '40%', left: '30%', background: 'radial-gradient(circle, rgba(236,72,153,0.06), transparent 70%)', animation: 'orbit-glow 30s ease-in-out infinite', filter: 'blur(60px)' }} />

        {/* Aurora ribbon */}
        <div className="absolute w-full h-[300px] pointer-events-none" style={{ top: '15%', background: 'linear-gradient(90deg, transparent 0%, rgba(124,58,237,0.08) 20%, rgba(167,139,250,0.12) 40%, rgba(52,211,153,0.06) 60%, rgba(167,139,250,0.08) 80%, transparent 100%)', animation: 'aurora-drift 12s ease-in-out infinite', filter: 'blur(80px)', transform: 'skewY(-3deg)' }} />

        {/* Grid overlay */}
        <div className="absolute inset-0 opacity-[0.03]" style={{ backgroundImage: 'linear-gradient(#a78bfa 1px, transparent 1px), linear-gradient(90deg, #a78bfa 1px, transparent 1px)', backgroundSize: '60px 60px' }} />
      </div>

      <Particles />
      <Header />

      {/* ═══ Hero Section ═══ */}
      <main className="relative z-10 flex flex-col items-center w-full min-h-screen px-6 md:px-12 max-w-[1400px] mx-auto pt-20 pb-10">
        <div className="flex flex-col lg:flex-row w-full gap-12 items-center justify-between flex-1">

          {/* Left Column */}
          <div
            className={`w-full lg:w-1/2 flex flex-col items-start text-left space-y-10 p-10 md:p-14 relative overflow-visible transition-all duration-1000 ${loaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}
            style={{ borderRadius: '1.25rem' }}
          >
            {/* ✦ Rotating gradient border glow */}
            <div className="absolute -inset-[1px] rounded-[1.25rem] pointer-events-none z-0" style={{ background: 'conic-gradient(from var(--glow-angle, 0deg), transparent 0%, rgba(167,139,250,0.5) 10%, transparent 20%, transparent 50%, rgba(52,211,153,0.3) 60%, transparent 70%, transparent 100%)', animation: 'rotate-glow 6s linear infinite', filter: 'blur(2px)' }} />
            {/* Outer soft aura */}
            <div className="absolute -inset-4 rounded-[2rem] pointer-events-none z-0" style={{ background: 'radial-gradient(ellipse at 30% 20%, rgba(167,139,250,0.12), transparent 60%)', animation: 'glow-pulse 4s ease-in-out infinite', filter: 'blur(20px)' }} />

            {/* Inner card fill */}
            <div className="absolute inset-0 rounded-[1.25rem] z-0" style={{ background: 'rgba(18,18,21,0.7)', backdropFilter: 'blur(24px)', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.8)' }} />

            {/* Internal violet glow — stronger */}
            <div className="absolute inset-0 rounded-[1.25rem] pointer-events-none z-0" style={{ background: 'radial-gradient(circle at top left, rgba(167,139,250,0.18), transparent 55%), radial-gradient(circle at bottom right, rgba(52,211,153,0.06), transparent 50%)', animation: 'glow-pulse 5s ease-in-out infinite' }} />

            <div className="space-y-5 relative z-10">
              <h1
                className="text-6xl md:text-7xl xl:text-8xl font-extrabold ax-fade-up"
                style={{ letterSpacing: '-0.04em', color: '#fafafa', lineHeight: '1.1', animationDelay: '0.1s', textShadow: '0 0 40px rgba(167,139,250,0.35), 0 0 80px rgba(167,139,250,0.15), 0 0 120px rgba(124,58,237,0.1)' }}
              >
                Authentix
              </h1>
              <p className="text-xl md:text-2xl font-semibold ax-fade-up" style={{ color: '#a78bfa', letterSpacing: '-0.01em', animationDelay: '0.25s' }}>
                <Typewriter words={['Multimodal Deepfake Detection', 'AI-Powered Video Analysis', ' Neural Networks, One Verdict', 'Real-Time Authenticity Guard']} speed={80} pause={2500} />
              </p>
              <p className="text-base md:text-lg leading-relaxed max-w-md ax-fade-up" style={{ color: '#a1a1aa', animationDelay: '0.4s' }}>
                Detect fake videos in real-time using AI-powered visual and audio analysis across <strong style={{ color: '#fafafa' }}></strong>.
              </p>
            </div>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto relative z-10 ax-fade-up" style={{ animationDelay: '0.55s' }}>
              <Link
                to="/detect"
                className="group flex items-center justify-center gap-2 font-semibold px-8 py-4 transition-all duration-300 hover:scale-[1.03] active:scale-95 relative overflow-hidden"
                style={{ background: 'linear-gradient(135deg, #a78bfa, #7c3aed)', color: '#fff', borderRadius: '0.5rem', boxShadow: '0 0 25px rgba(167,139,250,0.4), 0 0 60px rgba(124,58,237,0.15)', animation: 'btn-glow-pulse 3s ease-in-out infinite' }}
                onMouseEnter={e => e.currentTarget.style.boxShadow = '0 0 40px rgba(167,139,250,0.7), 0 0 80px rgba(124,58,237,0.3)'}
                onMouseLeave={e => e.currentTarget.style.boxShadow = '0 0 25px rgba(167,139,250,0.4), 0 0 60px rgba(124,58,237,0.15)'}
              >
                {/* Shimmer effect */}
                <div className="absolute inset-0 -translate-x-full group-hover:translate-x-full transition-transform duration-700" style={{ background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent)' }} />
                <span className="material-symbols-outlined text-lg">videocam</span>
                Start Detection
              </Link>
              <Link
                to="/analysis"
                className="group flex items-center justify-center gap-2 font-semibold px-8 py-4 transition-all duration-300 hover:scale-[1.03] active:scale-95"
                style={{ background: 'transparent', color: '#fafafa', borderRadius: '0.5rem', border: '1px solid #27272a' }}
                onMouseEnter={e => { e.currentTarget.style.background = '#121215'; e.currentTarget.style.borderColor = 'rgba(167,139,250,0.5)' }}
                onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.borderColor = '#27272a' }}
              >
                <span className="material-symbols-outlined text-lg" style={{ color: '#a78bfa' }}>folder</span>
                View Results
              </Link>
            </div>

            {/* Rainbow Banner */}
            <div className="w-full mt-4 z-20 relative ax-fade-up" style={{ animationDelay: '0.65s' }}>
              <Banner
                id="home-banner"
                variant="rainbow"
                className="shadow-lg bg-zinc-900/80 backdrop-blur-md"
                rainbowColors={[
                  "rgba(167,139,250,0.4)", "rgba(139,92,246,0.4)", "transparent",
                  "rgba(167,139,250,0.4)", "transparent", "rgba(139,92,246,0.4)", "transparent",
                ]}
              >
                🚀 Project evolving with more features soon!
              </Banner>
            </div>

            {/* Feature tags */}
            <div className="flex flex-wrap justify-start gap-3 pt-6 relative z-10 border-t border-zinc-800/50 w-full ax-fade-up" style={{ animationDelay: '0.75s' }}>
              {[
                { label: 'Real-Time', color: '#34d399', pulse: true },
                { label: 'Audio + Video', color: '#a1a1aa' },
                { label: 'Explainable AI', color: '#a1a1aa' },
                { label: ' AI Models', color: '#a78bfa' },
              ].map(tag => (
                <span
                  key={tag.label}
                  className="px-3 py-1.5 text-xs font-medium rounded-full flex items-center gap-1.5 transition-transform duration-200 hover:scale-105 cursor-default"
                  style={{ color: tag.color, background: tag.pulse ? 'rgba(52,211,153,0.1)' : '#121215', border: `1px solid ${tag.pulse ? 'rgba(52,211,153,0.2)' : '#27272a'}` }}
                >
                  {tag.pulse && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse inline-block" />}
                  {tag.label}
                </span>
              ))}
            </div>
          </div>

          {/* Right Column: Spline 3D */}
          <div className={`hidden lg:flex w-full lg:w-1/2 relative items-center justify-center h-[600px] transition-all duration-1000 delay-300 ${loaded ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-12'}`}>
            <Suspense fallback={
              <div className="w-full h-full flex flex-col items-center justify-center text-zinc-600 space-y-4">
                <div className="w-10 h-10 rounded-full border-4 border-t-transparent animate-spin" style={{ borderColor: '#8b5cf6', borderTopColor: 'transparent' }} />
                <p className="text-sm font-medium tracking-wide animate-pulse">Loading 3D Scene...</p>
              </div>
            }>
              <div id="spline-container" className="w-full h-full relative" style={{ mixBlendMode: 'screen' }}>
                <Spline scene="https://prod.spline.design/c6ryy5i3MC28oW9W/scene.splinecode" />
              </div>
            </Suspense>
          </div>
        </div>

        {/* ═══ Stats + Pipeline below hero ═══ */}
        <div className="w-full flex flex-col gap-12 mt-16 pb-12">
          <StatsSection />
          <PipelineStrip />
        </div>
      </main>

      {/* ═══ Keyframe animations ═══ */}
      <style>{`
        /* Fade-up entrance */
        .ax-fade-up {
          opacity: 0;
          transform: translateY(24px);
          animation: ax-fade-up 0.8s ease-out forwards;
        }
        @keyframes ax-fade-up {
          to { opacity: 1; transform: translateY(0); }
        }

        /* Floating particles */
        @keyframes float-particle {
          0%, 100% { transform: translate(0, 0) scale(1); opacity: var(--o, 0.2); }
          25% { transform: translate(30px, -40px) scale(1.3); }
          50% { transform: translate(-20px, -80px) scale(0.8); opacity: calc(var(--o, 0.2) + 0.15); }
          75% { transform: translate(40px, -30px) scale(1.1); }
        }

        /* Orbiting glow */
        @keyframes orbit-glow {
          0%, 100% { transform: translate(0, 0) scale(1); }
          33% { transform: translate(-60px, 40px) scale(1.1); }
          66% { transform: translate(40px, -30px) scale(0.9); }
        }

        /* Rotating conic gradient border — uses CSS @property workaround */
        @keyframes rotate-glow {
          from { filter: blur(2px) hue-rotate(0deg); }
          to { filter: blur(2px) hue-rotate(360deg); }
        }

        /* Inner glow pulse */
        @keyframes glow-pulse {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }

        /* CTA button glow pulse */
        @keyframes btn-glow-pulse {
          0%, 100% { box-shadow: 0 0 25px rgba(167,139,250,0.4), 0 0 60px rgba(124,58,237,0.15); }
          50% { box-shadow: 0 0 35px rgba(167,139,250,0.55), 0 0 80px rgba(124,58,237,0.25); }
        }

        /* Aurora drift */
        @keyframes aurora-drift {
          0%, 100% { transform: skewY(-3deg) translateX(0); opacity: 0.6; }
          50% { transform: skewY(-1deg) translateX(-40px); opacity: 1; }
        }

        /* Pipeline connecting line glow */
        @keyframes pipeline-line-glow {
          0%, 100% { opacity: 0.5; }
          50% { opacity: 1; box-shadow: 0 0 12px rgba(167,139,250,0.5); }
        }

        /* Cursor blink */
        @keyframes blink-cursor {
          0% { opacity: 1; }
          50% { opacity: 0; }
        }

        /* Pipeline nodes pop in */
        @keyframes pipeline-pop {
          from { opacity: 0; transform: scale(0.5) translateY(10px); }
          to { opacity: 1; transform: scale(1) translateY(0); }
        }
      `}</style>
    </div>
  )
}
