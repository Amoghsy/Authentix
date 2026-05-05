import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { ThemeProvider } from './context/ThemeContext'
import { AuthProvider } from './context/AuthContext'
import { DetectionProvider } from './context/DetectionContext'
import Home from './pages/Home'
import RealTimeDetection from './pages/RealTimeDetection'
import AnalysisResults from './pages/AnalysisResults'
import About from './pages/About'
import Reports from './pages/Reports'
import History from './pages/History'
import Settings from './pages/Settings'
import Profile from './pages/Profile'

export default function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <DetectionProvider>
          <BrowserRouter>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/detect" element={<RealTimeDetection />} />
              <Route path="/analysis" element={<AnalysisResults />} />
              <Route path="/about" element={<About />} />
              <Route path="/reports" element={<Reports />} />
              <Route path="/history" element={<History />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/profile" element={<Profile />} />
            </Routes>
          </BrowserRouter>
        </DetectionProvider>
      </AuthProvider>
    </ThemeProvider>
  )
}
