/**
 * AppLayout.jsx
 * Shared layout wrapper for all authenticated sidebar pages.
 * Eliminates duplicated Sidebar + Header boilerplate across every page.
 */
import Sidebar from './Sidebar'
import Header from './Header'

export default function AppLayout({ children, mainClassName = '' }) {
  return (
    <div
      className="flex h-screen overflow-hidden"
      style={{ backgroundColor: '#09090b', color: '#fafafa', fontFamily: 'Geist, sans-serif' }}
    >
      {/* Fixed-width sidebar — hidden on mobile */}
      <Sidebar />

      {/* Right column: sticky header + scrollable content */}
      <div className="flex-1 flex flex-col h-full overflow-hidden min-w-0">
        <Header sticky />
        <main className={`flex-1 overflow-y-auto ${mainClassName}`}>
          {children}
        </main>
      </div>
    </div>
  )
}
