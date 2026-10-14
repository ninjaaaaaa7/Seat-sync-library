import { useState, useEffect } from 'react'
import DetectPanel from './DetectPanel.jsx'

// Local fixed service base. Overridable via Vite env.
const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

const INITIAL_ZONES = [
  {
    id: 'zone-1',
    name: 'Main Reading Hall',
    floor: 'Floor 1 — West Wing',
    type: 'Silent Study',
    totalSeats: 36,
    availSeats: 22,
    noiseLevel: 'Silent',
    powerOutlets: 'All Desks',
    lastSync: '2 mins ago',
  },
  {
    id: 'zone-2',
    name: 'STEM & Reference Commons',
    floor: 'Floor 2 — North Wing',
    type: 'Study Desks',
    totalSeats: 24,
    availSeats: 7,
    noiseLevel: 'Quiet',
    powerOutlets: 'Available',
    lastSync: '5 mins ago',
  },
  {
    id: 'zone-3',
    name: 'Collaborative Study Pods',
    floor: 'Ground Floor — Central',
    type: 'Group Work',
    totalSeats: 16,
    availSeats: 0,
    noiseLevel: 'Moderate',
    powerOutlets: 'Available',
    lastSync: 'Just now',
  },
  {
    id: 'zone-4',
    name: 'Periodicals & Journal Mezzanine',
    floor: 'Floor 3 — Quiet Stacks',
    type: 'Armchairs & Desks',
    totalSeats: 20,
    availSeats: 14,
    noiseLevel: 'Silent',
    powerOutlets: 'Limited',
    lastSync: '10 mins ago',
  }
]

export default function App() {
  const [activePortal, setActivePortal] = useState('student') // 'student' | 'admin'
  const [zones, setZones] = useState(INITIAL_ZONES)
  const [selectedZoneId, setSelectedZoneId] = useState('zone-1')
  const [serverOnline, setServerOnline] = useState(false)
  const [telemetry, setTelemetry] = useState(null)

  // Probe backend status periodically for university system status bar
  useEffect(() => {
    let mounted = true
    const checkServer = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2000) })
        if (res.ok && mounted) {
          setServerOnline(true)
          fetch(`${API_BASE}/status`, { signal: AbortSignal.timeout(2000) })
            .then(r => r.ok ? r.json() : null)
            .then(data => { if (mounted && data) setTelemetry(data) })
            .catch(() => {})
        } else if (mounted) {
          setServerOnline(false)
        }
      } catch {
        if (mounted) setServerOnline(false)
      }
    }

    checkServer()
    const id = setInterval(checkServer, 5000)
    return () => {
      mounted = false
      clearInterval(id)
    }
  }, [])

  // Callback whenever admin scans an image/video to update zone occupancy
  const handleScanCompleted = (scanData) => {
    if (!scanData) return
    const chairCount = scanData.chair_count || (scanData.counts?.chair ?? 0)
    const availCount = scanData.available_chair_count != null 
      ? scanData.available_chair_count 
      : Math.max(0, chairCount - (scanData.person_count ?? 0))

    if (chairCount > 0) {
      setZones(prev => prev.map(z => {
        if (z.id === selectedZoneId) {
          return {
            ...z,
            totalSeats: chairCount,
            availSeats: availCount,
            lastSync: 'Just now (AI Verified)'
          }
        }
        return z
      }))
    }
  }

  const selectedZone = zones.find(z => z.id === selectedZoneId) || zones[0]
  const totalLibrarySeats = zones.reduce((acc, z) => acc + z.totalSeats, 0)
  const totalLibraryAvail = zones.reduce((acc, z) => acc + z.availSeats, 0)
  const overallPct = Math.round(((totalLibrarySeats - totalLibraryAvail) / totalLibrarySeats) * 100)

  return (
    <div className="app">
      {/* University Institutional Header */}
      <header className="univ-header">
        <div className="univ-topbar">
          <div className="univ-identity">
            <div className="univ-crest" title="University Library System">UL</div>
            <div className="univ-title">
              <h1>University Central Library</h1>
              <div className="univ-sub">SeatSync — Real-Time Spatial Occupancy Management</div>
            </div>
          </div>

          <nav className="portal-tabs" aria-label="Portal Mode">
            <button
              className={`portal-btn ${activePortal === 'student' ? 'active' : ''}`}
              onClick={() => setActivePortal('student')}
            >
              🎓 Student Seat Finder
            </button>
            <button
              className={`portal-btn ${activePortal === 'admin' ? 'active' : ''}`}
              onClick={() => setActivePortal('admin')}
            >
              🛡️ Library Admin Portal
              <span className="badge-admin">Audit</span>
            </button>
          </nav>
        </div>
      </header>

      {/* Institutional Status & Notice Bar */}
      <div className="univ-notice-bar">
        <div className="univ-notice-inner">
          <div className="notice-pill">
            <span className={`status-dot ${serverOnline ? '' : 'offline'}`} />
            {serverOnline ? 'Vision Inference Engine Online (CUDA RTX 2050)' : 'Vision Engine Offline (Host Service Standby)'}
          </div>
          <div>
            System Library Capacity: <strong>{totalLibraryAvail} / {totalLibrarySeats} Available</strong> ({overallPct}% Occupied)
          </div>
          <div>
            Building Hours: <strong>07:30 – 23:00</strong> · Quiet Hours active
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <main className="univ-container">
        {activePortal === 'student' ? (
          <StudentPortal
            zones={zones}
            selectedZone={selectedZone}
            onSelectZone={(id) => setSelectedZoneId(id)}
            serverOnline={serverOnline}
            onSwitchToAdmin={() => setActivePortal('admin')}
          />
        ) : (
          <AdminPortal
            apiBase={API_BASE}
            zones={zones}
            selectedZone={selectedZone}
            onSelectZone={(id) => setSelectedZoneId(id)}
            telemetry={telemetry}
            serverOnline={serverOnline}
            onScanResult={handleScanCompleted}
          />
        )}
      </main>

      {/* Institutional Footer */}
      <footer className="univ-footer">
        <div className="univ-footer-inner">
          <div>
            © University Library Spatial Information Services · Smart Seat Sync (Automated Occupancy Verification)
          </div>
          <div>
            Powered by YOLOv8/YOLO26 Spatial Geometry Engine · Normalized Layout Mapping
          </div>
        </div>
      </footer>
    </div>
  )
}

/**
 * Student Portal: Clean, distraction-free spatial vacancy viewer.
 * Helps students decide where to study without navigating AI diagnostic controls.
 */
function StudentPortal({ zones, selectedZone, onSelectZone, serverOnline, onSwitchToAdmin }) {
  const percentOccupied = Math.round(((selectedZone.totalSeats - selectedZone.availSeats) / selectedZone.totalSeats) * 100)
  const isFull = selectedZone.availSeats === 0
  const isModerate = !isFull && percentOccupied > 65

  const statusClass = isFull ? 'full' : isModerate ? 'moderate' : 'avail'
  const statusLabel = isFull ? 'At Full Capacity' : isModerate ? 'Limited Seating' : 'Seats Readily Available'

  // Generate deterministic mock desks for the selected zone layout
  const desks = Array.from({ length: selectedZone.totalSeats }, (_, i) => {
    const isOccupied = i >= selectedZone.availSeats
    const rowChar = String.fromCharCode(65 + Math.floor(i / 6))
    const colNum = (i % 6) + 1
    return {
      code: `${rowChar}${colNum}`,
      occupied: isOccupied,
      hasPower: i % 2 === 0,
      nearWindow: i % 4 === 0
    }
  })

  return (
    <div className="student-view">
      {/* Dynamic Headline Availability Banner */}
      <section className={`availability-banner ${statusClass}`}>
        <div className="banner-left">
          <span className="banner-zone-tag">{selectedZone.floor} · {selectedZone.type}</span>
          <div className="banner-metric">
            <span className="banner-count">{selectedZone.availSeats}</span>
            <span className="banner-label">
              open seats out of {selectedZone.totalSeats} total
            </span>
          </div>
          <div className="banner-meta">
            {statusLabel} · Last verified {selectedZone.lastSync} · Noise policy: <strong>{selectedZone.noiseLevel}</strong>
          </div>
        </div>

        <div className="recommend-box">
          <div><strong>Study Suggestion:</strong></div>
          {isFull ? (
            <div>This area is currently packed. We recommend heading to <em>Floor 3 Mezzanine</em> or <em>Floor 1</em>.</div>
          ) : selectedZone.availSeats < 5 ? (
            <div>Few remaining seats. Recommended to proceed immediately to secure an outlet desk.</div>
          ) : (
            <div>Great study conditions with ample desk space and verified power connectivity.</div>
          )}
        </div>
      </section>

      {/* Main Two-Column Layout */}
      <div className="student-grid">
        {/* Left: Zone Selection Directory */}
        <div className="academic-card">
          <div className="card-header">
            <h2>Library Study Halls</h2>
            <span style={{ fontSize: '11px', color: 'var(--text-light)' }}>{zones.length} Zones</span>
          </div>
          <div className="card-body">
            <div className="zone-list">
              {zones.map(z => {
                const pct = Math.round(((z.totalSeats - z.availSeats) / z.totalSeats) * 100)
                const zClass = z.availSeats === 0 ? 'full' : pct > 65 ? 'moderate' : 'avail'
                const zBadge = z.availSeats === 0 ? 'FULL' : `${z.availSeats} Free`

                return (
                  <div
                    key={z.id}
                    className={`zone-item ${z.id === selectedZone.id ? 'selected' : ''}`}
                    onClick={() => onSelectZone(z.id)}
                  >
                    <div className="zone-item-top">
                      <span className="zone-name">{z.name}</span>
                      <span className={`zone-status-badge ${zClass}`}>{zBadge}</span>
                    </div>
                    <div className="zone-floor">{z.floor} · {z.type}</div>
                    <div className="zone-bar-bg">
                      <div
                        className="zone-bar-fill"
                        style={{
                          width: `${pct}%`,
                          background: z.availSeats === 0 ? 'var(--status-full)' : pct > 65 ? 'var(--status-busy)' : 'var(--status-avail)'
                        }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        {/* Right: Interactive Desk Grid Layout */}
        <div className="academic-card">
          <div className="card-header">
            <h3>{selectedZone.name} — Desk Layout Matrix</h3>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              Outlets: <strong>{selectedZone.powerOutlets}</strong>
            </span>
          </div>

          <div className="card-body floor-layout-view">
            <div className="grid-legend">
              <div className="legend-item">
                <div className="legend-box seat-empty" />
                <span>Vacant Desk</span>
              </div>
              <div className="legend-item">
                <div className="legend-box seat-occ" />
                <span>Occupied</span>
              </div>
              <div className="legend-item" style={{ marginLeft: 'auto' }}>
                <button
                  className="btn-secondary"
                  style={{ padding: '3px 8px', fontSize: '11px' }}
                  onClick={onSwitchToAdmin}
                >
                  Upload CCTV Scan (Staff) →
                </button>
              </div>
            </div>

            <div className="seat-matrix">
              {desks.map(desk => (
                <div
                  key={desk.code}
                  className={`desk-cell ${desk.occupied ? 'taken' : 'free'}`}
                >
                  <div className="desk-header">
                    <span className="desk-code">{desk.code}</span>
                    <span className="desk-status">{desk.occupied ? 'Occupied' : 'Vacant'}</span>
                  </div>
                  <div className="desk-attr">
                    {desk.hasPower ? '⚡ AC Outlet' : 'Standard'}
                    {desk.nearWindow ? ' · Window' : ''}
                  </div>
                </div>
              ))}
            </div>

            <div style={{ fontSize: '11px', color: 'var(--text-light)', borderTop: '1px solid var(--border-light)', paddingTop: '10px', marginTop: '6px' }}>
              ℹ️ Spatial map reflects spatial inference from the latest camera sweep. Desks auto-refresh as students come and go.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

/**
 * Admin & Staff Portal: Full diagnostic verification, CCTV photo/video upload,
 * YOLO confidence controls, and live telemetry.
 */
function AdminPortal({ apiBase, zones, selectedZone, onSelectZone, telemetry, serverOnline, onScanResult }) {
  return (
    <div className="admin-view">
      {/* Institutional Telemetry Row */}
      <div className="admin-summary-row">
        <div className="stat-tile">
          <span className="stat-tile-title">Inference Engine</span>
          <span className="stat-tile-val">{telemetry?.device ? `CUDA:${telemetry.device}` : (serverOnline ? 'Online' : 'Standby')}</span>
          <span className="stat-tile-sub">YOLO Model: COCO Library Filtered</span>
        </div>

        <div className="stat-tile">
          <span className="stat-tile-title">Frames Processed</span>
          <span className="stat-tile-val">{telemetry?.frames_processed ?? 0}</span>
          <span className="stat-tile-sub">Throughput: {telemetry?.ema_fps ? `${telemetry.ema_fps.toFixed(1)} FPS` : 'Ready'}</span>
        </div>

        <div className="stat-tile">
          <span className="stat-tile-title">Peak GPU VRAM</span>
          <span className="stat-tile-val">{telemetry?.peak_vram_mb ? `${telemetry.peak_vram_mb.toFixed(0)} MB` : '< 400 MB'}</span>
          <span className="stat-tile-sub">4GB VRAM Hardware Budget</span>
        </div>

        <div className="stat-tile">
          <span className="stat-tile-title">Managed Library Zones</span>
          <span className="stat-tile-val">{zones.length}</span>
          <span className="stat-tile-sub">Active: {selectedZone.name}</span>
        </div>
      </div>

      {/* Main Admin Grid */}
      <div className="admin-main-grid">
        {/* Left: Deep Detection & Video Upload Scanner */}
        <div className="academic-card">
          <div className="card-header">
            <h2>CCTV Room Inspection & Seating Verification (YOLO)</h2>
            <span style={{ fontSize: '11px', color: 'var(--text-light)' }}>Image & Video Ingestion</span>
          </div>

          <div className="card-body">
            <DetectPanel
              apiBase={apiBase}
              selectedZoneName={selectedZone.name}
              onScanSuccess={onScanResult}
            />
          </div>
        </div>

        {/* Right: Zone Target Configuration & Hardware Metrics */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div className="academic-card controls-card">
            <div className="card-header">
              <h3>Target Zone Calibration</h3>
            </div>
            <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div className="control-group">
                <label htmlFor="zone-select">Assign Inspection to Zone</label>
                <select
                  id="zone-select"
                  className="select-input"
                  value={selectedZone.id}
                  onChange={(e) => onSelectZone(e.target.value)}
                >
                  {zones.map(z => (
                    <option key={z.id} value={z.id}>{z.name} ({z.floor})</option>
                  ))}
                </select>
                <span className="help-text">Uploaded images or video will sync occupancy directly to this hall.</span>
              </div>

              <div className="control-group">
                <label>Active Zone Specs</label>
                <div style={{ fontSize: '12px', background: 'var(--bg-card-alt)', padding: '10px', border: '1px solid var(--border-light)', borderRadius: 'var(--radius-sm)' }}>
                  <div>Floor: <strong>{selectedZone.floor}</strong></div>
                  <div>Category: <strong>{selectedZone.type}</strong></div>
                  <div>Current Seats: <strong>{selectedZone.availSeats} available / {selectedZone.totalSeats} capacity</strong></div>
                </div>
              </div>
            </div>
          </div>

          <div className="academic-card">
            <div className="card-header">
              <h3>System Diagnostics</h3>
            </div>
            <div className="card-body">
              <table className="telemetry-table">
                <tbody>
                  <tr>
                    <td>Target Classes</td>
                    <td>Person (0), Chair (56), Table (60)</td>
                  </tr>
                  <tr>
                    <td>IoU Threshold</td>
                    <td>0.30 (Overlap Matching)</td>
                  </tr>
                  <tr>
                    <td>Video Extraction</td>
                    <td>Evenly Sampled Frames (Max 24)</td>
                  </tr>
                  <tr>
                    <td>Image Decoding</td>
                    <td>OpenCV + AVIF/HEIC Native</td>
                  </tr>
                  <tr>
                    <td>Local Service</td>
                    <td>{apiBase}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

