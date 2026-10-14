import { useCallback, useEffect, useRef, useState } from 'react'
import SeatAvailability from './SeatAvailability.jsx'

const VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
const VIDEO_MAX_FRAMES = 24

function isVideoFile(file) {
  if (!file) return false
  if (file.type?.startsWith('video/')) return true
  const name = file.name?.toLowerCase() || ''
  return VIDEO_EXTS.some((ext) => name.endsWith(ext))
}

function agoText(ts) {
  if (!ts) return null
  const secs = Math.max(0, Math.round((Date.now() - ts) / 1000))
  if (secs < 2) return 'just now'
  if (secs < 60) return `${secs}s ago`
  return `${Math.round(secs / 60)}m ago`
}

/**
 * The whole app: upload a photo or video, get a real seat-availability
 * reading for it. There is no fixed seat layout and no per-seat calibration
 * — the hero number is estimated directly from /detect's own chair/person
 * counts for whatever was just checked, so it's honest about what it
 * actually saw instead of pretending to know where your room's seats are.
 */
export default function DetectPanel({ apiBase, selectedZoneName = 'Main Reading Hall', onScanSuccess }) {
  const [kind, setKind] = useState('image') // 'image' | 'video'
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [videoResult, setVideoResult] = useState(null)
  const [activeFrame, setActiveFrame] = useState(0)
  const [fileName, setFileName] = useState(null)
  const [checkedAt, setCheckedAt] = useState(null)
  const [conf, setConf] = useState(0.25)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)
  const [, forceTick] = useState(0)

  // re-render once a second so the "checked Xs ago" caption ages
  useEffect(() => {
    const id = setInterval(() => forceTick((n) => n + 1), 1000)
    return () => clearInterval(id)
  }, [])

  const submitImage = useCallback(async (file) => {
    setBusy(true); setError(null)
    setResult(null); setVideoResult(null)
    setFileName(file.name)
    try {
      const form = new FormData(); form.append('file', file)
      const res = await fetch(`${apiBase}/detect?conf=${conf}`, { method: 'POST', body: form })
      if (!res.ok) throw new Error(`server responded ${res.status}`)
      const data = await res.json()
      setResult(data)
      setCheckedAt(Date.now())
      if (onScanSuccess) {
        onScanSuccess(data)
      }
    } catch (e) {
      setError(e.message || 'detection failed')
    } finally {
      setBusy(false)
    }
  }, [apiBase, conf, onScanSuccess])

  const submitVideo = useCallback(async (file) => {
    setBusy(true); setError(null)
    setResult(null); setVideoResult(null)
    setFileName(file.name)
    setActiveFrame(0)
    try {
      const form = new FormData(); form.append('file', file)
      const res = await fetch(`${apiBase}/detect/video?conf=${conf}&max_frames=${VIDEO_MAX_FRAMES}`, {
        method: 'POST', body: form,
      })
      if (!res.ok) {
        const body = await res.json().catch(() => null)
        throw new Error(body?.detail || `server responded ${res.status}`)
      }
      const data = await res.json()
      setVideoResult(data)
      setCheckedAt(Date.now())
      if (onScanSuccess && data.timeline?.length > 0) {
        // Feed the first or peak frame to update zone occupancy
        const peakFrame = data.timeline.reduce((p, c) => (c.person_count > p.person_count ? c : p), data.timeline[0])
        onScanSuccess(peakFrame)
      }
    } catch (e) {
      setError(e.message || 'video detection failed')
    } finally {
      setBusy(false)
    }
  }, [apiBase, conf, onScanSuccess])

  const submit = useCallback((file) => {
    if (!file) return
    if (isVideoFile(file)) {
      setKind('video')
      submitVideo(file)
    } else {
      setKind('image')
      submitImage(file)
    }
  }, [submitImage, submitVideo])

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) submit(file)
  }, [submit])

  const frame = videoResult?.timeline?.[activeFrame]
  const current = kind === 'video' ? frame : result
  const everChecked = checkedAt !== null

  return (
    <div className="detection-stage">
      {/* Target & Mode Selector Row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
        <div style={{ display: 'flex', gap: '6px' }}>
          <button
            className={`btn-secondary ${kind === 'image' ? 'active' : ''}`}
            onClick={() => { setKind('image'); setError(null) }}
          >
            📷 Static Frame / Photo
          </button>
          <button
            className={`btn-secondary ${kind === 'video' ? 'active' : ''}`}
            onClick={() => { setKind('video'); setError(null) }}
          >
            🎥 Surveillance Video Clip
          </button>
        </div>

        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
          Assigned Destination: <strong>{selectedZoneName}</strong>
        </div>
      </div>

      {/* Upload Dropzone */}
      <div
        className={`univ-dropzone ${dragging ? 'dragover' : ''} ${busy ? 'busy' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept={kind === 'video' ? 'video/*' : 'image/*'}
          hidden
          onChange={(e) => submit(e.target.files?.[0])}
        />
        <div className="dropzone-icon">📁</div>
        <div className="dropzone-title">
          {busy
            ? (kind === 'video' ? 'Processing frames on GPU…' : 'Executing YOLO Spatial Inference…')
            : `Click to select or drop ${kind === 'video' ? 'a CCTV video recording' : 'a room inspection photo'}`}
        </div>
        <div className="dropzone-sub">
          {kind === 'video'
            ? 'Accepts MP4, MOV, AVI, MKV, WebM (Automatically samples up to 24 timeline checkpoints)'
            : 'Accepts JPEG, PNG, WebP, AVIF, HEIC, BMP (Native automated decoding)'}
        </div>
      </div>

      {/* Parameter Fine Tuning */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px', background: 'var(--bg-card-alt)', padding: '10px 14px', border: '1px solid var(--border-light)', borderRadius: 'var(--radius-sm)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flex: 1, maxWidth: '400px' }}>
          <label htmlFor="conf-slider" style={{ fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', color: 'var(--text-muted)' }}>
            Confidence Gate:
          </label>
          <input
            id="conf-slider"
            type="range" min="0.05" max="0.8" step="0.05"
            value={conf}
            onChange={(e) => setConf(parseFloat(e.target.value))}
            style={{ flex: 1, accentColor: 'var(--univ-blue)' }}
          />
          <span style={{ fontSize: '12px', fontWeight: '700', fontVariantNumeric: 'tabular-nums', width: '32px' }}>
            {conf.toFixed(2)}
          </span>
        </div>
        <span style={{ fontSize: '11px', color: 'var(--text-light)' }}>
          Default: 0.25 (Higher reduces false chair matches)
        </span>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="alert-box error">
          ⚠️ <strong>System Alert:</strong> {error} — Please verify FastAPI server at <code>{apiBase}</code> is active.
        </div>
      )}

      {/* Analysis Headline when checked */}
      {everChecked && current && (
        <div style={{ border: '1px solid var(--border-light)', borderRadius: 'var(--radius-sm)', padding: '12px', background: 'var(--bg-card-alt)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
          <div>
            <div style={{ fontSize: '13px', fontWeight: '700', color: 'var(--text-main)' }}>
              Scan Result: {current.available_chair_count ?? Math.max(0, (current.chair_count || 0) - (current.person_count || 0))} Available / {current.chair_count || 0} Total Chairs
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              File: <strong>{fileName}</strong> · Analyzed {agoText(checkedAt)}
              {kind === 'video' && videoResult && (
                <> · Frame {activeFrame + 1} of {videoResult.timeline.length}</>
              )}
            </div>
          </div>
          <div className="class-tags">
            <span className="tag count-badge">{current.n_detections} Total BBoxes</span>
            <span className="tag person">{current.person_count ?? 0} Persons</span>
            <span className="tag chair">{current.chair_count ?? current.counts?.chair ?? 0} Chairs</span>
            <span className="tag table">{current.table_count ?? current.counts?.table ?? 0} Tables</span>
          </div>
        </div>
      )}

      {/* Image Inspection View */}
      {kind === 'image' && result && (
        <div className="detection-image-container">
          <img src={result.annotated_image} alt="Inference Inspection Output" />
        </div>
      )}

      {/* Video Inspection View & Frame Timeline */}
      {kind === 'video' && videoResult && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <span className="tag count-badge">Video Summary</span>
            <span className="tag person">Peak: {videoResult.peak_person_count} Persons</span>
            <span className="tag">Avg: {videoResult.avg_person_count} Persons/Frame</span>
            <span className="tag" style={{ color: 'var(--text-muted)' }}>{videoResult.sampled_frames} Sampled Frames</span>
            {videoResult.duration_sec != null && (
              <span className="tag" style={{ color: 'var(--text-muted)' }}>{videoResult.duration_sec.toFixed(1)}s Duration</span>
            )}
          </div>

          {frame && (
            <div className="detection-image-container">
              <img src={frame.annotated_image} alt={`Frame ${frame.frame_index}`} />
            </div>
          )}

          {videoResult.timeline.length > 1 && (
            <div>
              <div style={{ fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '4px' }}>
                Extracted Timeline Checkpoints (Click frame to inspect):
              </div>
              <div className="timeline-strip">
                {videoResult.timeline.map((f, i) => (
                  <div
                    key={f.frame_index}
                    className={`timeline-frame ${i === activeFrame ? 'selected' : ''}`}
                    onClick={() => setActiveFrame(i)}
                    title={f.timestamp_sec != null ? `t=${f.timestamp_sec.toFixed(1)}s` : `Frame ${f.frame_index}`}
                  >
                    <img src={f.annotated_image} alt={`frame ${f.frame_index}`} />
                    <span className="timeline-frame-label">
                      {f.timestamp_sec != null ? `${f.timestamp_sec.toFixed(1)}s` : `#${i + 1}`}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {!everChecked && !busy && (
        <div className="empty-state">
          <div className="empty-icon">📷</div>
          <div className="empty-title">Ready for Hall Inspection</div>
          <div className="empty-desc">
            Upload room footage to audit seat occupancy with YOLO spatial overlap analysis. Results will automatically update student vacancies.
          </div>
        </div>
      )}
    </div>
  )
}
