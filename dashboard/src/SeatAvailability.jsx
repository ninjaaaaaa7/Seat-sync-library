/**
 * The headline "can I get a seat" answer, computed server-side by actually
 * checking which detected chair has a person's box covering it — not a
 * naive chairs-minus-people subtraction, which gets it wrong the moment
 * there are more people in frame than chairs (standing visitors, people
 * clustered near an empty seat, etc). See core/seat_geometry.py's
 * match_chairs_to_occupants and service/app.py's run_detection.
 *
 * chairCount:    total 'chair' detections in the current frame.
 * availableCount: chairs with no person covering them, per the backend.
 * personCount:   total 'person' detections, shown only for context.
 * loading:       true while a check is actively in flight.
 * idle:          true when nothing has ever been checked yet.
 */
export default function SeatAvailability({ chairCount = 0, availableCount = 0, personCount = 0, loading, idle, size = 'hero' }) {
  if (idle) {
    return (
      <div className={`availability ${size} idle`}>
        <div className="avail-number">?</div>
        <div className="avail-sub">Upload a photo or video to check seat availability.</div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className={`availability ${size} loading`}>
        <div className="avail-number">···</div>
        <div className="avail-sub">Checking seat availability…</div>
      </div>
    )
  }

  if (chairCount === 0) {
    return (
      <div className={`availability ${size}`}>
        <div className="avail-number">—</div>
        <div className="avail-sub">
          No chairs detected{personCount > 0 ? ` (saw ${personCount} people, but no seating)` : ''}.
        </div>
      </div>
    )
  }

  const occupiedCount = chairCount - availableCount
  const full = availableCount === 0

  return (
    <div className={`availability ${size} ${full ? 'full' : 'ok'}`}>
      <div className="avail-headline">
        <span className="avail-number">{availableCount}</span>
        <span className="avail-sub">of {chairCount} seats available</span>
      </div>
      <div className="avail-detail">
        {occupiedCount} chair{occupiedCount === 1 ? '' : 's'} occupied · {personCount} person{personCount === 1 ? '' : 's'} detected total
      </div>
    </div>
  )
}
