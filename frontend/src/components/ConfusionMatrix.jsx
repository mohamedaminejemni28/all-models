/**
 * ConfusionMatrix — renders a 2×2 confusion matrix from a string like
 * "[[tn, fp], [fn, tp]]" or "[[40, 5], [3, 52]]"
 */

function parseMatrix(raw) {
  if (!raw) return null;
  try {
    // Handle numpy-style string: "[[40  5]\n [ 3 52]]"
    const cleaned = raw
      .replace(/\n/g, ',')
      .replace(/\s+/g, ',')
      .replace(/\[,/g, '[')
      .replace(/,\]/g, ']')
      .replace(/,,+/g, ',');
    const parsed = JSON.parse(cleaned);
    if (Array.isArray(parsed) && parsed.length === 2) {
      return { tn: parsed[0][0], fp: parsed[0][1], fn: parsed[1][0], tp: parsed[1][1] };
    }
  } catch {
    // Try regex fallback
    const nums = raw.match(/\d+/g);
    if (nums && nums.length === 4) {
      return { tn: +nums[0], fp: +nums[1], fn: +nums[2], tp: +nums[3] };
    }
  }
  return null;
}

export default function ConfusionMatrix({ raw, classes = ['Negative', 'Positive'] }) {
  const cm = parseMatrix(raw);
  if (!cm) {
    return <div className="empty-state"><p>No confusion matrix data.</p></div>;
  }

  return (
    <div style={{ display: 'inline-block' }}>
      <div className="confusion-matrix">
        {/* Header row */}
        <div className="cm-header" />
        <div className="cm-header">Pred: {classes[0]}</div>
        <div className="cm-header">Pred: {classes[1]}</div>

        {/* Row 1: Actual Negative */}
        <div className="cm-header" style={{ writingMode: 'vertical-lr', transform: 'rotate(180deg)' }}>
          Act: {classes[0]}
        </div>
        <div className="cm-cell tn">{cm.tn}</div>
        <div className="cm-cell fp">{cm.fp}</div>

        {/* Row 2: Actual Positive */}
        <div className="cm-header" style={{ writingMode: 'vertical-lr', transform: 'rotate(180deg)' }}>
          Act: {classes[1]}
        </div>
        <div className="cm-cell fn">{cm.fn}</div>
        <div className="cm-cell tp">{cm.tp}</div>
      </div>

      <div style={{
        display: 'flex', gap: '1rem', marginTop: '0.75rem', fontSize: '0.72rem',
        color: 'var(--text-muted)', flexWrap: 'wrap',
      }}>
        <span><span style={{ color: '#22d3ee' }}>■</span> TN = {cm.tn}</span>
        <span><span style={{ color: '#fb7185' }}>■</span> FP = {cm.fp}</span>
        <span><span style={{ color: '#fbbf24' }}>■</span> FN = {cm.fn}</span>
        <span><span style={{ color: '#34d399' }}>■</span> TP = {cm.tp}</span>
      </div>
    </div>
  );
}
