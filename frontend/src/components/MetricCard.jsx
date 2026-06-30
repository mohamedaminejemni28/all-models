export default function MetricCard({ label, value, sub, accent = '' }) {
  const formatted =
    value === null || value === undefined
      ? '—'
      : typeof value === 'number'
        ? value >= 1
          ? value.toFixed(2)
          : (value * 100).toFixed(1) + '%'
        : value;

  return (
    <div className="metric-card slide-up">
      <span className="metric-label">{label}</span>
      <span className={`metric-value ${accent}`}>{formatted}</span>
      {sub && <span className="metric-sub">{sub}</span>}
    </div>
  );
}
