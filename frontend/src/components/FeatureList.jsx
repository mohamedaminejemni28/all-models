export default function FeatureList({ features = [] }) {
  if (!features.length) {
    return (
      <div className="empty-state">
        <div className="empty-icon">📋</div>
        <p>No feature data available.</p>
      </div>
    );
  }

  return (
    <ul className="feature-list">
      {features.map((f, i) => (
        <li key={i} className="slide-up" style={{ animationDelay: `${i * 0.03}s` }}>
          <span className="feature-rank">{f.rank}</span>
          <span className="feature-name">{f.name}</span>
          {f.score !== null && f.score !== undefined && (
            <span style={{
              marginLeft: 'auto',
              color: 'var(--accent-cyan)',
              fontSize: '0.82rem',
              fontWeight: 600,
            }}>
              {(f.score * 100).toFixed(1)}%
            </span>
          )}
        </li>
      ))}
    </ul>
  );
}
