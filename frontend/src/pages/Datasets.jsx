import { useState, useEffect } from 'react';
import { getDatasets, getExperiments } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';

export default function Datasets() {
  const [datasets, setDatasets] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([getDatasets(), getExperiments()])
      .then(([dsData, expData]) => {
        setDatasets(dsData);
        setExperiments(expData);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <LoadingSpinner />;
  if (error) return <div className="empty-state">Error: {error}</div>;

  return (
    <div className="slide-up">
      <div className="page-header">
        <h1>Datasets & Results</h1>
        <p>Information about the underlying data and available experiment result files.</p>
      </div>

      <h2 style={{ fontSize: '1.4rem', marginBottom: '1rem', color: 'var(--text-heading)' }}>Available Datasets</h2>
      <div className="card-grid card-grid-3" style={{ marginBottom: '3rem' }}>
        {datasets.map((ds, i) => (
          <div key={ds.id} className={`card slide-up-delay-${i + 1}`}>
            <h3 style={{ marginBottom: '0.5rem', color: 'var(--accent-blue)' }}>{ds.name}</h3>
            <p style={{ fontSize: '0.85rem', marginBottom: '1rem', minHeight: '40px' }}>{ds.description}</p>

            <div style={{ marginBottom: '1rem', padding: '10px', background: 'rgba(0,0,0,0.2)', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '4px' }}>CLASSES</div>
              <div style={{ display: 'flex', gap: '1rem' }}>
                {Object.entries(ds.classes).map(([val, name]) => (
                  <span key={val} style={{ fontSize: '0.85rem' }}>
                    <strong style={{ color: 'var(--text-primary)' }}>{val}:</strong> {name}
                  </span>
                ))}
              </div>
            </div>

            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '4px' }}>MODELS EVALUATED</div>
              <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                {ds.models_available.map(m => (
                  <span key={m} className="badge badge-emerald">{m}</span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>

      <h2 style={{ fontSize: '1.4rem', marginBottom: '1rem', color: 'var(--text-heading)' }}>Raw Experiment Files Discovered</h2>
      <div className="card slide-up-delay-3">
        <div className="data-table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Model</th>
                <th>Category</th>
                <th>Filename</th>
                <th>Size (KB)</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((f, i) => (
                <tr key={i}>
                  <td><span className="badge badge-blue">{f.dataset}</span></td>
                  <td style={{ textTransform: 'capitalize' }}>{f.model.replace('_', ' ')}</td>
                  <td>{f.category}</td>
                  <td style={{ fontFamily: 'monospace', fontSize: '0.8rem', color: 'var(--accent-cyan)' }}>{f.filename}</td>
                  <td>{f.size_bytes ? (f.size_bytes / 1024).toFixed(1) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
