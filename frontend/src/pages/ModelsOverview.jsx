import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getModels } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';

export default function ModelsOverview() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    getModels()
      .then(data => {
        setModels(data);
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
        <h1>Machine Learning Models</h1>
        <p>Overview of the classification models evaluated in this study.</p>
      </div>

      <div className="card-grid card-grid-3">
        {models.map((model, i) => (
          <div
            key={model.id}
            className={`card model-card slide-up-delay-${i + 1}`}
            onClick={() => navigate(`/models/${model.id}`)}
          >
            <div className="model-icon">{model.icon}</div>
            <h3 className="model-name" style={{ color: model.color }}>{model.name}</h3>
            <p className="model-desc">{model.description}</p>

            <div style={{ marginTop: '1.5rem' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Available Datasets
              </div>
              <div className="model-datasets">
                {model.datasets_available.map(ds => (
                  <span key={ds} className="badge badge-blue">{ds}</span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
