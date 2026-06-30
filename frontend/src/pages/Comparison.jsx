import { useState, useEffect } from 'react';
import { getComparison } from '../services/api';
import DatasetSelector from '../components/DatasetSelector';
import { ComparisonBarChart, ComparisonRadarChart } from '../components/ComparisonChart';
import MetricsTable from '../components/MetricsTable';
import LoadingSpinner from '../components/LoadingSpinner';

export default function Comparison() {
  const [dataset, setDataset] = useState('autism');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    getComparison(dataset)
      .then(res => {
        setData(res);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [dataset]);

  if (loading) return <LoadingSpinner />;
  if (error) return <div className="empty-state">Error: {error}</div>;

  const models = data?.models || [];

  return (
    <div className="slide-up">
      <div className="page-header">
        <h1>Model Comparison</h1>
        <p>Compare the best performing runs from all available model families side-by-side.</p>
      </div>

      <DatasetSelector selected={dataset} onChange={setDataset} />

      <div className="card-grid card-grid-2" style={{ marginBottom: '2rem' }}>
        <ComparisonBarChart models={models} />
        <ComparisonRadarChart models={models} />
      </div>

      <div className="card slide-up-delay-2">
        <h3 style={{ marginBottom: '1.5rem' }}>Performance Summary</h3>
        <MetricsTable rows={models} highlightBest={false} />
      </div>
    </div>
  );
}
