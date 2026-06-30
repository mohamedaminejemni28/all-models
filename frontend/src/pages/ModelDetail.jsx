import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { getModelMetrics, getModelFeatures, getModelFigures } from '../services/api';
import DatasetSelector from '../components/DatasetSelector';
import MetricCard from '../components/MetricCard';
import MetricsTable from '../components/MetricsTable';
import ConfusionMatrix from '../components/ConfusionMatrix';
import FigureGallery from '../components/FigureGallery';
import FeatureList from '../components/FeatureList';
import LoadingSpinner from '../components/LoadingSpinner';

const TABS = ['Overview', 'Features', 'Interpretability (SHAP)', 'Time Series Plots'];

export default function ModelDetail() {
  const { modelName } = useParams();
  const [dataset, setDataset] = useState('autism');
  const [tab, setTab] = useState('Overview');

  const [metricsData, setMetricsData] = useState(null);
  const [featuresData, setFeaturesData] = useState(null);
  const [figuresData, setFiguresData] = useState(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      getModelMetrics(modelName, dataset),
      getModelFeatures(modelName, dataset),
      getModelFigures(modelName) // Figures endpoint returns all datasets
    ])
      .then(([metrics, features, figures]) => {
        setMetricsData(metrics);
        setFeaturesData(features);
        setFiguresData(figures);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [modelName, dataset]);

  if (loading) return <LoadingSpinner />;
  if (error) return <div className="empty-state">Error: {error}</div>;

  const topModel = metricsData?.top3?.[0];

  // Filter figures for the current dataset
  const currentFigures = figuresData?.figures?.filter(f => f.dataset === dataset) || [];
  const shapFigures = currentFigures.filter(f => f.category === 'shap');
  const scatterFigures = currentFigures.filter(f => f.category === 'scatter');

  return (
    <div className="slide-up">
      <div className="page-header">
        <h1>{metricsData?.model_name || modelName}</h1>
        <p>Detailed performance metrics, selected features, and interpretability.</p>
      </div>

      <DatasetSelector selected={dataset} onChange={setDataset} />

      <div className="tabs">
        {TABS.map(t => (
          <button
            key={t}
            className={`tab-btn${tab === t ? ' active' : ''}`}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </div>

      <div style={{ marginTop: '2rem' }}>
        {tab === 'Overview' && (
          <div className="slide-up">
            <div className="card-grid card-grid-4" style={{ marginBottom: '2rem' }}>
              <MetricCard
                label="Best Test Accuracy"
                value={topModel?.test_accuracy}
                accent="accent-emerald"
              />
              <MetricCard
                label="Sensitivity"
                value={topModel?.sensitivity}
                accent="accent-blue"
              />
              <MetricCard
                label="Specificity"
                value={topModel?.specificity}
                accent="accent-cyan"
              />
              <MetricCard
                label="F1 Score"
                value={topModel?.f1}
                accent="accent-amber"
              />
            </div>

            <div className="card" style={{ marginBottom: '2rem' }}>
              <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem' }}>Top 3 Model Runs</h3>
              <MetricsTable rows={metricsData?.top3 || []} />
            </div>

            {topModel?.confusion_matrix && (
              <div className="card">
                <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem' }}>Confusion Matrix (Best Model)</h3>
                <ConfusionMatrix raw={topModel.confusion_matrix} />
              </div>
            )}
          </div>
        )}

        {tab === 'Features' && (
          <div className="card slide-up">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <h3 style={{ fontSize: '1.1rem' }}>Sequential Feature Selection (SFS)</h3>
              <span className="badge badge-cyan">{featuresData?.features?.length || 0} Features</span>
            </div>
            <FeatureList features={featuresData?.features || []} />
          </div>
        )}

        {tab === 'Interpretability (SHAP)' && (
          <div className="slide-up">
            <div style={{ marginBottom: '1.5rem' }}>
              <p style={{ color: 'var(--text-secondary)' }}>
                SHAP summary plots show the global importance of each feature and how higher or lower
                feature values impact the model's output.
              </p>
            </div>
            <FigureGallery figures={shapFigures} />
          </div>
        )}

        {tab === 'Time Series Plots' && (
          <div className="slide-up">
            <div style={{ marginBottom: '1.5rem' }}>
              <p style={{ color: 'var(--text-secondary)' }}>
                Scatter plots showing the distribution of time-variable features across the dataset.
              </p>
            </div>
            <FigureGallery figures={scatterFigures} />
          </div>
        )}
      </div>
    </div>
  );
}
