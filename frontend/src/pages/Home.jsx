import { Link } from 'react-router-dom';

export default function Home() {
  return (
    <div className="slide-up home-page">
      {/* Hero with gait cycle below */}
      <section className="hero hero-with-bg">
        <div className="hero-content">
          <h1>Biomechanical Gait Analysis<br />ML Results Platform</h1>
          <p>
            Visualize, compare, and explain the performance of six machine learning models
            (SVM, XGBoost, Random Forest, MLP, LightGBM, and LSTM/GRU) across multiple gait biomechanics datasets.
          </p>
          <div className="hero-actions">
            <Link to="/models" className="btn btn-primary">Explore Models</Link>
            <Link to="/comparison" className="btn btn-outline">Compare Performance</Link>
          </div>
        </div>
        <div className="hero-image-strip">
          <img
            src="/biomecanics/intro-to-gait-analysis.png"
            alt="Gait Cycle"
          />
          <div className="hero-image-fade-top" />
        </div>
      </section>

      <div className="card-grid card-grid-3">
        <div className="card slide-up-delay-1">
          <h3 style={{ marginBottom: '1rem', color: 'var(--accent-blue)' }}>Datasets</h3>
          <p style={{ fontSize: '0.9rem', marginBottom: '1rem' }}>
            Results are available for three primary datasets:
          </p>
          <ul style={{ paddingLeft: '1.2rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            <li style={{ marginBottom: '0.5rem' }}>Autism Study (Control vs Autism)</li>
            <li style={{ marginBottom: '0.5rem' }}>Age Comparison (Young vs Older)</li>
            <li>Flatfoot (Control vs Pes Planus)</li>
          </ul>
        </div>

        <div className="card slide-up-delay-2">
          <h3 style={{ marginBottom: '1rem', color: 'var(--accent-cyan)' }}>Methodology</h3>
          <p style={{ fontSize: '0.9rem' }}>
            The pipeline employs Sequential Feature Selection (SFS) to identify
            optimal feature subsets, followed by hyperparameter tuning and model
            evaluation using cross-validation.
          </p>
        </div>

        <div className="card slide-up-delay-3">
          <h3 style={{ marginBottom: '1rem', color: 'var(--accent-emerald)' }}>Interpretability</h3>
          <p style={{ fontSize: '0.9rem' }}>
            SHAP (SHapley Additive exPlanations) values are used to explain model
            predictions, highlighting the most influential biomechanical features
            for each classification task.
          </p>
        </div>
      </div>
    </div>
  );
}
