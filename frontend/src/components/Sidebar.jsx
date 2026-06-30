import { NavLink } from 'react-router-dom';

const navItems = [
  { section: 'Overview' },
  { path: '/', icon: '', label: 'Home' },
  { path: '/models', icon: '', label: 'Models' },
  { path: '/comparison', icon: '', label: 'Comparison' },
  { section: 'Data' },
  { path: '/datasets', icon: '', label: 'Datasets & Results' },
  { path: '/biomechanics', icon: '', label: 'Biomechanics' },
  { section: 'Models' },
  { path: '/models/svm', icon: '', label: 'SVM (RBF)' },
  { path: '/models/xgboost', icon: '', label: 'XGBoost' },
  { path: '/models/random_forest', icon: '', label: 'Random Forest' },
  { path: '/models/mlp', icon: '', label: 'MLP' },
  { path: '/models/lightgbm', icon: '', label: 'LightGBM' },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <h2>GaitML Platform</h2>
        <span>Biomechanical Analysis</span>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item, i) => {
          if (item.section) {
            return (
              <div key={i} className="sidebar-section-label">
                {item.section}
              </div>
            );
          }
          return (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === '/'}
              className={({ isActive }) =>
                `sidebar-link${isActive ? ' active' : ''}`
              }
            >
              <span className="icon">{item.icon}</span>
              {item.label}
            </NavLink>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        Gait ML Platform v1.0
      </div>
    </aside>
  );
}
