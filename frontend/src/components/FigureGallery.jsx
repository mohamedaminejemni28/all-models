import { useState } from 'react';

export default function FigureGallery({ figures = [] }) {
  const [lightbox, setLightbox] = useState(null);

  if (!figures.length) {
    return (
      <div className="empty-state">
        <div className="empty-icon">📸</div>
        <p>No figures available for this selection.</p>
      </div>
    );
  }

  return (
    <>
      <div className="figure-gallery">
        {figures.map((fig, i) => (
          <div
            key={i}
            className="figure-item slide-up"
            style={{ animationDelay: `${i * 0.05}s` }}
            onClick={() => setLightbox(fig.url)}
          >
            <img
              src={fig.url}
              alt={fig.filename}
              loading="lazy"
            />
            <div className="figure-label">
              <strong>{fig.category === 'shap' ? 'SHAP Summary' : 'Scatter Plot'}</strong>
              {' · '}
              {fig.filename}
            </div>
          </div>
        ))}
      </div>

      {lightbox && (
        <div className="lightbox-overlay" onClick={() => setLightbox(null)}>
          <img src={lightbox} alt="Enlarged figure" />
        </div>
      )}
    </>
  );
}
