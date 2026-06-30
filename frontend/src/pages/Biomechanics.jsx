import { useState } from 'react';

const mediaItems = [
  {
    type: 'image',
    src: '/biomecanics/intro-to-gait-analysis.png',
    title: 'The Gait Cycle',
    description:
      'An overview of the human gait cycle showing stance and swing phases — from initial contact through toe-off to the next initial contact.',
  },
  {
    type: 'image',
    src: '/biomecanics/Visual3D.jpg',
    title: 'Visual3D Skeletal Model',
    description:
      'A biomechanical skeleton rendered in Visual3D v5, used for 3D motion analysis and joint kinematics computation during gait trials.',
  },
  {
    type: 'image',
    src: '/biomecanics/Screenshot 2026-06-29 162225.png',
    title: 'Motion Capture Marker Model',
    description:
      'A 3D marker-based motion capture model showing reflective marker placement for foot and lower limb tracking.',
  },
  {
    type: 'image',
    src: '/biomecanics/Screenshot 2026-06-29 162146.png',
    title: 'Foot Marker Placement',
    description:
      'Reflective markers placed on the foot alongside a skeletal reference model, used for biomechanical gait data collection.',
  },
  {
    type: 'image',
    src: '/biomecanics/Screenshot 2026-06-29 162204.png',
    title: 'Lateral Foot Marker View',
    description:
      'A lateral perspective of the foot with retroreflective markers for capturing joint angles during walking trials.',
  },
  {
    type: 'video',
    src: '/biomecanics/mediahandler.mp4',
    title: 'Gait Analysis Recording',
    description:
      'A recorded gait analysis session demonstrating the motion capture process for biomechanical assessment.',
  },
];

export default function Biomechanics() {
  const [lightbox, setLightbox] = useState(null);

  return (
    <div className="slide-up">
      {/* Page header */}
      <div className="page-header">
        <h1>Biomechanics Gallery</h1>
        <p>
          Visual resources from gait biomechanics research — motion capture
          setups, marker placements, skeletal models, and recorded analysis
          sessions.
        </p>
      </div>

      {/* Overview stats */}
      <div className="card-grid card-grid-3" style={{ marginBottom: '2rem' }}>
        <div className="metric-card slide-up-delay-1">
          <span className="metric-label">Images</span>
          <span className="metric-value accent-blue">
            {mediaItems.filter((m) => m.type === 'image').length}
          </span>
          <span className="metric-sub">Photographs &amp; diagrams</span>
        </div>
        <div className="metric-card slide-up-delay-2">
          <span className="metric-label">Videos</span>
          <span className="metric-value accent-cyan">
            {mediaItems.filter((m) => m.type === 'video').length}
          </span>
          <span className="metric-sub">Recorded sessions</span>
        </div>
        <div className="metric-card slide-up-delay-3">
          <span className="metric-label">Total Media</span>
          <span className="metric-value accent-emerald">
            {mediaItems.length}
          </span>
          <span className="metric-sub">Gallery items</span>
        </div>
      </div>

      {/* Gallery */}
      <div className="bio-gallery">
        {mediaItems.map((item, i) => (
          <div
            key={i}
            className={`bio-gallery-item slide-up slide-up-delay-${(i % 4) + 1}`}
            onClick={() => item.type === 'image' && setLightbox(item)}
          >
            {item.type === 'image' ? (
              <div className="bio-gallery-media">
                <img src={item.src} alt={item.title} loading="lazy" />
                <div className="bio-gallery-overlay">
                  <span className="bio-zoom-icon">🔍</span>
                </div>
              </div>
            ) : (
              <div className="bio-gallery-media bio-video-wrapper">
                <video controls preload="metadata" poster="">
                  <source src={item.src} type="video/mp4" />
                  Your browser does not support the video tag.
                </video>
              </div>
            )}
            <div className="bio-gallery-info">
              <div className="bio-gallery-badge">
                <span className={`badge ${item.type === 'image' ? 'badge-blue' : 'badge-cyan'}`}>
                  {item.type === 'image' ? '📷 Image' : '🎬 Video'}
                </span>
              </div>
              <h3 className="bio-gallery-title">{item.title}</h3>
              <p className="bio-gallery-desc">{item.description}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Lightbox */}
      {lightbox && (
        <div className="lightbox-overlay" onClick={() => setLightbox(null)}>
          <div
            className="bio-lightbox-content"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="bio-lightbox-close"
              onClick={() => setLightbox(null)}
            >
              ✕
            </button>
            <img src={lightbox.src} alt={lightbox.title} />
            <div className="bio-lightbox-caption">
              <h3>{lightbox.title}</h3>
              <p>{lightbox.description}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
