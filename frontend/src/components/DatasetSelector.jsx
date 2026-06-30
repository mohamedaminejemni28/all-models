const DATASET_LABELS = {
  autism: 'Autism 2024',
  young_old: 'Young vs Older 2024',
  flatfoot: 'Flatfoot (Older) 2024',
};

export default function DatasetSelector({ selected, onChange, datasets = null }) {
  const items = datasets || Object.keys(DATASET_LABELS);

  return (
    <div className="dataset-selector">
      {items.map(ds => {
        const dsId = typeof ds === 'string' ? ds : ds.id;
        const label = DATASET_LABELS[dsId] || dsId;
        return (
          <button
            key={dsId}
            className={`dataset-btn${selected === dsId ? ' active' : ''}`}
            onClick={() => onChange(dsId)}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}
