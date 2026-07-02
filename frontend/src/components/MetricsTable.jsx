const METRIC_COLS = [
  { key: 'name_model',     altKey: 'model_name', label: 'Model Run' },
  { key: 'num_features',   label: '# Features' },
  { key: 'cv_accuracy',    label: 'CV Accuracy', fmt: 'pct' },
  { key: 'test_accuracy',  label: 'Test Accuracy', fmt: 'pct' },
  { key: 'sensitivity',    label: 'Sensitivity', fmt: 'pct' },
  { key: 'specificity',    label: 'Specificity', fmt: 'pct' },
  { key: 'f1',             label: 'F1 Score', fmt: 'pct' },
  { key: 'mcc',            label: 'MCC', fmt: 'dec' },
  { key: 'ppv',            label: 'PPV', fmt: 'pct' },
  { key: 'npv',            label: 'NPV', fmt: 'pct' },
  { key: 'parameters',     label: 'Parameters', fmt: 'params' },
];

const PARAM_FIELDS = [
  { key: 'c_value', label: 'C' },
  { key: 'gamma_value', label: 'Gamma' },
  { key: 'n_estimators', label: 'Trees' },
  { key: 'max_depth', label: 'Depth' },
  { key: 'learning_rate', label: 'LR' },
  { key: 'subsample', label: 'Subsample' },
  { key: 'colsample_bytree', label: 'Colsample' },
  { key: 'min_samples_split', label: 'Min split' },
  { key: 'min_samples_leaf', label: 'Min leaf' },
  { key: 'max_features', label: 'Max features' },
  { key: 'architecture', label: 'Arch' },
  { key: 'cell_type', label: 'Cell' },
  { key: 'hidden_size', label: 'Hidden size' },
  { key: 'num_layers', label: 'Layers' },
  { key: 'bidirectional', label: 'BiDir' },
  { key: 'hidden_layer_sizes', label: 'Hidden' },
  { key: 'batch_size', label: 'Batch' },
  { key: 'dropout', label: 'Dropout' },
  { key: 'learning_rate_init', label: 'LR' },
  { key: 'weight_decay', label: 'Weight decay' },
  { key: 'best_epoch_median', label: 'Best epoch' },
  { key: 'num_leaves', label: 'Leaves' },
  { key: 'min_child_samples', label: 'Min child' },
  { key: 'reg_alpha', label: 'Reg alpha' },
  { key: 'reg_lambda', label: 'Reg lambda' },
  { key: 'boosting_type', label: 'Boosting' },
];

function fmtValue(val, fmt) {
  if (val === null || val === undefined) return '-';
  if (fmt === 'pct') return (val * 100).toFixed(1) + '%';
  if (fmt === 'dec') return Number(val).toFixed(3);
  return String(val);
}

function fmtParamValue(value) {
  if (Array.isArray(value)) return `[${value.join(', ')}]`;
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : Number(value).toPrecision(4);
  return String(value);
}

function formatParameters(row) {
  const parts = PARAM_FIELDS
    .filter(({ key }) => row[key] !== null && row[key] !== undefined && row[key] !== '')
    .map(({ key, label }) => `${label}: ${fmtParamValue(row[key])}`);

  return parts.length ? parts.join(' | ') : '-';
}

export default function MetricsTable({ rows = [], highlightBest = true }) {
  if (!rows.length) {
    return <div className="empty-state"><p>No metrics data available.</p></div>;
  }

  return (
    <div className="data-table-wrapper">
      <table className="data-table">
        <thead>
          <tr>
            <th>#</th>
            {METRIC_COLS.map(col => (
              <th key={col.key}>{col.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              <td style={{ color: 'var(--accent-blue)', fontWeight: 700 }}>
                {i + 1}
              </td>
              {METRIC_COLS.map(col => {
                let val = row[col.key];
                if ((val === null || val === undefined) && col.altKey) {
                  val = row[col.altKey];
                }
                const isTop = highlightBest && i === 0 && col.fmt && col.fmt !== 'params';
                return (
                  <td
                    key={col.key}
                    className={isTop ? 'highlight' : ''}
                  >
                    {col.fmt === 'params' ? formatParameters(row) : fmtValue(val, col.fmt)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}