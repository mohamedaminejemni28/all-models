import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer,
} from 'recharts';

const METRIC_KEYS = [
  { key: 'cv_accuracy',   label: 'CV Accuracy' },
  { key: 'test_accuracy',  label: 'Test Accuracy' },
  { key: 'sensitivity',    label: 'Sensitivity' },
  { key: 'specificity',    label: 'Specificity' },
  { key: 'f1',             label: 'F1 Score' },
  { key: 'ppv',            label: 'PPV' },
  { key: 'npv',            label: 'NPV' },
];

export function ComparisonBarChart({ models = [] }) {
  if (!models.length) return null;

  const data = METRIC_KEYS.map(({ key, label }) => {
    const entry = { metric: label };
    models.forEach(m => {
      entry[m.model_name] = m[key] !== null && m[key] !== undefined
        ? +(m[key] * 100).toFixed(1)
        : 0;
    });
    return entry;
  });

  return (
    <div className="chart-container slide-up">
      <h3>Model Comparison — Key Metrics (%)</h3>
      <ResponsiveContainer width="100%" height={380}>
        <BarChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
          <XAxis
            dataKey="metric"
            tick={{ fontSize: 11 }}
            angle={-20}
            textAnchor="end"
            height={60}
          />
          <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
          <Tooltip
            contentStyle={{
              background: '#ffffff',
              border: '1px solid rgba(0,0,0,0.1)',
              borderRadius: 8,
              fontSize: 13,
              boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
            }}
          />
          <Legend />
          {models.map((m, i) => (
            <Bar
              key={m.model_id}
              dataKey={m.model_name}
              fill={m.color}
              radius={[4, 4, 0, 0]}
              barSize={28}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ComparisonRadarChart({ models = [] }) {
  if (!models.length) return null;

  const data = METRIC_KEYS.map(({ key, label }) => {
    const entry = { metric: label };
    models.forEach(m => {
      entry[m.model_name] = m[key] !== null && m[key] !== undefined
        ? +(m[key] * 100).toFixed(1)
        : 0;
    });
    return entry;
  });

  return (
    <div className="chart-container slide-up">
      <h3>Radar Overview</h3>
      <ResponsiveContainer width="100%" height={380}>
        <RadarChart data={data} cx="50%" cy="50%" outerRadius="75%">
          <PolarGrid stroke="rgba(0,0,0,0.08)" />
          <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: '#5a5a7a' }} />
          <PolarRadiusAxis
            angle={30}
            domain={[0, 100]}
            tick={{ fontSize: 10, fill: '#9494b0' }}
          />
          {models.map((m) => (
            <Radar
              key={m.model_id}
              name={m.model_name}
              dataKey={m.model_name}
              stroke={m.color}
              fill={m.color}
              fillOpacity={0.12}
              strokeWidth={2}
            />
          ))}
          <Legend />
          <Tooltip
            contentStyle={{
              background: '#ffffff',
              border: '1px solid rgba(0,0,0,0.1)',
              borderRadius: 8,
              fontSize: 13,
              boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
            }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
